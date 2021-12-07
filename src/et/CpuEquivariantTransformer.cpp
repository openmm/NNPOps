/**
 * Copyright (c) 2020-2021 Stanford University and the Authors
 * Authors: Peter Eastman
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "CpuEquivariantTransformer.h"
#include <cmath>
#include <cstring>

using namespace std;

template <bool PERIODIC, bool TRICLINIC>
static void computeDisplacement(const float* pos1, const float* pos2, float* delta, float& r2, const float* periodicBoxVectors, const float* invBoxSize) {
    delta[0] = pos2[0]-pos1[0];
    delta[1] = pos2[1]-pos1[1];
    delta[2] = pos2[2]-pos1[2];
    if (PERIODIC) {
        if (TRICLINIC) {
            float scale3 = round(delta[2]*invBoxSize[2]);
            delta[0] -= scale3*periodicBoxVectors[6];
            delta[1] -= scale3*periodicBoxVectors[7];
            delta[2] -= scale3*periodicBoxVectors[8];
            float scale2 = round(delta[1]*invBoxSize[1]);
            delta[0] -= scale2*periodicBoxVectors[3];
            delta[1] -= scale2*periodicBoxVectors[4];
            float scale1 = round(delta[0]*invBoxSize[0]);
            delta[0] -= scale1*periodicBoxVectors[0];
        }
        else {
            delta[0] -= round(delta[0]*invBoxSize[0])*periodicBoxVectors[0];
            delta[1] -= round(delta[1]*invBoxSize[1])*periodicBoxVectors[4];
            delta[2] -= round(delta[2]*invBoxSize[2])*periodicBoxVectors[8];
        }
    }
    r2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
}

CpuEquivariantTransformerNeighbors::CpuEquivariantTransformerNeighbors(int numAtoms, float cutoff, bool periodic) :
            EquivariantTransformerNeighbors(numAtoms, cutoff, periodic) {
    neighbors.resize(numAtoms);
    neighborDistances.resize(numAtoms);
}

void CpuEquivariantTransformerNeighbors::build(const float* positions, const float* periodicBoxVectors) {
    // Determine whether we have a rectangular or triclinic periodic box.
    
    triclinic = false;
    if (getPeriodic())
        for (int i = 0 ; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (i != j && periodicBoxVectors[3*i+j] != 0)
                    triclinic = true;

    // Clear the output arrays.

    for (int i = 0; i < getNumAtoms(); i++) {
        neighbors[i].clear();
        neighborDistances[i].clear();
    }

    // Build the neighbor list.

    if (getPeriodic()) {
        if (triclinic)
            findNeighbors<true, true>(positions, periodicBoxVectors);
        else
            findNeighbors<true, false>(positions, periodicBoxVectors);
    }
    else {
        findNeighbors<false, false>(positions, periodicBoxVectors);
    }
}

template <bool PERIODIC, bool TRICLINIC>
void CpuEquivariantTransformerNeighbors::findNeighbors(const float* positions, const float* periodicBoxVectors) {
    float invBoxSize[3];
    if (getPeriodic()) {
        invBoxSize[0] = 1/periodicBoxVectors[0];
        invBoxSize[1] = 1/periodicBoxVectors[4];
        invBoxSize[2] = 1/periodicBoxVectors[8];
    }

    // Loop over all pairs of atoms using an O(N^2) algorithms.  This is efficient for the small molecules
    // we are currently targeting.  If we want to scale to larger systems, a voxel based algorithm would
    // be more efficient.

    float cutoff2 = getCutoff()*getCutoff();
    for (int atom1 = 0; atom1 < getNumAtoms(); atom1++) {
        neighbors[atom1].push_back(atom1);
        neighborDistances[atom1].push_back(0);
        for (int atom2 = atom1+1; atom2 < getNumAtoms(); atom2++) {
            float delta[3];
            float r2;
            computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta, r2, periodicBoxVectors, invBoxSize);
            if (r2 < cutoff2) {
                neighbors[atom1].push_back(atom2);
                neighbors[atom2].push_back(atom1);
                neighborDistances[atom1].push_back(sqrtf(r2));
                neighborDistances[atom2].push_back(sqrtf(r2));
            }
        }
    }
}

CpuEquivariantTransformerLayer::CpuEquivariantTransformerLayer(int numAtoms, int width, int numHeads, int numRBF, const float* rbfMus,
            const float* rbfBetas, const float* qw, const float* qb, const float* kw, const float* kb, const float* vw, const float* vb,
            const float* ow, const float* ob, const float* uw, const float* ub, const float* dkw, const float* dkb, const float* dvw, const float* dvb) :
            EquivariantTransformerLayer(numAtoms, width, numHeads, numRBF) {
    for (int i = 0; i < numRBF; i++) {
        this->rbfMus.push_back(rbfMus[i]);
        this->rbfBetas.push_back(rbfBetas[i]);
    }
    for (int i = 0; i < width*width; i++) {
        this->qw.push_back(qw[i]);
        this->kw.push_back(kw[i]);
    }
    for (int i = 0; i < 3*width*width; i++) {
        this->vw.push_back(vw[i]);
        this->ow.push_back(ow[i]);
        this->uw.push_back(uw[i]);
    }
    for (int i = 0; i < width*numRBF; i++)
        this->dkw.push_back(dkw[i]);
    for (int i = 0; i < 3*width*numRBF; i++)
        this->dvw.push_back(dvw[i]);
    for (int i = 0; i < width; i++) {
        this->qb.push_back(qb[i]);
        this->kb.push_back(kb[i]);
        this->dkb.push_back(dkb[i]);
    }
    for (int i = 0; i < 3*width; i++) {
        this->vb.push_back(vb[i]);
        this->ob.push_back(ob[i]);
        this->ub.push_back(ub[i]);
        this->dvb.push_back(dvb[i]);
    }
    q.resize(numAtoms);
    k.resize(numAtoms);
    v.resize(numAtoms);
    o.resize(numAtoms);
    u.resize(numAtoms);
    s3.resize(numAtoms);
    s.resize(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        q[i].resize(width);
        k[i].resize(width);
        v[i].resize(3*width);
        o[i].resize(3*width);
        u[i].resize(9*width);
        s3[i].resize(width);
        s[i].resize(3*width);
    }
}

void CpuEquivariantTransformerLayer::compute(const EquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                 const float* x, const float* vec, float* dx, float* dvec) {
    // Apply the various transforms.

    for (int atom = 0; atom < numAtoms; atom++) {
        for (int i = 0; i < width; i++) {
            q[atom][i] = qb[i];
            k[atom][i] = kb[i];
            for (int j = 0; j < width; j++) {
                q[atom][i] += qw[i*width+j]*x[atom*width+j];
                k[atom][i] += kw[i*width+j]*x[atom*width+j];
            }
        }
        for (int i = 0; i < 3*width; i++) {
            v[atom][i] = vb[i];
            for (int j = 0; j < width; j++)
                v[atom][i] += vw[i*width+j]*x[atom*width+j];
        }
        for (int i = 0; i < 3*width; i++) {
            for (int k = 0; k < 3; k++) {
                u[atom][i+3*width*k] = ub[i];
                for (int j = 0; j < width; j++)
                    u[atom][i+3*width*k] += uw[i*width+j]*vec[atom*3*width+k*width+j];
            }
        }
    }

    // Prepare for the loop over interactions.

    const CpuEquivariantTransformerNeighbors& cpuNeighbors = dynamic_cast<const CpuEquivariantTransformerNeighbors&>(neighbors);
    int numRBF = rbfMus.size();
    int headWidth = width/numHeads;
    vector<float> rbf(numRBF), dk(width), dv(3*width), attention(numHeads);
    for (int atom = 0; atom < numAtoms; atom++) {
        for (int i = 0; i < width; i++)
            s3[atom][i] = 0;
        for (int i = 0; i < 3*width; i++)
            s[atom][i] = 0;
    }

    // Loop over pairs of atoms from the neighbor list.

    for (int atom1 = 0; atom1 < numAtoms; atom1++) {
        for (int neighborIndex = 0; neighborIndex < cpuNeighbors.getNeighbors()[atom1].size(); neighborIndex++) {
            int atom2 = cpuNeighbors.getNeighbors()[atom1][neighborIndex];
            float r = cpuNeighbors.getNeighborDistances()[atom1][neighborIndex];
            float rInv = (r > 0 ? 1/r : 0);
            float delta[3] = {(positions[3*atom2]-positions[3*atom1])*rInv,
                              (positions[3*atom2+1]-positions[3*atom1+1])*rInv,
                              (positions[3*atom2+2]-positions[3*atom1+2])*rInv};

            // Compute the radial basis functions.

            float alpha = 5.0f/neighbors.getCutoff();
            float cutoffScale = cutoffFunction(r, neighbors.getCutoff());
            for (int i = 0; i < numRBF; i++) {
                float expTerm = exp(-alpha*r);
                float expDiff = expTerm-rbfMus[i];
                rbf[i] = cutoffScale*exp(-rbfBetas[i]*expDiff*expDiff);
            }

            // Compute the filters.

            for (int i = 0; i < width; i++) {
                dk[i] = dkb[i];
                for (int j = 0; j < numRBF; j++)
                    dk[i] += dkw[i*numRBF+j]*rbf[j];
                dk[i] = dk[i]/(1+exp(-dk[i]));
            }
            for (int i = 0; i < 3*width; i++) {
                dv[i] = dvb[i];
                for (int j = 0; j < numRBF; j++)
                    dv[i] += dvw[i*numRBF+j]*rbf[j];
                dv[i] = dv[i]/(1+exp(-dv[i]));
            }

            // Compute the attention weights.

            for (int head = 0; head < numHeads; head++) {
                attention[head] = 0;
                for (int i = 0; i < headWidth; i++)
                    attention[head] += q[atom1][headWidth*head+i]*k[atom2][headWidth*head+i]*dk[headWidth*head+i];
                attention[head] = cutoffScale*attention[head]/(1+exp(-attention[head]));
            }

            // Compute contributions to the output values.

            for (int head = 0; head < numHeads; head++)
                for (int i = 0; i < headWidth; i++)
                    s3[atom1][headWidth*head+i] += v[atom2][3*head*headWidth+i]*dv[3*head*headWidth+i]*attention[head];
            for (int j = 0; j < 3; j++) {
                for (int head = 0; head < numHeads; head++)
                    for (int i = 0; i < headWidth; i++) {
                        float vec1 = v[atom2][(3*head+1)*headWidth+i]*dv[(3*head+1)*headWidth+i];
                        float vec2 = v[atom2][(3*head+2)*headWidth+i]*dv[(3*head+2)*headWidth+i];
                        s[atom1][width*j+headWidth*head+i] += vec1*vec[3*width*atom2+width*j+headWidth*head+i] + vec2*delta[j];
                    }
                }
        }
    }

    // Compute the final outputs.

    for (int atom = 0; atom < numAtoms; atom++) {
        for (int i = 0; i < 3*width; i++) {
            o[atom][i] = ob[i];
            for (int j = 0; j < width; j++)
                o[atom][i] += ow[i*width+j]*s3[atom][j];
        }
        for (int i = 0; i < width; i++) {
            float vecDot = u[atom][i]*u[atom][i+width] + u[atom][i+3*width]*u[atom][i+4*width] + u[atom][i+6*width]*u[atom][i+7*width];
            dx[width*atom+i] = vecDot*o[atom][width+i] + o[atom][2*width+i];
            for (int j = 0; j < 3; j++)
                dvec[3*width*atom+width*j+i] = u[atom][width*(3*j+2)+i]*o[atom][i] + s[atom][width*j+i];
        }
    }
}

void CpuEquivariantTransformerLayer::backprop(const EquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                  const float* x, const float* vec, const float* dxDeriv, const float* dvecDeriv, float* xDeriv, float* vecDeriv, float* positionDeriv) {
    // Clear the output array.

    memset(xDeriv, 0, numAtoms*width*sizeof(float));
    memset(vecDeriv, 0, numAtoms*3*width*sizeof(float));
    memset(positionDeriv, 0, numAtoms*3*sizeof(float));

    // Backpropagate through the layer.

    const CpuEquivariantTransformerNeighbors& cpuNeighbors = dynamic_cast<const CpuEquivariantTransformerNeighbors&>(neighbors);
    if (neighbors.getPeriodic()) {
        if (neighbors.getTriclinic())
            backpropImpl<true, true>(cpuNeighbors, positions, periodicBoxVectors, x, vec, dxDeriv, dvecDeriv, xDeriv, vecDeriv, positionDeriv);
        else
            backpropImpl<true, false>(cpuNeighbors, positions, periodicBoxVectors, x, vec, dxDeriv, dvecDeriv, xDeriv, vecDeriv, positionDeriv);
    }
    else {
        backpropImpl<false, false>(cpuNeighbors, positions, periodicBoxVectors, x, vec, dxDeriv, dvecDeriv, xDeriv, vecDeriv, positionDeriv);
    }
}

template <bool PERIODIC, bool TRICLINIC>
void CpuEquivariantTransformerLayer::backpropImpl(const CpuEquivariantTransformerNeighbors& neighbors, const float* positions, const float* periodicBoxVectors,
                      const float* x, const float* vec, const float* dxDeriv, const float* dvecDeriv, float* xDeriv, float* vecDeriv, float* positionDeriv) {
    const CpuEquivariantTransformerNeighbors& cpuNeighbors = dynamic_cast<const CpuEquivariantTransformerNeighbors&>(neighbors);
    int numRBF = rbfMus.size();
    int headWidth = width/numHeads;

    // Backprop through the final outputs.

    vector<vector<float> > dEdO(numAtoms, vector<float>(3*width));
    vector<vector<float> > dEdU(numAtoms, vector<float>(9*width));
    vector<vector<float> > dEdS3(numAtoms, vector<float>(width, 0));
    vector<vector<float> > dEdS(numAtoms, vector<float>(3*width));
    for (int atom = 0; atom < numAtoms; atom++) {
        for (int i = 0; i < width; i++) {
            float vecDot = u[atom][i]*u[atom][i+width] + u[atom][i+3*width]*u[atom][i+4*width] + u[atom][i+6*width]*u[atom][i+7*width];
            // dx[width*atom+i] = vecDot*o[atom][width+i] + o[atom][2*width+i];
            float dEdDX = dxDeriv[width*atom+i];
            dEdO[atom][width+i] = dEdDX*vecDot;
            dEdO[atom][2*width+i] = dEdDX;
            float dEdDot = dEdDX*o[atom][width+i];
            dEdU[atom][i] = dEdDot*u[atom][i+width];
            dEdU[atom][i+width] = dEdDot*u[atom][i];
            dEdU[atom][i+3*width] = dEdDot*u[atom][i+4*width];
            dEdU[atom][i+4*width] = dEdDot*u[atom][i+3*width];
            dEdU[atom][i+6*width] = dEdDot*u[atom][i+7*width];
            dEdU[atom][i+7*width] = dEdDot*u[atom][i+6*width];
            for (int j = 0; j < 3; j++) {
                // dvec[3*width*atom+width*j+i] = u[atom][width*(3*j+2)+i]*o[atom][i] + s[atom][width*j+i];
                float dEdDVEC = dvecDeriv[3*width*atom+width*j+i];
                dEdO[atom][i] = dEdDVEC*u[atom][width*(3*j+2)+i];
                dEdU[atom][width*(3*j+2)+i] = dEdDVEC*o[atom][i];
                dEdS[atom][width*j+i] = dEdDVEC;
            }
        }
    }
    for (int atom = 0; atom < numAtoms; atom++) {
        for (int i = 0; i < 3*width; i++) {
            for (int j = 0; j < width; j++)
                // o[atom][i] += ow[i*width+j]*s3[atom][j];
                dEdS3[atom][j] = dEdO[atom][i]*ow[i*width+j];
        }
    }

    // Backprop through the list of interactions from the neighbor list.

    vector<float> rbf(numRBF), dk(width), dv(3*width), attention(numHeads);
    vector<float> dRBFdR(numRBF), dDKdR(width), dDVdR(3*width), dATTENTIONdR(numHeads);
    for (int atom1 = 0; atom1 < numAtoms; atom1++) {
        for (int neighborIndex = 0; neighborIndex < cpuNeighbors.getNeighbors()[atom1].size(); neighborIndex++) {
            int atom2 = cpuNeighbors.getNeighbors()[atom1][neighborIndex];
            float r = cpuNeighbors.getNeighborDistances()[atom1][neighborIndex];
            float rInv = (r > 0 ? 1/r : 0);
            float delta[3] = {(positions[3*atom2]-positions[3*atom1])*rInv,
                              (positions[3*atom2+1]-positions[3*atom1+1])*rInv,
                              (positions[3*atom2+2]-positions[3*atom1+2])*rInv};

            // Compute the radial basis functions.

            float alpha = 5.0f/neighbors.getCutoff();
            float cutoffScale = cutoffFunction(r, neighbors.getCutoff());
            for (int i = 0; i < numRBF; i++) {
                float expTerm = exp(-alpha*r);
                float expDiff = expTerm-rbfMus[i];
                rbf[i] = cutoffScale*exp(-rbfBetas[i]*expDiff*expDiff);
                dRBFdR[i] = rbf[i]*2*rbfBetas[i]*expDiff*expTerm*alpha;
            }

            // Compute the filters.

            for (int i = 0; i < width; i++) {
                dk[i] = dkb[i];
                dDKdR[i] = 0;
                for (int j = 0; j < numRBF; j++) {
                    dk[i] += dkw[i*numRBF+j]*rbf[j];
                    dDKdR[i] += dkw[i*numRBF+j]*dRBFdR[j];
                }
                float expTerm = exp(-dk[i]);
                float scale = 1/(1+expTerm);
                dk[i] = dk[i]*scale;
                dDKdR[i] = (1 + dk[i]*expTerm)*scale*dDKdR[i];
            }
            for (int i = 0; i < 3*width; i++) {
                dv[i] = dvb[i];
                dDVdR[i] = 0;
                for (int j = 0; j < numRBF; j++) {
                    dv[i] += dvw[i*numRBF+j]*rbf[j];
                    dDVdR[i] += dvw[i*numRBF+j]*dRBFdR[j];
                }
                float expTerm = exp(-dv[i]);
                float scale = 1/(1+expTerm);
                dv[i] = dv[i]*scale;
                dDVdR[i] = (1 + dv[i]*expTerm)*scale*dDVdR[i];
            }

            // Compute the attention weights.

            for (int head = 0; head < numHeads; head++) {
                attention[head] = 0;
                dATTENTIONdR[head] = 0;
                for (int i = 0; i < headWidth; i++) {
                    attention[head] += q[atom1][headWidth*head+i]*k[atom2][headWidth*head+i]*dk[headWidth*head+i];
                    dATTENTIONdR[head] += q[atom1][headWidth*head+i]*k[atom2][headWidth*head+i]*dDKdR[headWidth*head+i];
                }
                float expTerm = exp(-attention[head]);
                float scale = 1/(1+expTerm);
                attention[head] = cutoffScale*attention[head]*scale;
                dATTENTIONdR[head] = (cutoffScale + attention[head]*expTerm)*scale*dATTENTIONdR[head];
            }

            // Compute contributions to the output values.

            float dEdR = 0;
            for (int head = 0; head < numHeads; head++)
                for (int i = 0; i < headWidth; i++) {
                    // s3[atom1][headWidth*head+i] += v[atom2][3*head*headWidth+i]*dv[3*head*headWidth+i]*attention[head];
                    dEdR += dEdS3[atom1][headWidth*head+i]*v[atom2][3*head*headWidth+i]*(dv[3*head*headWidth+i]*dATTENTIONdR[head] + dDVdR[3*head*headWidth+i]*attention[head]);
                }
            for (int j = 0; j < 3; j++) {
                for (int head = 0; head < numHeads; head++)
                    for (int i = 0; i < headWidth; i++) {
                        float vec1 = v[atom2][(3*head+1)*headWidth+i]*dv[(3*head+1)*headWidth+i];
                        float vec2 = v[atom2][(3*head+2)*headWidth+i]*dv[(3*head+2)*headWidth+i];
                        float dVEC1dR = v[atom2][(3*head+1)*headWidth+i]*dDVdR[(3*head+1)*headWidth+i];
                        float dVEC2dR = v[atom2][(3*head+2)*headWidth+i]*dDVdR[(3*head+2)*headWidth+i];
                        // s[atom1][width*j+headWidth*head+i] += vec1*vec[3*width*atom2+width*j+headWidth*head+i] + vec2*delta[j];
                        dEdR += dEdS[atom1][width*j+headWidth*head+i]*(dVEC1dR*vec[3*width*atom2+width*j+headWidth*head+i] + dVEC2dR*delta[j]);
                        positionDeriv[atom1*3+j] -= dEdS[atom1][width*j+headWidth*head+i]*vec2;
                        positionDeriv[atom2*3+j] += dEdS[atom1][width*j+headWidth*head+i]*vec2;
                    }
                float dVdX = dEdR*delta[j];
                positionDeriv[atom1*3+j] -= dVdX;
                positionDeriv[atom2*3+j] += dVdX;
            }
        }
    }




    // vector<vector<float> > q(numAtoms, vector<float>(width));
    // vector<vector<float> > k(numAtoms, vector<float>(width));
    // vector<vector<float> > v(numAtoms, vector<float>(3*width));

    // // Apply the various transforms.

    // for (int atom = 0; atom < numAtoms; atom++) {
    //     for (int i = 0; i < width; i++) {
    //         q[atom][i] = qb[i];
    //         k[atom][i] = kb[i];
    //         for (int j = 0; j < width; j++) {
    //             q[atom][i] += qw[i*width+j]*x[atom*width+j];
    //             k[atom][i] += kw[i*width+j]*x[atom*width+j];
    //         }
    //     }
    //     for (int i = 0; i < 3*width; i++) {
    //         v[atom][i] = vb[i];
    //         for (int j = 0; j < width; j++)
    //             v[atom][i] += vw[i*width+j]*x[atom*width+j];
    //     }
    //     for (int i = 0; i < 3*width; i++) {
    //         for (int k = 0; k < 3; k++) {
    //             u[atom][i+3*width*k] = ub[i];
    //             for (int j = 0; j < width; j++)
    //                 u[atom][i+3*width*k] += uw[i*width+j]*vec[atom*3*width+k*width+j];
    //         }
    //     }
    // }




    // int numRBF = rbfMus.size();
    // vector<float> rbf(numRBF), dRBFdR(numRBF);
    // vector<float> y1(width), dY1dR(width);
    // vector<float> y2(width), dY2dR(width);
    // float invBoxSize[3];
    // if (PERIODIC) {
    //     invBoxSize[0] = 1/periodicBoxVectors[0];
    //     invBoxSize[1] = 1/periodicBoxVectors[4];
    //     invBoxSize[2] = 1/periodicBoxVectors[8];
    // }

    // // Clear the output arrays.

    // memset(inputDeriv, 0, getNumAtoms()*width*sizeof(float));
    // memset(positionDeriv, 0, getNumAtoms()*3*sizeof(float));

    // // Loop over pairs of atoms from the neighbor list.

    // for (int atom1 = 0; atom1 < getNumAtoms(); atom1++) {
    //     for (int atom2 : neighbors.getNeighbors()[atom1]) {
    //         // Compute the displacement.

    //         float delta[3];
    //         float r2;
    //         computeDisplacement<PERIODIC, TRICLINIC>(&positions[3*atom1], &positions[3*atom2], delta, r2, periodicBoxVectors, invBoxSize);
    //         float r = sqrtf(r2);
    //         float rInv = 1/r;

    //         // Compute the radial basis functions.

    //         float alpha = 5.0f/(neighbors.getUpperCutoff()-neighbors.getLowerCutoff());
    //         float cutoffScale = cutoffFunction(r, neighbors.getUpperCutoff());
    //         for (int i = 0; i < numRBF; i++) {
    //             float expTerm = exp(-alpha*(r-neighbors.getLowerCutoff());
    //             float expDiff = expTerm-rbfMus[i];
    //             rbf[i] = cutoffScale*exp(-rbfBetas[i]*expDiff*expDiff);
    //             dRBFdR[i] = rbf[i]*2*rbfBetas[i]*expDiff*expTerm*alpha;
    //         }

    //         // Apply the first dense layer.

    //         for (int i = 0; i < width; i++) {
    //             float sum = b1[i], dSumdR = 0;
    //             for (int j = 0; j < numRBF; j++) {
    //                 sum += rbf[j]*w1[i*numRBF+j];
    //                 dSumdR += dRBFdR[j]*w1[i*numRBF+j];
    //             }
    //             if (getActivation() == ShiftedSoftplus) {
    //                 float expSum = expf(sum);
    //                 y1[i] = logf(0.5f*expSum + 0.5f);
    //                 dY1dR[i] = dSumdR*expSum/(expSum + 1);
    //             }
    //             else {
    //                 float th = tanhf(sum);
    //                 y1[i] = th;
    //                 dY1dR[i] = dSumdR*(1-th*th);
    //             }
    //         }

    //         // Apply the second dense layer.

    //         float cutoffScale = cutoffFunction(r);
    //         float dCutoffdR = cutoffDeriv(r);
    //         for (int i = 0; i < width; i++) {
    //             float sum = b2[i], dSumdR = 0;
    //             for (int j = 0; j < width; j++) {
    //                 sum += y1[j]*w2[i*width+j];
    //                 dSumdR += dY1dR[j]*w2[i*width+j];
    //             }
    //             y2[i] = cutoffScale*sum;
    //             dY2dR[i] = dCutoffdR*sum + cutoffScale*dSumdR;
    //         }

    //         // Add it to the output.

    //         for (int i = 0; i < width; i++) {
    //             int index1 = atom1*width+i;
    //             int index2 = atom2*width+i;
    //             inputDeriv[index1] += y2[i]*outputDeriv[index2];
    //             inputDeriv[index2] += y2[i]*outputDeriv[index1];
    //             float scale = rInv*dY2dR[i]*(input[index2]*outputDeriv[index1] + input[index1]*outputDeriv[index2]);
    //             for (int j = 0; j < 3; j++) {
    //                 float dVdX = scale * delta[j];
    //                 positionDeriv[atom1*3+j] -= dVdX;
    //                 positionDeriv[atom2*3+j] += dVdX;
    //             }
    //         }
    //     }
    // }
}

float CpuEquivariantTransformerLayer::cutoffFunction(float r, float cutoff) {
    return 0.5f * cosf(M_PI*r/cutoff) + 0.5f;
}

float CpuEquivariantTransformerLayer::cutoffDeriv(float r, float cutoff) {
    return -(0.5f*M_PI/cutoff) * sinf(M_PI*r/cutoff);
}
