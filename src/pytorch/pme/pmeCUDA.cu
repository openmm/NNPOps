#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <array>
#include <cmath>
#include <vector>

#include "common/accessor.cuh"

using namespace std;
using namespace torch::autograd;
using torch::Tensor;
using torch::TensorOptions;
using torch::Scalar;

#define CHECK_RESULT(result) \
    if (result != cudaSuccess) { \
        throw runtime_error(string("Encountered error ")+cudaGetErrorName(result)+" at "+__FILE__+":"+to_string(__LINE__));\
    }

static int getMaxBlocks() {
    // Get an upper limit on how many thread blocks we try to launch based on the size of the GPU.

    int device, numMultiprocessors;
    CHECK_RESULT(cudaGetDevice(&device));
    CHECK_RESULT(cudaDeviceGetAttribute(&numMultiprocessors, cudaDevAttrMultiProcessorCount, device));
    return numMultiprocessors*4;
}

__global__ void computeDirect(const Accessor<float, 2> pos, const Accessor<float, 1> charge, Accessor<int, 2> neighbors, const Accessor<float, 2> deltas,
                              const Accessor<float, 1> distances, const Accessor<int, 2> exclusions, Accessor<float, 2> posDeriv,
                              Accessor<float, 1> chargeDeriv, Accessor<float, 1> energyBuffer, float alpha, float coulomb) {
    int numAtoms = pos.size(0);
    int numNeighbors = neighbors.size(1);
    int maxExclusions = exclusions.size(1);
    double energy = 0;

    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < numNeighbors; index += blockDim.x*gridDim.x) {
        int atom1 = neighbors[0][index];
        int atom2 = neighbors[1][index];
        float r = distances[index];
        bool include = (atom1 > -1);
        for (int j = 0; include && j < maxExclusions && exclusions[atom1][j] >= atom2; j++)
            if (exclusions[atom1][j] == atom2)
                include = false;
        if (include) {
            float invR = 1/r;
            float alphaR = alpha*r;
            float expAlphaRSqr = expf(-alphaR*alphaR);
            float erfcAlphaR = erfcf(alphaR);
            float prefactor = coulomb*invR;
            float c1 = charge[atom1];
            float c2 = charge[atom2];
            energy += prefactor*erfcAlphaR*c1*c2;
            atomicAdd(&chargeDeriv[atom1], prefactor*erfcAlphaR*c2);
            atomicAdd(&chargeDeriv[atom2], prefactor*erfcAlphaR*c1);
            float dEdR = prefactor*c1*c2*(erfcAlphaR+alphaR*expAlphaRSqr*M_2_SQRTPI)*invR*invR;
            for (int j = 0; j < 3; j++) {
                atomicAdd(&posDeriv[atom1][j], -dEdR*deltas[index][j]);
                atomicAdd(&posDeriv[atom2][j], dEdR*deltas[index][j]);
            }
        }
    }

    // Subtract excluded interactions to compensate for the part that is
    // incorrectly added in reciprocal space.

    float dr[3];
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < numAtoms*maxExclusions; index += blockDim.x*gridDim.x) {
        int atom1 = index/maxExclusions;
        int atom2 = exclusions[atom1][index-atom1*maxExclusions];
        if (atom2 > atom1) {
            for (int j = 0; j < 3; j++)
                dr[j] = pos[atom1][j]-pos[atom2][j];
            float r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
            float invR = rsqrtf(r2);
            float r = invR*r2;
            float alphaR = alpha*r;
            float expAlphaRSqr = expf(-alphaR*alphaR);
            float erfAlphaR = erff(alphaR);
            float prefactor = coulomb*invR;
            float c1 = charge[atom1];
            float c2 = charge[atom2];
            energy -= prefactor*erfAlphaR*c1*c2;
            atomicAdd(&chargeDeriv[atom1], -prefactor*erfAlphaR*c2);
            atomicAdd(&chargeDeriv[atom2], -prefactor*erfAlphaR*c1);
            float dEdR = prefactor*c1*c2*(erfAlphaR-alphaR*expAlphaRSqr*M_2_SQRTPI)*invR*invR;
            for (int j = 0; j < 3; j++) {
                atomicAdd(&posDeriv[atom1][j], dEdR*dr[j]);
                atomicAdd(&posDeriv[atom2][j], -dEdR*dr[j]);
            }
        }
    }
    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] = energy;
}

__device__ void invertBoxVectors(const Accessor<float, 2>& box, float recipBoxVectors[3][3]) {
    float determinant = box[0][0]*box[1][1]*box[2][2];
    float scale = 1.0f/determinant;
    recipBoxVectors[0][0] = box[1][1]*box[2][2]*scale;
    recipBoxVectors[0][1] = 0;
    recipBoxVectors[0][2] = 0;
    recipBoxVectors[1][0] = -box[1][0]*box[2][2]*scale;
    recipBoxVectors[1][1] = box[0][0]*box[2][2]*scale;
    recipBoxVectors[1][2] = 0;
    recipBoxVectors[2][0] = (box[1][0]*box[2][1]-box[1][1]*box[2][0])*scale;
    recipBoxVectors[2][1] = -box[0][0]*box[2][1]*scale;
    recipBoxVectors[2][2] = box[0][0]*box[1][1]*scale;
}

__device__ void computeSpline(int atom, const Accessor<float, 2> pos, const Accessor<float, 2> box,
                          const float recipBoxVectors[3][3], const int gridSize[3], int gridIndex[3], float data[][3],
                          float ddata[][3], int pmeOrder) {
    // Find the position relative to the nearest grid point.

    float posInBox[3] = {pos[atom][0], pos[atom][1], pos[atom][2]};
    for (int i = 2; i >= 0; i--) {
        float scale = floor(posInBox[i]*recipBoxVectors[i][i]);
        for (int j = 0; j < 3; j++)
            posInBox[j] -= scale*box[i][j];
    }
    float t[3], dr[3];
    int ti[3];
    for (int i = 0; i < 3; i++) {
        t[i] = posInBox[0]*recipBoxVectors[0][i] + posInBox[1]*recipBoxVectors[1][i] + posInBox[2]*recipBoxVectors[2][i];
        t[i] = (t[i]-floor(t[i]))*gridSize[i];
        ti[i] = (int) t[i];
        dr[i] = t[i]-ti[i];
        gridIndex[i] = ti[i]%gridSize[i];
    }

    // Compute the B-spline coefficients.

    float scale = 1.0f/(pmeOrder-1);
    for (int i = 0; i < 3; i++) {
        data[pmeOrder-1][i] = 0;
        data[1][i] = dr[i];
        data[0][i] = 1-dr[i];
        for (int j = 3; j < pmeOrder; j++) {
            float div = 1.0f/(j-1);
            data[j-1][i] = div*dr[i]*data[j-2][i];
            for (int k = 1; k < j-1; k++)
                data[j-k-1][i] = div*((dr[i]+k)*data[j-k-2][i]+(j-k-dr[i])*data[j-k-1][i]);
            data[0][i] = div*(1-dr[i])*data[0][i];
        }
        if (ddata != NULL) {
            ddata[0][i] = -data[0][i];
            for (int j = 1; j < pmeOrder; j++)
                ddata[j][i] = data[j-1][i]-data[j][i];
        }
        data[pmeOrder-1][i] = scale*dr[i]*data[pmeOrder-2][i];
        for (int j = 1; j < pmeOrder-1; j++)
            data[pmeOrder-j-1][i] = scale*((dr[i]+j)*data[pmeOrder-j-2][i]+(pmeOrder-j-dr[i])*data[pmeOrder-j-1][i]);
        data[0][i] = scale*(1-dr[i])*data[0][i];
    }
}

template <int PME_ORDER>
__global__ void spreadCharge(const Accessor<float, 2> pos, const Accessor<float, 1> charge, const Accessor<float, 2> box,
                             Accessor<float, 3> grid, int gridx, int gridy, int gridz, float sqrtCoulomb) {
    __shared__ float recipBoxVectors[3][3];
    if (threadIdx.x == 0)
        invertBoxVectors(box, recipBoxVectors);
    __syncthreads();
    float data[PME_ORDER][3];
    int numAtoms = pos.size(0);
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x) {
        int gridIndex[3];
        int gridSize[3] = {gridx, gridy, gridz};
        computeSpline(atom, pos, box,recipBoxVectors, gridSize, gridIndex, data, NULL, PME_ORDER);

        // Spread the charge from this atom onto each grid point.

        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xindex = gridIndex[0]+ix;
            xindex -= (xindex >= gridx ? gridx : 0);
            float dx = charge[atom]*sqrtCoulomb*data[ix][0];
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int yindex = gridIndex[1]+iy;
                yindex -= (yindex >= gridy ? gridy : 0);
                float dxdy = dx*data[iy][1];
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex[2]+iz;
                    zindex -= (zindex >= gridz ? gridz : 0);
                    atomicAdd(&grid[xindex][yindex][zindex], dxdy*data[iz][2]);
                }
            }
        }
    }
}

__global__ void reciprocalConvolution(const Accessor<float, 2> box, Accessor<c10::complex<float>, 3> grid, int gridx, int gridy, int gridz,
                                      const Accessor<float, 1> xmoduli, const Accessor<float, 1> ymoduli, const Accessor<float, 1> zmoduli,
                                      float recipExpFactor, Accessor<float, 1> energyBuffer) {
    float recipBoxVectors[3][3];
    invertBoxVectors(box, recipBoxVectors);
    const unsigned int gridSize = gridx*gridy*(gridz/2+1);
    const float recipScaleFactor = recipBoxVectors[0][0]*recipBoxVectors[1][1]*recipBoxVectors[2][2]/M_PI;
    double energy = 0;

    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < gridSize; index += blockDim.x*gridDim.x) {
        int kx = index/(gridy*(gridz/2+1));
        int remainder = index-kx*gridy*(gridz/2+1);
        int ky = remainder/(gridz/2+1);
        int kz = remainder-ky*(gridz/2+1);
        int mx = (kx < (gridx+1)/2) ? kx : (kx-gridx);
        int my = (ky < (gridy+1)/2) ? ky : (ky-gridy);
        int mz = (kz < (gridz+1)/2) ? kz : (kz-gridz);
        float mhx = mx*recipBoxVectors[0][0];
        float mhy = mx*recipBoxVectors[1][0]+my*recipBoxVectors[1][1];
        float mhz = mx*recipBoxVectors[2][0]+my*recipBoxVectors[2][1]+mz*recipBoxVectors[2][2];
        float bx = xmoduli[kx];
        float by = ymoduli[ky];
        float bz = zmoduli[kz];
        c10::complex<float>& g = grid[kx][ky][kz];
        float m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        float denom = m2*bx*by*bz;
        float eterm = (index == 0 ? 0 : recipScaleFactor*exp(-recipExpFactor*m2)/denom);
        float scale = (kz > 0 && kz <= (gridz-1)/2 ? 2 : 1);
        energy += scale * eterm * (g.real()*g.real() + g.imag()*g.imag());
        g *= eterm;
    }
    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] = 0.5f*energy;
}

template <int PME_ORDER>
__global__ void interpolateForce(const Accessor<float, 2> pos, const Accessor<float, 1> charge, const Accessor<float, 2> box,
                                 const Accessor<float, 3> grid, int gridx, int gridy, int gridz, float sqrtCoulomb,
                                 Accessor<float, 2> posDeriv, Accessor<float, 1> chargeDeriv) {
    __shared__ float recipBoxVectors[3][3];
    if (threadIdx.x == 0)
        invertBoxVectors(box, recipBoxVectors);
    __syncthreads();
    float data[PME_ORDER][3];
    float ddata[PME_ORDER][3];
    int numAtoms = pos.size(0);
    
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x) {
        int gridIndex[3];
        int gridSize[3] = {gridx, gridy, gridz};
        computeSpline(atom, pos, box,recipBoxVectors, gridSize, gridIndex, data, ddata, PME_ORDER);

        // Compute the derivatives on this atom.

        float dpos[3] = {0, 0, 0};
        float dq = 0;
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xindex = gridIndex[0]+ix;
            xindex -= (xindex >= gridx ? gridx : 0);
            float dx = data[ix][0];
            float ddx = ddata[ix][0];
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int yindex = gridIndex[1]+iy;
                yindex -= (yindex >= gridy ? gridy : 0);
                float dy = data[iy][1];
                float ddy = ddata[iy][1];
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex[2]+iz;
                    zindex -= (zindex >= gridz ? gridz : 0);
                    float dz = data[iz][2];
                    float ddz = ddata[iz][2];
                    float g = grid[xindex][yindex][zindex];
                    dpos[0] += ddx*dy*dz*g;
                    dpos[1] += dx*ddy*dz*g;
                    dpos[2] += dx*dy*ddz*g;
                    dq += dx*dy*dz*g;
                }
            }
        }
        float scale = charge[atom]*sqrtCoulomb;
        posDeriv[atom][0] = scale*(dpos[0]*gridSize[0]*recipBoxVectors[0][0]);
        posDeriv[atom][1] = scale*(dpos[0]*gridSize[0]*recipBoxVectors[1][0] + dpos[1]*gridSize[1]*recipBoxVectors[1][1]);
        posDeriv[atom][2] = scale*(dpos[0]*gridSize[0]*recipBoxVectors[2][0] + dpos[1]*gridSize[1]*recipBoxVectors[2][1] + dpos[2]*gridSize[2]*recipBoxVectors[2][2]);
        chargeDeriv[atom] = dq*sqrtCoulomb;
    }
}

class PmeDirectFunctionCuda : public Function<PmeDirectFunctionCuda> {
public:
    static Tensor forward(AutogradContext *ctx,
                          const Tensor& positions,
                          const Tensor& charges,
                          const Tensor& neighbors,
                          const Tensor& deltas,
                          const Tensor& distances,
                          const Tensor& exclusions,
                          const Scalar& alpha,
                          const Scalar& coulomb) {
        const auto stream = c10::cuda::getCurrentCUDAStream(positions.get_device());
        const c10::cuda::CUDAStreamGuard guard(stream);
        int numAtoms = charges.size(0);
        int numPairs = neighbors.size(1);
        TensorOptions options = torch::TensorOptions().device(neighbors.device());
        Tensor posDeriv = torch::zeros({numAtoms, 3}, options);
        Tensor chargeDeriv = torch::zeros({numAtoms}, options);
        int blockSize = 128;
        int numBlocks = max(1, min(getMaxBlocks(), (numPairs+blockSize-1)/blockSize));
        Tensor energy = torch::zeros(numBlocks*blockSize, options);
        computeDirect<<<numBlocks, blockSize, 0, stream>>>(get_accessor<float, 2>(positions), get_accessor<float, 1>(charges),
            get_accessor<int, 2>(neighbors), get_accessor<float, 2>(deltas), get_accessor<float, 1>(distances),
            get_accessor<int, 2>(exclusions), get_accessor<float, 2>(posDeriv), get_accessor<float, 1>(chargeDeriv),
            get_accessor<float, 1>(energy), alpha.toDouble(), coulomb.toDouble());

        // Store data for later use.

        ctx->save_for_backward({posDeriv, chargeDeriv});
        return {torch::sum(energy)};
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        Tensor posDeriv = saved[0];
        Tensor chargeDeriv = saved[1];
        torch::Tensor ignore;
        return {posDeriv*grad_outputs[0], chargeDeriv*grad_outputs[0], ignore, ignore, ignore, ignore, ignore, ignore};
    }
};

class PmeFunctionCuda : public Function<PmeFunctionCuda> {
public:
    static Tensor forward(AutogradContext *ctx,
                          const Tensor& positions,
                          const Tensor& charges,
                          const Tensor& box_vectors,
                          const Scalar& gridx,
                          const Scalar& gridy,
                          const Scalar& gridz,
                          const Scalar& order,
                          const Scalar& alpha,
                          const Scalar& coulomb,
                          const Tensor& xmoduli,
                          const Tensor& ymoduli,
                          const Tensor& zmoduli) {
        const auto stream = c10::cuda::getCurrentCUDAStream(positions.get_device());
        const c10::cuda::CUDAStreamGuard guard(stream);
        int numAtoms = positions.size(0);
        int pmeOrder = (int) order.toInt();
        int gridSize[3] = {(int) gridx.toInt(), (int) gridy.toInt(), (int) gridz.toInt()};
        float sqrtCoulomb = sqrt(coulomb.toDouble());

        // Spread the charge on the grid.

        TensorOptions options = torch::TensorOptions().device(positions.device());
        Tensor realGrid = torch::zeros({gridSize[0], gridSize[1], gridSize[2]}, options);
        int blockSize = 128;
        int numBlocks = max(1, min(getMaxBlocks(), (numAtoms+blockSize-1)/blockSize));
        TORCH_CHECK(pmeOrder == 4 || pmeOrder == 5, "Only pmeOrder 4 or 5 is supported with CUDA");
        auto spread = (pmeOrder == 4 ? spreadCharge<4> : spreadCharge<5>);
        spread<<<numBlocks, blockSize, 0, stream>>>(get_accessor<float, 2>(positions), get_accessor<float, 1>(charges),
                get_accessor<float, 2>(box_vectors), get_accessor<float, 3>(realGrid), gridSize[0], gridSize[1], gridSize[2], sqrtCoulomb);

        // Take the Fourier transform.

        Tensor recipGrid = torch::fft::rfftn(realGrid);

        // Perform the convolution and calculate the energy.

        Tensor energy = torch::zeros(numBlocks*blockSize, options);
        reciprocalConvolution<<<numBlocks, blockSize, 0, stream>>>(get_accessor<float, 2>(box_vectors), get_accessor<c10::complex<float>, 3>(recipGrid),
                gridSize[0], gridSize[1], gridSize[2], get_accessor<float, 1>(xmoduli), get_accessor<float, 1>(ymoduli), get_accessor<float, 1>(zmoduli),
                M_PI*M_PI/(alpha.toDouble()*alpha.toDouble()), get_accessor<float, 1>(energy));

        // Store data for later use.

        ctx->save_for_backward({positions, charges, box_vectors, xmoduli, ymoduli, zmoduli, recipGrid});
        ctx->saved_data["gridx"] = gridx;
        ctx->saved_data["gridy"] = gridy;
        ctx->saved_data["gridz"] = gridz;
        ctx->saved_data["order"] = order;
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["coulomb"] = coulomb;
        return {torch::sum(energy)};
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        Tensor positions = saved[0];
        Tensor charges = saved[1];
        Tensor box_vectors = saved[2];
        Tensor xmoduli = saved[3];
        Tensor ymoduli = saved[4];
        Tensor zmoduli = saved[5];
        Tensor recipGrid = saved[6];
        int gridSize[3] = {(int) ctx->saved_data["gridx"].toInt(), (int) ctx->saved_data["gridy"].toInt(), (int) ctx->saved_data["gridz"].toInt()};
        int pmeOrder = (int) ctx->saved_data["order"].toInt();
        float alpha = (float) ctx->saved_data["alpha"].toDouble();
        float sqrtCoulomb = sqrt(ctx->saved_data["coulomb"].toDouble());
        const auto stream = c10::cuda::getCurrentCUDAStream(positions.get_device());
        const c10::cuda::CUDAStreamGuard guard(stream);
        int numAtoms = positions.size(0);

        // Take the inverse Fourier transform.

        Tensor realGrid = torch::fft::irfftn(recipGrid, {gridSize[0], gridSize[1], gridSize[2]}, c10::nullopt, "forward");

        // Compute the derivatives.

        TensorOptions options = torch::TensorOptions().device(positions.device());
        Tensor posDeriv = torch::empty({numAtoms, 3}, options);
        Tensor chargeDeriv = torch::empty({numAtoms}, options);
        int blockSize = 128;
        int numBlocks = max(1, min(getMaxBlocks(), (numAtoms+blockSize-1)/blockSize));
        TORCH_CHECK(pmeOrder == 4 || pmeOrder == 5, "Only pmeOrder 4 or 5 is supported with CUDA");
        if (pmeOrder == 4)
            interpolateForce<4><<<numBlocks, blockSize, 0, stream>>>(get_accessor<float, 2>(positions), get_accessor<float, 1>(charges),
                        get_accessor<float, 2>(box_vectors), get_accessor<float, 3>(realGrid), gridSize[0], gridSize[1], gridSize[2], sqrtCoulomb,
                        get_accessor<float, 2>(posDeriv), get_accessor<float, 1>(chargeDeriv));
        else
            interpolateForce<5><<<numBlocks, blockSize, 0, stream>>>(get_accessor<float, 2>(positions), get_accessor<float, 1>(charges),
                        get_accessor<float, 2>(box_vectors), get_accessor<float, 3>(realGrid), gridSize[0], gridSize[1], gridSize[2], sqrtCoulomb,
                        get_accessor<float, 2>(posDeriv), get_accessor<float, 1>(chargeDeriv));
        posDeriv *= grad_outputs[0];
        chargeDeriv *= grad_outputs[0];
        torch::Tensor ignore;
        return {posDeriv, chargeDeriv, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore};
    }
};

Tensor pme_direct_cuda(const Tensor& positions,
                       const Tensor& charges,
                       const Tensor& neighbors,
                       const Tensor& deltas,
                       const Tensor& distances,
                       const Tensor& exclusions,
                       const Scalar& alpha,
                       const Scalar& coulomb) {
    return PmeDirectFunctionCuda::apply(positions, charges, neighbors, deltas, distances, exclusions, alpha, coulomb);
}

Tensor pme_reciprocal_cuda(const Tensor& positions,
                           const Tensor& charges,
                           const Tensor& box_vectors,
                           const Scalar& gridx,
                           const Scalar& gridy,
                           const Scalar& gridz,
                           const Scalar& order,
                           const Scalar& alpha,
                           const Scalar& coulomb,
                           const Tensor& xmoduli,
                           const Tensor& ymoduli,
                           const Tensor& zmoduli) {
    return PmeFunctionCuda::apply(positions, charges, box_vectors, gridx, gridy, gridz, order, alpha, coulomb, xmoduli, ymoduli, zmoduli);
}

TORCH_LIBRARY_IMPL(pme, AutogradCUDA, m) {
    m.impl("pme_direct", pme_direct_cuda);
    m.impl("pme_reciprocal", pme_reciprocal_cuda);
}
