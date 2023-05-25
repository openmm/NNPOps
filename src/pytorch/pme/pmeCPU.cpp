#include <torch/extension.h>
#include <array>
#include <vector>

using namespace std;
using namespace torch::autograd;
using torch::Tensor;
using torch::TensorOptions;
using torch::Scalar;

static void invertBoxVectors(const Tensor& box_vectors, float recipBoxVectors[3][3]) {
    auto box = box_vectors.accessor<float,2>();
    float determinant = box[0][0]*box[1][1]*box[2][2];
    float scale = 1.0/determinant;
    recipBoxVectors[0][0] = box[1][1]*box[2][2]*scale;
    recipBoxVectors[0][1] = 0;
    recipBoxVectors[0][2] = 0;
    recipBoxVectors[1][0] = -box[1][0]*box[2][2]*scale;
    recipBoxVectors[1][1] = box[0][0]*box[2][2]*scale;
    recipBoxVectors[1][2] = 0;
    recipBoxVectors[2][0] = box[1][0]*box[2][1]-box[1][1]*box[2][0]*scale;
    recipBoxVectors[2][1] = -box[0][0]*box[2][1]*scale;
    recipBoxVectors[2][2] = box[0][0]*box[1][1]*scale;
}

static void computeSpline(int atom, const torch::TensorAccessor<float, 2>& pos, const torch::TensorAccessor<float, 2>& box,
                          const float recipBoxVectors[3][3], const int gridSize[3], int gridIndex[3], vector<array<float, 3> >& data,
                          vector<array<float, 3> >& ddata, int pmeOrder) {
    // Find the position relative to the nearest grid point.

    float posInBox[3], t[3], dr[3];
    int ti[3];
    for (int i = 0; i < 3; i++)
         posInBox[i] = pos[atom][i]-box[i][i]*floor(pos[atom][i]*recipBoxVectors[i][i]);
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
        if (ddata.size() > 0) {
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

class PmeFunctionCpu : public Function<PmeFunctionCpu> {
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
        int numAtoms = positions.size(0);
        int pmeOrder = (int) order.toInt();
        auto box = box_vectors.accessor<float,2>();
        auto pos = positions.accessor<float,2>();
        auto charge = charges.accessor<float,1>();
        int gridSize[3] = {(int) gridx.toInt(), (int) gridy.toInt(), (int) gridz.toInt()};
        float recipBoxVectors[3][3];
        invertBoxVectors(box_vectors, recipBoxVectors);
        vector<float> grid(gridSize[0]*gridSize[1]*gridSize[2], 0);
        float sqrtCoulomb = sqrt(coulomb.toDouble());

        // Spread the charge on the grid.

        for (int atom = 0; atom < numAtoms; atom++) {
            // Compute the B-spline coefficients.

            int gridIndex[3];
            vector<array<float, 3> > data(pmeOrder), ddata;
            computeSpline(atom, pos, box, recipBoxVectors, gridSize, gridIndex, data, ddata, pmeOrder);

            // Spread the charge from this atom onto each grid point.

            for (int ix = 0; ix < pmeOrder; ix++) {
                int xindex = (gridIndex[0]+ix) % gridSize[0];
                float dx = charge[atom]*sqrtCoulomb*data[ix][0];
                for (int iy = 0; iy < pmeOrder; iy++) {
                    int yindex = (gridIndex[1]+iy) % gridSize[1];
                    float dxdy = dx*data[iy][1];
                    for (int iz = 0; iz < pmeOrder; iz++) {
                        int zindex = (gridIndex[2]+iz) % gridSize[2];
                        int index = xindex*gridSize[1]*gridSize[2] + yindex*gridSize[2] + zindex;
                        grid[index] += dxdy*data[iz][2];
                    }
                }
            }
        }

        // Take the Fourier transform.

        TensorOptions options = torch::TensorOptions().device(positions.device()); // Data type of float by default
        Tensor realGrid = torch::from_blob(grid.data(), {gridSize[0], gridSize[1], gridSize[2]}, options);
        Tensor recipGrid = torch::fft::rfftn(realGrid);
        auto recip = recipGrid.accessor<c10::complex<float>,3>();

        // Perform the convolution and calculate the energy.

        double energy = 0.0;
        int zsize = gridSize[2]/2+1;
        int yzsize = gridSize[1]*zsize;
        float scaleFactor = (float) (M_PI*box[0][0]*box[1][1]*box[2][2]);
        float recipExpFactor = (float) (M_PI*M_PI/(alpha.toDouble()*alpha.toDouble()));
        auto xmod = xmoduli.accessor<float,1>();
        auto ymod = ymoduli.accessor<float,1>();
        auto zmod = zmoduli.accessor<float,1>();
        for (int kx = 0; kx < gridSize[0]; kx++) {
            int mx = (kx < (gridSize[0]+1)/2) ? kx : kx-gridSize[0];
            float mhx = mx*recipBoxVectors[0][0];
            float bx = scaleFactor*xmod[kx];
            for (int ky = 0; ky < gridSize[1]; ky++) {
                int my = (ky < (gridSize[1]+1)/2) ? ky : ky-gridSize[1];
                float mhy = mx*recipBoxVectors[1][0] + my*recipBoxVectors[1][1];
                float mhx2y2 = mhx*mhx + mhy*mhy;
                float bxby = bx*ymod[ky];
                for (int kz = 0; kz < zsize; kz++) {
                    int index = kx*yzsize + ky*zsize + kz;
                    int mz = (kz < (gridSize[2]+1)/2) ? kz : kz-gridSize[2];
                    float mhz = mx*recipBoxVectors[2][0] + my*recipBoxVectors[2][1] + mz*recipBoxVectors[2][2];
                    float bz = zmod[kz];
                    float m2 = mhx2y2 + mhz*mhz;
                    float denom = m2*bxby*bz;
                    float eterm = (index == 0 ? 0 : exp(-recipExpFactor*m2)/denom);
                    float scale = (kz > 0 && kz <= (gridSize[2]-1)/2 ? 2 : 1);
                    c10::complex<float>& g = recip[kx][ky][kz];
                    energy += scale * eterm * (g.real()*g.real() + g.imag()*g.imag());
                    g *= eterm;
                }
            }
        }

        // Store data for later use.

        ctx->save_for_backward({positions, charges, box_vectors, xmoduli, ymoduli, zmoduli, recipGrid});
        ctx->saved_data["gridx"] = gridx;
        ctx->saved_data["gridy"] = gridy;
        ctx->saved_data["gridz"] = gridz;
        ctx->saved_data["order"] = order;
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["coulomb"] = coulomb;
        return {torch::tensor(0.5*energy, options)};
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
        int numAtoms = positions.size(0);
        auto box = box_vectors.accessor<float,2>();
        auto pos = positions.accessor<float,2>();
        auto charge = charges.accessor<float,1>();
        float recipBoxVectors[3][3];
        invertBoxVectors(box_vectors, recipBoxVectors);

        // Take the inverse Fourier transform.

        Tensor realGrid = torch::fft::irfftn(recipGrid)*(gridSize[0]*gridSize[1]*gridSize[2]);
        auto grid = realGrid.accessor<float,3>();

        // Compute the derivatives.

        TensorOptions options = torch::TensorOptions().device(positions.device()); // Data type of float by default
        Tensor posDeriv = torch::empty({numAtoms, 3}, options);
        Tensor chargeDeriv = torch::empty({numAtoms}, options);
        auto posDeriv_a = posDeriv.accessor<float,2>();
        auto chargeDeriv_a = chargeDeriv.accessor<float,1>();
        for (int atom = 0; atom < numAtoms; atom++) {
            // Compute the B-spline coefficients.

            int gridIndex[3];
            vector<array<float, 3> > data(pmeOrder), ddata(pmeOrder);
            computeSpline(atom, pos, box, recipBoxVectors, gridSize, gridIndex, data, ddata, pmeOrder);

            // Compute the derivatives on this atom.

            float dpos[3] = {0, 0, 0};
            float dq = 0;
            for (int ix = 0; ix < pmeOrder; ix++) {
                int xindex = (gridIndex[0]+ix) % gridSize[0];
                float dx = data[ix][0];
                float ddx = ddata[ix][0];
                for (int iy = 0; iy < pmeOrder; iy++) {
                    int yindex = (gridIndex[1]+iy) % gridSize[1];
                    float dy = data[iy][1];
                    float ddy = ddata[iy][1];
                    for (int iz = 0; iz < pmeOrder; iz++) {
                        int zindex = (gridIndex[2]+iz) % gridSize[2];
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
            posDeriv_a[atom][0] = scale*(dpos[0]*gridSize[0]*recipBoxVectors[0][0]);
            posDeriv_a[atom][1] = scale*(dpos[0]*gridSize[0]*recipBoxVectors[1][0] + dpos[1]*gridSize[1]*recipBoxVectors[1][1]);
            posDeriv_a[atom][2] = scale*(dpos[0]*gridSize[0]*recipBoxVectors[2][0] + dpos[1]*gridSize[1]*recipBoxVectors[2][1] + dpos[2]*gridSize[2]*recipBoxVectors[2][2]);
            chargeDeriv_a[atom] = dq*sqrtCoulomb;
        }
        posDeriv *= grad_outputs[0];
        chargeDeriv *= grad_outputs[0];
        torch::Tensor ignore;
        return {posDeriv, chargeDeriv, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore};
    }
};

Tensor pme_reciprocal(const Tensor& positions,
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
    return PmeFunctionCpu::apply(positions, charges, box_vectors, gridx, gridy, gridz, order, alpha, coulomb, xmoduli, ymoduli, zmoduli);
}

TORCH_LIBRARY_IMPL(pme, CPU, m) {
    m.impl("pme_reciprocal", pme_reciprocal);
}
