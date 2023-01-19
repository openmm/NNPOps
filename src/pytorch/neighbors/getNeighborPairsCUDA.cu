#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <exception>
#include <stdexcept>
#include <torch/extension.h>
#include <algorithm>
#include <tuple>

#include "common/accessor.cuh"
#include "common/atomicAdd.cuh"

using c10::cuda::CUDAStreamGuard;
using c10::cuda::getCurrentCUDAStream;
using std::make_tuple;
using std::max;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;
using torch::empty;
using torch::full;
using torch::kInt32;
using torch::Scalar;
using torch::Tensor;
using torch::TensorOptions;
using torch::zeros;

template <typename scalar_t> __device__ __forceinline__ scalar_t sqrt_(scalar_t x) {};
template<> __device__ __forceinline__ float sqrt_(float x) { return ::sqrtf(x); };
template<> __device__ __forceinline__ double sqrt_(double x) { return ::sqrt(x); };

__device__ __managed__ int32_t tooManyNeighborsErrorFlag; //Error flag for forward_kernel

template <typename scalar_t> __global__ void forward_kernel(
    const int32_t num_all_pairs,
    const Accessor<scalar_t, 2> positions,
    const scalar_t cutoff2,
    const bool store_all_pairs,
    const bool use_periodic,
    Accessor<int32_t, 1> i_curr_pair,
    Accessor<int32_t, 2> neighbors,
    Accessor<scalar_t, 2> deltas,
    Accessor<scalar_t, 1> distances,
    Accessor<scalar_t, 2> box_vectors
) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_all_pairs) return;

    int32_t row = floor((sqrtf(8 * index + 1) + 1) / 2);
    if (row * (row - 1) > 2 * index) row--;
    const int32_t column = index - row * (row - 1) / 2;

    scalar_t delta_x = positions[row][0] - positions[column][0];
    scalar_t delta_y = positions[row][1] - positions[column][1];
    scalar_t delta_z = positions[row][2] - positions[column][2];
    if (use_periodic) {
        scalar_t scale3 = round(delta_z/box_vectors[2][2]);
        delta_x -= scale3*box_vectors[2][0];
        delta_y -= scale3*box_vectors[2][1];
        delta_z -= scale3*box_vectors[2][2];
        scalar_t scale2 = round(delta_y/box_vectors[1][1]);
        delta_x -= scale2*box_vectors[1][0];
        delta_y -= scale2*box_vectors[1][1];
        scalar_t scale1 = round(delta_x/box_vectors[0][0]);
        delta_x -= scale1*box_vectors[0][0];
    }
    const scalar_t distance2 = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

    if (distance2 > cutoff2) return;

    const int32_t i_pair = store_all_pairs ? index : atomicAdd(&i_curr_pair[0], 1);
    //If the maximum number of neighbours is surpassed encode the
    //number of pairs found in a flag and exit
    if(i_pair >= neighbors.size(1)){
      atomicMin(&tooManyNeighborsErrorFlag, -i_pair);
      return;
    }

    neighbors[0][i_pair] = row;
    neighbors[1][i_pair] = column;
    deltas[i_pair][0] = delta_x;
    deltas[i_pair][1] = delta_y;
    deltas[i_pair][2] = delta_z;
    distances[i_pair] = sqrt_(distance2);
}

template <typename scalar_t> __global__ void backward_kernel(
    const Accessor<int32_t, 2> neighbors,
    const Accessor<scalar_t, 2> deltas,
    const Accessor<scalar_t, 2> grad_deltas,
    const Accessor<scalar_t, 1> distances,
    const Accessor<scalar_t, 1> grad_distances,
    Accessor<scalar_t, 2> grad_positions
) {
    const int32_t i_pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t num_pairs = neighbors.size(1);
    if (i_pair >= num_pairs) return;

    const int32_t i_dir = blockIdx.y;
    const int32_t i_atom = neighbors[i_dir][i_pair];
    if (i_atom < 0) return;

    const int32_t i_comp = blockIdx.z;
    const scalar_t grad_deltas_ = grad_deltas[i_pair][i_comp];
    const scalar_t grad_distances_ = deltas[i_pair][i_comp] / distances[i_pair] * grad_distances[i_pair];
    const scalar_t grad = (i_dir ? -1 : 1) * (grad_deltas_ + grad_distances_);
    atomicAdd(&grad_positions[i_atom][i_comp], grad);
}

namespace detail{
  static std::exception_ptr tooManyNeighborsException = nullptr;
  // Checks the  too many  neighbors flag and  stores an  exception if
  // necessary to detail::tooManyNeighborsException.  This function is
  // intended to be launched via cudaLaunchHostFunc.
  //data  is a  void pointer  to a  std::tuple<int,bool>, storing  the
  // maximum number  of neighbors and  whether to throw  an uncatchable
  // exception here (false) or store it for later (true).
  void CUDART_CB checkTooManyNeighbors(void *data){
    int max_num_neighbors;
    bool checkErrors;
    std::tie(max_num_neighbors, checkErrors) = *static_cast<std::tuple<int, bool>*>(data);
    // An exception thrown  in a stream callback is  not catchable (it
    // runs in another thread), so we store it in an exception_ptr for
    // it  to be  processed sometime  later in  the main  thread.  For
    // performance   reasons,    the   exception   is    thrown   here
    // asynchronously if the checkErrors flag is set to false
    try{
      const int tooMan = tooManyNeighborsErrorFlag;
      TORCH_CHECK(tooMan == 0,
		  "Some particle has too many neighbors, found " +
		  std::to_string(-tooMan) + " but max is " +
		  std::to_string(max_num_neighbors));
    }
    catch(...){
      if(not checkErrors)
	throw;
      else
	tooManyNeighborsException = std::current_exception();
    }
  }
}

class Autograd : public Function<Autograd> {
public:
    static tensor_list forward(AutogradContext* ctx,
                               const Tensor& positions,
                               const Scalar& cutoff,
                               const Scalar& max_num_neighbors,
                               const Tensor& box_vectors,
			       bool checkErrors) {

        TORCH_CHECK(positions.dim() == 2, "Expected \"positions\" to have two dimensions");
        TORCH_CHECK(positions.size(0) > 0, "Expected the 1nd dimension size of \"positions\" to be more than 0");
        TORCH_CHECK(positions.size(1) == 3, "Expected the 2nd dimension size of \"positions\" to be 3");
        TORCH_CHECK(positions.is_contiguous(), "Expected \"positions\" to be contiguous");

        const int max_num_neighbors_ = max_num_neighbors.to<int>();
        TORCH_CHECK(max_num_neighbors_ > 0 || max_num_neighbors_ == -1,
            "Expected \"max_num_neighbors\" to be positive or equal to -1");

        const bool use_periodic = (box_vectors.size(0) != 0);
        if (use_periodic) {
            TORCH_CHECK(box_vectors.dim() == 2, "Expected \"box_vectors\" to have two dimensions");
            TORCH_CHECK(box_vectors.size(0) == 3 && box_vectors.size(1) == 3, "Expected \"box_vectors\" to have shape (3, 3)");
        }

        // Decide the algorithm
        const bool store_all_pairs = max_num_neighbors_ == -1;
        const int num_atoms = positions.size(0);
        const int num_all_pairs = num_atoms * (num_atoms - 1) / 2;
        const int num_pairs = store_all_pairs ? num_all_pairs : num_atoms * max_num_neighbors_;

        const int num_threads = 128;
        const int num_blocks = max((num_all_pairs + num_threads - 1) / num_threads, 1);
        const auto stream = getCurrentCUDAStream(positions.get_device());

        const TensorOptions options = positions.options();
        const Tensor i_curr_pair = zeros(1, options.dtype(kInt32));
        const Tensor neighbors = full({2, num_pairs}, -1, options.dtype(kInt32));
        const Tensor deltas = full({num_pairs, 3}, NAN, options);
        const Tensor distances = full(num_pairs, NAN, options);
	//Advice CUDA on expected usage of the error flag
	cudaMemAdvise(&tooManyNeighborsErrorFlag, sizeof(int),
		      cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
	cudaMemAdvise(&tooManyNeighborsErrorFlag, sizeof(int),
		      cudaMemAdviseSetReadMostly, 0);
	tooManyNeighborsErrorFlag = 0;
	const CUDAStreamGuard guard(stream);
        AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "getNeighborPairs::forward", [&]() {
	  const scalar_t cutoff_ = cutoff.to<scalar_t>();
	  TORCH_CHECK(cutoff_ > 0, "Expected \"cutoff\" to be positive");
            forward_kernel<<<num_blocks, num_threads, 0, stream>>>(
                num_all_pairs,
                get_accessor<scalar_t, 2>(positions),
                cutoff_ * cutoff_,
                store_all_pairs,
                use_periodic,
                get_accessor<int32_t, 1>(i_curr_pair),
                get_accessor<int32_t, 2>(neighbors),
                get_accessor<scalar_t, 2>(deltas),
                get_accessor<scalar_t, 1>(distances),
                get_accessor<scalar_t, 2>(box_vectors));
        });
	//Check the error flag via cudaLaunchHostFunction so it is compatible with cuda graphs
	cudaHostFn_t h_fn = detail::checkTooManyNeighbors;
	static std::tuple<int, bool> h_fn_data;
	h_fn_data = {max_num_neighbors_, checkErrors};
	cudaLaunchHostFunc(stream, h_fn, (void*)&h_fn_data);
	//Errors are thrown as exceptions  asynchronously and in a way
	//compatible with CUDA graphs.   However, this way of throwing
	//an exception  makes it  uncatchable, crashing the  code.  If
	//the   checkErrors  flag   is   set  to   true  an   explicit
	//synchronization barrier  here forces to throw  the exception
	//from the main thread, making  it catchable at the expense of
	//a performance penalty each time the function is called.
	if(checkErrors){
	  cudaStreamSynchronize(stream);
	  if(detail::tooManyNeighborsException)
	    std::rethrow_exception(detail::tooManyNeighborsException);
	}
	ctx->save_for_backward({neighbors, deltas, distances});
        ctx->saved_data["num_atoms"] = num_atoms;
        return {neighbors, deltas, distances};
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_inputs) {

        const Tensor grad_deltas = grad_inputs[1];
        const Tensor grad_distances = grad_inputs[2];
        const int num_atoms = ctx->saved_data["num_atoms"].toInt();
        const int num_pairs = grad_distances.size(0);
        const int num_threads = 128;
        const int num_blocks_x = max((num_pairs + num_threads - 1) / num_threads, 1);
        const dim3 blocks(num_blocks_x, 2, 3);
        const auto stream = getCurrentCUDAStream(grad_distances.get_device());

        const tensor_list data = ctx->get_saved_variables();
        const Tensor neighbors = data[0];
        const Tensor deltas = data[1];
        const Tensor distances = data[2];
        const Tensor grad_positions = zeros({num_atoms, 3}, grad_distances.options());

        AT_DISPATCH_FLOATING_TYPES(grad_distances.scalar_type(), "getNeighborPairs::backward", [&]() {
            const CUDAStreamGuard guard(stream);
            backward_kernel<<<blocks, num_threads, 0, stream>>>(
                get_accessor<int32_t, 2>(neighbors),
                get_accessor<scalar_t, 2>(deltas),
                get_accessor<scalar_t, 2>(grad_deltas),
                get_accessor<scalar_t, 1>(distances),
                get_accessor<scalar_t, 1>(grad_distances),
                get_accessor<scalar_t, 2>(grad_positions));
        });

        return {grad_positions, Tensor(), Tensor(), Tensor(), Tensor()};
      }
};

TORCH_LIBRARY_IMPL(neighbors, AutogradCUDA, m) {
  m.impl("getNeighborPairs",
	 [](const Tensor& positions, const Scalar& cutoff, const Scalar& max_num_neighbors,
	    const Tensor& box_vectors, const bool &checkErrors){
	   const tensor_list results = Autograd::apply(positions, cutoff, max_num_neighbors,
						       box_vectors, checkErrors);
	   return make_tuple(results[0], results[1], results[2]);
	 });
}
