#include "fuse_gemv_cmp_cuda.h"
#include "reduce_kernel_utils.cuh"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <torch/torch.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>



#define N 4096
#define M 14336

using namespace at::native;
using namespace cooperative_groups;

constexpr float log2e = 1.44269504088896340736f;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__forceinline__ __device__ float ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  }

__device__ __managed__ int length = 0;

__global__ void fuse_gemv_cmp_kernel(half* input, half* weight, half* output, int64_t* idx, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[64];
    half2 w_0 = make_half2(weight[m * N + warp_id * 128 + lane_id], weight[m * N + warp_id * 128 + lane_id + 32]);
    half2 w_1 = make_half2(weight[m * N + warp_id * 128 + lane_id + 64], weight[m * N + warp_id * 128 + lane_id + 96]);
    half2 x_0 = make_half2(input[warp_id * 128 + lane_id], input[warp_id * 128 + lane_id + 32]);
    half2 x_1 = make_half2(input[warp_id * 128 + lane_id + 64], input[warp_id * 128 + lane_id + 96]);


    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum_0 = __hmul2(w_0, x_0);
    half2 sum_1 = __hmul2(w_1, x_1);
    for (int i = 16; i > 0; i /= 2) {
        sum_0 = __hadd2(sum_0, group.shfl_down(sum_0, i));
        sum_1 = __hadd2(sum_1, group.shfl_down(sum_1, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_0;
        temp_result[warp_id + 32] = sum_1;
    }
    __syncthreads();
    if (warp_id == 0) {
        half2 sum_0 = temp_result[lane_id];
        half2 sum_1 = temp_result[lane_id + 32];
        for (int i = 16; i > 0; i /= 2) {
            sum_0 = __hadd2(sum_0, group.shfl_down(sum_0, i));
            sum_1 = __hadd2(sum_1, group.shfl_down(sum_1, i));
        }
        if (lane_id == 0) {
            half2 sum = __hadd2(sum_0, sum_1);
            float sum_f = __half2float(__hadd(sum.x, sum.y));
            sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
            if (fabsf(sum_f) > threshold) {
                int offset = atomicAdd(&length, 1);
                idx[offset] = (int64_t)m;
                output[offset] = __float2half_rn(sum_f);
            }
        }
    } 

}

__global__ void fuse_gemv_flag_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[64];
    half2 w_0 = make_half2(weight[m * N + warp_id * 128 + lane_id], weight[m * N + warp_id * 128 + lane_id + 32]);
    half2 w_1 = make_half2(weight[m * N + warp_id * 128 + lane_id + 64], weight[m * N + warp_id * 128 + lane_id + 96]);
    half2 x_0 = make_half2(input[warp_id * 128 + lane_id], input[warp_id * 128 + lane_id + 32]);
    half2 x_1 = make_half2(input[warp_id * 128 + lane_id + 64], input[warp_id * 128 + lane_id + 96]);


    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum_0 = __hmul2(w_0, x_0);
    half2 sum_1 = __hmul2(w_1, x_1);
    for (int i = 16; i > 0; i /= 2) {
        sum_0 = __hadd2(sum_0, group.shfl_down(sum_0, i));
        sum_1 = __hadd2(sum_1, group.shfl_down(sum_1, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_0;
        temp_result[warp_id + 32] = sum_1;
    }
    __syncthreads();
    if (warp_id == 0) {
        half2 sum_0 = temp_result[lane_id];
        half2 sum_1 = temp_result[lane_id + 32];
        for (int i = 16; i > 0; i /= 2) {
            sum_0 = __hadd2(sum_0, group.shfl_down(sum_0, i));
            sum_1 = __hadd2(sum_1, group.shfl_down(sum_1, i));
        }
        if (lane_id == 0) {
            half2 sum = __hadd2(sum_0, sum_1);
            float sum_f = __half2float(__hadd(sum.x, sum.y));
            sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
            if (fabsf(sum_f) > threshold) {
                flag[m] = true;
                output[m] = __float2half_rn(sum_f);
            }
        }
    } 
}

__forceinline__ __device__ uint64_t load_4half_to_uint64(const half* src) {
    half combined[4] = {src[0], src[32], src[64], src[96]};
    return *(uint64_t*)combined;
}

__forceinline__ __device__ uint64_t load_2half2_to_uint64(const half2* src) {
    half2 combined[2] = {src[0], src[32]};
    return *(uint64_t*)combined; 
}

__forceinline__ __device__ void store_uint64_to_2half2(uint64_t src, half2* dst) {
    half2* combined = (half2*)&src;
    dst[0] = combined[0];
    dst[32] = combined[1];
}

__forceinline__ __device__ uint64_t add_2half2_uint64(const uint64_t a, const uint64_t b) {
    half2 a_half[2];
    half2 b_half[2];
    half2 c_half[2];
    *(uint64_t*)a_half = a;
    *(uint64_t*)b_half = b;
    c_half[0] = __hadd2(a_half[0], b_half[0]);
    c_half[1] = __hadd2(a_half[1], b_half[1]);
    return *(uint64_t*)c_half;
}

__forceinline__ __device__ uint64_t mul_2half2_uint64(const uint64_t a, const uint64_t b) {
    half2 a_half[2];
    half2 b_half[2];
    half2 c_half[2];
    *(uint64_t*)a_half = a;
    *(uint64_t*)b_half = b;
    c_half[0] = __hmul2(a_half[0], b_half[0]);
    c_half[1] = __hmul2(a_half[1], b_half[1]);
    return *(uint64_t*)c_half;
}

__forceinline__ __device__ half2 reduce_2half2_uint64(const uint64_t a) {
    half2 a_half[2];
    *(uint64_t*)a_half = a;
    half2 sum = __hadd2(a_half[0], a_half[1]);
    return sum;
}

__global__ void fuse_gemv_flag_opt_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[64];
    uint64_t w = load_4half_to_uint64(weight + m * N + warp_id * 128 + lane_id);
    uint64_t x = load_4half_to_uint64(input + warp_id * 128 + lane_id);


    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    uint64_t sum = mul_2half2_uint64(w, x);
    for (int i = 16; i > 0; i /= 2) {
        sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        store_uint64_to_2half2(sum, temp_result + warp_id);
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = load_2half2_to_uint64(temp_result + lane_id);
        for (int i = 16; i > 0; i /= 2) {
            sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
        }
        if (lane_id == 0) {
            half2 sum_hf2 = reduce_2half2_uint64(sum);
            float sum_f = __half2float(__hadd(sum_hf2.x, sum_hf2.y));
            sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
            // if (fabsf(sum_f) > threshold) {
            flag[m] = fabsf(sum_f) > threshold;
            output[m] = __float2half_rn(sum_f);
            // }
        }
    } 
}

__global__ void fuse_gemv_flag_mem_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    half2* local_weight_ptr = (half2*)(weight + m * N + 512 * warp_id + 16 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 16 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        float sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum_f) > threshold;
        output[m] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_inter_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ uint64_t temp_result[32];
    uint64_t w = load_4half_to_uint64(weight + m * N + warp_id * 128 + lane_id);
    uint64_t x = load_4half_to_uint64(input + warp_id * 128 + lane_id);


    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    uint64_t sum = mul_2half2_uint64(w, x);
    for (int i = 16; i > 0; i /= 2) {
        sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = temp_result[lane_id];
        for (int i = 16; i > 0; i /= 2) {
            sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
        }
        if (lane_id == 0) {
            half2 sum_hf2 = reduce_2half2_uint64(sum);
            float sum_f = __half2float(__hadd(sum_hf2.x, sum_hf2.y));
            sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
            // if (fabsf(sum_f) > threshold) {
            flag[m] = fabsf(sum_f) > threshold;
            output[m] = __float2half_rn(sum_f);
            // }
        }
    } 
}

__forceinline__ __device__ uint64_t load_cont4half_to_uint64(const half* src) {
    half combined[4] = {src[0], src[1], src[2], src[3]};
    return *(uint64_t*)combined;
}

__global__ void fuse_gemv_flag_cont_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[64];
    uint64_t w = load_cont4half_to_uint64(weight + m * N + warp_id * 128 + lane_id * 4);
    uint64_t x = load_cont4half_to_uint64(input + warp_id * 128 + lane_id * 4);


    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    uint64_t sum = mul_2half2_uint64(w, x);
    for (int i = 16; i > 0; i /= 2) {
        sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        store_uint64_to_2half2(sum, temp_result + warp_id);
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = load_2half2_to_uint64(temp_result + lane_id);
        for (int i = 16; i > 0; i /= 2) {
            sum = add_2half2_uint64(sum, group.shfl_down(sum, i));
        }
        if (lane_id == 0) {
            half2 sum_hf2 = reduce_2half2_uint64(sum);
            float sum_f = __half2float(__hadd(sum_hf2.x, sum_hf2.y));
            sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
            // if (fabsf(sum_f) > threshold) {
            flag[m] = fabsf(sum_f) > threshold;
            output[m] = __float2half_rn(sum_f);
            // }
        }
    } 
}

__global__ void fuse_gemv_flag_mem1024_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[32];
    half2* local_weight_ptr = (half2*)(weight + m * N + 128 * warp_id + 4 * lane_id);
    half2* local_input_ptr = (half2*)(input + 128 * warp_id + 4 * lane_id);

    
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = temp_result[lane_id];
        #pragma unroll
        for (int i = 16; i > 0; i /= 2) {
            sum = __hadd2(sum, __shfl_down_sync(0xffffffff, sum, i));
        }
        if (lane_id == 0) {
            float sum_f = __half2float(__hadd(sum.x, sum.y));
            sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
            flag[m] = fabsf(sum_f) > threshold;
            output[m] = __float2half_rn(sum_f);
        }
    }
}

__global__ void fuse_gemv_flag_mem128_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[4];
    half2* local_weight_ptr = (half2*)(weight + m * N + 1024 * warp_id + 32 * lane_id);
    half2* local_input_ptr = (half2*)(input + 1024 * warp_id + 32 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 4; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        float sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum_f) > threshold;
        output[m] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_mem256_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    half2* local_weight_ptr = (half2*)(weight + m * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        float sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum_f) > threshold;
        output[m] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_mem256_fp32_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ float temp_result[8];
    half2* local_weight_ptr = (half2*)(weight + m * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        float2 w = __half22float2(local_weight_ptr[i]);
        float2 x = __half22float2(local_input_ptr[i]);
        sum = sum + w.y * x.y + w.x * x.x;
    }
    for (int i = 16; i > 0; i /= 2) {
        sum += group.shfl_down(sum, i);
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum += temp_result[i];
        }
        sum = sum / (1 + ptx_exp2(-sum * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum) > threshold;
        output[m] = __float2half_rn(sum);
    }
}

__global__ void fuse_gemv_flag_mem256_2_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int mz = threadIdx.z;
    int m = blockIdx.x * 2 + mz;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[16];
    half2* local_weight_ptr = (half2*)(weight + m * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id + mz * 8] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i + mz * 8]);
        }
        float sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum_f) > threshold;
        output[m] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_mem256_4_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int mz = threadIdx.z;
    int m = blockIdx.x * 4 + mz;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[32];
    half2* local_weight_ptr = (half2*)(weight + m * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id + mz * 8] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i + mz * 8]);
        }
        float sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum_f) > threshold;
        output[m] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_gemv_mem256_kernel(half* input, half* wgate, half* wup, half* output, bool* flag, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    __shared__ bool flag_shared[1];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum = __float2half2_rn(0.0f);
    float sum_f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        // flag[blockIdx.x] = fabsf(sum_f) > threshold;
        flag_shared[0] = fabsf(sum_f) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0] ;
    if (!flag_local) return;
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = (half2*)(input + 512 * warp_id + 2 * lane_id);
    half2 sum_up = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        sum_up = __hfma2(w, x, sum_up);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up = __hadd2(sum_up, group.shfl_down(sum_up, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum_up = __hadd2(sum_up, temp_result[i]);
        }
        sum_f = sum_f * __half2float(__hadd(sum_up.x, sum_up.y));
        flag[blockIdx.x] = flag_local;
        output[blockIdx.x] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_gemv_mem256_kernel1(half* input, half* wgate, half* wup, half* output, bool* flag, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    __shared__ bool flag_shared[1];
    __shared__ half2 input_shared[2048];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum = __float2half2_rn(0.0f);
    float sum_f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
        input_shared[warp_id * 256 + lane_id + i] = x;
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        // flag[blockIdx.x] = fabsf(sum_f) > threshold;
        flag_shared[0] = fabsf(sum_f) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0] ;
    if (!flag_local) return;
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = input_shared + warp_id * 256 + lane_id;
    half2 sum_up = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        sum_up = __hfma2(w, x, sum_up);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up = __hadd2(sum_up, group.shfl_down(sum_up, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum_up = __hadd2(sum_up, temp_result[i]);
        }
        sum_f = sum_f * __half2float(__hadd(sum_up.x, sum_up.y));
        flag[blockIdx.x] = flag_local;
        output[blockIdx.x] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_gemv_mem128_kernel(half* input, half* wgate, half* wup, half* output, bool* flag, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[4];
    __shared__ bool flag_shared[1];
    __shared__ half2 input_shared[2048];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 1024 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 1024 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum = __float2half2_rn(0.0f);
    float sum_f;
#pragma unroll
    for (int i = 0; i < 512; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
        input_shared[warp_id * 512 + lane_id + i] = x;
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 4; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        // flag[blockIdx.x] = fabsf(sum_f) > threshold;
        flag_shared[0] = fabsf(sum_f) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0] ;
    if (!flag_local) return;
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 1024 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = input_shared + warp_id * 512 + lane_id;
    half2 sum_up = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < 512; i+=32) {
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        sum_up = __hfma2(w, x, sum_up);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up = __hadd2(sum_up, group.shfl_down(sum_up, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 4; i++) {
            sum_up = __hadd2(sum_up, temp_result[i]);
        }
        sum_f = sum_f * __half2float(__hadd(sum_up.x, sum_up.y));
        flag[blockIdx.x] = flag_local;
        output[blockIdx.x] = __float2half_rn(sum_f);
    }
}

__global__ void fuse_gemv_flag_gemv_gemv_mem256_kernel(half* input, half* wgate, half* wup, half* wdown, float* output, bool* flag, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    __shared__ bool flag_shared[1];
    __shared__ float value_shared[1];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum = __float2half2_rn(0.0f);
    float sum_f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        // flag[blockIdx.x] = fabsf(sum_f) > threshold;
        flag_shared[0] = fabsf(sum_f) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0];
    if (!flag_local) return;

    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = (half2*)(input + 512 * warp_id + 2 * lane_id);
    half2 sum_up = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        sum_up = __hfma2(w, x, sum_up);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up = __hadd2(sum_up, group.shfl_down(sum_up, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum_up = __hadd2(sum_up, temp_result[i]);
        }
        sum_f = sum_f * __half2float(__hadd(sum_up.x, sum_up.y));
        flag[blockIdx.x] = true;
        value_shared[0] = sum_f;
        // output[blockIdx.x] = __float2half_rn(sum_f);
    }
    __syncthreads();
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_down = (half2*)(wdown + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    // half2* global_output_ptr = (half2*)(output + 512 * warp_id + 2 * lane_id);
    float* global_output_ptr = output + 512 * warp_id + 2 * lane_id;
    half2 sum_down = __float2half2_rn(value_shared[0]);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = __hmul2(local_weight_ptr_down[i], sum_down);
        // atomicAdd(global_output_ptr + i, __hmul2(w, sum_down));
        // atomicAdd(global_output_ptr + i * 2, __float2half2_rn(1.0f).x);
        // atomicAdd(global_output_ptr + i * 2 + 1, __float2half2_rn(1.0f).y);
        // atomicAdd(global_output_ptr + i * 2, w.x);
        // atomicAdd(global_output_ptr + i * 2 + 1, w.y);
        // atomicAdd(global_output_ptr + i * 2, __float2half_rn(0.000001f * (float)blockIdx.x));
        // atomicAdd(global_output_ptr + i * 2, 0.000001f * (float)blockIdx.x);
        atomicAdd(global_output_ptr + i * 2, __half2float(w.x));
        atomicAdd(global_output_ptr + i * 2 + 1, __half2float(w.y));
    }
    
}

__global__ void fuse_gemv_gemv_gemv_mem256_kernel(half* input, half* wgate, half* wup, half* wdown, float* output, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    __shared__ bool flag_shared[1];
    __shared__ float value_shared[1];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    half2 sum = __float2half2_rn(0.0f);
    float sum_f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        // flag[blockIdx.x] = fabsf(sum_f) > threshold;
        flag_shared[0] = fabsf(sum_f) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0];
    if (!flag_local) return;

    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = (half2*)(input + 512 * warp_id + 2 * lane_id);
    half2 sum_up = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        sum_up = __hfma2(w, x, sum_up);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up = __hadd2(sum_up, group.shfl_down(sum_up, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum_up = __hadd2(sum_up, temp_result[i]);
        }
        sum_f = sum_f * __half2float(__hadd(sum_up.x, sum_up.y));
        value_shared[0] = sum_f;
        // output[blockIdx.x] = __float2half_rn(sum_f);
    }
    __syncthreads();
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_down = (half2*)(wdown + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    // half2* global_output_ptr = (half2*)(output + 512 * warp_id + 2 * lane_id);
    float* global_output_ptr = output + 512 * warp_id + 2 * lane_id;
    half2 sum_down = __float2half2_rn(value_shared[0]);
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = __hmul2(local_weight_ptr_down[i], sum_down);
        atomicAdd(global_output_ptr + i * 2, __half2float(w.x));
        atomicAdd(global_output_ptr + i * 2 + 1, __half2float(w.y));
    }
    
}

__global__ void fuse_gemv_gemv_gemv_mem256_fp32_kernel(half* input, half* wgate, half* wup, half* wdown, float* output, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ float temp_result[8];
    __shared__ bool flag_shared[1];
    __shared__ float value_shared[1];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        // float2 w = __half22float2(local_weight_ptr[i]);
        // float2 x = __half22float2(local_input_ptr[i]);
        // sum = fmaf(w.x, x.x, sum);
        // sum = fmaf(w.y, x.y, sum);
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        float2 tmp = __half22float2(__hmul2(w, x));
        sum += (tmp.x + tmp.y);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum += group.shfl_down(sum, i);
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum += temp_result[i];
        }
        sum = sum / (1 + ptx_exp2(-sum * log2e)); // x * sigmoid(x)
        flag_shared[0] = fabsf(sum) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0];
    if (!flag_local) return;

    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = (half2*)(input + 512 * warp_id + 2 * lane_id);
    float sum_up = 0.0f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        // float2 w = __half22float2(local_weight_ptr_up[i]);
        // float2 x = __half22float2(local_input_ptr_up[i]);
        // sum_up = fmaf(w.x, x.x, sum_up);
        // sum_up = fmaf(w.y, x.y, sum_up);
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        float2 tmp = __half22float2(__hmul2(w, x));
        sum_up += (tmp.x + tmp.y);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up += group.shfl_down(sum_up, i);
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum_up += temp_result[i];
        }
        // sum *= sum_up;
        // value_shared[0] = __hmul(__float2half(sum), __float2half(sum_up));
        value_shared[0] = sum * sum_up;
    }
    __syncthreads();
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_down = (half2*)(wdown + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    float* global_output_ptr = output + 512 * warp_id + 2 * lane_id;
    float sum_down = value_shared[0];
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        // half2 w = local_weight_ptr_down[i];
        // atomicAdd(global_output_ptr + i * 2, __half2float(__hmul(w.x, sum_down)));
        // atomicAdd(global_output_ptr + i * 2 + 1, __half2float(__hmul(w.y, sum_down)));
        float2 w = __half22float2(local_weight_ptr_down[i]);
        atomicAdd(global_output_ptr + i * 2, w.x * sum_down);
        atomicAdd(global_output_ptr + i * 2 + 1, w.y * sum_down);
    }
    
}

__global__ void fuse_gemv_gemv_gemv_mem256_fp16_kernel(half* input, half* wgate, half* wup, half* wdown, half* output, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ float temp_result[8];
    __shared__ bool flag_shared[1];
    __shared__ float value_shared[1];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        // float2 w = __half22float2(local_weight_ptr[i]);
        // float2 x = __half22float2(local_input_ptr[i]);
        // sum = fmaf(w.x, x.x, sum);
        // sum = fmaf(w.y, x.y, sum);
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        float2 tmp = __half22float2(__hmul2(w, x));
        sum += (tmp.x + tmp.y);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum += group.shfl_down(sum, i);
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum += temp_result[i];
        }
        sum = sum / (1 + ptx_exp2(-sum * log2e)); // x * sigmoid(x)
        flag_shared[0] = fabsf(sum) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0];
    if (!flag_local) return;

    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = (half2*)(input + 512 * warp_id + 2 * lane_id);
    float sum_up = 0.0f;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        // float2 w = __half22float2(local_weight_ptr_up[i]);
        // float2 x = __half22float2(local_input_ptr_up[i]);
        // sum_up = fmaf(w.x, x.x, sum_up);
        // sum_up = fmaf(w.y, x.y, sum_up);
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        float2 tmp = __half22float2(__hmul2(w, x));
        sum_up += (tmp.x + tmp.y);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up += group.shfl_down(sum_up, i);
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum_up += temp_result[i];
        }
        // sum *= sum_up;
        // value_shared[0] = __hmul(__float2half(sum), __float2half(sum_up));
        value_shared[0] = sum * sum_up;
    }
    __syncthreads();
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_down = (half2*)(wdown + blockIdx.x * N + 512 * warp_id + 2 * lane_id);
    half* global_output_ptr = output + 512 * warp_id + 2 * lane_id;
    float sum_down = value_shared[0];
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        // half2 w = local_weight_ptr_down[i];
        // atomicAdd(global_output_ptr + i * 2, __half2float(__hmul(w.x, sum_down)));
        // atomicAdd(global_output_ptr + i * 2 + 1, __half2float(__hmul(w.y, sum_down)));
        float2 w = __half22float2(local_weight_ptr_down[i]);
        atomicAdd(global_output_ptr + i * 2, __float2half(w.x * sum_down));
        atomicAdd(global_output_ptr + i * 2 + 1, __float2half(w.y * sum_down));
    }
    
}

__global__ void fuse_gemv_gemv_gemv_mem512_fp32_kernel(half* input, half* wgate, half* wup, half* wdown, float* output, float threshold) {
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ float temp_result[16];
    __shared__ bool flag_shared[1];
    __shared__ float value_shared[1];
    half2* local_weight_ptr = (half2*)(wgate + blockIdx.x * N + 256 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 256 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 128; i+=32) {
        // float2 w = __half22float2(local_weight_ptr[i]);
        // float2 x = __half22float2(local_input_ptr[i]);
        // sum = fmaf(w.x, x.x, sum);
        // sum = fmaf(w.y, x.y, sum);
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        float2 tmp = __half22float2(__hmul2(w, x));
        sum += (tmp.x + tmp.y);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum += group.shfl_down(sum, i);
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 16; i++) {
            sum += temp_result[i];
        }
        sum = sum / (1 + ptx_exp2(-sum * log2e)); // x * sigmoid(x)
        flag_shared[0] = fabsf(sum) > threshold;
    }
    __syncthreads();
    bool flag_local = flag_shared[0];
    if (!flag_local) return;

    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_up = (half2*)(wup + blockIdx.x * N + 256 * warp_id + 2 * lane_id);
    half2* local_input_ptr_up = (half2*)(input + 256 * warp_id + 2 * lane_id);
    float sum_up = 0.0f;
#pragma unroll
    for (int i = 0; i < 128; i+=32) {
        // float2 w = __half22float2(local_weight_ptr_up[i]);
        // float2 x = __half22float2(local_input_ptr_up[i]);
        // sum_up = fmaf(w.x, x.x, sum_up);
        // sum_up = fmaf(w.y, x.y, sum_up);
        half2 w = local_weight_ptr_up[i];
        half2 x = local_input_ptr_up[i];
        float2 tmp = __half22float2(__hmul2(w, x));
        sum_up += (tmp.x + tmp.y);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum_up += group.shfl_down(sum_up, i);
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum_up;
    }
    __syncthreads();
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 16; i++) {
            sum_up += temp_result[i];
        }
        // sum *= sum_up;
        // value_shared[0] = __hmul(__float2half(sum), __float2half(sum_up));
        value_shared[0] = sum * sum_up;
    }
    __syncthreads();
    warp_id = threadIdx.y;
    lane_id = threadIdx.x;
    half2* local_weight_ptr_down = (half2*)(wdown + blockIdx.x * N + 256 * warp_id + 2 * lane_id);
    float* global_output_ptr = output + 256 * warp_id + 2 * lane_id;
    float sum_down = value_shared[0];
#pragma unroll
    for (int i = 0; i < 128; i+=32) {
        // half2 w = local_weight_ptr_down[i];
        // atomicAdd(global_output_ptr + i * 2, __half2float(__hmul(w.x, sum_down)));
        // atomicAdd(global_output_ptr + i * 2 + 1, __half2float(__hmul(w.y, sum_down)));
        float2 w = __half22float2(local_weight_ptr_down[i]);
        atomicAdd(global_output_ptr + i * 2, w.x * sum_down);
        atomicAdd(global_output_ptr + i * 2 + 1, w.y * sum_down);
    }
    
}

__global__ void fuse_gemv_flag_mem256_batch_kernel(half* input, half* weight, half* output, bool* flag, float threshold) {
    int m = blockIdx.x;
    int batch = blockIdx.y;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    __shared__ half2 temp_result[8];
    input = input + batch * N;
    flag = flag + batch * M;
    output = output + batch * M;
    half2* local_weight_ptr = (half2*)(weight + m * N + 512 * warp_id + 2 * lane_id);
    half2* local_input_ptr = (half2*)(input + 512 * warp_id + 2 * lane_id);

    thread_block_tile<32, thread_block> group = tiled_partition<32>(this_thread_block());
    // Accumulate to local (first land)
    // First lane stores the accumulated result to the shared memory
    // First warp reduces the shared memory, and atomically add to the global memory
    const half2 zerohf2 = __float2half2_rn(0.0f);
    half2 sum = zerohf2;
#pragma unroll
    for (int i = 0; i < 256; i+=32) {
        half2 w = local_weight_ptr[i];
        half2 x = local_input_ptr[i];
        sum = __hfma2(w, x, sum);
    }
    for (int i = 16; i > 0; i /= 2) {
        sum = __hadd2(sum, group.shfl_down(sum, i));
    }
    if (lane_id == 0) {
        temp_result[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0 and lane_id == 0) {
        #pragma unroll
        for (int i = 1; i < 8; i++) {
            sum = __hadd2(sum, temp_result[i]);
        }
        float sum_f = __half2float(__hadd(sum.x, sum.y));
        sum_f = sum_f / (1 + ptx_exp2(-sum_f * log2e)); // x * sigmoid(x)
        flag[m] = fabsf(sum_f) > threshold;
        output[m] = __float2half_rn(sum_f);
    }
}

extern __shared__ float cgBlockReduceSumElements_shm[];

// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/matrix_vector_multiplication.cu#L118
template<int m, int nPerThread>
__global__ void fuse_gemv_flag_local_kernel(const half4* input, const half4* weight, void* output, void* flag, float threshold, const int k_4) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    using array_half = struct ARRAY<nPerThread, half>;
    using array_bool = struct ARRAY<nPerThread, bool>;
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        half2 input_val_0[m];
        half2 input_val_1[m];
#pragma unroll 
        for (int m_i = 0; m_i < m; m_i++) {
            const half4 input_val = input[k_idx + m_i * k_4];
            input_val_0[m_i] = {input_val.x, input_val.y};
            input_val_1[m_i] = {input_val.z, input_val.w};
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const half4 weight_val = weight[b_offset + i * k_4 + k_idx];
            const half2 weight_val_0 = {weight_val.x, weight_val.y};
            const half2 weight_val_1 = {weight_val.z, weight_val.w};
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                const half2 weight_val_2 = 
                    __hadd2(__hmul2(input_val_0[m_i], weight_val_0), __hmul2(input_val_1[m_i], weight_val_1));
                sum_list[m_i].data[i] += __half2float(weight_val_2.x) + __half2float(weight_val_2.y);
            }
        }
    }

#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
        float tmp;
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array_half sum_list_half;
            array_bool flag_list;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                tmp = sum_list[m_i].data[i] / (1 + ptx_exp2(-sum_list[m_i].data[i] * log2e));
                sum_list_half.data[i] = __float2half_rn(tmp);
                flag_list.data[i] = fabsf(tmp) > threshold;
            }
            *((array_half*)output + bidx + m_i * gridDim.x) = sum_list_half;
            *((array_bool*)flag + bidx + m_i * gridDim.x) = flag_list;
        }
    }
}

// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/matrix_vector_multiplication.cu#L118
template<int m, int nPerThread>
__global__ void fuse_gemv_flag_local_kernel(const float4* input, const float4* weight, float* output, bool* flag, float threshold, const int k_4) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    using array_bool = struct ARRAY<nPerThread, bool>;
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        float input_val_x[m];
        float input_val_y[m];
        float input_val_z[m];
        float input_val_w[m];
#pragma unroll 
        for (int m_i = 0; m_i < m; m_i++) {
            const float4 input_val = input[k_idx + m_i * k_4];
            input_val_x[m_i] = input_val.x;
            input_val_y[m_i] = input_val.y;
            input_val_z[m_i] = input_val.z;
            input_val_w[m_i] = input_val.w;
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const float4 weight_val = weight[b_offset + i * k_4 + k_idx];
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                sum_list[m_i].data[i] += input_val_x[m_i] * weight_val.x + input_val_y[m_i] * weight_val.y + input_val_z[m_i] * weight_val.z + input_val_w[m_i] * weight_val.w;
            }
        }
    }

#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
        float tmp;
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array local_sum_list;
            array_bool flag_list;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                tmp = sum_list[m_i].data[i] / (1 + ptx_exp2(-sum_list[m_i].data[i] * log2e));
                flag_list.data[i] = fabsf(tmp) > threshold;
                local_sum_list.data[i] = tmp;
            }
            *((array*)output + bidx + m_i * gridDim.x) = local_sum_list;
            *((array_bool*)flag + bidx + m_i * gridDim.x) = flag_list;
        }
    }
}

// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/matrix_vector_multiplication.cu#L118
template<int m, int nPerThread>
__global__ void fuse_gemv_flag_local_kernel(const bf16_4* input, const bf16_4* weight, bf16* output, bool* flag, float threshold, const int k_4) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    using array_bf16 = struct ARRAY<nPerThread, bf16>;
    using array_bool = struct ARRAY<nPerThread, bool>;
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        bf16_2 input_val_0[m];
        bf16_2 input_val_1[m];
#pragma unroll 
        for (int m_i = 0; m_i < m; m_i++) {
            const bf16_4 input_val = input[k_idx + m_i * k_4];
            input_val_0[m_i] = {input_val.x, input_val.y};
            input_val_1[m_i] = {input_val.z, input_val.w};
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const bf16_4 weight_val = weight[b_offset + i * k_4 + k_idx];
            const bf16_2 weight_val_0 = {weight_val.x, weight_val.y};
            const bf16_2 weight_val_1 = {weight_val.z, weight_val.w};
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                const bf16_2 weight_val_2 = 
                    __hadd2(__hmul2(input_val_0[m_i], weight_val_0), __hmul2(input_val_1[m_i], weight_val_1));
                sum_list[m_i].data[i] += __bfloat162float(weight_val_2.x) + __bfloat162float(weight_val_2.y);
            }
        }
    }

#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
        float tmp;
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array_bf16 sum_list_half;
            array_bool flag_list;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                tmp = sum_list[m_i].data[i] / (1 + ptx_exp2(-sum_list[m_i].data[i] * log2e));
                sum_list_half.data[i] = __float2bfloat16_rn(tmp);
                flag_list.data[i] = fabsf(tmp) > threshold;
            }
            *((array_bf16*)output + bidx + m_i * gridDim.x) = sum_list_half;
            *((array_bool*)flag + bidx + m_i * gridDim.x) = flag_list;
        }
    }
}

template<typename T>
__inline__ __device__ bool loadExist(T* flag, const int m, const int i) {
#pragma unroll
    for (int j = 0; j < m; j++) {
        if (flag[j].data[i]) return true;
    }
    return false;
}    

template<typename T>
__inline__ __device__ bool storeExist(T* flag, const int m_i, const int i) {
#pragma unroll
    for (int j = 0; j < i; j++) {
        if (flag[m_i].data[j]) return true;
    }
    return false;
}   

// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/matrix_vector_multiplication.cu#L118
template<int m, int nPerThread>
__global__ void fuse_flag_gemv_local_kernel(const half4* input, const half4* wup, void* output, void* flag, const int k_4) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    using array_half = struct ARRAY<nPerThread, half>;
    using array_bool = struct ARRAY<nPerThread, bool>;
    array sum_list[m];
    array_bool flag_list[m];
    array_half gate_list[m];

#pragma unroll 
    for (int m_i = 0; m_i < m; m_i++) {
        flag_list[m_i] = *((array_bool*)flag + bidx + m_i * gridDim.x);
        gate_list[m_i] = *((array_half*)output + bidx + m_i * gridDim.x); // The output array stores the x * Wgate
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        half2 input_val_0[m];
        half2 input_val_1[m];
#pragma unroll 
        for (int m_i = 0; m_i < m; m_i++) {
            const half4 input_val = input[k_idx + m_i * k_4];
            input_val_0[m_i] = {input_val.x, input_val.y};
            input_val_1[m_i] = {input_val.z, input_val.w};
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            if (!loadExist<array_bool>(flag_list, m, i)) continue;
            const half4 wup_val = wup[b_offset + i * k_4 + k_idx];
            const half2 wup_val_0 = {wup_val.x, wup_val.y};
            const half2 wup_val_1 = {wup_val.z, wup_val.w};
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                if (!flag_list[m_i].data[i]) continue;
                const half2 wup_val_2 = 
                    __hadd2(__hmul2(input_val_0[m_i], wup_val_0), __hmul2(input_val_1[m_i], wup_val_1));
                sum_list[m_i].data[i] += __half2float(wup_val_2.x) + __half2float(wup_val_2.y);
            }
        }
    }

#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgFlagBlockReduceSumElements<nPerThread>(sum_list[m_i].data, flag_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array_half sum_list_half;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                if (flag_list[m_i].data[i]) {
                    sum_list_half.data[i] = __float2half_rn(sum_list[m_i].data[i] * __half2float(gate_list[m_i].data[i]));
                } else {
                    sum_list_half.data[i] = __float2half_rn(0.0f);
                }
            }
            *((array_half*)output + bidx + m_i * gridDim.x) = sum_list_half;
        }
    }

}

// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/matrix_vector_multiplication.cu#L118
template<int m, int nPerThread>
__global__ void fuse_flag_gemv_local_kernel(const float4* input, const float4* wup, float* output, bool* flag, const int k_4) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    using array_bool = struct ARRAY<nPerThread, bool>;
    array sum_list[m];
    array_bool flag_list[m];
    array gate_list[m];

#pragma unroll 
    for (int m_i = 0; m_i < m; m_i++) {
        flag_list[m_i] = *((array_bool*)flag + bidx + m_i * gridDim.x);
        gate_list[m_i] = *((array*)output + bidx + m_i * gridDim.x); // The output array stores the x * Wgate
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        float input_val_x[m];
        float input_val_y[m];
        float input_val_z[m];
        float input_val_w[m];
#pragma unroll 
        for (int m_i = 0; m_i < m; m_i++) {
            const float4 input_val = input[k_idx + m_i * k_4];
            input_val_x[m_i] = input_val.x;
            input_val_y[m_i] = input_val.y;
            input_val_z[m_i] = input_val.z;
            input_val_w[m_i] = input_val.w;
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            if (!loadExist<array_bool>(flag_list, m, i)) continue;
            const float4 wup_val = wup[b_offset + i * k_4 + k_idx];
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                if (!flag_list[m_i].data[i]) continue;
                sum_list[m_i].data[i] += input_val_x[m_i] * wup_val.x + input_val_y[m_i] * wup_val.y + input_val_z[m_i] * wup_val.z + input_val_w[m_i] * wup_val.w;
            }
        }
    }

#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgFlagBlockReduceSumElements<nPerThread>(sum_list[m_i].data, flag_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array local_sum_list;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                if (flag_list[m_i].data[i]) {
                    local_sum_list.data[i] = sum_list[m_i].data[i] * gate_list[m_i].data[i];
                } else {
                    local_sum_list.data[i] = 0.0f;
                }
            }
            *((array*)output + bidx + m_i * gridDim.x) = local_sum_list;
        }
    }

}

// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/matrix_vector_multiplication.cu#L118
template<int m, int nPerThread>
__global__ void fuse_flag_gemv_local_kernel(const bf16_4* input, const bf16_4* wup, bf16* output, bool* flag, const int k_4) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    using array_bf16 = struct ARRAY<nPerThread, bf16>;
    using array_bool = struct ARRAY<nPerThread, bool>;
    array sum_list[m];
    array_bool flag_list[m];
    array_bf16 gate_list[m];

#pragma unroll 
    for (int m_i = 0; m_i < m; m_i++) {
        flag_list[m_i] = *((array_bool*)flag + bidx + m_i * gridDim.x);
        gate_list[m_i] = *((array_bf16*)output + bidx + m_i * gridDim.x); // The output array stores the x * Wgate
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        bf16_2 input_val_0[m];
        bf16_2 input_val_1[m];
#pragma unroll 
        for (int m_i = 0; m_i < m; m_i++) {
            const bf16_4 input_val = input[k_idx + m_i * k_4];
            input_val_0[m_i] = {input_val.x, input_val.y};
            input_val_1[m_i] = {input_val.z, input_val.w};
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            if (!loadExist<array_bool>(flag_list, m, i)) continue;
            const bf16_4 wup_val = wup[b_offset + i * k_4 + k_idx];
            const bf16_2 wup_val_0 = {wup_val.x, wup_val.y};
            const bf16_2 wup_val_1 = {wup_val.z, wup_val.w};
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                if (!flag_list[m_i].data[i]) continue;
                const bf16_2 wup_val_2 = 
                    __hadd2(__hmul2(input_val_0[m_i], wup_val_0), __hmul2(input_val_1[m_i], wup_val_1));
                sum_list[m_i].data[i] += __bfloat162float(wup_val_2.x) + __bfloat162float(wup_val_2.y);
            }
        }
    }

#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgFlagBlockReduceSumElements<nPerThread>(sum_list[m_i].data, flag_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array_bf16 sum_list_half;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                if (flag_list[m_i].data[i]) {
                    sum_list_half.data[i] = __float2bfloat16_rn(sum_list[m_i].data[i] * __bfloat162float(gate_list[m_i].data[i]));
                } else {
                    sum_list_half.data[i] = __float2bfloat16_rn(0.0f);
                }
            }
            *((array_bf16*)output + bidx + m_i * gridDim.x) = sum_list_half;
        }
    }

}

template<int m, int kPerThread>
__global__ void atomic_gemv_kernel(const half* input, const half4* wdown, float* output, const bool* flag, const int n_4) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * kPerThread;
    const size_t b_offset = row_idx * n_4;

    using array = struct ARRAY<4, float>; // Output is grouped by 4
    using array_half = struct ARRAY<kPerThread, half>;
    using array_bool = struct ARRAY<kPerThread, bool>; // "Row-balance" for flag
    array_half input_list[m];
    array_bool flag_list[m];
    bool loadexists[kPerThread];
    bool storeexists[m];

#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        flag_list[m_i] = *((array_bool*)flag + bidx + m_i * gridDim.x);
        storeexists[m_i] = storeExist<array_bool>(flag_list, m_i, kPerThread);
#pragma unroll
        for (int k_idx = 0; k_idx < kPerThread; k_idx++) {
            input_list[m_i].data[k_idx] = input[row_idx + k_idx + m_i * gridDim.x * kPerThread];
        }
    }
#pragma unroll
    for (int k_idx = 0; k_idx < kPerThread; k_idx++) {
        loadexists[k_idx] = loadExist<array_bool>(flag_list, m, k_idx);
    }
#pragma unroll
    for (int n_idx = tidx; n_idx < n_4; n_idx += blockDim.x) {
        array sum_list[m];
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll 
            for (int i = 0; i < 4; i++) {
                sum_list[m_i].data[i] = 0.0f;
            }
        }
        for (int k_idx = 0; k_idx < kPerThread; k_idx++) {
            if(!loadexists[k_idx]) continue;
            const half4 wdown_val = wdown[b_offset + k_idx * n_4 + n_idx];
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                // if(!flag_list[m_i].data[k_idx]) continue;
                sum_list[m_i].data[0] += __half2float(__hmul(wdown_val.x, input_list[m_i].data[k_idx]));
                sum_list[m_i].data[1] += __half2float(__hmul(wdown_val.y, input_list[m_i].data[k_idx]));
                sum_list[m_i].data[2] += __half2float(__hmul(wdown_val.z, input_list[m_i].data[k_idx]));
                sum_list[m_i].data[3] += __half2float(__hmul(wdown_val.w, input_list[m_i].data[k_idx]));
            }
        }
        for (int m_i = 0; m_i < m; m_i++) {
            if (!storeexists[m_i]) continue;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                atomicAdd(output + (n_idx + m_i * n_4) * 4 + i, sum_list[m_i].data[i]);
            }
        }

    }
}

// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/matrix_vector_multiplication.cu#L118
template<int m, int nPerThread>
__global__ void gemv_local_kernel(const half4* input, const half4* weight, half* output, const int k_4) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    using array_half = struct ARRAY<nPerThread, half>;
    using array_bool = struct ARRAY<nPerThread, bool>;
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        half2 input_val_0[m];
        half2 input_val_1[m];
#pragma unroll 
        for (int m_i = 0; m_i < m; m_i++) {
            const half4 input_val = input[k_idx + m_i * k_4];
            input_val_0[m_i] = {input_val.x, input_val.y};
            input_val_1[m_i] = {input_val.z, input_val.w};
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const half4 weight_val = weight[b_offset + i * k_4 + k_idx];
            const half2 weight_val_0 = {weight_val.x, weight_val.y};
            const half2 weight_val_1 = {weight_val.z, weight_val.w};
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                const half2 weight_val_2 = 
                    __hadd2(__hmul2(input_val_0[m_i], weight_val_0), __hmul2(input_val_1[m_i], weight_val_1));
                sum_list[m_i].data[i] += __half2float(weight_val_2.x) + __half2float(weight_val_2.y);
            }
        }
    }

#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array_half sum_list_half;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                sum_list_half.data[i] = __float2half_rn(sum_list[m_i].data[i]);
            }
            *((array_half*)output + bidx + m_i * gridDim.x) = sum_list_half;
        }
    }
}

// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/matrix_vector_multiplication.cu#L118
template<int m, int nPerThread>
__global__ void gemv_local_kernel(const float4* input, const float4* weight, float* output, const int k_4) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    using array_bool = struct ARRAY<nPerThread, bool>;
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        float input_val_x[m];
        float input_val_y[m];
        float input_val_z[m];
        float input_val_w[m];
#pragma unroll 
        for (int m_i = 0; m_i < m; m_i++) {
            const float4 input_val = input[k_idx + m_i * k_4];
            input_val_x[m_i] = input_val.x;
            input_val_y[m_i] = input_val.y;
            input_val_z[m_i] = input_val.z;
            input_val_w[m_i] = input_val.w;
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const float4 weight_val = weight[b_offset + i * k_4 + k_idx];
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                sum_list[m_i].data[i] += input_val_x[m_i] * weight_val.x + input_val_y[m_i] * weight_val.y + input_val_z[m_i] * weight_val.z + input_val_w[m_i] * weight_val.w;
            }
        }
    }

#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            *((array*)output + bidx + m_i * gridDim.x) = sum_list[m_i];
        }
    }
}

// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/matrix_vector_multiplication.cu#L118
template<int m, int nPerThread>
__global__ void gemv_local_kernel(const bf16_4* input, const bf16_4* weight, bf16* output, const int k_4) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int row_idx = bidx * nPerThread;
    const size_t b_offset = row_idx * k_4;

    using array = struct ARRAY<nPerThread, float>;
    using array_bf16 = struct ARRAY<nPerThread, bf16>;
    using array_bool = struct ARRAY<nPerThread, bool>;
    array sum_list[m];
#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            sum_list[m_i].data[i] = 0.0f;
        }
    }

    for (int k_idx = tidx; k_idx < k_4; k_idx += blockDim.x) {
        bf16_2 input_val_0[m];
        bf16_2 input_val_1[m];
#pragma unroll 
        for (int m_i = 0; m_i < m; m_i++) {
            const bf16_4 input_val = input[k_idx + m_i * k_4];
            input_val_0[m_i] = {input_val.x, input_val.y};
            input_val_1[m_i] = {input_val.z, input_val.w};
        }
#pragma unroll
        for (int i = 0; i < nPerThread; i++) {
            const bf16_4 weight_val = weight[b_offset + i * k_4 + k_idx];
            const bf16_2 weight_val_0 = {weight_val.x, weight_val.y};
            const bf16_2 weight_val_1 = {weight_val.z, weight_val.w};
#pragma unroll
            for (int m_i = 0; m_i < m; m_i++) {
                const bf16_2 weight_val_2 = 
                    __hadd2(__hmul2(input_val_0[m_i], weight_val_0), __hmul2(input_val_1[m_i], weight_val_1));
                sum_list[m_i].data[i] += __bfloat162float(weight_val_2.x) + __bfloat162float(weight_val_2.y);
            }
        }
    }

#pragma unroll
    for (int m_i = 0; m_i < m; m_i++) {
        cgBlockReduceSumElements<nPerThread>(sum_list[m_i].data, cgBlockReduceSumElements_shm);
        __syncthreads();
    }
    if (tidx == 0) {
#pragma unroll
        for (int m_i = 0; m_i < m; m_i++) {
            array_bf16 sum_list_half;
#pragma unroll
            for (int i = 0; i < nPerThread; i++) {
                sum_list_half.data[i] = __float2bfloat16_rn(sum_list[m_i].data[i]);
            }
            *((array_bf16*)output + bidx + m_i * gridDim.x) = sum_list_half;
        }
    }
}

int64_t fuse_gemv_cmp_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &idx, const double threshold) {
    length = 0;
    // printf("threshold: %f\n", threshold);
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* weight_data = reinterpret_cast<half*>(weight.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    int64_t* idx_data = reinterpret_cast<int64_t*>(idx.data_ptr());

    dim3 gridDim(M,1,1);
    dim3 blockDim(32,32,1);

    fuse_gemv_cmp_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, idx_data, static_cast<float>(threshold));
    cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
    return length;
}

void fuse_gemv_flag_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* weight_data = reinterpret_cast<half*>(weight.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    bool* flag_data = reinterpret_cast<bool*>(flag.data_ptr());

    dim3 gridDim(M,1,1);
    // dim3 blockDim(32,4,1);
    // fuse_gemv_flag_mem128_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    dim3 blockDim(32,8,1);
    // fuse_gemv_flag_mem_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // fuse_gemv_flag_mem256_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    fuse_gemv_flag_mem256_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // dim3 blockDim(32,32,1);
    // fuse_gemv_flag_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // fuse_gemv_flag_opt_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // fuse_gemv_flag_inter_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // fuse_gemv_flag_cont_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // fuse_gemv_flag_mem1024_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    
    // dim3 gridDim(M/2,1,1);
    // dim3 blockDim(32,8,2);
    // fuse_gemv_flag_mem256_2_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // dim3 gridDim(M/4,1,1);
    // dim3 blockDim(32,8,4);
    // fuse_gemv_flag_mem256_4_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    // cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
}

void fuse_gemv_flag_gemv_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, at::Tensor &output, at::Tensor &flag, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* wgate_data = reinterpret_cast<half*>(wgate.data_ptr());
    half* wup_data = reinterpret_cast<half*>(wup.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    bool* flag_data = reinterpret_cast<bool*>(flag.data_ptr());

    dim3 gridDim(M,1,1);
    // dim3 blockDim(32,8,1);
    // fuse_gemv_flag_gemv_mem256_kernel1<<<gridDim, blockDim>>>(input_data, wgate_data, wup_data, output_data, flag_data, static_cast<float>(threshold));
    dim3 blockDim(32,4,1);
    fuse_gemv_flag_gemv_mem128_kernel<<<gridDim, blockDim>>>(input_data, wgate_data, wup_data, output_data, flag_data, static_cast<float>(threshold));
    cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
}

void fuse_gemv_flag_gemv_gemv_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, at::Tensor &flag, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* wgate_data = reinterpret_cast<half*>(wgate.data_ptr());
    half* wup_data = reinterpret_cast<half*>(wup.data_ptr());
    half* wdown_data = reinterpret_cast<half*>(wdown.data_ptr());
    float* output_data = reinterpret_cast<float*>(output.data_ptr());
    bool* flag_data = reinterpret_cast<bool*>(flag.data_ptr());

    dim3 gridDim(M,1,1);
    dim3 blockDim(32,8,1);
    fuse_gemv_flag_gemv_gemv_mem256_kernel<<<gridDim, blockDim>>>(input_data, wgate_data, wup_data, wdown_data, output_data, flag_data, static_cast<float>(threshold));
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void fuse_gemv_gemv_gemv_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* wgate_data = reinterpret_cast<half*>(wgate.data_ptr());
    half* wup_data = reinterpret_cast<half*>(wup.data_ptr());
    half* wdown_data = reinterpret_cast<half*>(wdown.data_ptr());
    float* output_data = reinterpret_cast<float*>(output.data_ptr());
    // half* output_data = reinterpret_cast<half*>(output.data_ptr());

    dim3 gridDim(M,1,1);
    // dim3 blockDim(32,8,1);
    // fuse_gemv_gemv_gemv_mem256_fp32_kernel<<<gridDim, blockDim>>>(input_data, wgate_data, wup_data, wdown_data, output_data, static_cast<float>(threshold));
    // fuse_gemv_gemv_gemv_mem256_fp16_kernel<<<gridDim, blockDim>>>(input_data, wgate_data, wup_data, wdown_data, output_data, static_cast<float>(threshold));
    dim3 blockDim(32,16,1);
    fuse_gemv_gemv_gemv_mem512_fp32_kernel<<<gridDim, blockDim>>>(input_data, wgate_data, wup_data, wdown_data, output_data, static_cast<float>(threshold));
    // cudaDeviceSynchronize();
    // gpuErrchk(cudaGetLastError());
}

void fuse_gemv_flag_batch_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* weight_data = reinterpret_cast<half*>(weight.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    bool* flag_data = reinterpret_cast<bool*>(flag.data_ptr());
    int batch_size = 1;
    if (input.dim() == 3) {
        batch_size = input.size(0) * input.size(1);
    } else if (input.dim() == 2) {
        batch_size = input.size(0);
    } else {
        printf("Input dimension is not supported\n");
        return;
    }
    dim3 gridDim(M,batch_size,1);
    dim3 blockDim(32,8,1);
    fuse_gemv_flag_mem256_batch_kernel<<<gridDim, blockDim>>>(input_data, weight_data, output_data, flag_data, static_cast<float>(threshold));
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

#define RUN(M, TYPE)                                                                                                   \
    fuse_gemv_flag_local_kernel<M, nPerThread><<<grid, block, shm_size>>>(                           \
        (const TYPE*)input_data, (const TYPE*)weight_data, (void*)output_data, (void*)flag_data, thf, k / 4);

void fuse_gemv_flag_local_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* weight_data = reinterpret_cast<half*>(weight.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    bool* flag_data = reinterpret_cast<bool*>(flag.data_ptr());
    float thf = static_cast<float>(threshold);
    
    int batch_size = 1;
    if (input.dim() == 3) {
        batch_size = input.size(0) * input.size(1);
    } else if (input.dim() == 2) {
        batch_size = input.size(0);
    } else {
        printf("[ERROR][fuse_gemv_flag_local_dispatch] Input dimension: %d is not supported.\n", input.dim());
        exit(-1);
    }
    const int n = weight.size(0);
    const int k = weight.size(1);
    const int nPerThread = 2;
    dim3 grid(n / nPerThread);
    dim3 block;
    block.x = 128;
    // printf("Launching grid (%d, %d, %d) block (%d, %d, %d) k_4 (%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, k / 4);
    const size_t shm_size = block.x * nPerThread * sizeof(float);
    if (batch_size == 1) {
        RUN(1, half4)
    } else if (batch_size == 2) {
        RUN(2, half4)
    } else {
        printf("[ERROR][fuse_gemv_flag_local_dispatch] not support batch == %d.\n", batch_size);
        exit(-1);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

#define RUN_FLAG_GEMV(M, TYPE)                                                                                                   \
    fuse_flag_gemv_local_kernel<M, nPerThread><<<grid, block, shm_size>>>(                           \
        (const TYPE*)input_data, (const TYPE*)weight_data, (void*)output_data, (void*)flag_data, k / 4);

void fuse_flag_gemv_local_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* weight_data = reinterpret_cast<half*>(weight.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    bool* flag_data = reinterpret_cast<bool*>(flag.data_ptr());
    
    int batch_size = 1;
    if (input.dim() == 3) {
        batch_size = input.size(0) * input.size(1);
    } else if (input.dim() == 2) {
        batch_size = input.size(0);
    } else {
        printf("[ERROR][fuse_gemv_flag_local_dispatch] Input dimension: %d is not supported.\n", input.dim());
        exit(-1);
    }
    const int n = weight.size(0);
    const int k = weight.size(1);
    const int nPerThread = 4;
    dim3 grid(n / nPerThread);
    dim3 block;
    block.x = 128;
    // printf("Launching grid (%d, %d, %d) block (%d, %d, %d) k_4 (%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, k / 4);
    const size_t shm_size = block.x * nPerThread * sizeof(float);
    if (batch_size == 1) {
        RUN_FLAG_GEMV(1, half4)
    } else if (batch_size == 2) {
        RUN_FLAG_GEMV(2, half4)
    } else {
        printf("[ERROR][fuse_gemv_flag_local_dispatch] not support batch == %d.\n", batch_size);
        exit(-1);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

#define RUN_ATOMIC_GEMV(M, TYPE)                                                                                                   \
    atomic_gemv_kernel<M, kPerThread><<<grid, block>>>(                           \
        (const half*)input_data, (const TYPE*)weight_data, output_data, (const bool*)flag_data, n / 4);

void atomic_gemv_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, const at::Tensor &flag) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* weight_data = reinterpret_cast<half*>(weight.data_ptr());
    float* output_data = reinterpret_cast<float*>(output.data_ptr());
    bool* flag_data = reinterpret_cast<bool*>(flag.data_ptr());
    
    int batch_size = 1;
    if (input.dim() == 3) {
        batch_size = input.size(0) * input.size(1);
    } else if (input.dim() == 2) {
        batch_size = input.size(0);
    } else {
        printf("[ERROR][atomic_gemv_dispatch] Input dimension: %d is not supported.\n", input.dim());
        exit(-1);
    }
    const int k = weight.size(0);
    const int n = weight.size(1);
    const int kPerThread = 16;
    dim3 grid(k / kPerThread);
    dim3 block;
    block.x = 128;
    if (batch_size == 1) {
        RUN_ATOMIC_GEMV(1, half4)
    } else if (batch_size == 2) {
        // printf("Launching grid (%d, %d, %d) block (%d, %d, %d) n_4 (%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, n / 4);
        RUN_ATOMIC_GEMV(2, half4)
    } else {
        printf("[ERROR][atomic_gemv_dispatch] not support batch == %d.\n", batch_size);
        exit(-1);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void flag_gemv_gemv_atomic_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* wgate_data = reinterpret_cast<half*>(wgate.data_ptr());
    half* wup_data = reinterpret_cast<half*>(wup.data_ptr());
    half* wdown_data = reinterpret_cast<half*>(wdown.data_ptr());
    float* output_data = reinterpret_cast<float*>(output.data_ptr());
    float thf = static_cast<float>(threshold);

    int batch_size = 1;
    if (input.dim() == 3) {
        batch_size = input.size(0) * input.size(1);
    } else if (input.dim() == 2) {
        batch_size = input.size(0);
    } else {
        printf("[ERROR][flag_gemv_gemv_atomic_dispatch] Input dimension: %d is not supported.\n", input.dim());
        exit(-1);
    }
    const int n = wgate.size(0);
    const int k = wgate.size(1);
    // Allocate a temporary memory for flag with elements batch_size * n
    bool* d_flag;
    cudaMalloc((void **)&d_flag, sizeof(float) * batch_size * n);
    half* d_intermediate;
    cudaMalloc((void **)&d_intermediate, sizeof(half) * batch_size * n);
    const int nPerThread = 2;
    dim3 grid(n / nPerThread);
    dim3 block;
    block.x = 128;
    // printf("Launching grid (%d, %d, %d) block (%d, %d, %d) k_4 (%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, k / 4);
    const size_t shm_size = block.x * nPerThread * sizeof(float);
    const int down_k = wdown.size(0);
    const int down_n = wdown.size(1);
    const int kPerThread = 16;
    dim3 grid_down(down_k / kPerThread);
    dim3 block_down;
    block_down.x = 128;
    if (batch_size == 1) {
        fuse_gemv_flag_local_kernel<1, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wgate_data, (void*)d_intermediate, (void*)d_flag, thf, k / 4);
        fuse_flag_gemv_local_kernel<1, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wup_data, (void*)d_intermediate, (void*)d_flag, k / 4);
        atomic_gemv_kernel<1, kPerThread><<<grid_down, block_down>>>(
            (const half*)d_intermediate, (const half4*)wdown_data, output_data, (const bool*)d_flag, down_n / 4);
    } else if (batch_size == 2) {
        fuse_gemv_flag_local_kernel<2, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wgate_data, (void*)d_intermediate, (void*)d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<2, nPerThread><<<grid, block, shm_size>>>(
            (const half4*)input_data, (const half4*)wup_data, (void*)d_intermediate, (void*)d_flag, k / 4);  
        atomic_gemv_kernel<2, kPerThread><<<grid_down, block_down>>>(
            (const half*)d_intermediate, (const half4*)wdown_data, output_data, (const bool*)d_flag, down_n / 4);
    } else {
        printf("[ERROR][flag_gemv_gemv_atomic_dispatch] not support batch == %d.\n", batch_size);
        exit(-1);
    }
    cudaDeviceSynchronize();
    cudaFree(d_flag);
    cudaFree(d_intermediate);
}

void flag_gemv_gemv_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, const double threshold) {
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* wgate_data = reinterpret_cast<half*>(wgate.data_ptr());
    half* wup_data = reinterpret_cast<half*>(wup.data_ptr());
    half* wdown_data = reinterpret_cast<half*>(wdown.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    float thf = static_cast<float>(threshold);

    int batch_size = 1;
    if (input.dim() == 3) {
        batch_size = input.size(0) * input.size(1);
    } else if (input.dim() == 2) {
        batch_size = input.size(0);
    } else {
        printf("[ERROR][flag_gemv_gemv_atomic_dispatch] Input dimension: %d is not supported.\n", input.dim());
        exit(-1);
    }
    const int n = wgate.size(0);
    const int k = wgate.size(1);
    // Allocate a temporary memory for flag with elements batch_size * n
    bool* d_flag;
    cudaMalloc((void **)&d_flag, sizeof(float) * batch_size * n);
    half* d_intermediate;
    cudaMalloc((void **)&d_intermediate, sizeof(half) * batch_size * n);
    const int nPerThread = 2;
    dim3 grid(n / nPerThread);
    dim3 block;
    block.x = 128;
    // printf("Launching grid (%d, %d, %d) block (%d, %d, %d) k_4 (%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, k / 4);
    const size_t shm_size = block.x * nPerThread * sizeof(float);
    const int down_n = wdown.size(0);
    const int down_k = wdown.size(1);
    const int down_nPerThread = 2;
    dim3 grid_down(down_n / down_nPerThread);
    dim3 block_down;
    block_down.x = 256;
    const size_t down_shm_size = block_down.x * down_nPerThread * sizeof(float);
    if (batch_size == 1) {
        fuse_gemv_flag_local_kernel<1, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wgate_data, (void*)d_intermediate, (void*)d_flag, thf, k / 4);
        fuse_flag_gemv_local_kernel<1, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wup_data, (void*)d_intermediate, (void*)d_flag, k / 4);
        gemv_local_kernel<1, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const half4*)d_intermediate, (const half4*)wdown_data, output_data, down_k / 4);
    } else if (batch_size == 2) {
        fuse_gemv_flag_local_kernel<2, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wgate_data, (void*)d_intermediate, (void*)d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<2, nPerThread><<<grid, block, shm_size>>>(
            (const half4*)input_data, (const half4*)wup_data, (void*)d_intermediate, (void*)d_flag, k / 4);  
        gemv_local_kernel<2, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const half4*)d_intermediate, (const half4*)wdown_data, output_data, down_k / 4);
    } else if (batch_size == 8) {
        fuse_gemv_flag_local_kernel<8, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wgate_data, (void*)d_intermediate, (void*)d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<8, nPerThread><<<grid, block, shm_size>>>(
            (const half4*)input_data, (const half4*)wup_data, (void*)d_intermediate, (void*)d_flag, k / 4);  
        gemv_local_kernel<8, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const half4*)d_intermediate, (const half4*)wdown_data, output_data, down_k / 4);        
    } else if (batch_size == 16) {
        fuse_gemv_flag_local_kernel<16, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wgate_data, (void*)d_intermediate, (void*)d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<16, nPerThread><<<grid, block, shm_size>>>(
            (const half4*)input_data, (const half4*)wup_data, (void*)d_intermediate, (void*)d_flag, k / 4);  
        gemv_local_kernel<16, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const half4*)d_intermediate, (const half4*)wdown_data, output_data, down_k / 4);
    }
    else {
        printf("[ERROR][flag_gemv_gemv_atomic_dispatch] not support batch == %d.\n", batch_size);
        exit(-1);
    }
    cudaDeviceSynchronize();
    cudaFree(d_flag);
    cudaFree(d_intermediate);
}

at::Tensor flag_gemv_gemv_inner_cuda_impl(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, const double threshold) {
    auto output = at::empty(input.sizes(), input.options());
    half* input_data = reinterpret_cast<half*>(input.data_ptr());
    half* wgate_data = reinterpret_cast<half*>(wgate.data_ptr());
    half* wup_data = reinterpret_cast<half*>(wup.data_ptr());
    half* wdown_data = reinterpret_cast<half*>(wdown.data_ptr());
    half* output_data = reinterpret_cast<half*>(output.data_ptr());
    float thf = static_cast<float>(threshold);

    int batch_size = 1;
    if (input.dim() == 3) {
        batch_size = input.size(0) * input.size(1);
    } else if (input.dim() == 2) {
        batch_size = input.size(0);
    } else {
        printf("[ERROR][flag_gemv_gemv_atomic_dispatch] Input dimension: %d is not supported.\n", input.dim());
        exit(-1);
    }
    const int n = wgate.size(0);
    const int k = wgate.size(1);
    // Allocate a temporary memory for flag with elements batch_size * n
    bool* d_flag;
    cudaMalloc((void **)&d_flag, sizeof(bool) * batch_size * n);
    half* d_intermediate;
    cudaMalloc((void **)&d_intermediate, sizeof(half) * batch_size * n);
    const int nPerThread = 2;
    dim3 grid(n / nPerThread);
    dim3 block;
    block.x = 128;
    // printf("Launching grid (%d, %d, %d) block (%d, %d, %d) k_4 (%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, k / 4);
    const size_t shm_size = block.x * nPerThread * sizeof(float);
    const int down_n = wdown.size(0);
    const int down_k = wdown.size(1);
    const int down_nPerThread = 2;
    dim3 grid_down(down_n / down_nPerThread);
    dim3 block_down;
    block_down.x = 256;
    const size_t down_shm_size = block_down.x * down_nPerThread * sizeof(float);
    if (batch_size == 1) {
        fuse_gemv_flag_local_kernel<1, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wgate_data, (void*)d_intermediate, (void*)d_flag, thf, k / 4);
        fuse_flag_gemv_local_kernel<1, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wup_data, (void*)d_intermediate, (void*)d_flag, k / 4);
        gemv_local_kernel<1, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const half4*)d_intermediate, (const half4*)wdown_data, output_data, down_k / 4);
    } else if (batch_size == 2) {
        fuse_gemv_flag_local_kernel<2, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wgate_data, (void*)d_intermediate, (void*)d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<2, nPerThread><<<grid, block, shm_size>>>(
            (const half4*)input_data, (const half4*)wup_data, (void*)d_intermediate, (void*)d_flag, k / 4);  
        gemv_local_kernel<2, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const half4*)d_intermediate, (const half4*)wdown_data, output_data, down_k / 4);
    } else if (batch_size == 8) {
        fuse_gemv_flag_local_kernel<8, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wgate_data, (void*)d_intermediate, (void*)d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<8, nPerThread><<<grid, block, shm_size>>>(
            (const half4*)input_data, (const half4*)wup_data, (void*)d_intermediate, (void*)d_flag, k / 4);  
        gemv_local_kernel<8, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const half4*)d_intermediate, (const half4*)wdown_data, output_data, down_k / 4);        
    } else if (batch_size == 16) {
        fuse_gemv_flag_local_kernel<16, nPerThread><<<grid, block, shm_size>>>(                           
            (const half4*)input_data, (const half4*)wgate_data, (void*)d_intermediate, (void*)d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<16, nPerThread><<<grid, block, shm_size>>>(
            (const half4*)input_data, (const half4*)wup_data, (void*)d_intermediate, (void*)d_flag, k / 4);  
        gemv_local_kernel<16, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const half4*)d_intermediate, (const half4*)wdown_data, output_data, down_k / 4);
    }
    else {
        printf("[ERROR][flag_gemv_gemv_atomic_dispatch] not support batch == %d.\n", batch_size);
        exit(-1);
    }
    // cudaDeviceSynchronize();
    cudaFree(d_flag);
    cudaFree(d_intermediate);
    return output;
}

at::Tensor flag_gemv_gemv_inner_cuda_impl_fp32(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, const double threshold) {
    auto output = at::empty(input.sizes(), input.options());
    float* input_data = reinterpret_cast<float*>(input.data_ptr());
    float* wgate_data = reinterpret_cast<float*>(wgate.data_ptr());
    float* wup_data = reinterpret_cast<float*>(wup.data_ptr());
    float* wdown_data = reinterpret_cast<float*>(wdown.data_ptr());
    float* output_data = reinterpret_cast<float*>(output.data_ptr());
    float thf = static_cast<float>(threshold);

    int batch_size = 1;
    if (input.dim() == 3) {
        batch_size = input.size(0) * input.size(1);
    } else if (input.dim() == 2) {
        batch_size = input.size(0);
    } else {
        printf("[ERROR][flag_gemv_gemv_inner_cuda_impl_fp32] Input dimension: %d is not supported.\n", input.dim());
        exit(-1);
    }
    const int n = wgate.size(0);
    const int k = wgate.size(1);
    // Allocate a temporary memory for flag with elements batch_size * n
    bool* d_flag;
    cudaMalloc((void **)&d_flag, sizeof(float) * batch_size * n);
    float* d_intermediate;
    cudaMalloc((void **)&d_intermediate, sizeof(float) * batch_size * n);
    const int nPerThread = 2;
    dim3 grid(n / nPerThread);
    dim3 block;
    block.x = 128;
    // printf("Launching grid (%d, %d, %d) block (%d, %d, %d) k_4 (%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, k / 4);
    const size_t shm_size = block.x * nPerThread * sizeof(float);
    const int down_n = wdown.size(0);
    const int down_k = wdown.size(1);
    const int down_nPerThread = 2;
    dim3 grid_down(down_n / down_nPerThread);
    dim3 block_down;
    block_down.x = 256;
    const size_t down_shm_size = block_down.x * down_nPerThread * sizeof(float);
    if (batch_size == 1) {
        fuse_gemv_flag_local_kernel<1, nPerThread><<<grid, block, shm_size>>>(                           
            (const float4*)input_data, (const float4*)wgate_data, d_intermediate, d_flag, thf, k / 4);
        fuse_flag_gemv_local_kernel<1, nPerThread><<<grid, block, shm_size>>>(                           
            (const float4*)input_data, (const float4*)wup_data, d_intermediate, d_flag, k / 4);
        gemv_local_kernel<1, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const float4*)d_intermediate, (const float4*)wdown_data, output_data, down_k / 4);
    } else if (batch_size == 2) {
        fuse_gemv_flag_local_kernel<2, nPerThread><<<grid, block, shm_size>>>(                           
            (const float4*)input_data, (const float4*)wgate_data, d_intermediate, d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<2, nPerThread><<<grid, block, shm_size>>>(
            (const float4*)input_data, (const float4*)wup_data, d_intermediate, d_flag, k / 4);  
        gemv_local_kernel<2, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const float4*)d_intermediate, (const float4*)wdown_data, output_data, down_k / 4);
    } else if (batch_size == 8) {
        fuse_gemv_flag_local_kernel<8, nPerThread><<<grid, block, shm_size>>>(                           
            (const float4*)input_data, (const float4*)wgate_data, d_intermediate, d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<8, nPerThread><<<grid, block, shm_size>>>(
            (const float4*)input_data, (const float4*)wup_data, d_intermediate, d_flag, k / 4);  
        gemv_local_kernel<8, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const float4*)d_intermediate, (const float4*)wdown_data, output_data, down_k / 4);        
    } else if (batch_size == 16) {
        fuse_gemv_flag_local_kernel<16, nPerThread><<<grid, block, shm_size>>>(                           
            (const float4*)input_data, (const float4*)wgate_data, d_intermediate, d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<16, nPerThread><<<grid, block, shm_size>>>(
            (const float4*)input_data, (const float4*)wup_data, d_intermediate, d_flag, k / 4);  
        gemv_local_kernel<16, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const float4*)d_intermediate, (const float4*)wdown_data, output_data, down_k / 4);
    }
    else {
        printf("[ERROR][flag_gemv_gemv_inner_cuda_impl_fp32] not support batch == %d.\n", batch_size);
        exit(-1);
    }
    // cudaDeviceSynchronize();
    cudaFree(d_flag);
    cudaFree(d_intermediate);
    return output;
}

at::Tensor flag_gemv_gemv_inner_cuda_impl_bf16(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, const double threshold) {
    auto output = at::empty(input.sizes(), input.options());
    bf16* input_data = reinterpret_cast<bf16*>(input.data_ptr());
    bf16* wgate_data = reinterpret_cast<bf16*>(wgate.data_ptr());
    bf16* wup_data = reinterpret_cast<bf16*>(wup.data_ptr());
    bf16* wdown_data = reinterpret_cast<bf16*>(wdown.data_ptr());
    bf16* output_data = reinterpret_cast<bf16*>(output.data_ptr());
    float thf = static_cast<float>(threshold);

    int batch_size = 1;
    if (input.dim() == 3) {
        batch_size = input.size(0) * input.size(1);
    } else if (input.dim() == 2) {
        batch_size = input.size(0);
    } else {
        printf("[ERROR][flag_gemv_gemv_inner_cuda_impl_bf16] Input dimension: %d is not supported.\n", input.dim());
        exit(-1);
    }
    const int n = wgate.size(0);
    const int k = wgate.size(1);
    // Allocate a temporary memory for flag with elements batch_size * n
    bool* d_flag;
    cudaMalloc((void **)&d_flag, sizeof(bool) * batch_size * n);
    bf16* d_intermediate;
    cudaMalloc((void **)&d_intermediate, sizeof(bf16) * batch_size * n);
    const int nPerThread = 2;
    dim3 grid(n / nPerThread);
    dim3 block;
    block.x = 128;
    // printf("Launching grid (%d, %d, %d) block (%d, %d, %d) k_4 (%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, k / 4);
    const size_t shm_size = block.x * nPerThread * sizeof(float);
    const int down_n = wdown.size(0);
    const int down_k = wdown.size(1);
    const int down_nPerThread = 2;
    dim3 grid_down(down_n / down_nPerThread);
    dim3 block_down;
    block_down.x = 256;
    const size_t down_shm_size = block_down.x * down_nPerThread * sizeof(float);
    if (batch_size == 1) {
        fuse_gemv_flag_local_kernel<1, nPerThread><<<grid, block, shm_size>>>(                           
            (const bf16_4*)input_data, (const bf16_4*)wgate_data, d_intermediate, d_flag, thf, k / 4);
        fuse_flag_gemv_local_kernel<1, nPerThread><<<grid, block, shm_size>>>(                           
            (const bf16_4*)input_data, (const bf16_4*)wup_data, d_intermediate, d_flag, k / 4);
        gemv_local_kernel<1, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const bf16_4*)d_intermediate, (const bf16_4*)wdown_data, output_data, down_k / 4);
    } else if (batch_size == 2) {
        fuse_gemv_flag_local_kernel<2, nPerThread><<<grid, block, shm_size>>>(                           
            (const bf16_4*)input_data, (const bf16_4*)wgate_data, d_intermediate, d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<2, nPerThread><<<grid, block, shm_size>>>(
            (const bf16_4*)input_data, (const bf16_4*)wup_data, d_intermediate, d_flag, k / 4);  
        gemv_local_kernel<2, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const bf16_4*)d_intermediate, (const bf16_4*)wdown_data, output_data, down_k / 4);
    } else if (batch_size == 8) {
        fuse_gemv_flag_local_kernel<8, nPerThread><<<grid, block, shm_size>>>(                           
            (const bf16_4*)input_data, (const bf16_4*)wgate_data, d_intermediate, d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<8, nPerThread><<<grid, block, shm_size>>>(
            (const bf16_4*)input_data, (const bf16_4*)wup_data, d_intermediate, d_flag, k / 4);  
        gemv_local_kernel<8, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const bf16_4*)d_intermediate, (const bf16_4*)wdown_data, output_data, down_k / 4);        
    } else if (batch_size == 16) {
        fuse_gemv_flag_local_kernel<16, nPerThread><<<grid, block, shm_size>>>(                           
            (const bf16_4*)input_data, (const bf16_4*)wgate_data, d_intermediate, d_flag, thf, k / 4);      
        fuse_flag_gemv_local_kernel<16, nPerThread><<<grid, block, shm_size>>>(
            (const bf16_4*)input_data, (const bf16_4*)wup_data, d_intermediate, d_flag, k / 4);  
        gemv_local_kernel<16, down_nPerThread><<<grid_down, block_down, down_shm_size>>>(
            (const bf16_4*)d_intermediate, (const bf16_4*)wdown_data, output_data, down_k / 4);
    }
    else {
        printf("[ERROR][flag_gemv_gemv_inner_cuda_impl_bf16] not support batch == %d.\n", batch_size);
        exit(-1);
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(d_flag);
    cudaFree(d_intermediate);
    return output;
}