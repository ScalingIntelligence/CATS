#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

typedef struct half4 {
    half x, y, z, w;
} half4;

typedef struct bf16_4 {
    bf16 x, y, z, w;
} bf16_4;

template<int NUM, typename T>
struct ARRAY {
    T data[NUM];
};

// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/reduce_kernel_utils.cuh#L244
template<int NUM>
__inline__ __device__ void cgBlockReduceSumElements(float* element_list, float* cgBlockReduceSumElements_shm)
{
    cg::thread_block          cta  = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    const int tid    = cta.thread_rank();
    const int blockz = blockDim.x;
    for (int i = 0; i < NUM; i++) {
        cgBlockReduceSumElements_shm[i * blockz + tid] = cg::reduce(tile, element_list[i], cg::plus<float>());
    }
    cg::sync(cta);
    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < NUM; i++) {
            float beta = 0.0f;
            for (int j = 0; j < blockz; j += 32) {
                beta += cgBlockReduceSumElements_shm[i * blockz + j];
            }
            element_list[i] = beta;
        }
    }
}

template<int NUM>
__inline__ __device__ void cgFlagBlockReduceSumElements(float* element_list, bool* flag_list, float* cgBlockReduceSumElements_shm)
{
    cg::thread_block          cta  = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    const int tid    = cta.thread_rank();
    const int blockz = blockDim.x;
    for (int i = 0; i < NUM; i++) {
        if (!flag_list[i]) continue;
        cgBlockReduceSumElements_shm[i * blockz + tid] = cg::reduce(tile, element_list[i], cg::plus<float>());
    }
    cg::sync(cta);
    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < NUM; i++) {
            if (!flag_list[i]) continue;
            float beta = 0.0f;
            for (int j = 0; j < blockz; j += 32) {
                beta += cgBlockReduceSumElements_shm[i * blockz + j];
            }
            element_list[i] = beta;
        }
    }
}