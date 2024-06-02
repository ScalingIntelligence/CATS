#pragma once

#include <torch/torch.h>
int64_t fuse_gemv_cmp_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &idx, double threshold);
void fuse_gemv_flag_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag, const double threshold);
void fuse_gemv_flag_gemv_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, at::Tensor &output, at::Tensor &flag, const double threshold);
void fuse_gemv_flag_gemv_gemv_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, at::Tensor &flag, const double threshold);
void fuse_gemv_gemv_gemv_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, const double threshold);
void fuse_gemv_flag_batch_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag, const double threshold);
void fuse_gemv_flag_local_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag, const double threshold);
void fuse_flag_gemv_local_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag);
void atomic_gemv_dispatch(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, const at::Tensor &flag);
void flag_gemv_gemv_atomic_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, const double threshold);
void flag_gemv_gemv_dispatch(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, const double threshold);
at::Tensor flag_gemv_gemv_inner_cuda_impl(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, const double threshold);
at::Tensor flag_gemv_gemv_inner_cuda_impl_fp32(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, const double threshold);
at::Tensor flag_gemv_gemv_inner_cuda_impl_bf16(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, const double threshold);
