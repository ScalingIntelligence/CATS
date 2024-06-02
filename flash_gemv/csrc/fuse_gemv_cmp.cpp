#include "./cuda/fuse_gemv_cmp_cuda.h"
#include <cuda_fp16.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <stdio.h>

int64_t fuse_gemv_cmp_cuda_impl(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &idx, const double threshold) {
  // call the cuda kernel if the input tensor is on cuda
  // printf("fuse_gemv_cmp_cuda_impl\n");
  TORCH_CHECK(input.sizes()[0] == 1 && input.sizes()[1] == 1 && input.sizes()[2] == 4096, "Input has to be (1,1,4096)");
  TORCH_CHECK(weight.sizes()[0] == 14336 && weight.sizes()[1] == 4096, "Input has to be (14336,4096)");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half, "Input has to be half");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half, "Weight has to be half");
  return fuse_gemv_cmp_dispatch(input, weight, output, idx, threshold);
}

void fuse_gemv_flag_cuda_impl(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag, const double threshold) {
  // TORCH_CHECK(input.sizes()[0] == 1 && input.sizes()[1] == 1 && input.sizes()[2] == 4096, "Input has to be (1,1,4096)");
  // TORCH_CHECK(weight.sizes()[0] == 14336 && weight.sizes()[1] == 4096, "Input has to be (14336,4096)");
  // TORCH_CHECK(input.scalar_type() == at::ScalarType::Half, "Input has to be half");
  // TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half, "Weight has to be half");
  fuse_gemv_flag_dispatch(input, weight, output, flag, threshold);
}

void fuse_gemv_flag_gemv_cuda_impl(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, at::Tensor &output, at::Tensor &flag, const double threshold) {
  TORCH_CHECK(input.sizes()[0] == 1 && input.sizes()[1] == 1 && input.sizes()[2] == 4096, "Input has to be (1,1,4096)");
  TORCH_CHECK(wgate.sizes()[0] == 14336 && wgate.sizes()[1] == 4096, "Gate weight has to be (14336,4096)");
  TORCH_CHECK(wup.sizes()[0] == 14336 && wup.sizes()[1] == 4096, "Up weight has to be (14336,4096)");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half, "Input has to be half");
  TORCH_CHECK(wgate.scalar_type() == at::ScalarType::Half, "Weight has to be half");
  TORCH_CHECK(wup.scalar_type() == at::ScalarType::Half, "Weight has to be half");
  fuse_gemv_flag_gemv_dispatch(input, wgate, wup, output, flag, threshold);
}

void fuse_gemv_flag_gemv_gemv_cuda_impl(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, at::Tensor &flag, const double threshold) {
  TORCH_CHECK(input.sizes()[0] == 1 && input.sizes()[1] == 1 && input.sizes()[2] == 4096, "Input has to be (1,1,4096)");
  TORCH_CHECK(wgate.sizes()[0] == 14336 && wgate.sizes()[1] == 4096, "Gate weight has to be (14336,4096)");
  TORCH_CHECK(wup.sizes()[0] == 14336 && wup.sizes()[1] == 4096, "Up weight has to be (14336,4096)");
  TORCH_CHECK(wdown.sizes()[0] == 14336 && wdown.sizes()[1] == 4096, "Down weight has to be (14336,4096)");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half, "Input has to be half");
  TORCH_CHECK(wgate.scalar_type() == at::ScalarType::Half, "Weight has to be half");
  TORCH_CHECK(wup.scalar_type() == at::ScalarType::Half, "Weight has to be half");
  TORCH_CHECK(wdown.scalar_type() == at::ScalarType::Half, "Weight has to be half");
  fuse_gemv_flag_gemv_gemv_dispatch(input, wgate, wup, wdown, output, flag, threshold);
}

void fuse_gemv_gemv_gemv_cuda_impl(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, const double threshold) {
  // TORCH_CHECK(input.sizes()[0] == 1 && input.sizes()[1] == 1 && input.sizes()[2] == 4096, "Input has to be (1,1,4096)");
  // TORCH_CHECK(wgate.sizes()[0] == 14336 && wgate.sizes()[1] == 4096, "Gate weight has to be (14336,4096)");
  // TORCH_CHECK(wup.sizes()[0] == 14336 && wup.sizes()[1] == 4096, "Up weight has to be (14336,4096)");
  // TORCH_CHECK(wdown.sizes()[0] == 14336 && wdown.sizes()[1] == 4096, "Down weight has to be (14336,4096)");
  // TORCH_CHECK(input.scalar_type() == at::ScalarType::Half, "Input has to be half");
  // TORCH_CHECK(wgate.scalar_type() == at::ScalarType::Half, "Weight has to be half");
  // TORCH_CHECK(wup.scalar_type() == at::ScalarType::Half, "Weight has to be half");
  // TORCH_CHECK(wdown.scalar_type() == at::ScalarType::Half, "Weight has to be half");
  // auto output = at::empty({1, 1, 4096}, input.options());
  fuse_gemv_gemv_gemv_dispatch(input, wgate, wup, wdown, output, threshold);
  // return output;
}

void fuse_gemv_flag_batch_cuda_impl(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag, const double threshold) {
  fuse_gemv_flag_batch_dispatch(input, weight, output, flag, threshold);
}

void fuse_gemv_flag_local_cuda_impl(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag, const double threshold) {
  fuse_gemv_flag_local_dispatch(input, weight, output, flag, threshold);
}

void fuse_flag_gemv_local_cuda_impl(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag) {
  fuse_flag_gemv_local_dispatch(input, weight, output, flag);
}

void atomic_gemv_cuda_impl(const at::Tensor &input, const at::Tensor &weight, at::Tensor &output, at::Tensor &flag) {
  atomic_gemv_dispatch(input, weight, output, flag);
}

void flag_gemv_gemv_atomic_cuda_impl(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, const double threshold) {
  flag_gemv_gemv_atomic_dispatch(input, wgate, wup, wdown, output, threshold);
}

void flag_gemv_gemv_cuda_impl(const at::Tensor &input, const at::Tensor &wgate, const at::Tensor &wup, const at::Tensor &wdown, at::Tensor &output, const double threshold) {
  flag_gemv_gemv_dispatch(input, wgate, wup, wdown, output, threshold);
}

TORCH_LIBRARY(flash_gemv, m) {
  m.def("fuse_gemv_cmp", fuse_gemv_cmp_cuda_impl);
  m.def("fuse_gemv_flag", fuse_gemv_flag_cuda_impl);
  m.def("fuse_gemv_flag_gemv", fuse_gemv_flag_gemv_cuda_impl);
  m.def("fuse_gemv_flag_gemv_gemv", fuse_gemv_flag_gemv_gemv_cuda_impl);
  m.def("fuse_gemv_gemv_gemv", fuse_gemv_gemv_gemv_cuda_impl);
  m.def("fuse_gemv_flag_batch", fuse_gemv_flag_batch_cuda_impl);
  m.def("fuse_gemv_flag_local", fuse_gemv_flag_local_cuda_impl);
  m.def("fuse_flag_gemv_local", fuse_flag_gemv_local_cuda_impl);
  m.def("atomic_gemv", atomic_gemv_cuda_impl);
  m.def("flag_gemv_gemv_atomic", flag_gemv_gemv_atomic_cuda_impl);
  m.def("flag_gemv_gemv", flag_gemv_gemv_cuda_impl);
  m.def("flag_gemv_gemv_inner", flag_gemv_gemv_inner_cuda_impl);
  m.def("flag_gemv_gemv_inner_fp32", flag_gemv_gemv_inner_cuda_impl_fp32);
  m.def("flag_gemv_gemv_inner_bf16", flag_gemv_gemv_inner_cuda_impl_bf16);
}
