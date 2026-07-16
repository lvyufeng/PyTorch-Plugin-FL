// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written kernel
// or a structured_*_out_cuda subclass (which pins the build to CUDA-torch).

#include "../../log_softmax.h"
#include "../../device_boxing.h"

#include <ATen/ops/_log_softmax.h>
#include <ATen/ops/_log_softmax_backward_data.h>
#include <ATen/ops/_softmax_backward_data.h>

namespace at::native::flagos {
namespace {

at::Tensor LogSoftmaxKernelCuda(const at::Tensor& self, int64_t dim, bool half_to_float) {
  DeviceBoxingGuard guard(self);
  auto result = at::_log_softmax(self, dim, half_to_float);
  UnboxToFlagos(result);
  return result;
}

at::Tensor LogSoftmaxBackwardKernelCuda(
    const at::Tensor& grad_output, const at::Tensor& output,
    int64_t dim, at::ScalarType input_dtype) {
  DeviceBoxingGuard guard(grad_output, output);
  auto result = at::_log_softmax_backward_data(grad_output, output, dim, input_dtype);
  UnboxToFlagos(result);
  return result;
}

at::Tensor SoftmaxBackwardKernelCuda(
    const at::Tensor& grad_output, const at::Tensor& output,
    int64_t dim, at::ScalarType input_dtype) {
  DeviceBoxingGuard guard(grad_output, output);
  auto result = at::_softmax_backward_data(grad_output, output, dim, input_dtype);
  UnboxToFlagos(result);
  return result;
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(LogSoftmaxFn, log_softmax_dispatcher, Backend::kCuda, LogSoftmaxKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(LogSoftmaxBackwardFn, log_softmax_backward_dispatcher, Backend::kCuda, LogSoftmaxBackwardKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(SoftmaxBackwardFn, softmax_backward_dispatcher, Backend::kCuda, SoftmaxBackwardKernelCuda)
} // namespace at::native::flagos
