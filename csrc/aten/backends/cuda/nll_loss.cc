// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written kernel
// or a structured_*_out_cuda subclass (which pins the build to CUDA-torch).

#include "../../nll_loss.h"
#include "../../device_boxing.h"

#include <ATen/ops/nll_loss_forward.h>
#include <ATen/ops/nll_loss_backward.h>

namespace at::native::flagos {
namespace {

std::tuple<at::Tensor, at::Tensor> NllLossForwardKernelCuda(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index) {
  at::Tensor weight_t = weight.has_value() ? *weight : at::Tensor();
  DeviceBoxingGuard guard(self, target, weight_t);
  auto result = at::nll_loss_forward(self, target, weight, reduction, ignore_index);
  UnboxToFlagos(std::get<0>(result));
  UnboxToFlagos(std::get<1>(result));
  return result;
}

at::Tensor NllLossBackwardKernelCuda(
    const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, const at::Tensor& total_weight) {
  at::Tensor weight_t = weight.has_value() ? *weight : at::Tensor();
  DeviceBoxingGuard guard(grad_output, self, target, weight_t, total_weight);
  auto result = at::nll_loss_backward(
      grad_output, self, target, weight, reduction, ignore_index, total_weight);
  UnboxToFlagos(result);
  return result;
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(NllLossForwardFn, nll_loss_forward_dispatcher, Backend::kCuda, NllLossForwardKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(NllLossBackwardFn, nll_loss_backward_dispatcher, Backend::kCuda, NllLossBackwardKernelCuda)
} // namespace at::native::flagos
