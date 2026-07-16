// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written .cu.
// DeviceBoxingGuard rewrites PrivateUse1 (flagos) tensors' device metadata to
// CUDA so the public at:: op dispatches to the CUDA kernel; the result comes
// back tagged CUDA, so we unbox it to flagos before returning.

#include "../../silu_backward.h"
#include "../../device_boxing.h"

#include <ATen/ops/silu_backward.h>

namespace at::native::flagos {
namespace {

at::Tensor SiluBackwardKernelCuda(const at::Tensor& grad_output, const at::Tensor& self) {
  DeviceBoxingGuard guard(grad_output, self);
  auto result = at::silu_backward(grad_output, self);
  UnboxToFlagos(result);
  return result;
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(SiluBackwardFn, silu_backward_dispatcher, Backend::kCuda, SiluBackwardKernelCuda)
} // namespace at::native::flagos
