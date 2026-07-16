// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written kernel
// or a structured_*_out_cuda subclass (which pins the build to CUDA-torch).

#include "../../softmax.h"
#include "../../device_boxing.h"

#include <ATen/ops/_softmax.h>

namespace at::native::flagos {
namespace {

at::Tensor SoftmaxKernelCuda(const at::Tensor& self, int64_t dim, bool half_to_float) {
  DeviceBoxingGuard guard(self);
  auto result = at::_softmax(self, dim, half_to_float);
  UnboxToFlagos(result);
  return result;
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(SoftmaxFn, softmax_dispatcher, Backend::kCuda, SoftmaxKernelCuda)
} // namespace at::native::flagos
