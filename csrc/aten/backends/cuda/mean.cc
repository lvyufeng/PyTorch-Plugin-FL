// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written .cu.

#include "../../mean.h"
#include "../../device_boxing.h"

#include <ATen/ops/mean.h>

namespace at::native::flagos {
namespace {

at::Tensor MeanDimKernelCuda(
    const at::Tensor& self, at::OptionalIntArrayRef opt_dims,
    bool keepdim, std::optional<at::ScalarType> dtype) {
  DeviceBoxingGuard guard(self);
  auto result = at::mean(self, opt_dims, keepdim, dtype);
  UnboxToFlagos(result);
  return result;
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(MeanDimFn, mean_dim_dispatcher, Backend::kCuda, MeanDimKernelCuda)
} // namespace at::native::flagos
