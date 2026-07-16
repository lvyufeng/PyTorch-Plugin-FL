// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written .cu.
// DeviceBoxingGuard rewrites PrivateUse1 (flagos) tensors' device metadata to
// CUDA so the public at:: op dispatches to the CUDA kernel; the result comes
// back tagged CUDA, so we unbox it to flagos before returning.

#include "../../mul_scalar.h"
#include "../../device_boxing.h"

#include <ATen/ops/mul.h>

namespace at::native::flagos {
namespace {

at::Tensor MulScalarKernelCuda(const at::Tensor& self, const at::Scalar& other) {
  DeviceBoxingGuard guard(self);
  auto result = at::mul(self, other);
  UnboxToFlagos(result);
  return result;
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(MulScalarFn, mul_scalar_dispatcher, Backend::kCuda, MulScalarKernelCuda)
} // namespace at::native::flagos
