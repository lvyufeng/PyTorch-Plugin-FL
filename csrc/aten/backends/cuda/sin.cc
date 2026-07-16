// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written .cu.
// DeviceBoxingGuard rewrites PrivateUse1 (flagos) tensors' device metadata to
// CUDA so the public at:: op dispatches to the CUDA kernel; the result comes
// back tagged CUDA, so we unbox it to flagos before returning.

#include "../../sin.h"
#include "../../device_boxing.h"

#include <ATen/ops/sin.h>

namespace at::native::flagos {
namespace {

at::Tensor SinKernelCuda(const at::Tensor& self) {
  DeviceBoxingGuard guard(self);
  auto result = at::sin(self);
  UnboxToFlagos(result);
  return result;
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(SinFn, sin_dispatcher, Backend::kCuda, SinKernelCuda)
} // namespace at::native::flagos
