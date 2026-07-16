// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written .cu.
// DeviceBoxingGuard rewrites PrivateUse1 (flagos) tensors' device metadata to
// CUDA so the public at:: op dispatches to the CUDA kernel; the result comes
// back tagged CUDA, so we unbox it to flagos before returning.

#include "../../rsqrt.h"
#include "../../device_boxing.h"

#include <ATen/ops/rsqrt.h>

namespace at::native::flagos {
namespace {

at::Tensor RsqrtKernelCuda(const at::Tensor& self) {
  DeviceBoxingGuard guard(self);
  auto result = at::rsqrt(self);
  UnboxToFlagos(result);
  return result;
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(RsqrtFn, rsqrt_dispatcher, Backend::kCuda, RsqrtKernelCuda)
} // namespace at::native::flagos
