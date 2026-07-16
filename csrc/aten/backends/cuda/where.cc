// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written .cu.
// DeviceBoxingGuard rewrites PrivateUse1 (flagos) tensors' device metadata to
// CUDA so the public at:: op dispatches to the CUDA kernel; the result comes
// back tagged CUDA, so we unbox it to flagos before returning.

#include "../../where.h"
#include "../../device_boxing.h"

#include <ATen/ops/where.h>

namespace at::native::flagos {
namespace {

at::Tensor WhereKernelCuda(
    const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other) {
  DeviceBoxingGuard guard(condition, self, other);
  auto result = at::where(condition, self, other);
  UnboxToFlagos(result);
  return result;
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(WhereSelfFn, where_self_dispatcher, Backend::kCuda, WhereKernelCuda)
} // namespace at::native::flagos
