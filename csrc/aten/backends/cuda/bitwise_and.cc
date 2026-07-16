// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written .cu.
// DeviceBoxingGuard rewrites PrivateUse1 (flagos) tensors' device metadata to
// CUDA so the public at:: op dispatches to the CUDA kernel; the result comes
// back tagged CUDA, so we unbox it to flagos before returning.

#include "../../bitwise_and.h"
#include "../../device_boxing.h"

#include <ATen/ops/bitwise_and.h>

namespace at::native::flagos {
namespace {

at::Tensor BitwiseAndKernelCuda(const at::Tensor& self, const at::Tensor& other) {
  DeviceBoxingGuard guard(self, other);
  auto result = at::bitwise_and(self, other);
  UnboxToFlagos(result);
  return result;
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(BitwiseAndTensorFn, bitwise_and_tensor_dispatcher, Backend::kCuda, BitwiseAndKernelCuda)
} // namespace at::native::flagos
