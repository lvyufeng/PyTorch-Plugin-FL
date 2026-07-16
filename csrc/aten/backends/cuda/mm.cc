// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written .cu.
// out is pre-allocated (flagos), boxed to CUDA in place; at::mm_out writes into
// its storage (shared GPU memory). Guard unboxes self/mat2/out on destruction.

#include "../../mm.h"
#include "../../device_boxing.h"

#include <ATen/ops/mm.h>

namespace at::native::flagos {
namespace {

void MmKernelCuda(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  DeviceBoxingGuard guard(self, mat2, out);
  at::mm_out(out, self, mat2);
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kCuda, MmKernelCuda)
} // namespace at::native::flagos
