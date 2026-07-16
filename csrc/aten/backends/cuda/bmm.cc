// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written .cu.
// See mm.cc for the boxing rationale.

#include "../../bmm.h"
#include "../../device_boxing.h"

#include <ATen/ops/bmm.h>

namespace at::native::flagos {
namespace {

void BmmKernelCuda(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  DeviceBoxingGuard guard(self, mat2, out);
  at::bmm_out(out, self, mat2);
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(BmmFn, bmm_dispatcher, Backend::kCuda, BmmKernelCuda)
} // namespace at::native::flagos
