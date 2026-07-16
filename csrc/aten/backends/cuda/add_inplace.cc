// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written .cu.
// In-place: modifies self's storage directly (shared GPU memory), so we only
// unbox self/other back to flagos on guard destruction.

#include "../../add_inplace.h"
#include "../../device_boxing.h"

#include <ATen/ops/add.h>

namespace at::native::flagos {
namespace {

void AddInplaceKernelCuda(
    at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  DeviceBoxingGuard guard(self, other);
  self.add_(other, alpha);
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(AddInplaceTensorFn, add_inplace_tensor_dispatcher, Backend::kCuda, AddInplaceKernelCuda)
} // namespace at::native::flagos
