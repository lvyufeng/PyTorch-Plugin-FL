// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../add.h"
#include "../../device_boxing.h"

#include <ATen/ops/add.h>

namespace at::native::flagos {

namespace {

// Reuse PyTorch's native CUDA add kernel (registered at runtime by an
// externally loaded libtorch_cuda.so) instead of hand-writing a .cu kernel.
// DeviceBoxingGuard temporarily rewrites the PrivateUse1 (flagos) tensors'
// device metadata to CUDA so the public at::add op dispatches to the CUDA
// kernel; CPU scalar inputs are left untouched. The result comes back tagged
// as CUDA, so we unbox it to flagos before returning.
at::Tensor AddKernelCuda(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  DeviceBoxingGuard guard(self, other);
  auto result = at::add(self, other, alpha);
  UnboxToFlagos(result);
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(AddTensorFn, add_tensor_dispatcher, Backend::kCuda, AddKernelCuda)

} // namespace at::native::flagos
