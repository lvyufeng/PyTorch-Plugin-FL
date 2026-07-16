// Copyright (c) 2026, BAAI. All rights reserved.
//
// Reuse PyTorch's native CUDA kernel (registered at runtime by an externally
// loaded libtorch_cuda.so) via device boxing, instead of a hand-written kernel
// or a structured_*_out_cuda subclass (which pins the build to CUDA-torch).

#include "../../cat.h"
#include "../../device_boxing.h"

#include <ATen/ops/cat.h>

namespace at::native::flagos {
namespace {

at::Tensor CatKernelCuda(const at::ITensorListRef& tensors, int64_t dim) {
  std::vector<at::Tensor> boxed;
  for (const auto& t : tensors) {
    if (t.defined() && t.is_privateuseone()) {
      BoxToCuda(t);
      boxed.push_back(t);
    }
  }
  auto result = at::cat(tensors, dim);
  for (const auto& t : boxed) {
    UnboxToFlagos(t);
  }
  UnboxToFlagos(result);
  return result;
}

} // namespace
REGISTER_IMPL_TO_DISPATCHER(CatFn, cat_dispatcher, Backend::kCuda, CatKernelCuda)
} // namespace at::native::flagos
