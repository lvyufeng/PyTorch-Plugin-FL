// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../embedding_dense_backward.h"
#include "../../device_boxing.h"

#include <ATen/ops/embedding_dense_backward.h>

namespace at::native::flagos {

namespace {

at::Tensor EmbeddingDenseBackwardKernelCuda(
    const at::Tensor& grad_output, const at::Tensor& indices,
    int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  DeviceBoxingGuard guard(grad_output, indices);
  auto result = at::embedding_dense_backward(
      grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
  UnboxToFlagos(result);
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(EmbeddingDenseBackwardFn, embedding_dense_backward_dispatcher, Backend::kCuda, EmbeddingDenseBackwardKernelCuda)

} // namespace at::native::flagos
