// Copyright (c) 2026, BAAI. All rights reserved.
//
// scaled_dot_product_attention composite override for CUDA-boxing platforms
// (DCU via DTK libtorch_hip, MetaX/maca, PPU/tsingmicro, and any CUDA-
// compatible backend that reuses the per-op CUDA boxing path).
//
// WHY a composite override is needed (measured root cause,
// docs/bench_qwen3_dcu_route_perf.md): aten::scaled_dot_product_attention is a
// *composite* op (ATen/native/transformers/attention.cpp). Its fused-backend
// selection runs inside that composite and, once a backend is chosen, branches
// on the query's device type: only CUDA/XPU tensors are sent to
// at::_scaled_dot_product_flash_attention; every other device (including
// PrivateUse1) is sent to the CPU kernel _scaled_dot_product_flash_attention_for_cpu.
// So on the boxing route, routing the attention leaf ops per-op (conf
// "*_attention = cuda") can never reach the vendor fused kernel: the composite
// picks the CPU flash leaf *before* dispatch ever sees the flagos tensors. The
// earlier _fused_sdp_choice DispatchStub PrivateUse1 slot that this file
// registered had the same flaw -- the composite consults the stub only to choose
// a *backend*, then still routes the chosen backend's leaf by
// query_.device().type().
//
// The fix: intercept the composite itself at PrivateUse1, box q/k/v(+mask) to
// CUDA (one DeviceBoxingGuard round-trip for the whole attention), and call the
// native composite on the CUDA tensors. The composite then sees device == CUDA,
// runs the vendor's own _fused_sdp_choice selector (flash for Qwen3 decode
// shapes), and returns the exact tensor the vendor route returns -- numerics
// match by construction, and the decomposed math path (_safe_softmax + bmm) is
// never taken. Nothing about the composite is hardcoded here: shapes the vendor
// selector refuses fall back to its own math path exactly as on the CUDA route.
//
// Scope: inference only. Under grad mode with a grad-requiring input the wrapper
// falls through to the composite on the flagos tensors (the pre-change math
// decomposition over autograd-aware leaf kernels), because a boxed CUDA forward
// would leave backward with PrivateUse1 tensors where the autograd graph
// recorded CUDA. See WrapperScaledDotProductAttention below.
//
// Registered only on CUDA-boxing builds. Ascend (USE_ASCEND) keeps its own
// PrivateUse1 _fused_sdp_choice stub (returns efficient_attention for its
// aclnn kernel) in backends/ascend/scaled_dot_product_attention.cc; MUSA, GCU
// and BPU do not run CUDA-compatible kernels and must keep their own paths.
// Without this guard the PrivateUse1 stub registration below would collide with
// Ascend's registration of the same DispatchStub slot.

#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <torch/library.h>

#include <optional>

#include "device_boxing.h"

#if !defined(USE_ASCEND) && !defined(USE_GCU) && !defined(USE_MUSA) && \
    !defined(USE_BPU)

namespace at::flagos {
namespace {

at::Tensor WrapperScaledDotProductAttention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  // Training path: when a grad-requiring input is present the composite's flash
  // forward records its autograd node on the CUDA-boxed tensors, but the guard
  // below unboxes them on scope exit -- so backward would later see PrivateUse1
  // tensors where the graph recorded CUDA (ToCopyBackward0 device mismatch; see
  // perf/sdpa_train_smoke.py). Fall through to the composite on the flagos
  // tensors instead: it then takes its math decomposition over leaf ops that
  // dispatch through the generated, autograd-aware PrivateUse1 kernels --
  // exactly the semantics this override replaced. This call is the native
  // composite body, not a dispatch, so it cannot re-enter this impl (no
  // recursion). Inference (no grad required) takes the fast boxed-CUDA path.
  if (at::GradMode::is_enabled() &&
      (query.requires_grad() || key.requires_grad() || value.requires_grad())) {
    return at::native::scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  }
  // DeviceBoxingGuard records raw TensorImpl* and must only be handed named
  // lvalues that outlive the guard, so the optional mask is bound to a local
  // first. An undefined mask Tensor is left untouched by the guard (matching the
  // CUDA path where attn_mask is None).
  at::Tensor mask_t = attn_mask.has_value() ? *attn_mask : at::Tensor();
  at::native::flagos::DeviceBoxingGuard guard(query, key, value, mask_t);
  // Call the native composite directly (register.cc's WrapperMatmul does the
  // same for aten::matmul): inside the guard the tensors are CUDA, so dispatch
  // cannot re-enter this PrivateUse1 impl (no recursion), and the composite's
  // backend decision runs on the same device type the vendor route sees.
  at::Tensor output = at::native::scaled_dot_product_attention(
      query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  // The output was produced by CUDA kernels, so it is a fresh CUDA tensor (not
  // one of the boxed inputs) and must be explicitly unboxed back to flagos.
  at::native::flagos::UnboxToFlagos(output);
  return output;
}

} // namespace
} // namespace at::flagos

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl(
      "scaled_dot_product_attention",
      TORCH_FN(at::flagos::WrapperScaledDotProductAttention));
}

#endif // CUDA-boxing builds
