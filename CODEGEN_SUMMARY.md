# CUDA Operator Codegen — Complete

## Summary

**Goal**: Eliminate ~4600 lines of manual boilerplate across 71 CUDA operators by generating all dispatcher infrastructure, boxing kernels, and registration code from `native_functions.yaml` + `backends_cuda.conf`.

**Status**: ✅ **COMPLETE** — Codegen script produces 4 files totaling 3200+ lines, all compile cleanly.

## What Was Generated

### Input Files
- **torch_fl/backends_cuda.conf**: 71 operator entries (`op.name=cuda`)
- **native_functions.yaml**: PyTorch's authoritative operator schema (via torchgen)

### Output Files (csrc/aten/generated/)

1. **ops.h** (71 typedefs + dispatcher declarations)
   - `using AbsFn = at::Tensor (*)(const at::Tensor &);`
   - `DECLARE_DISPATCHER(AbsFn, abs_dispatcher)`
   - One typedef + declaration per operator
   - Uses CppSignature faithful form (exploded TensorOptions, not SymInt)

2. **ops.cc** (71 dispatcher definitions)
   - `ADD_IMPL_TO_DISPATCHER(AbsFn, abs_dispatcher, "abs")`
   - Creates the global dispatcher instance for each op

3. **cuda_kernels.cc** (71 boxing kernel implementations)
   - Pattern: `DeviceBoxingGuard` → `at::` public API → `UnboxToFlagos`
   - Example:
     ```cpp
     at::Tensor AbsKernelCuda(const at::Tensor& self) {
       DeviceBoxingGuard guard(self);
       auto result = at::abs(self);
       UnboxToFlagos(result);
       return result;
     }
     REGISTER_IMPL_TO_DISPATCHER(AbsFn, abs_dispatcher, Backend::kCuda, AbsKernelCuda)
     ```
   - Zero hand-written CUDA code, all delegate to PyTorch's optimized kernels

4. **register.inc** (71 wrappers + m.impl lines)
   - Wrappers: `at::Tensor WrapperAbs(const at::Tensor& self) { return abs_dispatcher(self); }`
   - Registrations: `m.impl("abs", WrapperAbs);`
   - Included by `register.cc` with `#define FLAGOS_GEN_WRAPPERS / FLAGOS_GEN_IMPLS`

## Architecture & Key Design Decisions

### 1. Schema-Driven Naming (Unified Across Backends)

**Decision**: Use PyTorch schema as single source of truth for symbol names.
- `add.Tensor` → `AddTensorFn` / `add_tensor_dispatcher`
- `add_.Tensor` → `AddInplaceTensorFn` / `add_inplace_tensor_dispatcher`  
- `mm.out` → `MmOutFn` / `mm_out_dispatcher`

**Why**: Eliminates manual naming conventions, ensures perfect alignment with PyTorch's operator registry.

### 2. Unified CppSignature (Faithful) Everywhere

**Decision**: Use torchgen's `CppSignature.faithful` (exploded TensorOptions) for typedef, kernel, AND wrapper.
- `IntArrayRef` (not `SymIntArrayRef`)
- `at::TensorList` (not `const at::ITensorListRef &`)
- All three must match for `REGISTER_IMPL_TO_DISPATCHER` type-checking

**Why**: Matches existing hand-written code ABI, avoids type mismatches. PyTorch's internal `SymInt` is an implementation detail not exposed in public `at::` API.

### 3. Seven Operator Categories

Each category has a distinct boxing pattern:

#### functional_pure (46 ops)
Standard: box inputs → call `at::op` → unbox result
```cpp
at::Tensor AbsKernelCuda(const at::Tensor& self) {
  DeviceBoxingGuard guard(self);
  auto result = at::abs(self);
  UnboxToFlagos(result);
  return result;
}
```

#### inplace (3 ops: add_.Tensor, fill_.Scalar, masked_fill_.Scalar)
Box → **method call** `self.op_()` → return self
```cpp
at::Tensor& AddInplaceTensorKernelCuda(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  DeviceBoxingGuard guard(self, other);
  self.add_(other, alpha);
  return self;
}
```

#### out_variant (2 ops: mm.out, bmm.out)
Box all (including out tensor) → `at::op_out(out, ...)` → return out
```cpp
at::Tensor& MmOutKernelCuda(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  DeviceBoxingGuard guard(self, mat2, out);
  at::mm_out(out, self, mat2);
  return out;
}
```

#### tuple_return (3 ops: topk, native_batch_norm, native_batch_norm_backward)
Box → call → **unbox each tuple element**
```cpp
::std::tuple<at::Tensor, at::Tensor> TopkKernelCuda(...) {
  DeviceBoxingGuard guard(self);
  auto result = at::topk(self, k, dim, largest, sorted);
  UnboxToFlagos(::std::get<0>(result));
  UnboxToFlagos(::std::get<1>(result));
  return result;
}
```

#### foreach_tensorlist (12 ops: _foreach_add.Scalar, _foreach_mul.Scalar, ...)
`TensorListBoxingGuard` → call → optional `UnboxTensorVecToFlagos` for non-inplace
```cpp
void ForeachAddInplaceScalarKernelCuda(at::TensorList self, const at::Scalar& scalar) {
  TensorListBoxingGuard guard;
  guard.box(self);
  at::_foreach_add_(self, scalar);  // inplace, no unbox needed
}
```

#### factory (5 ops: arange, zeros_like, ones_like, new_ones, scalar_tensor)
**No input boxing** (no input tensors or creates from scratch)
```cpp
at::Tensor ArangeKernelCuda(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, ...) {
  auto options = at::TensorOptions().dtype(...).device(...);
  return at::arange(start, end, step, options);
}
```
Special: `new_ones` uses `at::empty(size, options) + result.fill_(1)` (no public `at::new_ones` API)

#### special_optlist (1 op: index.Tensor)
Handle `optional<Tensor>[]` — box self + each element in the list
```cpp
at::Tensor IndexTensorKernelCuda(const at::Tensor& self, const ::std::optional<at::List<::std::optional<at::Tensor>>>& indices) {
  DeviceBoxingGuard guard(self);
  if (indices.has_value()) {
    for (const auto& opt_t : indices->vec()) {
      if (opt_t.has_value() && opt_t->defined()) BoxToCuda(*opt_t);
    }
  }
  auto result = at::index(self, indices);
  UnboxToFlagos(result);
  return result;
}
```

## Technical Details

### torchgen Integration

**API Pattern** (all signature extraction must run inside `local.parametrize` context):
```python
from torchgen.api.cpp import CppSignatureGroup
from torchgen.context import local

with local.parametrize(
    use_const_ref_for_mutable_tensors=False,
    use_ilistref_for_tensor_lists=False,
):
    grp = CppSignatureGroup.from_native_function(func, method=False)
    sig = grp.faithful_signature or grp.signature
    ptr_type = sig.ptr_type()    # "at::Tensor (*)(const at::Tensor &, ...)"
    args = sig.arguments()       # [(ConstRefCType(BaseCType(tensorT)), 'self'), ...]
    returns = sig.func.returns   # [Return(name=None, type=BaseType(BaseTy.Tensor), ...]
```

### Dispatcher Return Type Fix

**Problem**: `auto` in `Dispatcher::operator()` drops references (`at::Tensor&` → `at::Tensor`)

**Fix**: Use `decltype(auto)` to preserve exact return type:
```cpp
template <typename... Args>
decltype(auto) operator()(Args&&... args) const {
  return DispatchAs(op_name_, std::forward<Args>(args)...);
}
```

This ensures inplace/out ops correctly return `at::Tensor&` references.

## Verification

### Compile Tests (All Pass ✅)

1. **ops.h + ops.cc**: Standalone compile → 0 errors
2. **cuda_kernels.cc**: Standalone compile → 742KB object file
3. **register.inc**: Full integration test (wrappers + m.impl) → 0 errors

### Sample Correctness Checks

- **Signatures match existing ABI**: `CatFn = at::Tensor (*)(at::TensorList, int64_t)` (not ITensorListRef)
- **Inplace uses method syntax**: `self.add_(other, alpha)` (not `at::add_`)
- **Out variant boxes all**: `DeviceBoxingGuard guard(self, mat2, out)`
- **Factory special-casing**: `new_ones` → `at::empty + fill_(1)`
- **Tuple unboxing**: `topk` unboxes both returned tensors

## Migration Path (Not Yet Executed)

Generated code is ready but **not yet integrated** into the build to avoid collisions with existing hand-written files. Next steps:

1. **Backup existing code**:
   ```bash
   mkdir -p csrc/aten/archived/cuda_handwritten
   mv csrc/aten/backends/cuda/*.{cc,cu} csrc/aten/archived/cuda_handwritten/
   mv csrc/aten/*.{h,cc} csrc/aten/archived/  # 71 per-op headers/impls
   ```

2. **Wire generated files into build**:
   - Include `csrc/aten/generated/ops.cc` + `cuda_kernels.cc` in CMakeLists
   - Modify `register.cc` to `#include "generated/register.inc"` with both macros

3. **Full rebuild**:
   ```bash
   FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON \
     pip install -e . --no-build-isolation
   ```

4. **End-to-end validation**:
   ```bash
   FLAGOS_BACKEND_CONFIG=torch_fl/backends_cuda.conf \
     ./scripts/with_cuda_libtorch.sh \
     pytest tests/integration/ops/ -v
   ```

## Benefits

### Code Reduction
- **Before**: ~4600 lines of manual boilerplate (71 headers, 71 .cc/.cu files)
- **After**: 450-line codegen script → 3200 lines auto-generated
- **Maintenance**: Single script update propagates to all 71 ops

### Consistency
- Schema-driven naming eliminates typos and convention drift
- All 71 ops use identical boxing patterns (no ad-hoc variations)
- torchgen signature ensures perfect type alignment with PyTorch

### Extensibility
- **Add new ops**: Add one line to `backends_cuda.conf`, rerun codegen
- **New backends**: Add backend-specific templates to codegen, reuse same dispatcher/wrapper infrastructure
- **PyTorch upgrades**: Regenerate from updated `native_functions.yaml`, auto-sync signature changes

## Files

**Core Script**:
- `scripts/codegen_ops.py` — 450 lines, 7 category generators + torchgen integration

**Generated Output** (not yet integrated):
- `csrc/aten/generated/ops.h` — 213 lines
- `csrc/aten/generated/ops.cc` — 143 lines
- `csrc/aten/generated/cuda_kernels.cc` — 1034 lines
- `csrc/aten/generated/register.inc` — 1847 lines

**Infrastructure (modified)**:
- `csrc/aten/dispatcher.h` — Fixed return type deduction (`decltype(auto)`)

**Test Scripts**:
- `/tmp/test_register.cc` — Standalone compile harness for register.inc

## Next Steps

Task #10: **全量编译 + 端到端验证** (Full build + end-to-end validation)
- Archive existing hand-written code
- Wire generated files into CMakeLists + register.cc
- Full rebuild with CPU-only torch
- Run test suite with external libtorch_cuda.so
- Verify all 71 ops dispatch correctly and produce correct results

---

**Codegen completed**: 2026-07-16  
**Total operators generated**: 71 (46 functional, 12 foreach, 3 inplace, 3 tuple, 2 out, 5 factory, 1 optlist)  
**Lines generated**: 3237  
**Lines of codegen script**: 450  
**Compression ratio**: 7.2x
