# torch_fl 重构交接文档（给 Claude Code）

> 本文件是在 2080ti 服务器上继续 torch_fl 重构的入口说明。
> 它假设你（Claude Code）刚接手、没有之前的对话上下文。请先完整读本文，再动手。
> 最后更新：2026-07-16

## 0. 一句话背景

torch_fl 是一个基于 PyTorch **PrivateUse1** 机制的自定义设备后端（对外设备名计划叫 **`vm`**，当前代码里是 `flagos`），目标是做一层**统一的设备抽象**（类似 virtual machine），对上层屏蔽不同厂商芯片（CUDA / Ascend / MetaX）的差异，主要通过接入 **FlagGems**（Triton 通用算子库）+ **厂商原生算子库兜底** 来支持各硬件。

**重构的北极星**：torch_fl 应是一个**薄的路由/适配层**，**尽量不手写 kernel**，快速把它作为后端接入各种芯片，然后腾出人力去做 torch 的其他特性（torch.compile、distributed 等）。

## 1. 必读文档（按顺序）

1. `docs/torch_fl_current_issues.md` — **现状问题全面分析**（11 个问题 + codegen 设计草案 + 架构演进方向）。这是重构的问题清单和优先级来源。
2. `docs/cpu_torch_external_libtorch_cuda.md` — **一个已实测验证的关键结论**（见下 §3），决定 NVIDIA 后端的接入方式。
3. `README_zh.md` / `README.md` — 项目原始文档（安装、后端配置、目录结构）。

## 2. 项目架构速览（重构前必须理解）

- **dispatch 机制**：torch_fl **没有绕过** PyTorch 的 `c10::Dispatcher`。它用官方 `TORCH_LIBRARY_IMPL(aten, PrivateUse1)`（`csrc/aten/register.cc`）接管算子，然后在算子内部用自研的 `Dispatcher`（`csrc/aten/dispatcher.h`）**替换了原生的 `DispatchStub`**——即"按 backend 选 kernel"这一层。后端选择由 `.conf` 文件在**运行时按算子粒度**驱动（`csrc/aten/common.cc` 的 `GetBackendForOp`）。
- **后端枚举**：`kCuda / kFlagOs(=flaggems) / kFlagOsPython(=flaggems_python) / kAscend / kMetax / kMusa`。conf 里 `flaggems`→C++ native API wrapper，`flagos_python`→pybind11 调 `flag_gems.ops`，`cuda`→原生 CUDA kernel。
- **两个"零 per-op kernel"的通用适配器（最宝贵资产，应贯彻）**：
  - `csrc/aten/backends/flagos/python_op_caller.cc` — 通用调用任意 FlagGems Python 算子。
  - `csrc/aten/backends/ascend/op_api_common.h` — 通用调用任意 aclnn（Ascend）算子。
- **运行时抽象**：`include/flagos.h` 定义厂商中立 C ABI（Malloc/Stream/Event…），各芯片在 `csrc/runtime/accelerator/<hw>/` 用自己 SDK 实现。**此层不吃 torch 类型，天然版本解耦，应保持。**

## 3. 已实测验证的关键结论（决定 NVIDIA 接入方式）

**结论：pip 只装 CPU 版 torch、不装 CUDA torch，通过外挂一个版本匹配的 `libtorch_cuda.so`，即可复用 PyTorch 全套已注册的 CUDA kernel，实测计算正确。**（完整细节见 `docs/cpu_torch_external_libtorch_cuda.md`）

这条路让 NVIDIA 后端 **零手写 kernel**（直接 boxing 调 `structured_*_out_cuda`）且 **不依赖 pip CUDA torch**。三个约束：
1. **硬约束**：`libtorch_cuda.so` 必须在 `import torch` **之前**载入（CUDAHooks 缓存问题），用 `LD_PRELOAD` 或在 `__init__.py` 里于 `import torch` 前 `ctypes.CDLL` 解决。
2. Python 层 `torch.cuda._lazy_init` gate 与 torch_fl **无关**（boxing 全程在 C++，不碰 `torch.cuda`）。
3. torch 版本必须与 `libtorch_cuda.so` 逐位匹配；依赖"CPU wheel 符号完整"这一无官方承诺的性质，升级 torch 需复测。

## 4. 服务器上已就绪的测试环境

- **仓库**：`/mnt/data1/PyTorch-Plugin-FL`（本仓库，已 clone，分支 `main`，HEAD `95d46ff`）。
- **conda 环境**：`libtorch_test`（`conda activate libtorch_test`）。已装：
  - CPU torch `2.13.0+cpu`（`torch.cuda.is_available()==False`）
  - `nvidia-*-cu12` runtime 包（cudart/cublas/cudnn/nvshmem/… 全套）
- **CUDA .so 资产（已固化，勿删）**：`/mnt/data1/PyTorch-Plugin-FL/.libtorch_cuda_assets/`
  - 从 torch `2.13.0+cu126` wheel 抽取的、CPU torch 没有的 5 个 CUDA 独有 `.so`：
    `libtorch_cuda.so`（≈1GB）、`libc10_cuda.so`、`libtorch_cuda_linalg.so`、`libtorch_nvshmem.so`、`libcaffe2_nvrtc.so`
  - 这些必须放在**同一目录**（`libtorch_cuda.so` 的 RPATH=`$ORIGIN`，靠同目录解析 `libtorch_nvshmem.so` 等依赖）
- **验证脚本**：`bash docs/verify_external_cuda.sh` —— 一键复现 §3 结论（跑通即证明环境 OK）。
- **机器**：4× RTX 2080 Ti，driver 550.163.01。
- **网络**：pip 走清华镜像（`~/.pip/pip.conf` 已配）；conda 默认源 SSL 挂，建环境用 `-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main --override-channels`。

> **第一件事**：跑 `bash docs/verify_external_cuda.sh`，确认输出 `=== PASS ===`，证明环境完好再开始重构。

## 5. 重构优先级（来自 docs/torch_fl_current_issues.md §六）

| 优先级 | 事项 | 出处 |
|---|---|---|
| **P0** | 默认 conf 与 backend 实现的一致性校验（避免静默失效/崩溃） | 问题 6/7 |
| **P0** | codegen 层：声明式表自动生成 per-op 胶水（可扩展性生死线） | 问题 2 + §七 |
| **P1** | NVIDIA 接入定型：用 §3 的"CPU torch + 外挂 libtorch_cuda.so + boxing"，减少手写 `.cu` | 问题 1/5 |
| **P1** | device_boxing 平台隔离（明确 CUDA-only） | 问题 5 |
| **P1** | 贯彻通用适配器 + 修正默认路由（FlagGems-first） | 问题 3/4 |
| **P2** | 配置重载、文档、命名别名、死代码清理 | 问题 8/9/10/11 |

## 6. 重要约束与纪律（务必遵守）

- **不手写 CUDA kernel**：NVIDIA 走 §3 的 boxing 复用 native，不要新增 `.cu`。厂商芯片（Ascend/MetaX）走 dlopen 厂商 SDK C-ABI（aclnn/maca），不手写 kernel。
- **只依赖两个"版本稳定层"**：向下依赖厂商 SDK C-ABI（CANN/MACA/cuBLAS 等），向上依赖 PyTorch 公开扩展 API（PrivateUse1/torch.library）。避免依赖 `torch_npu` 等厂商 torch 插件（会引入 PrivateUse1 独占冲突 + torch 版本锁）。
- **PrivateUse2 方案已否决**：算子按 device key 派发，PU2 张量调不到 PU1 kernel；且加载 torch_npu 会引入版本锁。不要走这条路。
- **改动前先读代码、先跑验证脚本**；改动后跑 `tests/integration/ops/` 相关测试确认不回退。
- **git**：在 `main` 之外开分支工作；只有用户明确要求才 commit/push。
- **保留 #15 成果**：§3 已证明 boxing + structured 复用在"CPU torch + 外挂 so"下成立，**无需回退 #15 的训练性能优化**。

## 7. 建议的第一步（供参考，最终以用户指令为准）

1. `conda activate libtorch_test && cd /mnt/data1/PyTorch-Plugin-FL`
2. `bash docs/verify_external_cuda.sh` 确认 PASS。
3. 与用户确认本轮重构要先做哪一项（P0 的一致性校验，还是 P0 的 codegen，还是先把 NVIDIA 接入按 §3 定型）。
4. 若做 codegen：先做"反向抽取"——写脚本从现有 `register.cc` 的 90 个 `m.impl` + 各 backend `.cc` 反向生成初版声明式表（`ops.yaml`），再用生成器重生成 1~2 个算子与手写版 diff 对齐。详见 `docs/torch_fl_current_issues.md §七`。

---

**给 Claude Code 的提醒**：本文 §3 的实测结论此前与直觉相反（一度被判断为"走不通"），是经过在本机逐关卡实测才确认的。涉及 NVIDIA 接入/版本耦合的判断，请以 `docs/cpu_torch_external_libtorch_cuda.md` 的实测记录为准，不要凭记忆推翻。有疑问先跑 `verify_external_cuda.sh` 或写最小复现脚本实测，再下结论。
