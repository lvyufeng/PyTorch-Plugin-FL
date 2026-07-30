"""Direct end-to-end Qwen3 generation throughput benchmark.

This benchmark uses the same model, random input, precision, batch/sequence
shape, warmup and iteration counts as the three-route inference benchmark in
``沐曦三路线性能与模型兼容性完整报告_2026-07-29.md``.  Unlike that benchmark,
it times only one aggregate ``model.generate`` region and reports
``all generated tokens / total wall-clock time`` directly.

Environment variables:
    MODE: native or torch_fl
    MODEL: model path (default: /data/nfs/Qwen3-0.6B)
    BS: batch size (default: 2)
    SEQ: input sequence length (default: 128)
    WARMUP: warmup generate calls (default: 3)
    ITERS: timed generate calls (default: 10)
    GEN_TOKENS: maximum new tokens per sample (default: 128)
"""

import os
import time


MODE = os.environ.get("MODE", "native")
if MODE not in {"native", "torch_fl"}:
    raise ValueError(f"Unsupported MODE={MODE!r}; expected native or torch_fl")

# MetaX torch_fl must be imported before torch so its runtime shims are ready.
if MODE == "torch_fl":
    import torch_fl

import torch


if MODE == "torch_fl" and os.environ.get("FLAGOS_USE_FLAGGEMS") == "1":

    def _do_bench_wall(fn, *args, quantiles=None, return_mode="mean", **kwargs):
        fn()
        torch.cuda.synchronize()
        iterations = 5
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
        if quantiles is not None:
            return [elapsed_ms] * len(quantiles)
        return elapsed_ms

    import triton.runtime.autotuner as _triton_autotuner
    import triton.testing as _triton_testing

    _triton_testing.do_bench = _do_bench_wall
    _triton_autotuner.do_bench = _do_bench_wall


MODEL = os.environ.get("MODEL", "/data/nfs/Qwen3-0.6B")
BS = int(os.environ.get("BS", "2"))
SEQ = int(os.environ.get("SEQ", "128"))
WARMUP = int(os.environ.get("WARMUP", "3"))
ITERS = int(os.environ.get("ITERS", "10"))
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "128"))


if MODE == "torch_fl":
    DEVICE = "flagos:0"

    def synchronize():
        torch_fl.flagos.synchronize()

else:
    DEVICE = "cuda:0"

    def synchronize():
        torch.cuda.synchronize()


def main():
    from transformers import AutoModelForCausalLM

    torch.__future__.set_swap_module_params_on_conversion(True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    model = model.to(DEVICE).eval()

    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(
        0, model.config.vocab_size, (BS, SEQ), generator=generator
    ).to(DEVICE)
    attention_mask = torch.ones((BS, SEQ), dtype=torch.long).to(DEVICE)
    generate_args = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": GEN_TOKENS,
        "do_sample": False,
        "use_cache": True,
    }

    print(
        f"mode={MODE} device={DEVICE} model={MODEL} dtype=fp16 "
        f"bs={BS} seq={SEQ} new_tokens={GEN_TOKENS}"
    )
    print(f"warmup={WARMUP} timed_iterations={ITERS}")
    if MODE == "torch_fl":
        print(f"backend_config={os.environ.get('FLAGOS_BACKEND_CONFIG')}")

    with torch.no_grad():
        for _ in range(WARMUP):
            model.generate(**generate_args)
        synchronize()

        total_new_tokens = 0
        start = time.perf_counter()
        for _ in range(ITERS):
            output = model.generate(**generate_args)
            total_new_tokens += (output.shape[1] - SEQ) * BS
        synchronize()
        elapsed = time.perf_counter() - start

    throughput = total_new_tokens / elapsed
    print("\n=== Direct E2E Generate TPS ===")
    print(f"total_time: {elapsed:.6f} s")
    print(f"total_new_tokens: {total_new_tokens}")
    print(f"average_time_per_generate: {elapsed / ITERS * 1000:.3f} ms")
    print(f"throughput: {throughput:.2f} tok/s")


if __name__ == "__main__":
    main()
