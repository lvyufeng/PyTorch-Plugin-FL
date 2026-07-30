#!/usr/bin/env python
"""在 flagos 设备上批量扫描 transformers 模型架构的前向覆盖度。

思路:对每个架构构造一个"迷你" config(层数/隐层维度都调到最小),
用 from_config 随机初始化(不下载权重),跑一次 forward。

每个架构先在 CPU 跑一遍作对照,再在 flagos 跑,自动三态分类:
  - CPU 失败                → SCRIPT_ERR(本脚本 config/输入构造问题,与插件无关)
  - CPU 通过 + flagos 失败  → PLUGIN_FAIL(真实插件问题)
  - CPU 通过 + flagos 通过  → PASS
目标:结果里只剩 PASS 与 PLUGIN_FAIL,SCRIPT_ERR 应为 0。
子进程隔离运行:一次 GPU 非法访问会污染整个 CUDA 上下文,必须隔离才得真实结果。
"""
import os
import sys
import traceback

# (model_type, 额外 config 覆盖) —— 覆盖主流文本 encoder/decoder/seq2seq 架构
SMALL = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=64,
)

MODELS = [
    ("bert", {}),
    ("roberta", {}),
    ("albert", dict(embedding_size=64, num_hidden_groups=1)),
    ("electra", dict(embedding_size=64)),
    ("distilbert", dict(dim=64, hidden_dim=128, n_layers=2, n_heads=2)),
    ("deberta", {}),
    ("deberta-v2", {}),
    ("mobilebert", {}),
    ("xlm-roberta", {}),
    ("gpt2", dict(n_embd=64, n_layer=2, n_head=2, n_positions=64)),
    ("gpt_neo", dict(attention_types=[[["global", "local"], 1]])),
    ("gptj", dict(rotary_dim=16)),
    ("bloom", {}),
    ("opt", dict(word_embed_proj_dim=64, ffn_dim=128)),
    ("llama", dict(num_key_value_heads=2)),
    ("mistral", dict(num_key_value_heads=2)),
    ("qwen2", dict(num_key_value_heads=2)),
    ("qwen3", dict(num_key_value_heads=2, head_dim=32)),
    ("gemma", dict(num_key_value_heads=2, head_dim=32)),
    ("phi", {}),
    ("falcon", {}),
    ("mpt", dict(d_model=64, n_heads=2, n_layers=2)),
    ("stablelm", dict(num_key_value_heads=2)),
    ("starcoder2", dict(num_key_value_heads=2)),
    ("t5", dict(d_model=64, d_ff=128, num_layers=2, num_heads=2, d_kv=32)),
    ("bart", dict(d_model=64, encoder_layers=2, decoder_layers=2,
                  encoder_attention_heads=2, decoder_attention_heads=2,
                  encoder_ffn_dim=128, decoder_ffn_dim=128)),
    ("mbart", dict(d_model=64, encoder_layers=2, decoder_layers=2,
                   encoder_attention_heads=2, decoder_attention_heads=2,
                   encoder_ffn_dim=128, decoder_ffn_dim=128)),
    ("pegasus", dict(d_model=64, encoder_layers=2, decoder_layers=2,
                     encoder_attention_heads=2, decoder_attention_heads=2,
                     encoder_ffn_dim=128, decoder_ffn_dim=128)),
    ("marian", dict(d_model=64, encoder_layers=2, decoder_layers=2,
                    encoder_attention_heads=2, decoder_attention_heads=2,
                    encoder_ffn_dim=128, decoder_ffn_dim=128,
                    pad_token_id=1, decoder_start_token_id=2)),
    # 视觉
    ("vit", dict(image_size=32, patch_size=16, num_channels=3)),
    ("deit", dict(image_size=32, patch_size=16, num_channels=3)),
    ("beit", dict(image_size=32, patch_size=16, num_channels=3)),
    ("swin", dict(image_size=32, patch_size=4, embed_dim=48,
                  depths=[2, 2], num_heads=[3, 6], window_size=2,
                  num_attention_heads=None, hidden_size=None)),
    ("convnext", dict(num_channels=3)),
    ("clip", {}),
]


def make_config(mtype, extra):
    kw = dict(SMALL)
    kw.update(extra)
    # extra 里值为 None 的键表示"删除这个 SMALL 通用键"(避免与架构特有键冲突,
    # 例如 swin 的 num_heads=[..] 会被通用键 num_attention_heads 反向覆盖成标量)
    kw = {k: v for k, v in kw.items() if v is not None}
    # 逐个键尝试:有些架构不认识某些键,失败就丢掉再来
    while True:
        try:
            return AutoConfig.for_model(mtype, **kw)
        except TypeError as e:
            # 解析出不被接受的关键字并移除
            msg = str(e)
            removed = False
            for k in list(kw.keys()):
                if k in msg:
                    kw.pop(k)
                    removed = True
                    break
            if not removed:
                raise


def make_inputs(model, cfg, mtype, dev):
    import inspect
    # 视觉类:pixel_values
    vision = mtype in {"vit", "deit", "beit", "swin", "convnext"}
    sig = inspect.signature(model.forward)
    params = sig.parameters
    inputs = {}
    if mtype == "clip":
        inputs["input_ids"] = torch.randint(0, 100, (2, 8), device=dev)
        inputs["attention_mask"] = torch.ones(2, 8, dtype=torch.long, device=dev)
        img = getattr(cfg.vision_config, "image_size", 224)
        inputs["pixel_values"] = torch.randn(2, 3, img, img, device=dev)
        return inputs
    if vision:
        img = getattr(cfg, "image_size", 32)
        ch = getattr(cfg, "num_channels", 3)
        inputs["pixel_values"] = torch.randn(2, ch, img, img, device=dev)
        return inputs
    inputs["input_ids"] = torch.randint(0, 100, (2, 8), device=dev)
    if "attention_mask" in params:
        inputs["attention_mask"] = torch.ones(2, 8, dtype=torch.long, device=dev)
    # encoder-decoder 需要 decoder_input_ids
    if "decoder_input_ids" in params and getattr(cfg, "is_encoder_decoder", False):
        inputs["decoder_input_ids"] = torch.randint(0, 100, (2, 8), device=dev)
    return inputs


def run_one(mtype, extra, dev):
    cfg = make_config(mtype, extra)
    model = AutoModel.from_config(cfg).to(dev).eval()
    inputs = make_inputs(model, cfg, mtype, dev)
    with torch.no_grad():
        out = model(**inputs)
    # 触发一次读回,确保 kernel 真的算了
    t = out[0] if isinstance(out, tuple) else getattr(out, "last_hidden_state", None)
    if t is None:
        t = list(out.values())[0] if hasattr(out, "values") else out
    _ = t.float().sum().item()
    return tuple(t.shape)


MODEL_MAP = {m: e for m, e in MODELS}


def _last_err():
    return traceback.format_exc().strip().splitlines()[-1][:200]


def run_single(mtype):
    """CPU 对照 + flagos,输出三态分类结果供父进程解析。"""
    extra = MODEL_MAP[mtype]
    # 1) CPU 对照:失败 → 本脚本构造问题
    try:
        run_one(mtype, extra, "cpu")
    except Exception:
        print(f"RESULT:SCRIPT_ERR:{mtype}:CPU 即失败: {_last_err()}", flush=True)
        return
    # 2) flagos
    try:
        shape = run_one(mtype, extra, "flagos")
        print(f"RESULT:PASS:{mtype}:out={shape}", flush=True)
    except Exception:
        print(f"RESULT:PLUGIN_FAIL:{mtype}:{_last_err()}", flush=True)


def main():
    """父进程:每个模型 fork 一个子进程隔离运行,避免 GPU 非法状态串扰。"""
    import subprocess
    buckets = {"PASS": [], "PLUGIN_FAIL": [], "SCRIPT_ERR": []}
    for mtype, _ in MODELS:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--single", mtype],
            capture_output=True, text=True, timeout=180,
        )
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("RESULT:")), "")
        parts = line.split(":", 3)
        if len(parts) == 4 and parts[1] in buckets:
            kind, msg = parts[1], parts[3]
            buckets[kind].append((mtype, msg))
            tag = {"PASS": "PASS  ", "PLUGIN_FAIL": "PLUGIN", "SCRIPT_ERR": "SCRIPT"}[kind]
            print(f"{tag} {mtype:16s} {msg}", flush=True)
        else:
            tail = (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1][:200]
            buckets["PLUGIN_FAIL"].append((mtype, f"[子进程崩溃 rc={proc.returncode}] {tail}"))
            print(f"CRASH  {mtype:16s} rc={proc.returncode} {tail}", flush=True)
    total = len(MODELS)
    n_pass, n_plugin, n_script = (len(buckets[k]) for k in ("PASS", "PLUGIN_FAIL", "SCRIPT_ERR"))
    print("\n" + "=" * 64)
    print(f"总计 {total}  通过 {n_pass}  真实插件失败 {n_plugin}  脚本构造错误 {n_script}")
    denom = total - n_script
    if denom:
        print(f"真实通过率(剔除脚本错误) {n_pass}/{denom} = {n_pass / denom * 100:.1f}%")
    if buckets["PLUGIN_FAIL"]:
        print("\n真实插件失败:")
        for mtype, err in buckets["PLUGIN_FAIL"]:
            print(f"  - {mtype}: {err}")
    if buckets["SCRIPT_ERR"]:
        print("\n脚本构造错误(应清零):")
        for mtype, err in buckets["SCRIPT_ERR"]:
            print(f"  - {mtype}: {err}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--single":
        import torch_fl  # noqa: F401  设备注册
        import torch  # noqa
        from transformers import AutoConfig, AutoModel  # noqa
        globals().update(torch=torch, AutoConfig=AutoConfig, AutoModel=AutoModel)
        run_single(sys.argv[2])
    else:
        main()
