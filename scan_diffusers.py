#!/usr/bin/env python
"""在 flagos 设备上扫描 diffusers 核心模型组件的前向覆盖度。

diffusers 的算力都在几个 nn.Module 上:UNet(2D/3D)、VAE(AutoencoderKL/VQ)、
各种 Transformer2D/DiT/SD3/Flux backbone、ControlNet、Prior。用最小 config 随机初始化
直接 forward,子进程隔离,避免单个 GPU 非法访问污染后续。

每个组件先在 CPU 跑一遍作对照,再在 flagos 跑,自动三态分类:
  - CPU 失败                → SCRIPT_ERR(本脚本 config/输入构造问题,与插件无关)
  - CPU 通过 + flagos 失败  → PLUGIN_FAIL(真实插件问题)
  - CPU 通过 + flagos 通过  → PASS
目标:结果里只剩 PASS 与 PLUGIN_FAIL,SCRIPT_ERR 应为 0。
"""
import os
import sys
import traceback


def m_unet2dcond(dev):
    from diffusers import UNet2DConditionModel
    net = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4,
        layers_per_block=1, block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8,
    ).to(dev).eval()
    s = torch.randn(1, 4, 16, 16, device=dev)
    t = torch.tensor([10], device=dev)
    enc = torch.randn(1, 4, 32, device=dev)
    with torch.no_grad():
        o = net(s, t, encoder_hidden_states=enc).sample
    return tuple(o.shape)


def m_unet2d(dev):
    from diffusers import UNet2DModel
    net = UNet2DModel(
        sample_size=16, in_channels=3, out_channels=3,
        layers_per_block=1, block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"), norm_num_groups=8,
    ).to(dev).eval()
    s = torch.randn(1, 3, 16, 16, device=dev)
    t = torch.tensor([5], device=dev)
    with torch.no_grad():
        o = net(s, t).sample
    return tuple(o.shape)


def m_unet3d(dev):
    from diffusers import UNet3DConditionModel
    net = UNet3DConditionModel(
        sample_size=16, in_channels=4, out_channels=4,
        layers_per_block=1, block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D"),
        cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8,
    ).to(dev).eval()
    s = torch.randn(1, 4, 2, 16, 16, device=dev)
    t = torch.tensor([10], device=dev)
    enc = torch.randn(1, 4, 32, device=dev)
    with torch.no_grad():
        o = net(s, t, encoder_hidden_states=enc).sample
    return tuple(o.shape)


def m_vae_kl(dev):
    from diffusers import AutoencoderKL
    net = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4,
        block_out_channels=(32,), layers_per_block=1,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",), norm_num_groups=8,
    ).to(dev).eval()
    x = torch.randn(1, 3, 32, 32, device=dev)
    with torch.no_grad():
        o = net(x).sample
    return tuple(o.shape)


def m_vqmodel(dev):
    from diffusers import VQModel
    net = VQModel(
        in_channels=3, out_channels=3, latent_channels=4,
        block_out_channels=(32,), layers_per_block=1,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",), norm_num_groups=8,
    ).to(dev).eval()
    x = torch.randn(1, 3, 32, 32, device=dev)
    with torch.no_grad():
        o = net(x).sample
    return tuple(o.shape)


def m_transformer2d(dev):
    from diffusers import Transformer2DModel
    net = Transformer2DModel(
        num_attention_heads=2, attention_head_dim=16, in_channels=4,
        num_layers=1, cross_attention_dim=32, norm_num_groups=4,
        sample_size=16,
    ).to(dev).eval()
    h = torch.randn(1, 4, 16, 16, device=dev)
    enc = torch.randn(1, 4, 32, device=dev)
    t = torch.tensor([1], device=dev)
    with torch.no_grad():
        o = net(h, encoder_hidden_states=enc, timestep=t).sample
    return tuple(o.shape)


def m_dit(dev):
    from diffusers import DiTTransformer2DModel
    net = DiTTransformer2DModel(
        num_attention_heads=2, attention_head_dim=16, in_channels=4,
        num_layers=1, sample_size=8, patch_size=2, num_embeds_ada_norm=10,
    ).to(dev).eval()
    h = torch.randn(1, 4, 8, 8, device=dev)
    t = torch.tensor([1], device=dev)
    cls = torch.tensor([0], device=dev)
    with torch.no_grad():
        o = net(h, timestep=t, class_labels=cls).sample
    return tuple(o.shape)


def m_sd3(dev):
    from diffusers import SD3Transformer2DModel
    net = SD3Transformer2DModel(
        sample_size=16, patch_size=2, in_channels=4, num_layers=1,
        attention_head_dim=8, num_attention_heads=2,
        joint_attention_dim=32, caption_projection_dim=16,
        pooled_projection_dim=32, out_channels=4,
    ).to(dev).eval()
    h = torch.randn(1, 4, 16, 16, device=dev)
    enc = torch.randn(1, 4, 32, device=dev)
    pooled = torch.randn(1, 32, device=dev)
    t = torch.tensor([1], device=dev)
    with torch.no_grad():
        o = net(hidden_states=h, encoder_hidden_states=enc,
                pooled_projections=pooled, timestep=t).sample
    return tuple(o.shape)


def m_flux(dev):
    from diffusers import FluxTransformer2DModel
    net = FluxTransformer2DModel(
        patch_size=1, in_channels=4, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2,
        joint_attention_dim=32, pooled_projection_dim=32,
        axes_dims_rope=(4, 2, 2), guidance_embeds=True,
    ).to(dev).eval()
    h = torch.randn(1, 16, 4, device=dev)
    enc = torch.randn(1, 4, 32, device=dev)
    pooled = torch.randn(1, 32, device=dev)
    img_ids = torch.zeros(16, 3, device=dev)
    txt_ids = torch.zeros(4, 3, device=dev)
    t = torch.tensor([1.0], device=dev)
    guidance = torch.tensor([1.0], device=dev)
    with torch.no_grad():
        o = net(hidden_states=h, encoder_hidden_states=enc,
                pooled_projections=pooled, timestep=t, img_ids=img_ids,
                txt_ids=txt_ids, guidance=guidance).sample
    return tuple(o.shape)


def m_controlnet(dev):
    from diffusers import ControlNetModel
    net = ControlNetModel(
        in_channels=4, down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        block_out_channels=(32, 64), layers_per_block=1,
        cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8,
        conditioning_embedding_out_channels=(16,),
    ).to(dev).eval()
    s = torch.randn(1, 4, 16, 16, device=dev)
    t = torch.tensor([10], device=dev)
    enc = torch.randn(1, 4, 32, device=dev)
    cond = torch.randn(1, 3, 16, 16, device=dev)
    with torch.no_grad():
        out = net(s, t, encoder_hidden_states=enc, controlnet_cond=cond)
    return tuple(out[0][0].shape)


def m_prior(dev):
    from diffusers import PriorTransformer
    net = PriorTransformer(
        num_attention_heads=2, attention_head_dim=16, num_layers=1,
        embedding_dim=32, num_embeddings=8, additional_embeddings=4,
    ).to(dev).eval()
    emb = torch.randn(1, 32, device=dev)
    t = torch.tensor([1], device=dev)
    proj = torch.randn(1, 32, device=dev)
    enc = torch.randn(1, 8, 32, device=dev)
    mask = torch.ones(1, 8, dtype=torch.bool, device=dev)
    with torch.no_grad():
        o = net(emb, timestep=t, proj_embedding=proj,
                encoder_hidden_states=enc, attention_mask=mask).predicted_image_embedding
    return tuple(o.shape)


MODELS = [
    ("UNet2DConditionModel", m_unet2dcond),
    ("UNet2DModel", m_unet2d),
    ("UNet3DConditionModel", m_unet3d),
    ("AutoencoderKL", m_vae_kl),
    ("VQModel", m_vqmodel),
    ("Transformer2DModel", m_transformer2d),
    ("DiTTransformer2DModel", m_dit),
    ("SD3Transformer2DModel", m_sd3),
    ("FluxTransformer2DModel", m_flux),
    ("ControlNetModel", m_controlnet),
    ("PriorTransformer", m_prior),
]

MODEL_MAP = {n: f for n, f in MODELS}


def _last_err():
    return traceback.format_exc().strip().splitlines()[-1][:200]


def run_single(name):
    """CPU 对照 + flagos,输出三态分类结果供父进程解析。"""
    fn = MODEL_MAP[name]
    # 1) CPU 对照:失败 → 本脚本构造问题
    try:
        fn("cpu")
    except Exception:
        print(f"RESULT:SCRIPT_ERR:{name}:CPU 即失败: {_last_err()}", flush=True)
        return
    # 2) flagos
    try:
        shape = fn("flagos")
        print(f"RESULT:PASS:{name}:out={shape}", flush=True)
    except Exception:
        print(f"RESULT:PLUGIN_FAIL:{name}:{_last_err()}", flush=True)


def main():
    import subprocess
    buckets = {"PASS": [], "PLUGIN_FAIL": [], "SCRIPT_ERR": []}
    for name, _ in MODELS:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--single", name],
            capture_output=True, text=True, timeout=300,
        )
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("RESULT:")), "")
        parts = line.split(":", 3)
        if len(parts) == 4 and parts[1] in buckets:
            kind, _, msg = parts[1], parts[2], parts[3]
            buckets[kind].append((name, msg))
            tag = {"PASS": "PASS ", "PLUGIN_FAIL": "PLUGIN", "SCRIPT_ERR": "SCRIPT"}[kind]
            print(f"{tag} {name:26s} {msg}", flush=True)
        else:
            tail = (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1][:200]
            buckets["PLUGIN_FAIL"].append((name, f"[子进程崩溃 rc={proc.returncode}] {tail}"))
            print(f"CRASH {name:26s} rc={proc.returncode} {tail}", flush=True)
    total = len(MODELS)
    n_pass, n_plugin, n_script = (len(buckets[k]) for k in ("PASS", "PLUGIN_FAIL", "SCRIPT_ERR"))
    print("\n" + "=" * 64)
    print(f"总计 {total}  通过 {n_pass}  真实插件失败 {n_plugin}  脚本构造错误 {n_script}")
    denom = total - n_script  # 真实通过率剔除脚本构造错误
    if denom:
        print(f"真实通过率(剔除脚本错误) {n_pass}/{denom} = {n_pass / denom * 100:.1f}%")
    if buckets["PLUGIN_FAIL"]:
        print("\n真实插件失败:")
        for name, err in buckets["PLUGIN_FAIL"]:
            print(f"  - {name}: {err}")
    if buckets["SCRIPT_ERR"]:
        print("\n脚本构造错误(应清零):")
        for name, err in buckets["SCRIPT_ERR"]:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--single":
        import torch_fl  # noqa: F401  设备注册
        import torch  # noqa: F811
        globals()["torch"] = torch
        run_single(sys.argv[2])
    else:
        main()
