import os
import torch
import numpy as np
from PIL import Image
import lpips
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_vgg = lpips.LPIPS(net="vgg").to(device)
lpips_alex = lpips.LPIPS(net="alex").to(device)

w, h = 1352, 1014

# Get GT image from exp2 (left side of val_step4999_0000.png)
gt_raw = Image.open("/home/tsaichenghan/GSCodec_Studio/results/exp2_our_base01_native/renders/val_step4999_0000.png").convert("RGB")
gt_img = gt_raw.crop((0, 0, w, h))
gt_t = torch.from_numpy(np.array(gt_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

# Render image paths for all 5 inside container
render_paths = {
    "Exp 1: EDGS 原生 (RoMa + EDGS)": "/workspace/output/exp1_edgs_native/renders/val_step5000_render.png",
    "Exp 2: 我們原生 (DA3 Base 1% + 我們FT)": "/home/tsaichenghan/GSCodec_Studio/results/exp2_our_base01_native/renders/val_step4999_0000.png",
    "Exp 3: COLMAP 218k 基線 (Phase 2 FT)": "/workspace/output/colmap_render.png",
    "Exp 4: 交叉 A (DA3 點雲 + EDGS 訓練)": "/workspace/output/exp4_our_pcd_on_edgs/renders/val_step5000_render.png",
    "Exp 5: 交叉 B (RoMa 點雲 + 我們FT)": "/home/tsaichenghan/GSCodec_Studio/results/exp5_edgs_pcd_on_our/renders/val_step4999_0000.png"
}

results = {}

for name, p in render_paths.items():
    if not os.path.exists(p):
        print(f"Missing {p}")
        continue
    raw = Image.open(p).convert("RGB")
    if raw.size == (2 * w, h):
        render_img = raw.crop((w, 0, 2 * w, h))
    else:
        render_img = raw
    
    r_t = torch.from_numpy(np.array(render_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
    
    # Calculate MSE on [0, 1] range: (r_t - gt_t)/2.0
    diff = (r_t - gt_t) / 2.0
    mse = torch.mean(diff ** 2).item()
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    
    with torch.no_grad():
        vgg = lpips_vgg(r_t, gt_t).item()
        alex = lpips_alex(r_t, gt_t).item()
    
    results[name] = {
        "psnr_cam00": float(psnr),
        "lpips_vgg": float(vgg),
        "lpips_alex": float(alex)
    }

print("\n=== STANDARDIZED LPIPS & PSNR BENCHMARK (cam00 Held-Out, Final Step 5000) ===")
print(f"{'Method':<38} | {'PSNR (dB)':<10} | {'LPIPS (VGG)':<12} | {'LPIPS (Alex)':<12}")
print("-" * 80)
for name, val in results.items():
    print(f"{name:<38} | {val['psnr_cam00']:<10.3f} | {val['lpips_vgg']:<12.4f} | {val['lpips_alex']:<12.4f}")

with open("/workspace/output/standardized_5exp_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

