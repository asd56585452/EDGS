import os
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

out_dir = "/home/tsaichenghan/EDGS/output/patch_analysis_80x80"
art_dir = "/home/tsaichenghan/.gemini/antigravity-ide/brain/cd1c9aec-32ca-4d5b-8491-fa8bfec9b350"
patch_size = 80
H, W = 1014, 1352
n_rows = int(np.ceil(H / patch_size))
n_cols = int(np.ceil(W / patch_size))

gt = np.array(Image.open('/home/tsaichenghan/EDGS/output/exp1_edgs_native/renders/val_step5000_gt.png')).astype(np.float32)/255.0
r1 = np.array(Image.open('/home/tsaichenghan/EDGS/output/exp1_edgs_native/renders/val_step5000_render.png')).astype(np.float32)/255.0
r4 = np.array(Image.open('/home/tsaichenghan/EDGS/output/exp4_our_pcd_on_edgs/renders/val_step5000_render.png')).astype(np.float32)/255.0
r7 = np.array(Image.open('/home/tsaichenghan/EDGS/output/exp7_da3_b01_no_reduce_opacity/renders/val_step5000_render.png')).astype(np.float32)/255.0

def compute_patch_psnr(img_render, img_gt):
    psnr_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    for r in range(n_rows):
        y1 = r * patch_size
        y2 = min((r + 1) * patch_size, H)
        for c in range(n_cols):
            x1 = c * patch_size
            x2 = min((c + 1) * patch_size, W)
            mse = np.mean((img_render[y1:y2, x1:x2] - img_gt[y1:y2, x1:x2]) ** 2)
            psnr_grid[r, c] = -10.0 * np.log10(max(mse, 1e-10))
    return psnr_grid

p1 = compute_patch_psnr(r1, gt)
p4 = compute_patch_psnr(r4, gt)
p7 = compute_patch_psnr(r7, gt)

delta_p7_4 = p7 - p4 # Exp 7 (No decay) vs Exp 4 (Decay)

fig, (ax_diff, ax_timeline) = plt.subplots(1, 2, figsize=(18, 7))

gain_vmax = max(abs(np.min(delta_p7_4)), abs(np.max(delta_p7_4)), 1.5)
g_norm = mcolors.TwoSlopeNorm(vmin=-gain_vmax, vcenter=0, vmax=gain_vmax)
im_diff = ax_diff.imshow(delta_p7_4, cmap='PRGn', norm=g_norm, interpolation='nearest')
n_improved = np.sum(delta_p7_4 > 0)
ax_diff.set_title(f"Exp 7 (No Decay) vs Exp 4 (Decay) Quality Delta\nPurple: Exp 7 Better ({n_improved}/221, {n_improved/221*100:.1f}%) | Green: Exp 4 Better", fontsize=12, fontweight='bold')
plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04, label="ΔPSNR (Exp 7 - Exp 4) [dB]")

for r in range(n_rows):
    for c in range(n_cols):
        val = delta_p7_4[r, c]
        color = 'black' if abs(val) < gain_vmax * 0.6 else 'white'
        ax_diff.text(c, r, f"{val:+.1f}", ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')
ax_diff.set_xticks(range(n_cols))
ax_diff.set_yticks(range(n_rows))

# Timeline comparison of Splats count
with open('/home/tsaichenghan/EDGS/output/exp1_edgs_native/eval_stats.json') as f: e1 = {d['step']: d for d in json.load(f)}
with open('/home/tsaichenghan/EDGS/output/exp4_our_pcd_on_edgs/eval_stats.json') as f: e4 = {d['step']: d for d in json.load(f)}
with open('/home/tsaichenghan/EDGS/output/exp7_da3_b01_no_reduce_opacity/eval_stats.json') as f: e7 = {d['step']: d for d in json.load(f)}

steps = sorted(e7.keys())
s1 = [e1[s]['splats'] for s in steps]
s4 = [e4[s]['splats'] for s in steps]
s7 = [e7[s]['splats'] for s in steps]

ax_timeline.plot(steps, s7, 'o-', color='#2ca02c', linewidth=2.5, markersize=8, label='Exp 7: No Opacity Decay (214.5k preserved)')
ax_timeline.plot(steps, s1, 's--', color='#ff7f0e', linewidth=2.0, markersize=7, label='Exp 1: RoMa (192.4k preserved)')
ax_timeline.plot(steps, s4, '^:', color='#d62728', linewidth=2.0, markersize=7, label='Exp 4: With Opacity Decay (bled to 158.0k)')

ax_timeline.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax_timeline.set_ylabel('Total Surviving Gaussians (#)', fontsize=12, fontweight='bold')
ax_timeline.set_title('Gaussian Point Retention Across 5,000 Steps\nDisabling reduce_opacity eliminates 94% of Point Loss!', fontsize=12, fontweight='bold')
ax_timeline.legend(fontsize=10.5)
ax_timeline.grid(True, alpha=0.3)
for i, s in enumerate(steps[::2]):
    ax_timeline.annotate(f"{s7[2*i]//1000}k", (s, s7[2*i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8.5, color='#2ca02c', fontweight='bold')

plt.tight_layout()
p_out = f"{out_dir}/comparison_decay_vs_nodecay.png"
plt.savefig(p_out, dpi=200)
plt.close()

import shutil
shutil.copyfile(p_out, f"{art_dir}/comparison_decay_vs_nodecay.png")
print("Saved comparison_decay_vs_nodecay.png successfully!")
