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

qvec = np.array([0.99929653, 0.03413274, 0.01005442, -0.0118453])
tvec = np.array([0.00400316, 1.30594596, 0.34566886])

def qvec2rotmat(q):
    return np.array([
        [1 - 2 * q[2]**2 - 2 * q[3]**2,
         2 * q[1] * q[2] - 2 * q[0] * q[3],
         2 * q[1] * q[3] + 2 * q[0] * q[2]],
        [2 * q[1] * q[2] + 2 * q[0] * q[3],
         1 - 2 * q[1]**2 - 2 * q[3]**2,
         2 * q[2] * q[3] - 2 * q[0] * q[1]],
        [2 * q[1] * q[3] - 2 * q[0] * q[2],
         2 * q[2] * q[3] + 2 * q[0] * q[1],
         1 - 2 * q[1]**2 - 2 * q[2]**2]
    ])

R = qvec2rotmat(qvec)
K = np.array([
    [731.0626, 0.0, 676.0],
    [0.0, 731.0626, 507.0],
    [0.0, 0.0, 1.0]
])

def load_ply_xyz(ply_p):
    with open(ply_p, 'rb') as f:
        header_lines = []
        for line in f:
            header_lines.append(line)
            if line.strip() == b'end_header':
                break
        header_text = b''.join(header_lines).decode('ascii', errors='ignore')
        num_vertices = 0
        prop_names = []
        for l in header_text.splitlines():
            if l.startswith('element vertex'):
                num_vertices = int(l.split()[-1])
            elif l.startswith('property float'):
                prop_names.append(l.split()[-1])
        raw_bytes = f.read()
        record_dtype = [(name, '<f4') for name in prop_names]
        data = np.frombuffer(raw_bytes, dtype=record_dtype, count=num_vertices)
        return np.stack([data['x'], data['y'], data['z']], axis=-1)

def project_and_count_grid(xyz):
    pts_cam = (R @ xyz.T + tvec[:, None]).T
    valid_z = pts_cam[:, 2] > 0.1
    pts_valid = pts_cam[valid_z]
    u = K[0, 0] * pts_valid[:, 0] / pts_valid[:, 2] + K[0, 2]
    v = K[1, 1] * pts_valid[:, 1] / pts_valid[:, 2] + K[1, 2]
    in_view = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    counts = np.zeros((n_rows, n_cols), dtype=np.int32)
    r_idx = np.clip((v[in_view] // patch_size).astype(int), 0, n_rows - 1)
    c_idx = np.clip((u[in_view] // patch_size).astype(int), 0, n_cols - 1)
    np.add.at(counts, (r_idx, c_idx), 1)
    return counts

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

gt = np.array(Image.open('/home/tsaichenghan/EDGS/output/exp1_edgs_native/renders/val_step5000_gt.png')).astype(np.float32)/255.0
r1 = np.array(Image.open('/home/tsaichenghan/EDGS/output/exp1_edgs_native/renders/val_step5000_render.png')).astype(np.float32)/255.0
r4 = np.array(Image.open('/home/tsaichenghan/EDGS/output/exp4_our_pcd_on_edgs/renders/val_step5000_render.png')).astype(np.float32)/255.0
r6 = np.array(Image.open('/home/tsaichenghan/EDGS/output/exp6_da3_restored_on_edgs/renders/val_step5000_render.png')).astype(np.float32)/255.0

p1 = compute_patch_psnr(r1, gt)
p4 = compute_patch_psnr(r4, gt)
p6 = compute_patch_psnr(r6, gt)

c1 = project_and_count_grid(load_ply_xyz('/home/tsaichenghan/EDGS/output/exp1_edgs_native/point_cloud_step_5000.ply'))
c4 = project_and_count_grid(load_ply_xyz('/home/tsaichenghan/EDGS/output/exp4_our_pcd_on_edgs/point_cloud_step_5000.ply'))
c6 = project_and_count_grid(load_ply_xyz('/home/tsaichenghan/EDGS/output/exp6_da3_restored_on_edgs/point_cloud_step_5000.ply'))

# 1. Figure: Exp 6 vs RoMa Step 5000 4-panel
delta_p6_1 = p6 - p1
delta_c6_1 = c6 - c1

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
vmax = max(abs(np.min(delta_p6_1)), abs(np.max(delta_p6_1)), 3.0)
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im0 = axes[0, 0].imshow(delta_p6_1, cmap='RdYlGn', norm=norm, interpolation='nearest')
win_pct6 = (np.sum(delta_p6_1 > 0) / delta_p6_1.size) * 100
axes[0, 0].set_title(f"Step 5000: Exp 6 (DA3 Restored) vs Exp 1 (RoMa) Win/Loss Map\nWin (Green): {win_pct6:.1f}% ({np.sum(delta_p6_1 > 0)}/221) | Loss (Red): {100-win_pct6:.1f}%", fontsize=13, fontweight='bold')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="ΔPSNR (dB)")
for r in range(n_rows):
    for c in range(n_cols):
        val = delta_p6_1[r, c]
        color = 'black' if abs(val) < vmax * 0.6 else 'white'
        axes[0, 0].text(c, r, f"{val:+.1f}", ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')

im1 = axes[0, 1].imshow(c6, cmap='viridis', interpolation='nearest')
axes[0, 1].set_title(f"Step 5000: Exp 6 (DA3 Restored) 3DGS Density per 80x80 Patch\nTotal: {np.sum(c6):,} Gaussians", fontsize=13, fontweight='bold')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="# Gaussians")
for r in range(n_rows):
    for c in range(n_cols):
        axes[0, 1].text(c, r, f"{c6[r, c]}", ha='center', va='center', fontsize=6.5, color='white')

im2 = axes[1, 0].imshow(c1, cmap='viridis', interpolation='nearest')
axes[1, 0].set_title(f"Step 5000: Exp 1 (RoMa) 3DGS Density per 80x80 Patch\nTotal: {np.sum(c1):,} Gaussians", fontsize=13, fontweight='bold')
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label="# Gaussians")
for r in range(n_rows):
    for c in range(n_cols):
        axes[1, 0].text(c, r, f"{c1[r, c]}", ha='center', va='center', fontsize=6.5, color='white')

c_vmax = max(abs(np.min(delta_c6_1)), abs(np.max(delta_c6_1)), 200)
c_norm = mcolors.TwoSlopeNorm(vmin=-c_vmax, vcenter=0, vmax=c_vmax)
im3 = axes[1, 1].imshow(delta_c6_1, cmap='coolwarm', norm=c_norm, interpolation='nearest')
axes[1, 1].set_title(f"Step 5000: Difference in Gaussian Density (Exp 6 - RoMa)\nBlue: DA3 denser | Red: RoMa denser", fontsize=13, fontweight='bold')
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label="ΔGS Count")
for r in range(n_rows):
    for c in range(n_cols):
        color = 'black' if abs(delta_c6_1[r, c]) < c_vmax * 0.6 else 'white'
        axes[1, 1].text(c, r, f"{delta_c6_1[r, c]:+d}", ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')

for ax in axes.flat:
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.grid(True, color='gray', linestyle=':', alpha=0.3)

plt.tight_layout()
p6_out = f"{out_dir}/patch_analysis_step_5000_exp6.png"
plt.savefig(p6_out, dpi=200)
plt.close()

# 2. Figure: Exp 6 (Restored) vs Exp 4 (Base 1%) Direct Comparison (Where did Restoring Background Score Help?)
delta_p6_4 = p6 - p4
fig, (ax_diff, ax_gain) = plt.subplots(1, 2, figsize=(18, 7))

gain_vmax = max(abs(np.min(delta_p6_4)), abs(np.max(delta_p6_4)), 1.5)
g_norm = mcolors.TwoSlopeNorm(vmin=-gain_vmax, vcenter=0, vmax=gain_vmax)
im_diff = ax_diff.imshow(delta_p6_4, cmap='PRGn', norm=g_norm, interpolation='nearest')
n_improved = np.sum(delta_p6_4 > 0)
ax_diff.set_title(f"Direct Quality Delta: Exp 6 (Restored) vs Exp 4 (Base 1%)\nPurple: Exp 6 Improved ({n_improved}/221, {n_improved/221*100:.1f}%) | Green: Exp 4 Higher", fontsize=12, fontweight='bold')
plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04, label="PSNR Gain (Exp6 - Exp4) [dB]")

for r in range(n_rows):
    for c in range(n_cols):
        val = delta_p6_4[r, c]
        color = 'black' if abs(val) < gain_vmax * 0.6 else 'white'
        ax_diff.text(c, r, f"{val:+.1f}", ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')
ax_diff.set_xticks(range(n_cols))
ax_diff.set_yticks(range(n_rows))

# Histogram / bar comparison of smooth plate/background vs steak
meat_mask = np.zeros((13, 17), dtype=bool)
meat_mask[3:8, 6:15] = True
bg_mask = ~meat_mask

categories = ['Steak Meat (45 patches)', 'Smooth Plate & BG (176 patches)', 'Overall (221 patches)']
exp4_means = [np.mean(p4[meat_mask]), np.mean(p4[bg_mask]), np.mean(p4)]
exp6_means = [np.mean(p6[meat_mask]), np.mean(p6[bg_mask]), np.mean(p6)]
roma_means = [np.mean(p1[meat_mask]), np.mean(p1[bg_mask]), np.mean(p1)]

x = np.arange(len(categories))
width = 0.25
rects1 = ax_gain.bar(x - width, roma_means, width, label='Exp 1 (RoMa)', color='#ff7f0e', alpha=0.85)
rects2 = ax_gain.bar(x, exp4_means, width, label='Exp 4 (DA3 Base 1%)', color='#d62728', alpha=0.85)
rects3 = ax_gain.bar(x + width, exp6_means, width, label='Exp 6 (DA3 Restored)', color='#2ca02c', alpha=0.85)

ax_gain.set_ylabel('Average Patch PSNR (dB)', fontsize=12, fontweight='bold')
ax_gain.set_title('Regional PSNR Breakdown: Restoring Background Score Impact', fontsize=12, fontweight='bold')
ax_gain.set_xticks(x)
ax_gain.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax_gain.set_ylim(32, 37)
ax_gain.legend(fontsize=10)
ax_gain.grid(True, axis='y', alpha=0.3)

for rect in rects1 + rects2 + rects3:
    h = rect.get_height()
    ax_gain.annotate(f'{h:.2f}',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
gain_out = f"{out_dir}/comparison_exp4_vs_exp6_gain.png"
plt.savefig(gain_out, dpi=200)
plt.close()

# Copy to art_dir
import shutil
shutil.copyfile(p6_out, f"{art_dir}/patch_analysis_step_5000_exp6.png")
shutil.copyfile(gain_out, f"{art_dir}/comparison_exp4_vs_exp6_gain.png")
print("Saved all Exp 6 visual comparison artifacts successfully!")
