import os
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

gt = np.array(Image.open('/home/tsaichenghan/EDGS/output/exp1_edgs_native/renders/val_step1500_gt.png')).astype(np.float32)/255.0
r4 = np.array(Image.open('/home/tsaichenghan/EDGS/output/exp4_our_pcd_on_edgs/renders/val_step1500_render.png')).astype(np.float32)/255.0
r1 = np.array(Image.open('/home/tsaichenghan/EDGS/output/exp1_edgs_native/renders/val_step3000_render.png')).astype(np.float32)/255.0

p4 = compute_patch_psnr(r4, gt)
p1 = compute_patch_psnr(r1, gt)
delta_p = p4 - p1

c4 = project_and_count_grid(load_ply_xyz('/home/tsaichenghan/EDGS/output/exp4_our_pcd_on_edgs/point_cloud_step_1500.ply'))
c1 = project_and_count_grid(load_ply_xyz('/home/tsaichenghan/EDGS/output/exp1_edgs_native/point_cloud_step_3000.ply'))
delta_c = c4 - c1

fig, axes = plt.subplots(2, 2, figsize=(18, 13))

vmax = max(abs(np.min(delta_p)), abs(np.max(delta_p)), 3.0)
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im0 = axes[0, 0].imshow(delta_p, cmap='RdYlGn', norm=norm, interpolation='nearest')
n_win = np.sum(delta_p > 0)
win_pct = (n_win / delta_p.size) * 100
axes[0, 0].set_title(f"Exp 4 (DA3 Step 1500) vs Exp 1 (RoMa Step 3000)\nWin (Green): {win_pct:.1f}% ({n_win}/221) | Loss (Red): {100-win_pct:.1f}% | Mean ΔPSNR: {np.mean(delta_p):+.2f}dB", fontsize=13, fontweight='bold')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="ΔPSNR (dB)")
for r in range(n_rows):
    for c in range(n_cols):
        val = delta_p[r, c]
        color = 'black' if abs(val) < vmax * 0.6 else 'white'
        axes[0, 0].text(c, r, f"{val:+.1f}", ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')

im1 = axes[0, 1].imshow(c4, cmap='viridis', interpolation='nearest')
axes[0, 1].set_title(f"Exp 4 (DA3 Step 1500) 3DGS Density per 80x80 Patch\nTotal: {np.sum(c4):,} Gaussians", fontsize=13, fontweight='bold')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="# Gaussians")
for r in range(n_rows):
    for c in range(n_cols):
        axes[0, 1].text(c, r, f"{c4[r, c]}", ha='center', va='center', fontsize=6.5, color='white')

im2 = axes[1, 0].imshow(c1, cmap='viridis', interpolation='nearest')
axes[1, 0].set_title(f"Exp 1 (RoMa Step 3000) 3DGS Density per 80x80 Patch\nTotal: {np.sum(c1):,} Gaussians", fontsize=13, fontweight='bold')
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label="# Gaussians")
for r in range(n_rows):
    for c in range(n_cols):
        axes[1, 0].text(c, r, f"{c1[r, c]}", ha='center', va='center', fontsize=6.5, color='white')

c_vmax = max(abs(np.min(delta_c)), abs(np.max(delta_c)), 200)
c_norm = mcolors.TwoSlopeNorm(vmin=-c_vmax, vcenter=0, vmax=c_vmax)
im3 = axes[1, 1].imshow(delta_c, cmap='coolwarm', norm=c_norm, interpolation='nearest')
axes[1, 1].set_title(f"Difference in Density (DA3 Step 1500 - RoMa Step 3000)\nBlue: DA3 denser | Red: RoMa denser", fontsize=13, fontweight='bold')
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label="ΔGS Count")
for r in range(n_rows):
    for c in range(n_cols):
        color = 'black' if abs(delta_c[r, c]) < c_vmax * 0.6 else 'white'
        axes[1, 1].text(c, r, f"{delta_c[r, c]:+d}", ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')

for ax in axes.flat:
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.grid(True, color='gray', linestyle=':', alpha=0.3)

plt.tight_layout()
p_out = f"{out_dir}/comparison_exp4_1500_vs_exp1_3000.png"
plt.savefig(p_out, dpi=200)
plt.close()

import shutil
shutil.copyfile(p_out, f"{art_dir}/comparison_exp4_1500_vs_exp1_3000.png")
print("Saved comparison_exp4_1500_vs_exp1_3000.png successfully!")
