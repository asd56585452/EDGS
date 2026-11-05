import os
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Camera parameters for cam00 in sear_steak (colmap_25)
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
W, H = 1352, 1014
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
        xyz = np.stack([data['x'], data['y'], data['z']], axis=-1)
        return xyz

def project_and_count_grid(xyz, H, W, patch_size=80):
    pts_cam = (R @ xyz.T + tvec[:, None]).T
    valid_z = pts_cam[:, 2] > 0.1
    pts_valid = pts_cam[valid_z]
    u = K[0, 0] * pts_valid[:, 0] / pts_valid[:, 2] + K[0, 2]
    v = K[1, 1] * pts_valid[:, 1] / pts_valid[:, 2] + K[1, 2]
    in_view = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_in = u[in_view]
    v_in = v[in_view]
    
    n_rows = int(np.ceil(H / patch_size))
    n_cols = int(np.ceil(W / patch_size))
    counts = np.zeros((n_rows, n_cols), dtype=np.int32)
    
    r_idx = np.clip((v_in // patch_size).astype(int), 0, n_rows - 1)
    c_idx = np.clip((u_in // patch_size).astype(int), 0, n_cols - 1)
    np.add.at(counts, (r_idx, c_idx), 1)
    return counts

def compute_patch_psnr(img_render, img_gt, patch_size=80):
    H, W, _ = img_render.shape
    n_rows = int(np.ceil(H / patch_size))
    n_cols = int(np.ceil(W / patch_size))
    psnr_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    mse_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    
    for r in range(n_rows):
        y1 = r * patch_size
        y2 = min((r + 1) * patch_size, H)
        for c in range(n_cols):
            x1 = c * patch_size
            x2 = min((c + 1) * patch_size, W)
            crop_rend = img_render[y1:y2, x1:x2]
            crop_gt = img_gt[y1:y2, x1:x2]
            mse = np.mean((crop_rend - crop_gt) ** 2)
            mse_grid[r, c] = mse
            psnr = -10.0 * np.log10(max(mse, 1e-10))
            psnr_grid[r, c] = psnr
    return psnr_grid, mse_grid

steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
out_dir = "/home/tsaichenghan/EDGS/output/patch_analysis_80x80"
os.makedirs(out_dir, exist_ok=True)

stats_all_steps = {}

print("Starting 80x80 patch analysis across all 10 steps...")

for step in steps:
    exp1_rend_p = f"/home/tsaichenghan/EDGS/output/exp1_edgs_native/renders/val_step{step}_render.png"
    exp4_rend_p = f"/home/tsaichenghan/EDGS/output/exp4_our_pcd_on_edgs/renders/val_step{step}_render.png"
    exp1_gt_p = f"/home/tsaichenghan/EDGS/output/exp1_edgs_native/renders/val_step{step}_gt.png"
    
    exp1_ply_p = f"/home/tsaichenghan/EDGS/output/exp1_edgs_native/point_cloud_step_{step}.ply"
    exp4_ply_p = f"/home/tsaichenghan/EDGS/output/exp4_our_pcd_on_edgs/point_cloud_step_{step}.ply"
    
    img_exp1 = np.array(Image.open(exp1_rend_p)).astype(np.float32) / 255.0
    img_exp4 = np.array(Image.open(exp4_rend_p)).astype(np.float32) / 255.0
    img_gt = np.array(Image.open(exp1_gt_p)).astype(np.float32) / 255.0
    
    psnr1, mse1 = compute_patch_psnr(img_exp1, img_gt, patch_size=80)
    psnr4, mse4 = compute_patch_psnr(img_exp4, img_gt, patch_size=80)
    delta_psnr = psnr4 - psnr1  # positive: DA3 wins, negative: DA3 loses
    
    xyz1 = load_ply_xyz(exp1_ply_p)
    xyz4 = load_ply_xyz(exp4_ply_p)
    
    count1 = project_and_count_grid(xyz1, H, W, patch_size=80)
    count4 = project_and_count_grid(xyz4, H, W, patch_size=80)
    delta_count = count4 - count1  # positive: DA3 has more GS, negative: DA3 has fewer
    
    n_patches = delta_psnr.size
    win_mask = delta_psnr > 0
    loss_mask = delta_psnr <= 0
    n_win = int(np.sum(win_mask))
    n_loss = int(np.sum(loss_mask))
    win_pct = (n_win / n_patches) * 100.0
    
    avg_delta_psnr_win = float(np.mean(delta_psnr[win_mask])) if n_win > 0 else 0.0
    avg_delta_psnr_loss = float(np.mean(delta_psnr[loss_mask])) if n_loss > 0 else 0.0
    avg_delta_gs_win = float(np.mean(delta_count[win_mask])) if n_win > 0 else 0.0
    avg_delta_gs_loss = float(np.mean(delta_count[loss_mask])) if n_loss > 0 else 0.0
    
    stats_all_steps[step] = {
        "n_win": n_win,
        "n_loss": n_loss,
        "win_pct": win_pct,
        "loss_pct": 100.0 - win_pct,
        "mean_delta_psnr": float(np.mean(delta_psnr)),
        "avg_win_delta_psnr": avg_delta_psnr_win,
        "avg_loss_delta_psnr": avg_delta_psnr_loss,
        "total_gs_exp1": int(len(xyz1)),
        "total_gs_exp4": int(len(xyz4)),
        "inview_gs_exp1": int(np.sum(count1)),
        "inview_gs_exp4": int(np.sum(count4)),
        "avg_delta_gs_win": avg_delta_gs_win,
        "avg_delta_gs_loss": avg_delta_gs_loss,
        "delta_psnr_grid": delta_psnr.tolist(),
        "psnr1_grid": psnr1.tolist(),
        "psnr4_grid": psnr4.tolist(),
        "count1_grid": count1.tolist(),
        "count4_grid": count4.tolist(),
        "delta_count_grid": delta_count.tolist()
    }
    
    print(f"Step {step:4d} | Win: {n_win:3d} ({win_pct:5.1f}%) | Loss: {n_loss:3d} | Mean ΔPSNR: {np.mean(delta_psnr):+5.2f}dB | DA3 In-view GS: {np.sum(count4):6d} (RoMa: {np.sum(count1):6d})")

# Save stats JSON
with open(f"{out_dir}/patch_80x80_stats.json", "w") as f:
    json.dump(stats_all_steps, f, indent=2)

print("\nGenerating Detailed Visualizations...")

# Generate detailed figures for Key Steps: 500, 1500, 3000, 5000
key_steps = [500, 1500, 3000, 5000]

for step in key_steps:
    d = stats_all_steps[step]
    delta_p = np.array(d["delta_psnr_grid"])
    c1 = np.array(d["count1_grid"])
    c4 = np.array(d["count4_grid"])
    dc = np.array(d["delta_count_grid"])
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    
    # 1. Win/Loss Map (ΔPSNR)
    # Use Diverging colormap: Red for Loss (negative), Green for Win (positive)
    vmax = max(abs(np.min(delta_p)), abs(np.max(delta_p)), 3.0)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im0 = axes[0, 0].imshow(delta_p, cmap='RdYlGn', norm=norm, interpolation='nearest')
    axes[0, 0].set_title(f"Step {step}: Regional Win/Loss Map (ΔPSNR = DA3 - RoMa [dB])\nWin (Green): {d['win_pct']:.1f}% | Loss (Red): {d['loss_pct']:.1f}%", fontsize=13, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="ΔPSNR (dB)")
    
    # Add text labels on cells if grid isn't too huge
    for r in range(delta_p.shape[0]):
        for c in range(delta_p.shape[1]):
            val = delta_p[r, c]
            color = 'black' if abs(val) < vmax * 0.6 else 'white'
            axes[0, 0].text(c, r, f"{val:+.1f}", ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')
    
    # 2. DA3 Regional GS Count (80x80)
    im1 = axes[0, 1].imshow(c4, cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title(f"Step {step}: Exp 4 (DA3) 3DGS Count per 80x80 Patch\nTotal In-view: {d['inview_gs_exp4']:,} Gaussians", fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="# Gaussians")
    for r in range(c4.shape[0]):
        for c in range(c4.shape[1]):
            axes[0, 1].text(c, r, f"{c4[r, c]}", ha='center', va='center', fontsize=6.5, color='white')
            
    # 3. RoMa Regional GS Count (80x80)
    im2 = axes[1, 0].imshow(c1, cmap='viridis', interpolation='nearest')
    axes[1, 0].set_title(f"Step {step}: Exp 1 (RoMa) 3DGS Count per 80x80 Patch\nTotal In-view: {d['inview_gs_exp1']:,} Gaussians", fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label="# Gaussians")
    for r in range(c1.shape[0]):
        for c in range(c1.shape[1]):
            axes[1, 0].text(c, r, f"{c1[r, c]}", ha='center', va='center', fontsize=6.5, color='white')
            
    # 4. Delta GS Count (DA3 - RoMa)
    c_vmax = max(abs(np.min(dc)), abs(np.max(dc)), 200)
    c_norm = mcolors.TwoSlopeNorm(vmin=-c_vmax, vcenter=0, vmax=c_vmax)
    im3 = axes[1, 1].imshow(dc, cmap='coolwarm', norm=c_norm, interpolation='nearest')
    axes[1, 1].set_title(f"Step {step}: Difference in Gaussian Density (DA3 - RoMa)\nBlue: DA3 denser | Red: DA3 sparser (Pruned)", fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label="ΔGS Count")
    for r in range(dc.shape[0]):
        for c in range(dc.shape[1]):
            color = 'black' if abs(dc[r, c]) < c_vmax * 0.6 else 'white'
            axes[1, 1].text(c, r, f"{dc[r, c]:+d}", ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')
            
    for ax in axes.flat:
        ax.set_xticks(range(delta_p.shape[1]))
        ax.set_yticks(range(delta_p.shape[0]))
        ax.grid(True, color='gray', linestyle=':', alpha=0.3)
        
    plt.tight_layout()
    step_plot_p = f"{out_dir}/patch_analysis_step_{step}.png"
    plt.savefig(step_plot_p, dpi=200)
    plt.close()
    print(f"Saved: {step_plot_p}")

# Figure 5: Comprehensive 10-Step Evolution Timeline
fig, (ax_win, ax_gs) = plt.subplots(1, 2, figsize=(16, 6))

step_list = [s for s in steps]
win_pcts = [stats_all_steps[s]["win_pct"] for s in steps]
loss_pcts = [stats_all_steps[s]["loss_pct"] for s in steps]
inview_da3 = [stats_all_steps[s]["inview_gs_exp4"] for s in steps]
inview_roma = [stats_all_steps[s]["inview_gs_exp1"] for s in steps]

ax_win.plot(step_list, win_pcts, 'o-', color='green', linewidth=2.5, markersize=8, label='DA3 Winning Area % (PSNR > RoMa)')
ax_win.plot(step_list, loss_pcts, 's--', color='crimson', linewidth=2.5, markersize=8, label='DA3 Losing Area % (PSNR < RoMa)')
ax_win.axhline(50, color='gray', linestyle=':', alpha=0.7)
ax_win.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax_win.set_ylabel('Patch Percentage (%)', fontsize=12, fontweight='bold')
ax_win.set_title('Evolution of Win vs Loss Territory Across 10 Steps (80x80 Patches)', fontsize=13, fontweight='bold')
ax_win.grid(True, alpha=0.3)
ax_win.legend(fontsize=11)
for i, s in enumerate(step_list):
    ax_win.annotate(f"{win_pcts[i]:.1f}%", (s, win_pcts[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, fontweight='bold', color='green')

ax_gs.plot(step_list, inview_da3, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='Exp 4 (DA3) In-view GS Count')
ax_gs.plot(step_list, inview_roma, 's-', color='#ff7f0e', linewidth=2.5, markersize=8, label='Exp 1 (RoMa) In-view GS Count')
ax_gs.set_xlabel('Training Step', fontsize=12, fontweight='bold')
ax_gs.set_ylabel('Total In-view Gaussians (#)', fontsize=12, fontweight='bold')
ax_gs.set_title('In-view Gaussian Count Trajectory (Aggressive Pruning in DA3)', fontsize=13, fontweight='bold')
ax_gs.grid(True, alpha=0.3)
ax_gs.legend(fontsize=11)
for i, s in enumerate(step_list):
    ax_gs.annotate(f"{inview_da3[i]//1000}k", (s, inview_da3[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#1f77b4', fontweight='bold')

plt.tight_layout()
timeline_plot_p = f"{out_dir}/win_loss_and_gs_evolution_timeline.png"
plt.savefig(timeline_plot_p, dpi=200)
plt.close()
print(f"Saved: {timeline_plot_p}")

print("ALL 80x80 patch analysis completed successfully!")
