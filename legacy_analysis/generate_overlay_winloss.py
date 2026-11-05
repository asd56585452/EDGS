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

steps = [500, 1500, 3000, 5000]

fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes_flat = axes.flat

for idx, step in enumerate(steps):
    ax = axes_flat[idx]
    exp1_rend_p = f"/home/tsaichenghan/EDGS/output/exp1_edgs_native/renders/val_step{step}_render.png"
    exp4_rend_p = f"/home/tsaichenghan/EDGS/output/exp4_our_pcd_on_edgs/renders/val_step{step}_render.png"
    gt_p = f"/home/tsaichenghan/EDGS/output/exp1_edgs_native/renders/val_step{step}_gt.png"
    
    img_gt = np.array(Image.open(gt_p)).astype(np.float32) / 255.0
    img_e1 = np.array(Image.open(exp1_rend_p)).astype(np.float32) / 255.0
    img_e4 = np.array(Image.open(exp4_rend_p)).astype(np.float32) / 255.0
    
    delta_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    overlay_mask = np.zeros((H, W, 4), dtype=np.float32)
    
    n_win, n_loss = 0, 0
    for r in range(n_rows):
        y1 = r * patch_size
        y2 = min((r + 1) * patch_size, H)
        for c in range(n_cols):
            x1 = c * patch_size
            x2 = min((c + 1) * patch_size, W)
            m1 = np.mean((img_e1[y1:y2, x1:x2] - img_gt[y1:y2, x1:x2]) ** 2)
            m4 = np.mean((img_e4[y1:y2, x1:x2] - img_gt[y1:y2, x1:x2]) ** 2)
            p1 = -10.0 * np.log10(max(m1, 1e-10))
            p4 = -10.0 * np.log10(max(m4, 1e-10))
            dp = p4 - p1
            delta_grid[r, c] = dp
            if dp > 0:
                n_win += 1
                # Green tint with alpha proportional to win margin
                alpha = min(0.6, 0.2 + abs(dp) * 0.1)
                overlay_mask[y1:y2, x1:x2] = [0.0, 0.9, 0.2, alpha]
            else:
                n_loss += 1
                # Red tint with alpha proportional to loss margin
                alpha = min(0.6, 0.2 + abs(dp) * 0.1)
                overlay_mask[y1:y2, x1:x2] = [0.95, 0.1, 0.1, alpha]
                
    win_pct = n_win / (n_win + n_loss) * 100.0
    
    # Render base GT image in grayscale background to make win/loss tint pop
    gray_gt = np.dot(img_gt[..., :3], [0.2989, 0.5870, 0.1140])
    base_rgb = np.stack([gray_gt * 0.7 + img_gt[..., 0] * 0.3,
                        gray_gt * 0.7 + img_gt[..., 1] * 0.3,
                        gray_gt * 0.7 + img_gt[..., 2] * 0.3], axis=-1)
    
    # Alpha blend
    alpha = overlay_mask[..., 3:4]
    blended = base_rgb * (1.0 - alpha) + overlay_mask[..., :3] * alpha
    
    ax.imshow(blended)
    
    # Draw 80x80 grid lines
    for x in range(0, W, patch_size):
        ax.axvline(x, color='white', linestyle=':', alpha=0.35, linewidth=0.8)
    for y in range(0, H, patch_size):
        ax.axhline(y, color='white', linestyle=':', alpha=0.35, linewidth=0.8)
        
    ax.set_title(f"Step {step}: Spatial Win/Loss Territory Map (80x80 Grid)\n"
                 f"🟩 DA3 Win Territory: {win_pct:.1f}% ({n_win} patches) | 🟥 DA3 Loss Territory: {100-win_pct:.1f}% ({n_loss} patches)",
                 fontsize=13, fontweight='bold', pad=10)
    ax.axis('off')

plt.tight_layout()
overlay_all_p = f"{out_dir}/spatial_win_loss_territory_evolution.png"
plt.savefig(overlay_all_p, dpi=200)
plt.close()

import shutil
shutil.copyfile(overlay_all_p, f"{art_dir}/spatial_win_loss_territory_evolution.png")
print("Saved spatial_win_loss_territory_evolution.png successfully!")
