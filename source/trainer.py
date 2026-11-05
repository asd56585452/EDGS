import torch
import os
from random import randint
from tqdm.rich import trange
from tqdm import tqdm as tqdm
from source.networks import Warper3DGS
import wandb
import sys

sys.path.append('./submodules/gaussian-splatting/')
import lpips
from source.losses import ssim, l1_loss, psnr
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red"
})

from source.corr_init import init_gaussians_with_corr, init_gaussians_with_corr_fast
from source.utils_aux import log_samples

from source.timer import Timer

class EDGSTrainer:
    def __init__(self,
                 GS: Warper3DGS,
                 training_config,
                 dataset_white_background=False,
                 device=torch.device('cuda'),
                 log_wandb=True,
                 ):
        self.GS = GS
        self.scene = GS.scene
        self.viewpoint_stack = GS.viewpoint_stack
        self.gaussians = GS.gaussians

        self.training_config = training_config
        self.GS_optimizer = GS.gaussians.optimizer
        self.dataset_white_background = dataset_white_background

        self.training_step = 1
        self.gs_step = 0
        self.CONSOLE = Console(width=120, theme=custom_theme)
        self.saving_iterations = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        self.evaluate_iterations = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        self.eval_history = []
        self.batch_size = training_config.batch_size
        self.ema_loss_for_log = 0.0

        # Logs in the format {step:{"loss1":loss1_value, "loss2":loss2_value}}
        self.logs_losses = {}
        self.lpips = lpips.LPIPS(net='vgg').to(device)
        self.device = device
        self.timer = Timer()
        self.log_wandb = log_wandb

    def load_checkpoints(self, load_cfg):
        # Load 3DGS checkpoint
        if load_cfg.gs:
            self.gs.gaussians.restore(
                torch.load(f"{load_cfg.gs}/chkpnt{load_cfg.gs_step}.pth")[0],
                self.training_config)
            self.GS_optimizer = self.GS.gaussians.optimizer
            self.CONSOLE.print(f"3DGS loaded from checkpoint for iteration {load_cfg.gs_step}",
                               style="info")
            self.training_step += load_cfg.gs_step
            self.gs_step += load_cfg.gs_step

    def train(self, train_cfg):
        # 3DGS training
        self.CONSOLE.print("Train 3DGS for {} iterations".format(train_cfg.gs_epochs), style="info")
        # 超過 5000 步的長訓練：每 500 步評估、每 2500 步存檔（5000 步內的清單不變）
        end_step = self.training_step + train_cfg.gs_epochs
        self.evaluate_iterations = sorted(set(self.evaluate_iterations) | set(range(5500, end_step, 500)))
        self.saving_iterations = sorted(set(self.saving_iterations) | set(range(7500, end_step, 2500)))
        self.final_step = end_step - 1
        self.save_final_only = bool(train_cfg.get("save_final_only", True))
        if self.save_final_only:
            self.saving_iterations = sorted(set(self.saving_iterations) | {self.final_step})
        with trange(self.training_step, self.training_step + train_cfg.gs_epochs, desc="[green]Train gaussians") as progress_bar:
            for self.training_step in progress_bar:
                radii = self.train_step_gs(max_lr=train_cfg.max_lr, no_densify=train_cfg.no_densify)
                with torch.no_grad():
                    if train_cfg.no_densify:
                        self.prune(radii)
                    else:
                        self.densify_and_prune(radii)
                    if train_cfg.reduce_opacity:
                        # Slightly reduce opacity every few steps:
                        if self.gs_step < self.training_config.densify_until_iter and self.gs_step % 10 == 0:
                            opacities_new = torch.log(torch.exp(self.GS.gaussians._opacity.data) * 0.99)
                            self.GS.gaussians._opacity.data = opacities_new
                    self.timer.pause()
                    # Progress bar
                    if self.training_step % 10 == 0:
                        progress_bar.set_postfix({"[red]Loss": f"{self.ema_loss_for_log:.{7}f}"}, refresh=True)
                    # Log and save
                    if self.training_step in self.saving_iterations:
                        self.save_model()
                    if self.evaluate_iterations is not None:
                        if self.training_step in self.evaluate_iterations:
                            self.evaluate()
                    else:
                        if (self.training_step <= 3000 and self.training_step % 500 == 0) or \
                            (self.training_step > 3000 and self.training_step % 1000 == 228) :
                            self.evaluate()

                    self.timer.start()


    def evaluate(self):
        torch.cuda.empty_cache()
        log_gen_images, log_real_images = [], []
        validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras(), 'cam_idx': self.training_config.TEST_CAM_IDX_TO_LOG},
                              {'name': 'train',
                               'cameras': [self.scene.getTrainCameras()[idx % len(self.scene.getTrainCameras())] for idx in
                                           range(0, 150, 5)], 'cam_idx': 10})
        if self.log_wandb:
            wandb.log({f"Number of Gaussians": len(self.GS.gaussians._xyz)}, step=self.training_step)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_splat_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(self.GS(viewpoint)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(self.device), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).double()
                    psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).double()
                    ssim_test += ssim(image, gt_image).double()
                    lpips_splat_test += self.lpips(image, gt_image).detach().double()
                    target_cam_idx = min(config['cam_idx'], len(config['cameras']) - 1)
                    if idx == target_cam_idx:
                        log_gen_images.append(image)
                        log_real_images.append(gt_image)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_splat_test /= len(config['cameras'])
                if self.log_wandb:
                    wandb.log({f"{config['name']}/L1": l1_test.item(), f"{config['name']}/PSNR": psnr_test.item(), \
                            f"{config['name']}/SSIM": ssim_test.item(), f"{config['name']}/LPIPS_splat": lpips_splat_test.item()}, step = self.training_step)
                self.CONSOLE.print("\n[ITER {}], #{} gaussians, Evaluating {}: L1={:.6f},  PSNR={:.6f}, SSIM={:.6f}, LPIPS_splat={:.6f} ".format(
                    self.training_step, len(self.GS.gaussians._xyz), config['name'], l1_test.item(), psnr_test.item(), ssim_test.item(), lpips_splat_test.item()), style="info")

                if config['name'] == 'test' and len(config['cameras']) > 0:
                    try:
                        render_dir = os.path.join(self.scene.model_path, "renders")
                        os.makedirs(render_dir, exist_ok=True)
                        from torchvision.utils import save_image
                        test_view = config['cameras'][0]
                        test_render = torch.clamp(self.GS(test_view)["render"], 0.0, 1.0)
                        test_gt = torch.clamp(test_view.original_image.to(self.device), 0.0, 1.0)
                        side_by_side = torch.cat([test_gt, test_render], dim=-1)
                        save_image(side_by_side, os.path.join(render_dir, f"val_step{self.training_step}_0000.png"))
                        save_image(test_render, os.path.join(render_dir, f"val_step{self.training_step}_render.png"))
                        save_image(test_gt, os.path.join(render_dir, f"val_step{self.training_step}_gt.png"))
                    except Exception as e:
                        print(f"Error saving test render: {e}")

                    self.eval_history.append({
                        "step": self.training_step,
                        "psnr": float(psnr_test.item()),
                        "ssim": float(ssim_test.item()),
                        "lpips": float(lpips_splat_test.item()),
                        "splats": len(self.GS.gaussians._xyz)
                    })
                    import json
                    with open(os.path.join(self.scene.model_path, "eval_stats.json"), "w") as f:
                        json.dump(self.eval_history, f, indent=2)
        if self.log_wandb:
            if len(log_real_images) > 0 and len(log_gen_images) > 0:
                with torch.no_grad():
                    log_samples(torch.stack((log_real_images[0],log_gen_images[0])) , [], self.training_step, caption="Real and Generated Samples")
            wandb.log({"time": self.timer.get_elapsed_time()}, step=self.training_step)
        torch.cuda.empty_cache()

    def train_step_gs(self, max_lr = False, no_densify = False):
        self.gs_step += 1
        if max_lr:
            self.GS.gaussians.update_learning_rate(max(self.gs_step, 8_000))
        else:
            self.GS.gaussians.update_learning_rate(self.gs_step)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.gs_step % 1000 == 0:
            self.GS.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
      
        render_pkg = self.GS(viewpoint_cam=viewpoint_cam)
        image = render_pkg["render"]
        # Loss
        gt_image = viewpoint_cam.original_image.to(self.device)
        L1_loss = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - self.training_config.lambda_dssim) * L1_loss + \
               self.training_config.lambda_dssim * ssim_loss
        self.timer.pause() 
        self.logs_losses[self.training_step] = {"loss": loss.item(),
                                                "L1_loss": L1_loss.item(),
                                                "ssim_loss": ssim_loss.item()}
        
        if self.log_wandb:
            for k, v in self.logs_losses[self.training_step].items():
                wandb.log({f"train/{k}": v}, step=self.training_step)
        self.ema_loss_for_log = 0.4 * self.logs_losses[self.training_step]["loss"] + 0.6 * self.ema_loss_for_log
        self.timer.start()
        self.GS_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            if self.gs_step < self.training_config.densify_until_iter and not no_densify:
                self.GS.gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(
                    self.GS.gaussians.max_radii2D[render_pkg["visibility_filter"]],
                    render_pkg["radii"][render_pkg["visibility_filter"]])
                self.GS.gaussians.add_densification_stats(render_pkg["viewspace_points"],
                                                                     render_pkg["visibility_filter"])

        # Optimizer step
        self.GS_optimizer.step()
        self.GS_optimizer.zero_grad(set_to_none=True)
        return render_pkg["radii"]

    def densify_and_prune(self, radii = None):
        # Densification or pruning
        if self.gs_step < self.training_config.densify_until_iter:
            if (self.gs_step > self.training_config.densify_from_iter) and \
                    (self.gs_step % self.training_config.densification_interval == 0):
                size_threshold = 20 if self.gs_step > self.training_config.opacity_reset_interval else None
                self.GS.gaussians.densify_and_prune(self.training_config.densify_grad_threshold,
                                                               0.005,
                                                               self.GS.scene.cameras_extent,
                                                               size_threshold, radii)
            if self.gs_step % self.training_config.opacity_reset_interval == 0 or (
                    self.dataset_white_background and self.gs_step == self.training_config.densify_from_iter):
                self.GS.gaussians.reset_opacity()             

          

    def save_model(self):
        if getattr(self, "alive_init_idx", None) is not None and len(self.alive_init_idx) == len(self.GS.gaussians._xyz):
            import numpy as np
            np.save(os.path.join(self.scene.model_path, f"alive_init_idx_step_{self.gs_step}.npy"),
                    self.alive_init_idx.cpu().numpy())
        # save_final_only：中間步只留小檔（alive_init_idx），checkpoint / 點雲 PLY 只在最後一步存，避免塞滿硬碟
        if getattr(self, "save_final_only", False) and self.training_step != getattr(self, "final_step", None):
            return
        print(f"[ITER {self.gs_step}] Saving Gaussians & Standard Universal PLY")
        self.scene.save(self.gs_step)
        root_ply_path = os.path.join(self.scene.model_path, f"point_cloud_step_{self.gs_step}.ply")
        self.GS.gaussians.save_ply(root_ply_path)
        print(f"[ITER {self.gs_step}] Saving Checkpoint: {root_ply_path}")
        try:
            import numpy as np
            xyz = self.GS.gaussians.get_xyz.detach().cpu().numpy()
            f_dc = self.GS.gaussians._features_dc.detach().cpu().numpy().reshape(-1, 3)
            rgb = np.clip((f_dc * 0.28209479177387814 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            header_lines = [
                "ply",
                "format binary_little_endian 1.0",
                f"element vertex {len(xyz)}",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header"
            ]
            header_bytes = "\n".join(header_lines).encode("ascii") + b"\n"
            out_data = np.empty(len(xyz), dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1")
            ])
            out_data["x"] = xyz[:, 0]
            out_data["y"] = xyz[:, 1]
            out_data["z"] = xyz[:, 2]
            out_data["red"] = rgb[:, 0]
            out_data["green"] = rgb[:, 1]
            out_data["blue"] = rgb[:, 2]
            rgb_ply_path = os.path.join(self.scene.model_path, f"point_cloud_step_{self.gs_step}_rgb.ply")
            with open(rgb_ply_path, "wb") as f_rgb:
                f_rgb.write(header_bytes)
                f_rgb.write(out_data.tobytes())
            print(f"[ITER {self.gs_step}] Saving MeshLab RGB PLY: {rgb_ply_path}")
        except Exception as e:
            print(f"Warning: failed to save RGB ply: {e}")
        torch.save((self.GS.gaussians.capture(), self.gs_step),
                self.scene.model_path + "/chkpnt" + str(self.gs_step) + ".pth")

    def init_with_corr(self, cfg, verbose=False, roma_model=None): 
        """
        Initializes image with matchings. Also removes SfM init points.
        Args:
            cfg: configuration part named init_wC. Check train.yaml
            verbose: whether you want to print intermediate results. Useful for debug.
            roma_model: optionally you can pass here preinit RoMA model to avoid reinit 
                it every time.  
        """
        if not cfg.use:
            with torch.no_grad():
                gaussians = self.gaussians
                # 可選：讓自訂 PLY 的初始 opacity / scale 與原生 RoMa 路徑一致（init_gaussians_with_corr）
                # custom_opacity / custom_scale 套用全部點；*_static / *_dyn 依 PLY 的 label 欄位
                # （0 = 靜態, 1 = 動態）覆寫各自群組，未設定則沿用全域值。
                n = len(gaussians._xyz)
                dev = gaussians._xyz.device
                labels = None
                ply_field = {}   # 'ply' 模式：逐點初始值 init_opacity（已 sigmoid）、init_scale（×0.5 前的世界尺寸）
                input_ply = os.path.join(self.scene.model_path, "input.ply")
                if os.path.exists(input_ply):
                    import numpy as np
                    from plyfile import PlyData
                    v = PlyData.read(input_ply)["vertex"]
                    if "label" in v.data.dtype.names and len(v["label"]) == n:
                        labels = torch.from_numpy(np.asarray(v["label"]).astype(np.int64)).to(dev)
                    for k in ("init_opacity", "init_scale"):
                        if k in v.data.dtype.names and len(v[k]) == n:
                            ply_field[k] = torch.from_numpy(np.asarray(v[k]).astype(np.float32)).to(dev)
                groups = {"all": torch.ones(n, dtype=torch.bool, device=dev)}
                if labels is not None:
                    groups = {"static": labels == 0, "dyn": labels == 1}
                elif any(cfg.get(f"custom_{k}_{g}", None) is not None
                         for k in ("opacity", "scale") for g in ("static", "dyn")):
                    print("[EDGS] ⚠️ custom_*_static/_dyn 已設定，但 input.ply 沒有 label 欄位，改用全域設定")
                d_cam = None
                msgs = []
                for g, m in groups.items():
                    op = cfg.get(f"custom_opacity_{g}", None) if g != "all" else None
                    op = cfg.get("custom_opacity", None) if op is None else op
                    sc = cfg.get(f"custom_scale_{g}", None) if g != "all" else None
                    sc = cfg.get("custom_scale", "knn") if sc is None else sc
                    if op == "ply":
                        assert "init_opacity" in ply_field, "custom_opacity=ply 但 input.ply 沒有 init_opacity"
                        o = ply_field["init_opacity"][m].clamp(1e-6, 1 - 1e-6).unsqueeze(1)
                        gaussians._opacity.data[m] = gaussians.inverse_opacity_activation(o)
                    elif op is not None:
                        gaussians._opacity.data[m] = gaussians.inverse_opacity_activation(
                            torch.full_like(gaussians._opacity.data[m], float(op)))
                    if sc == "ply":
                        assert "init_scale" in ply_field, "custom_scale=ply 但 input.ply 沒有 init_scale"
                        gaussians._scaling.data[m] = gaussians.scaling_inverse_activation(
                            ply_field["init_scale"][m].unsqueeze(1).repeat(1, 3))
                    elif sc == "cam_dist":
                        if d_cam is None:
                            centers = torch.stack([c.camera_center for c in self.scene.getTrainCameras()]).to(dev)
                            d_cam = torch.cdist(gaussians._xyz.data, centers).min(dim=1).values
                        gaussians._scaling.data[m] = gaussians.scaling_inverse_activation(
                            (d_cam[m] * cfg.scaling_factor).unsqueeze(1).repeat(1, 3))
                    msgs.append(f"{g}({int(m.sum())}): opacity={op if op is not None else 'default 0.1'}, scale={sc}")
                print("[EDGS] custom init: " + " | ".join(msgs))
                gaussians._scaling = gaussians.scaling_inverse_activation(gaussians.scaling_activation(gaussians._scaling)*0.5)
                if hasattr(self.scene, "model_path") and self.scene.model_path:
                    init_dir = os.path.join(self.scene.model_path, "point_cloud", "iteration_0")
                    os.makedirs(init_dir, exist_ok=True)
                    self.gaussians.save_ply(os.path.join(init_dir, "point_cloud.ply"))
                    self.gaussians.save_ply(os.path.join(self.scene.model_path, "custom_init.ply"))
            self.init_splats = len(self.gaussians._xyz)
            print(f"[EDGS] Initialized with custom point cloud: {self.init_splats} points")
            return None
        N_splats_at_init = len(self.GS.gaussians._xyz)
        print("N_splats_at_init:", N_splats_at_init)
        if cfg.nns_per_ref == 1:
            init_fn = init_gaussians_with_corr_fast
        else:
            init_fn = init_gaussians_with_corr
        camera_set, selected_indices, visualization_dict = init_fn(
            self.GS.gaussians, 
            self.scene, 
            cfg, 
            self.device,                                                                                    
            verbose=verbose,
            roma_model=roma_model)

        # Remove SfM points and leave only matchings inits
        if not cfg.add_SfM_init:
            with torch.no_grad():
                N_splats_after_init = len(self.GS.gaussians._xyz)
                print("N_splats_after_init:", N_splats_after_init)
                self.gaussians.tmp_radii = torch.zeros(self.gaussians._xyz.shape[0]).to(self.device)
                mask = torch.concat([torch.ones(N_splats_at_init, dtype=torch.bool),
                                    torch.zeros(N_splats_after_init-N_splats_at_init, dtype=torch.bool)],
                                axis=0)
                self.GS.gaussians.prune_points(mask)
        with torch.no_grad():
            gaussians =  self.gaussians
            gaussians._scaling =  gaussians.scaling_inverse_activation(gaussians.scaling_activation(gaussians._scaling)*0.5)
            
            # Auto-save RoMa initial point cloud as PLY
            if hasattr(self.scene, 'model_path') and self.scene.model_path:
                roma_iter0_dir = os.path.join(self.scene.model_path, "point_cloud", "iteration_0")
                os.makedirs(roma_iter0_dir, exist_ok=True)
                self.gaussians.save_ply(os.path.join(roma_iter0_dir, "point_cloud.ply"))
                roma_ply_path = os.path.join(self.scene.model_path, "roma_init.ply")
                self.gaussians.save_ply(roma_ply_path)
                
                # Also save standard XYZ+RGB PLY for generic 3D viewers (CloudCompare / MeshLab)
                import numpy as np
                from plyfile import PlyData, PlyElement
                xyz = self.gaussians._xyz.detach().cpu().numpy()
                f_dc = self.gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
                rgb = np.clip((0.5 + 0.28209479177387814 * f_dc) * 255, 0, 255).astype(np.uint8)
                vertex = np.empty(xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
                vertex['x'], vertex['y'], vertex['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                vertex['red'], vertex['green'], vertex['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
                roma_rgb_ply = os.path.join(self.scene.model_path, "roma_init_rgb.ply")
                PlyData([PlyElement.describe(vertex, 'vertex')]).write(roma_rgb_ply)
                print(f"[EDGS] RoMa initial point cloud saved: {roma_ply_path} & {roma_rgb_ply} ({len(xyz)} points)")
        self.init_splats = len(self.gaussians._xyz)
        return visualization_dict
    

    def prune(self, radii, min_opacity=0.005):
        self.GS.gaussians.tmp_radii = radii
        if self.gs_step < self.training_config.densify_until_iter:
            prune_mask = (self.GS.gaussians.get_opacity < min_opacity).squeeze()
            # 追蹤存活高斯對應的初始點索引（no_densify 時只有剪枝會改變點數，順序保留）
            if getattr(self, "alive_init_idx", None) is None:
                self.alive_init_idx = torch.arange(len(prune_mask), device=prune_mask.device)
            if len(self.alive_init_idx) == len(prune_mask):
                self.alive_init_idx = self.alive_init_idx[~prune_mask]
            self.GS.gaussians.prune_points(prune_mask)
            torch.cuda.empty_cache()
        self.GS.gaussians.tmp_radii = None

