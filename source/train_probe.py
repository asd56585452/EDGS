"""
訓練探針（全部預設關閉，不改變數值結果；freeze 例外，它就是要改訓練）：

  train.profile_steps : [start, end]  在這段 gs_step 開 torch.profiler，輸出
                        profile_kernels.txt（依 CUDA 時間排序的 kernel / op 表）、profile_phases.json（各階段時間、每步牆鐘時間）
  train.render_stats  : True          每步記 num_rendered（高斯 × tile 配對數 = 排序 key 數）、可見高斯數、總高斯數 → render_stats.npz
                        （包一層 diff_gaussian_rasterization._C.rasterize_gaussians，不需重編 CUDA）
  train.attr_log_every: N             每 N 步記各屬性 vs 初始值 / vs 上次快照的變化量與梯度範數（依群組）→ attr_log.json、attr_final_delta.npz
  train.freeze        : [xyz, ...]    把這些參數群組的 lr 設 0（xyz/f_dc/f_rest/opacity/scaling/rotation）
"""

import contextlib
import json
import os
import time
import types

import numpy as np
import torch

GROUPS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")
SH_C0 = 0.28209479177387814


# ----------------------------------------------------------------------------- render stats
class RenderStats:
    def __init__(self):
        import diff_gaussian_rasterization as dgr
        self.dgr = dgr
        self.orig = dgr._C
        self.last = None
        outer = self

        def rasterize_gaussians(*args):
            out = outer.orig.rasterize_gaussians(*args)
            outer.last = int(out[0])
            return out

        ns = types.SimpleNamespace(**{k: getattr(self.orig, k) for k in dir(self.orig) if not k.startswith("__")})
        ns.rasterize_gaussians = rasterize_gaussians
        dgr._C = ns
        self.rows = []  # (gs_step, num_rendered, n_visible, n_total, H*W)

    def record(self, step, radii, n_total, hw):
        self.rows.append((step, self.last if self.last is not None else -1, int((radii > 0).sum()), n_total, hw))

    def save(self, path):
        self.dgr._C = self.orig
        a = np.array(self.rows, dtype=np.int64)
        np.savez_compressed(path, step=a[:, 0], num_rendered=a[:, 1], n_visible=a[:, 2], n_total=a[:, 3], hw=a[:, 4])


# ----------------------------------------------------------------------------- profiler
class Profiler:
    def __init__(self, start, end, out_dir):
        self.start, self.end, self.out_dir = int(start), int(end), out_dir
        self.prof = None
        self.wall = []
        self._t = None

    def active(self, step):
        return self.start <= step <= self.end

    def step_begin(self, step):
        if step == self.start:
            torch.cuda.synchronize()
            from torch.profiler import ProfilerActivity, profile
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False)
            self.prof.__enter__()
            self._t0 = time.perf_counter()
        if self.active(step):
            torch.cuda.synchronize()
            self._t = time.perf_counter()

    def step_end(self, step):
        if not self.active(step):
            return
        torch.cuda.synchronize()
        self.wall.append(time.perf_counter() - self._t)
        if step == self.end:
            self.prof.__exit__(None, None, None)
            self.dump()

    def rf(self, name, step):
        if self.prof is not None and self.active(step):
            return torch.profiler.record_function(name)
        return contextlib.nullcontext()

    def dump(self):
        ka = self.prof.key_averages()
        n = len(self.wall)
        dev = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        with open(os.path.join(self.out_dir, "profile_kernels.txt"), "w") as f:
            f.write(f"steps {self.start}-{self.end} ({n} steps), wall per step {1e3 * np.mean(self.wall):.2f} ms\n\n")
            f.write(ka.table(sort_by="cuda_time_total" if not hasattr(ka[0], "device_time_total") else "device_time_total",
                             row_limit=60, max_name_column_width=90))
        phases = {}
        for e in ka:
            if e.key.startswith("probe/"):
                phases[e.key] = {"cpu_ms_per_step": e.cpu_time_total / 1e3 / n, "cuda_ms_per_step": dev(e) / 1e3 / n,
                                 "count": e.count}
        kern = sorted(((e.key, dev(e) / 1e3 / n) for e in ka if dev(e) > 0 and not e.key.startswith("probe/")
                       and e.cpu_time_total == 0), key=lambda x: -x[1])
        json.dump({"steps": [self.start, self.end], "wall_ms_per_step": 1e3 * float(np.mean(self.wall)),
                   "wall_ms_p50": 1e3 * float(np.median(self.wall)), "phases": phases,
                   "top_kernels_ms_per_step": kern[:40],
                   "sum_kernel_ms_per_step": float(sum(k[1] for k in kern))},
                  open(os.path.join(self.out_dir, "profile_phases.json"), "w"), indent=1)
        print(f"[PROBE] profile 寫入 {self.out_dir}/profile_kernels.txt、profile_phases.json")


# ----------------------------------------------------------------------------- attribute log
class AttrLogger:
    """變化量都只對存活點算（alive_init_idx 對回初始索引）。"""

    def __init__(self, trainer, every):
        self.t, self.every = trainer, int(every)
        g = trainer.GS.gaussians
        with torch.no_grad():
            self.init = {k: p.detach().clone() for k, p in self._params(g).items()}
            self.prev = {k: p.detach().clone() for k, p in self._params(g).items()}
            self.prev_idx = torch.arange(len(g._xyz), device=g._xyz.device)
            cams = trainer.scene.getTrainCameras()
            centers = torch.stack([c.camera_center for c in cams]).to(g._xyz.device)
            self.d_cam0 = torch.cdist(g._xyz.detach(), centers).min(1).values      # 到最近訓練相機
            fx = [c.image_width / (2 * np.tan(c.FoVx / 2)) for c in cams]
            self.fx = float(np.mean(fx))
        self.n0 = len(g._xyz)
        self.groups = self._load_groups(trainer.scene.model_path, self.n0)
        self.gnorm = {k: torch.zeros((), device=g._xyz.device) for k in GROUPS}
        self.gcount = 0
        self.log = {"every": self.every, "fx_mean": self.fx, "n_init": self.n0,
                    "groups": {k: int(m.sum()) for k, m in self.groups.items()}, "snapshots": []}

    @staticmethod
    def _params(g):
        return {"xyz": g._xyz, "f_dc": g._features_dc, "f_rest": g._features_rest, "opacity": g._opacity,
                "scaling": g._scaling, "rotation": g._rotation}

    @staticmethod
    def _load_groups(model_path, n):
        dev = "cuda"
        groups = {"all": torch.ones(n, dtype=torch.bool, device=dev)}
        ip = os.path.join(model_path, "input.ply")
        try:
            from plyfile import PlyData
            v = PlyData.read(ip)["vertex"]
            if "label" in v.data.dtype.names and len(v["label"]) == n:
                lab = torch.from_numpy(np.asarray(v["label"]).astype(np.int64)).to(dev)
                groups["static"], groups["dynamic"] = lab == 0, lab == 1
        except Exception:
            pass
        fp = os.path.join(model_path, "roma_init_flat.npy")
        if os.path.exists(fp):
            f = torch.from_numpy(np.load(fp)).to(dev)
            if len(f) == n:
                groups["flat"], groups["nonflat"] = f, ~f
        return groups

    def accumulate_grads(self):
        g = self.t.GS.gaussians
        for k, p in self._params(g).items():
            if p.grad is not None:
                self.gnorm[k] += p.grad.detach().float().norm()
        self.gcount += 1

    @staticmethod
    def _stats(x):
        x = x.float()
        if x.numel() == 0:
            return None
        q = torch.quantile(x[torch.randperm(len(x), device=x.device)[:200000]], torch.tensor([0.5, 0.9], device=x.device))
        return [round(float(x.mean()), 6), round(float(q[0]), 6), round(float(q[1]), 6)]

    def _deltas(self, cur, ref, idx_ref_rows):
        """cur：目前參數（存活點）；ref：參考參數已取對應列。回傳各屬性每點變化量。"""
        dxyz = (cur["xyz"] - ref["xyz"]).norm(dim=1)
        d = {
            "xyz_world": dxyz,
            "xyz_px": dxyz / self.d_cam0[idx_ref_rows].clamp_min(1e-6) * self.fx,       # 約略的畫面位移（px，原解析度訓練圖）
            "xyz_over_scale": dxyz / torch.exp(ref["scaling"]).mean(1).clamp_min(1e-9),
            "log_scale": (cur["scaling"] - ref["scaling"]).abs().mean(1),
            "rot_deg": torch.rad2deg(2 * torch.acos(
                (torch.nn.functional.normalize(cur["rotation"], dim=1) *
                 torch.nn.functional.normalize(ref["rotation"], dim=1)).sum(1).abs().clamp(max=1.0))),
            "opacity": (torch.sigmoid(cur["opacity"]) - torch.sigmoid(ref["opacity"])).abs().squeeze(1),
            "rgb255": ((cur["f_dc"] - ref["f_dc"]) * SH_C0 * 255).reshape(len(dxyz), -1).norm(dim=1),
            "f_rest": (cur["f_rest"] - ref["f_rest"]).reshape(len(dxyz), -1).norm(dim=1),
        }
        return d

    @torch.no_grad()
    def snapshot(self, step, final=False):
        g = self.t.GS.gaussians
        alive = getattr(self.t, "alive_init_idx", None)
        cur = {k: p.detach() for k, p in self._params(g).items()}
        if alive is None or len(alive) != len(g._xyz):
            alive = torch.arange(len(g._xyz), device=g._xyz.device)
        ref0 = {k: v[alive] for k, v in self.init.items()}
        d0 = self._deltas(cur, ref0, alive)
        # vs 上次快照：上次快照存的是 prev_idx 列；找目前存活點在 prev 中的位置
        pos = torch.full((self.n0,), -1, dtype=torch.long, device=alive.device)
        pos[self.prev_idx] = torch.arange(len(self.prev_idx), device=alive.device)
        rows = pos[alive]
        ok = rows >= 0
        refp = {k: v[rows[ok]] for k, v in self.prev.items()}
        dp = self._deltas({k: v[ok] for k, v in cur.items()}, refp, alive[ok])
        snap = {"step": int(step), "n_alive": int(len(alive)),
                "opacity_mean": round(float(torch.sigmoid(cur["opacity"]).mean()), 5),
                "grad_norm_mean": {k: round(float(v) / max(self.gcount, 1), 8) for k, v in self.gnorm.items()},
                "lr": {pg["name"]: pg["lr"] for pg in g.optimizer.param_groups},
                "vs_init": {}, "vs_prev": {}}
        for gname, m in self.groups.items():
            ma = m[alive]
            snap["vs_init"][gname] = {k: self._stats(v[ma]) for k, v in d0.items()}
            snap["vs_prev"][gname] = {k: self._stats(v[ma[ok]]) for k, v in dp.items()}
        self.log["snapshots"].append(snap)
        self.prev = {k: v.clone() for k, v in cur.items()}
        self.prev_idx = alive.clone()
        self.gnorm = {k: torch.zeros_like(v) for k, v in self.gnorm.items()}
        self.gcount = 0
        if final:
            np.savez_compressed(os.path.join(self.t.scene.model_path, "attr_final_delta.npz"),
                                alive=alive.cpu().numpy(), **{k: v.half().cpu().numpy() for k, v in d0.items()})

    def save(self):
        json.dump(self.log, open(os.path.join(self.t.scene.model_path, "attr_log.json"), "w"), indent=1)
        print(f"[PROBE] attr_log.json（{len(self.log['snapshots'])} 個快照）")


# ----------------------------------------------------------------------------- freeze
def apply_freeze(optimizer, names):
    for pg in optimizer.param_groups:
        if pg["name"] in names:
            pg["lr"] = 0.0
