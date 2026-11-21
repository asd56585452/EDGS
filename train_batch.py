"""一個 process 依序跑多個 train.py 工作（同一個 RoMa 模型只載入一次），輸出與逐一執行 train.py 相同。

用法（edgs 容器內，/workspace）：
  python train_batch.py <jobs.json> <timing.json>
jobs.json = [{"overrides": ["gs.dataset.source_path=...", ...], "log": "/path/train.log"}, ...]

作法：
  - 每個工作原樣呼叫 train.main()（hydra，sys.argv = overrides），stdout/stderr（fd 層級）導到該工作的 log。
  - source.corr_init.roma_outdoor / roma_indoor 換成快取版：第一次照常建立模型，並記下建立前後的 RNG 狀態
    （torch CPU / CUDA、numpy、random）。之後的工作檢查建立前的 RNG 狀態與第一次相同，再把 RNG 設成
    「建立後」的狀態，讓後續取樣與原本每次重建模型時完全一致（原本建立模型會消耗 RNG：層的隨機初始化）。
  - 計時（同 FreeTimeGS++ run_scene.py 的口徑：整個 process 牆鐘，含 Python 啟動後的 import 與模型載入）：
    process 總時間、每個工作時間；寫到 timing.json。
"""
import time

T_START = time.perf_counter()
import json, os, random, sys

import numpy as np
import torch

JOBS, TIMING = sys.argv[1], sys.argv[2]
jobs = json.load(open(JOBS))

import train  # noqa: E402  （先 import：source.trainer 會設定 utils 等路徑；train.main 由 hydra 包裝）
import source.corr_init as ci
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir


def _rng():
    return (torch.get_rng_state(), [s.clone() for s in torch.cuda.get_rng_state_all()], np.random.get_state(), random.getstate())


def _set_rng(st):
    torch.set_rng_state(st[0]); torch.cuda.set_rng_state_all(st[1]); np.random.set_state(st[2]); random.setstate(st[3])


def _eq(a, b):
    return (torch.equal(a[0], b[0]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))
            and a[2][0] == b[2][0] and np.array_equal(a[2][1], b[2][1]) and a[2][2:] == b[2][2:] and a[3] == b[3])


_cache, rng_checks = {}, []


def _cached(name, orig):
    def f(*args, **kw):
        key = (name, repr(args), repr(sorted(kw.items())))
        before = _rng()
        if key not in _cache:
            t = time.perf_counter(); m = orig(*args, **kw)
            _cache[key] = (m, before, _rng(), time.perf_counter() - t)
            rng_checks.append({"model": name, "created": True, "load_sec": _cache[key][3]})
        else:
            m, b0, a0, _ = _cache[key]
            ok = _eq(before, b0); rng_checks.append({"model": name, "created": False, "rng_before_same": ok})
            if not ok:
                raise RuntimeError("RNG 狀態與第一次建立模型前不同，無法保證輸出一致")
            _set_rng(a0)
        return m
    return f


ci.roma_outdoor = _cached("roma_outdoor", ci.roma_outdoor)
ci.roma_indoor = _cached("roma_indoor", ci.roma_indoor)


t_ready = time.perf_counter() - T_START
per = []
for j in jobs:
    GlobalHydra.instance().clear()
    sys.argv = ["train.py"] + j["overrides"]
    # hydra.main 在被 import 時找不到 configs；改用 compose API 組出同一份 cfg，呼叫 train.main 的原函式（__wrapped__）
    with initialize_config_dir(config_dir=os.path.join(os.path.dirname(os.path.abspath(train.__file__)), "configs"), version_base="1.2"):
        cfg = compose(config_name="train", overrides=j["overrides"])
    os.makedirs(os.path.dirname(j["log"]), exist_ok=True)
    sys.stdout.flush(); sys.stderr.flush()
    fd_o, fd_e = os.dup(1), os.dup(2)
    with open(j["log"], "w") as lf:
        os.dup2(lf.fileno(), 1); os.dup2(lf.fileno(), 2)
        t = time.perf_counter(); err = None
        try:
            train.main.__wrapped__(cfg)
        except SystemExit as e:
            err = None if not e.code else f"SystemExit {e.code}"
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); err = repr(e)
        sys.stdout.flush(); sys.stderr.flush()
        dt = time.perf_counter() - t
        os.dup2(fd_o, 1); os.dup2(fd_e, 2)
    os.close(fd_o); os.close(fd_e)
    per.append({"log": j["log"], "sec": dt, "error": err})
    print(f"[train_batch] {len(per)}/{len(jobs)} {dt:.1f}s {'OK' if err is None else err} → {j['log']}", flush=True)
    if err is not None:
        break
total = time.perf_counter() - T_START
json.dump({"process_sec": total, "startup_import_sec": t_ready, "jobs": per, "model_cache": rng_checks}, open(TIMING, "w"), indent=1)
print(f"[train_batch] process {total:.1f}s（import {t_ready:.1f}s）", flush=True)
sys.exit(0 if all(p["error"] is None for p in per) else 1)
