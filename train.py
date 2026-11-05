import os
from source.trainer import EDGSTrainer
from source.utils_aux import set_seed
import omegaconf
import wandb
import hydra
from argparse import Namespace
from omegaconf import OmegaConf


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    _ = wandb.init(entity=cfg.wandb.entity,
                   project=cfg.wandb.project,
                   config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                   tags=[cfg.wandb.tag], 
                   name = cfg.wandb.name,
                   mode = cfg.wandb.mode)
    omegaconf.OmegaConf.resolve(cfg)
    set_seed(cfg.seed)

    # Init output folder
    print("Output folder: {}".format(cfg.gs.dataset.model_path))
    os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)
    with open(os.path.join(cfg.gs.dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        params = {
                "sh_degree": 3,
                "source_path": cfg.gs.dataset.source_path,
                "model_path": cfg.gs.dataset.model_path,
                "images": cfg.gs.dataset.images,
                "depths": "",
                "resolution": cfg.gs.dataset.resolution if "resolution" in cfg.gs.dataset else -1,
                "_white_background": cfg.gs.dataset.white_background,
                "train_test_exp": False,
                "data_device": cfg.gs.dataset.data_device,
                "eval": False,
                "convert_SHs_python": False,
                "compute_cov3D_python": False,
                "debug": False,
                "antialiasing": False,
                "ply_path": cfg.gs.dataset.ply_path if "ply_path" in cfg.gs.dataset else ""   
                    }
        cfg_log_f.write(str(Namespace(**params)))

    # Forward test_views if set in top-level config
    if "test_views" in cfg and cfg.test_views is not None:
        cfg.gs.test_views = cfg.test_views

    # Init both agents
    gs = hydra.utils.instantiate(cfg.gs) 

    # Init trainer and launch training
    trainer = EDGSTrainer(GS=gs,
        training_config=cfg.gs.opt,
        device=cfg.device,
        log_wandb=(cfg.wandb.mode != "disabled"))
    
    trainer.load_checkpoints(cfg.load)
    trainer.timer.start()

    import time, json
    t0 = time.time()
    trainer.init_with_corr(cfg.init_wC)
    init_time = time.time() - t0
    print("[TIMING] RoMa Initialization Time: {:.2f}s ({:.2f} min)".format(init_time, init_time/60.0))

    t1 = time.time()
    trainer.train(cfg.train)
    train_time = time.time() - t1
    print("[TIMING] 3DGS Fine-tune Time: {:.2f}s ({:.2f} min)".format(train_time, train_time/60.0))

    timing_info = {
        "init_time_sec": init_time,
        "train_time_sec": train_time,
        "total_time_sec": init_time + train_time,
        "init_splats": getattr(trainer, "init_splats", len(trainer.GS.gaussians._xyz)),
        "final_splats": len(trainer.GS.gaussians._xyz)
    }
    with open(os.path.join(cfg.gs.dataset.model_path, "timing_summary.json"), "w") as f:
        json.dump(timing_info, f, indent=2)
    print("[TIMING] Summary saved to timing_summary.json: Total: {:.2f}s".format(init_time + train_time))
    
    # All done
    wandb.finish()
    print("\nTraining complete.")

if __name__ == "__main__":
    main()

