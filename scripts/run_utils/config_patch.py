import os
import shutil
import hydra
from hydra.core.hydra_config import HydraConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig
from omegaconf import DictConfig, OmegaConf


from habitat import logger
from habitat.config.read_write import read_write

def patch_exp_name(cfg):
    """
    Patch experiment name to hydra, logging and evaluation ckpt
    """
    # get the config name you read and mark it as the experiment name for logging
    hydra_cfg = HydraConfig.get()
    config_name_with_ext = hydra_cfg.job.config_name
    exp_name = os.path.splitext(config_name_with_ext)[0]
    exp_mode = cfg.experiment_mode

    if 'habitat_baselines' in cfg.keys():
        if cfg.habitat_baselines.evaluate:
            ckpt_source = cfg.habitat_baselines.ckpt_source
        if 'ckpt_no' in cfg.habitat_baselines.keys():
            ckpt_no = cfg.habitat_baselines.ckpt_no
            ckpt_no_flag = True
        else:
            ckpt_no_flag = False

        with read_write(cfg):
            cfg.habitat_baselines.tensorboard_dir = cfg.habitat_baselines.tensorboard_dir.format(exp_name=exp_name, exp_mode=exp_mode)
            cfg.habitat_baselines.video_dir = cfg.habitat_baselines.video_dir.format(exp_name=exp_name, exp_mode=exp_mode)
            cfg.habitat_baselines.checkpoint_folder = cfg.habitat_baselines.checkpoint_folder.format(exp_name=exp_name, exp_mode=exp_mode)
            cfg.habitat_baselines.log_file = cfg.habitat_baselines.log_file.format(exp_name=exp_name, exp_mode=exp_mode)
            cfg.habitat_baselines.wb.group = cfg.habitat_baselines.wb.group.format(exp_name=exp_name, exp_mode=exp_mode)
            if not ckpt_no_flag:
                cfg.habitat_baselines.eval_ckpt_path_dir = cfg.habitat_baselines.eval_ckpt_path_dir.format(exp_name=exp_name,
                                                                                                           exp_mode=exp_mode)
            else:
                if cfg.habitat_baselines.evaluate:
                    cfg.habitat_baselines.eval_ckpt_path_dir = cfg.habitat_baselines.eval_ckpt_path_dir.format(exp_name=exp_name,
                                                                                                               ckpt_source=ckpt_source,
                                                                                                               no=ckpt_no
                                                                                                           )

        # Evaluation phase
        if cfg.habitat_baselines.evaluate:
            with read_write(cfg):
                if 'ckpt_no' in cfg.habitat_baselines.keys():
                    cfg.habitat_baselines.eval_ckpt_path_dir = cfg.habitat_baselines.eval_ckpt_path_dir.format(
                        exp_name=exp_name,
                        exp_mode=exp_mode,
                        no=cfg.habitat_baselines.ckpt_no
                    )
                else:
                    # if ckpt_no is not defined, we assume it is the latest checkpoint
                    cfg.habitat_baselines.ckpt_no = "latest"
                    cfg.habitat_baselines.ckpt_no = cfg.habitat_baselines.ckpt_no.replace("latest", "0")
            logger.info("Evaluation mode enabled. Using checkpoint: {}".format(cfg.habitat_baselines.eval_ckpt_path_dir))

    with read_write(hydra_cfg):
        hydra_cfg.run.dir = hydra_cfg.run.dir.format(exp_name=exp_name, exp_mode=exp_mode)
        hydra_cfg.job.name = hydra_cfg.job.name.format(exp_name=exp_name, exp_mode=exp_mode)
        hydra_cfg.runtime.output_dir = hydra_cfg.runtime.output_dir.format(exp_name=exp_name, exp_mode=exp_mode)
    return cfg

def update_and_save_hydra_config(cfg):
    """
    Update Hydra config variables (e.g., exp_name) and save them in the correct directory.
    """
    hydra_cfg = HydraConfig.get()

    # delete the {exp_name} folder
    # shutil.rmtree(os.path.join(*hydra_cfg.run.dir.split("/")[:-1], '{exp_name}_{exp_mode}'))

    # Ensure the target directory exists
    os.makedirs(hydra_cfg.run.dir, exist_ok=True)

    # Save the updated configurations to the Hydra folder
    hydra_folder = os.path.join(hydra_cfg.run.dir, ".hydra")
    os.makedirs(hydra_folder, exist_ok=True)

    # Save updated configs back to the files
    OmegaConf.save(config=cfg, f=os.path.join(hydra_folder, "config.yaml"))
    OmegaConf.save(config=hydra_cfg,
                   f=os.path.join(hydra_folder, "hydra.yaml"))
    OmegaConf.save(config=OmegaConf.create(hydra_cfg.overrides.task),
                   f=os.path.join(hydra_folder, "overrides.yaml"))
    exp_name = os.path.basename(hydra_cfg.run.dir)
    log_file = os.path.join(hydra_cfg.run.dir, f"{os.path.basename(hydra_cfg.run.dir)}.log")

    logger.add_filehandler(log_file)

    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Hydra configuration updated and saved in {hydra_folder}")