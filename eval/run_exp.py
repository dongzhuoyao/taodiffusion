from pathlib import Path

from loguru import logger
from diffusion_utils.util import make_clean_dir
from dynamic_input.misc import assert_image_dir
from eval.eval_knn import get_knn_eval_dict


from eval.test_exps.common_stuff import get_save_dir, should_vis
from eval.test_exps.main_exp import (
    main_cond_scale_4test,
    main_cond_scale_4val,
)
from eval.test_exps.oracle_related import test_oracle
from pytorch_lightning.utilities import rank_zero_only

import os
import torch
import numpy as np


def run_test_vis(pl_module, fid_kwargs, exp_dict_kwargs):
    vis = pl_module.hparams.vis
    papervis_dir = get_save_dir(pl_module) + "_papervis"
    make_clean_dir(papervis_dir)

    if should_vis(vis, "knn_vis"):
        logger.warning("knn_vissssss...")
        sample_dir = "/home/thu/lab/data-efficiency-diffusion/outputs/v1.3.3.0_unet64c64_afdog389_iter150k_clipmix_n0.01_ddim50/26-10-2022/18-14-13/eval_test_ddim250_s0_sample_rank0"
        logger.warning(f"vis knn with sample_dir={sample_dir}")
        gt_dir_4fid = fid_kwargs["fid_val_image_dir"]
        get_knn_eval_dict(
            sample_dir,
            gt_dir_4fid,
            fid_kwargs,
            knn_k=10,
            q_num=10,
            width=2,
            debug=False,
            batch_size=128,
            papervis=True,
        )


def run_test_and_all_exploration(
    pl_module, wandb_rootdir, log_immediately=False, test_oracle=False
):
    assert_image_dir(pl_module=pl_module)
    log_dict = dict()
    debug = pl_module.hparams.debug
    logger.warning("*" * 100)
    logger.warning(
        f"congrats, run_all_exp start, debug={pl_module.hparams.debug}")
    logger.warning("*" * 100)
    condition_kwargs, sampling_kwargs, fid_kwargs = pl_module.get_default_config()
    _exp = pl_module.hparams.exp

    ######         update some common params #################################
    save_dir = get_save_dir(pl_module)
    current_gt_dir_4fid = fid_kwargs["fid_val_image_dir"]
    if _exp.dir4fid is not None:
        root_global = pl_module.hparams.data.root_global
        _dir4fid = os.path.join(root_global, _exp.dir4fid)
        current_gt_dir_4fid = str(Path(_dir4fid).expanduser().resolve())
        logger.warning(f"override gt_dir4fid, {current_gt_dir_4fid}")
        assert os.path.exists(
            current_gt_dir_4fid), f"{current_gt_dir_4fid} not exist"

    fid_kwargs.update(
        save_dir=save_dir,
        gt_dir_4fid=current_gt_dir_4fid,
        fid_num=pl_module.hparams.data.test_fid_num,
        condition_kwargs=condition_kwargs,
        exp=_exp,
    )

    sampling_kwargs.update(
        dict(
            save_dir=save_dir,
            num_timesteps=pl_module.hparams.model.num_timesteps_test,
            sampling_method=pl_module.hparams.model.sampling_test,
            exp=_exp,
        )
    )
    ######         update some common params #################################
    exp_dict_kwargs = dict(
        pl_module=pl_module,
        condition_kwargs=condition_kwargs,
        sampling_kwargs=sampling_kwargs,
        fid_kwargs=fid_kwargs,
        log_dict=log_dict,
        log_immediately=log_immediately,
        wandb_rootdir=wandb_rootdir,
        debug=debug,
    )

    run_test_vis(
        pl_module=pl_module, fid_kwargs=fid_kwargs, exp_dict_kwargs=exp_dict_kwargs
    )

    ########## main ###############
    if _exp.cond_scale:
        main_cond_scale_4test(**exp_dict_kwargs)
    ########## main ###############

    logger.warning("run_all_exp end..")
    if not log_immediately:
        pl_module.logger.experiment.log(log_dict, commit=True)
    return log_dict


# rank_zero_only
def run_validation(pl_module, wandb_rootdir, val_fid_num, log_immediately=False):

    assert_image_dir(pl_module=pl_module)
    log_dict = dict()

    debug = pl_module.hparams.debug
    if debug:
        val_fid_num = 100
    logger.warning("*" * 66)
    logger.warning(f"run_validation start, debug={debug}")
    condition_kwargs, sampling_kwargs, fid_kwargs = pl_module.get_default_config()

    ######  update some common params #################################
    fid_kwargs.update(
        save_dir=get_save_dir(pl_module),
        gt_dir_4fid=fid_kwargs["fid_train_image_dir"],
        fid_num=val_fid_num,
        condition_kwargs=condition_kwargs,
        exp=pl_module.hparams.exp
    )
    sampling_kwargs.update(
        dict(
            num_timesteps=pl_module.hparams.model.num_timesteps_val,
            sampling_method=pl_module.hparams.model.sampling_val,
        )
    )
    ######  update some common params #################################
    exp_dict_kwargs = dict(
        pl_module=pl_module,
        condition_kwargs=condition_kwargs,
        sampling_kwargs=sampling_kwargs,
        fid_kwargs=fid_kwargs,
        log_dict=log_dict,
        log_immediately=log_immediately,
        wandb_rootdir=wandb_rootdir,
        debug=debug,
    )

    fid_for_ckpt = main_cond_scale_4val(**exp_dict_kwargs)

    if pl_module.trainer.current_epoch == 0:
        logger.warning("oracle mode")
        test_oracle(**exp_dict_kwargs)

    # pl_module.trainer.strategy.broadcast(fid_for_ckpt)
    fid_for_ckpt = torch.tensor(fid_for_ckpt).float().to(pl_module.device)
    pl_module.log("val/fid_for_ckpt", fid_for_ckpt, on_epoch=True)
    logger.warning(f"run_validation end, debug={debug}")
    return fid_for_ckpt
