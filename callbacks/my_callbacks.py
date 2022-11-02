import copy
from pathlib import Path
from loguru import logger

import numpy as np
import torch
import wandb
from einops import rearrange
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import Callback
from dataset.ds_utils.dataset_common_utils import need_to_upsample256
from diffusion_utils.taokit.wandb_utils import wandb_scatter_fig
from eval.test_exps.common_stuff import sampling_cond_str
from diffusion_utils.taokit.vis_utils import upsample_pt


def update_cfg_dict(_kwargs, update_info_dict, cfg_name):
    _kwargs_new = copy.deepcopy(_kwargs)
    _kwargs_new.update(update_info_dict)
    logger.warning("{}: {}".format(cfg_name, _kwargs_new))
    return _kwargs_new


def _get_wandbimg(samples):
    assert isinstance(
        samples, torch.Tensor
    )  # if it is tensor, split along first dimension
    # n_log_step, n_row, C, H, W
    denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
    denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
    wandb_list = [wandb.Image(_i.float()) for _i in denoise_grid]
    return wandb_list


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
    ):
        super().__init__()

        self.batch_frequency = batch_frequency
        self.max_images = max_images

        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        is_train = pl_module.training#important issue!!!
        
        if self.check_frequency(check_idx) and self.max_images > 0:
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                if hasattr(pl_module, "log_images"):
                    batch = pl_module.log_images(batch)
                    self.log_wandb(pl_module, batch=batch)
                else:
                    logger.warning(
                        "log_images function is missing in pl_module, skip the vis.."
                    )

            if is_train:
                pl_module.train()

    def log_sample_and_prog(
        self, log, samples, inter_dict, _key, prog_vis_num, need_to_upsample256=False
    ):
        log[f"{_key}_max"] = samples.max().item()
        log[f"{_key}_min"] = samples.min().item()
        if need_to_upsample256:
            samples = [wandb.Image(upsample_pt(_i).float()) for _i in samples]
        else:
            samples = [wandb.Image(_i.float()) for _i in samples]
        log[_key] = samples

        # pred_x0_unclipped
        if "pred_x0_unclipped_max" in inter_dict:
            prog_pred_x0_unclipped_max = inter_dict["pred_x0_unclipped_max"]
            prog_pred_x0_unclipped_min = inter_dict["pred_x0_unclipped_min"]
            timestep = len(prog_pred_x0_unclipped_max)
            max_list = [prog_pred_x0_unclipped_max[i] for i in range(timestep)]
            min_list = [prog_pred_x0_unclipped_min[i] for i in range(timestep)]
            x_list = list(range(timestep))
            log.update(
                wandb_scatter_fig(
                    x_list=x_list, y_list=max_list, dict_key=f"{_key}_unclipped_max"
                )
            )
            log.update(
                wandb_scatter_fig(
                    x_list=x_list, y_list=min_list, dict_key=f"{_key}_unclipped_min"
                )
            )

        # progressive
        prog = inter_dict["pred_x0"]  # [timestep_vis, b, 3 ,w, h]
        prog = prog[
            :, :prog_vis_num
        ]  # [timestep_vis, b, 3 ,w, h], truncate to progressive_vis_num
        prog_wandblist = _get_wandbimg(prog)
        log[f"{_key}_prog"] = prog_wandblist
        return log

    def log_wandb(self, pl_module, batch, vis_num=16, prog_vis_num=9):
        logdict_wandb = dict()
        vis_num = min(len(batch["image"]), vis_num)
        logdict_wandb["inputs"] = batch["image"][:vis_num].to(
            pl_module.device)  # [B,3,W,H]
        _dataset = pl_module.trainer.datamodule.train_dataloader().dataset
        try:
            captions = [f"{_dataset.label_list[_dataset.targets[_id]]}:{_dataset.targets[_id]}:{_id}" for _id in batch['id'][:vis_num]]
        except:
            captions = ['dump caption'] * vis_num
        logdict_wandb["inputs"] = [wandb.Image(_img.float(), caption=_caption) for _img,_caption in zip(logdict_wandb["inputs"],captions)]
        pl_module.logger.experiment.log({"imagelogger/inputs_vis": logdict_wandb["inputs"]})


        batch = {k:v[:vis_num].to(pl_module.device) for k,v in batch.items()}

        condition_kwargs, sampling_kwargs, fid_kwargs = pl_module.get_default_config()
        ######         update some common params #################################
        fid_kwargs.update(
            save_dir=str(Path(pl_module.trainer.logdir).expanduser().resolve())
        )
        sampling_kwargs.update(
            dict(
                num_timesteps=pl_module.hparams.model.num_timesteps_imagelogger,
                sampling_method=pl_module.hparams.model.sampling_imagelogger,
            )
        )

        with pl_module.ema_scope("Plotting Native Sampling"):
            cond_scale_list = list(set([0, pl_module.hparams.cond_scale]))
            for cond_scale in cond_scale_list:
                condition_cfg_now = update_cfg_dict(
                    condition_kwargs,
                    update_info_dict=dict(cond_scale=cond_scale),
                    cfg_name="condition_kwargs",
                )
                sampling_cfg_now = update_cfg_dict(
                    sampling_kwargs,
                    update_info_dict=dict(return_inter_dict=True),
                    cfg_name="sampling_kwargs",
                )
                samples, inter_dict = pl_module.sampling_progressive(
                    batch_size=len(batch["image"]),
                    batch_data=batch,
                    condition_kwargs=condition_cfg_now,
                    sampling_kwargs=sampling_cfg_now,
                )

                logdict_wandb = self.log_sample_and_prog(
                    log=logdict_wandb,
                    _key=f"native_{sampling_cond_str(sampling_cfg_now,condition_cfg_now)}",
                    samples=samples,
                    inter_dict=inter_dict,
                    prog_vis_num=prog_vis_num,
                    need_to_upsample256=need_to_upsample256(
                        pl_module.hparams.data.name
                    ),
                )

        #########################
        wandb_dict = dict()
        for k, v in logdict_wandb.items():
            if isinstance(
                v, list
            ):  # already constructed a list of wandb.Image, so directly use it here.
                wandb_dict[k] = v
            elif isinstance(v, wandb.Image):
                wandb_dict[k] = v
            else:
                wandb_dict[k] = v

        wandb_dict = {f"imagelogger/{k}": v for k, v in wandb_dict.items()}
        pl_module.logger.experiment.log(wandb_dict)

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_frequency) == 0) and (
            check_idx > 0 or self.log_first_step
        ):
            return True
        else:
            return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")
