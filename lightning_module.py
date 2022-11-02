import imp
import os
from queue import Queue
import random
from functools import partial
from pathlib import Path
import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from contextlib import contextmanager
from diffusion_utils.taokit.pl_utils import FIDMetrics
from dynamic_input.misc import (
    assert_check,
    assert_image_dir,
    get_default_config,
    log_range,
)
from dynamic_input.condition import (
    prepare_denoise_fn_kwargs_4sharestep,
    prepare_denoise_fn_kwargs_4sampling,
)


from eval.run_exp import run_test_and_all_exploration, run_validation
from diffusion_utils.util import (
    count_params,
    instantiate_from_config,
    tensor_dict_copy,
)
from dynamic.ema import LitEma
from dynamic_input.image import prepare_image

from loguru import logger
from diffusion_utils.taokit.wandb_utils import wandb_scatter_fig
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat, reduce
from lightning_module_common import configure_optimizers, print_best_path


class DiffusionPL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = instantiate_from_config(self.hparams.dynamic).to(
            self.hparams.device
        )
        count_params(self.model, verbose=True)
        if self.hparams.use_ema:
            self.model_ema = LitEma(
                self.model, decay=self.hparams.ema_decay)
            logger.info(
                f"Keeping EMAs{self.hparams.ema_decay} of {len(list(self.model_ema.buffers()))} variables..")

        self.diffusion = instantiate_from_config(self.hparams.diffusion_model)

        self.diffusion.set_denoise_fn(
            self.model.forward, self.model.forward_with_cond_scale
        )

        self.fid_metric = FIDMetrics(prefix='fidmetric_eval_val')
        ##############################
        assert_check(pl_module=self)
        self.min_fid_for_ckpt = 100


    def get_default_config(
        self,
    ):
        condition_kwargs, sampling_kwargs, fid_kwargs = get_default_config(
            pl_module=self
        )
        return condition_kwargs, sampling_kwargs, fid_kwargs

    @contextmanager
    def ema_scope(self, context=None):
        if self.hparams.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.hparams.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    logger.info(f"{context}: Restored training weights")

    @torch.no_grad()
    def log_images(self, batch):
        batch = self.prepare_batch(batch)
        return batch

    def configure_optimizers(self):
        opt = configure_optimizers(pl_module=self)
        return opt

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            assert_image_dir(pl_module=self)
            self.logger.experiment.log(
                self.diffusion.vis_schedule(), commit=False)

    def prepare_batch(
        self,
        batch_data,
    ):
        batch_data = prepare_image(pl_module=self, batch_data=batch_data)
        return batch_data

    @torch.no_grad()
    def sampling_progressive(
        self,
        batch_size,
        batch_data=None,
        sampling_kwargs=None,
        condition_kwargs=None,
        denoise_sample_fn_kwargs=None,
        **kwargs,
    ):
        _shape = (
            batch_size,
            self.hparams.data.channels,
            self.hparams.data.image_size,
            self.hparams.data.image_size,
        )
        if denoise_sample_fn_kwargs is None:
            denoise_sample_fn_kwargs = prepare_denoise_fn_kwargs_4sampling(
                pl_module=self,
                batch_data=batch_data,
                sampling_kwargs=sampling_kwargs,
                cond_scale=condition_kwargs["cond_scale"],
            )
        #############################

        result = self.diffusion.p_sample_loop(
            sampling_method=sampling_kwargs["sampling_method"],
            shape=_shape,
            denoise_sample_fn_kwargs=denoise_sample_fn_kwargs,
            sampling_kwargs=sampling_kwargs,
            condition_kwargs=condition_kwargs,
            **kwargs,
        )
        samples, intermediate_dict = result
        if sampling_kwargs["return_inter_dict"]:
            return samples, intermediate_dict
        else:
            return samples, intermediate_dict["pred_x0"]
            # return samples, intermediate_dict["x_inter"]

    @torch.no_grad()
    def sampling(self, **kwargs):
        final, inter = self.sampling_progressive(**kwargs)
        return final



    def shared_step(self, batch_data):
        batch_data = self.prepare_batch(batch_data)
        log_range(self, batch_data, commit=False)
        batch_data, denoise_fn_kwargs = prepare_denoise_fn_kwargs_4sharestep(
            pl_module=self, batch_data=batch_data
        )
        loss, loss_dict = self.diffusion.forward_tao(
            x=batch_data['image'], **denoise_fn_kwargs
        )
        return loss, loss_dict

    def training_step(self, batch_data, batch_idx):
        loss, loss_dict = self.shared_step(batch_data)
        if batch_idx > 0:  # more elegant ?
            self.iters_per_sec = 1.0 / (time.time() - self.last_time)
            loss_dict.update(dict(iters_per_sec=self.iters_per_sec))
            self.last_time = time.time()

        if batch_idx == 0:
            self.last_time = time.time()
            

       
        loss_dict.update(
            dict(
                global_step=float(self.global_step),
                img_million=float(self.global_step *
                                  len(batch_data["image"]) / 1e6),
            )
        )
        if self.hparams.optim.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            loss_dict.update({"lr_abs": lr})

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )

        return loss

    def training_epoch_end(self, training_step_outputs):
        pass

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            with torch.no_grad():
                assert_image_dir(pl_module=self)
                wandb_rootdir = "eval_val"
                if self.global_step == 0:
                    return  # don't evaluation when very-first batch of your training
                if (self.current_epoch % self.hparams.data.fid_every_n_epoch == 0) or (self.trainer.ckpt_path is not None and not self.ckpt_path_has_run_first_time):
                    if self.current_epoch == 0:
                        #val_fid_num = max(min(int(self.hparams.data.val_fid_num * 0.1), 128),32)
                        val_fid_num = max(
                            int(self.hparams.data.val_fid_num * 0.1), 1001)
                        logger.warning(
                            f"val_fid_num is set to {val_fid_num} for the first epoch")
                    else:
                        val_fid_num = self.hparams.data.val_fid_num

                    if True:
                        self.fid_for_ckpt = run_validation(
                            self,
                            wandb_rootdir=wandb_rootdir,
                            val_fid_num=val_fid_num,
                            log_immediately=True,
                        )
                    else:  # used for debugging multi-gpu training
                        self.fid_for_ckpt = 1.0/self.current_epoch
                    
                if self.fid_for_ckpt < self.min_fid_for_ckpt:
                    self.min_fid_for_ckpt = self.fid_for_ckpt
                self.log("val/fid_for_ckpt", self.fid_for_ckpt, on_epoch=True)
                self.log("val/min_fid_for_ckpt",
                         self.min_fid_for_ckpt, on_epoch=True)
                if False:
                    tb_metrics = {
                        **self.fid_metric.compute(self.fid_for_ckpt),
                    }
                    self.log_dict(tb_metrics)
                print_best_path(self)

        _, loss_dict_no_ema = self.shared_step(
            tensor_dict_copy(batch))
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(
                tensor_dict_copy(batch))
            loss_dict_ema = {
                key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}

        self.log_dict(
            loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
        self.log_dict(
            loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )

    @torch.no_grad()
    def validation_epoch_end(
        self,
        val_step_outputs,
    ):
        self.fid_metric.reset()

    def on_train_batch_end(self, *args, **kwargs):
        if self.hparams.use_ema:
            self.model_ema(self.model)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            if not self.hparams.profile:
                assert_image_dir(pl_module=self)
                run_test_and_all_exploration(
                    self, wandb_rootdir="eval_test", log_immediately=True
                )
