from eval.test_exps.common_stuff import (
    get_condition_scale_list,
    get_condition_scale_main,
    sample_and_get_fid,
    sampling_cond_str,
    update_cfg_and_print,
)
from loguru import logger


def run_cond_scale_single(
    pl_module,
    condition_kwargs,
    sampling_kwargs,
    fid_kwargs,
    log_dict,
    log_immediately,
    wandb_rootdir,
    debug,
    cond_scale,
    vis_knn=False,
):
    condition_cfg_now = update_cfg_and_print(
        condition_kwargs, dict(cond_scale=cond_scale), cfg_name="condition_kwargs"
    )
    sampling_cfg_now = update_cfg_and_print(
        sampling_kwargs, dict(), cfg_name="sampling_kwargs"
    )
    fid_cfg_now = update_cfg_and_print(  # always sampling from train_dataloader for fid
        fid_kwargs,
        dict(
            dl_sample=pl_module.trainer.datamodule.train_dataloader(),
            # dl_sample=pl_module.trainer.datamodule.val_dataloader(),  # hutao
            vis_knn=vis_knn,
        ),
        cfg_name="fid_kwargs",
    )
    #logger.warning('use val_dataloader for fid')
    eval_dict, fid_for_ckpt = sample_and_get_fid(
        pl_module=pl_module,
        prefix=f"{wandb_rootdir}/{sampling_cond_str(sampling_cfg_now,condition_cfg_now)}",
        condition_cfg=condition_cfg_now,
        fid_cfg=fid_cfg_now,
        sampling_cfg=sampling_cfg_now,
        debug=debug,
    )
    log_dict.update(eval_dict)

    if log_immediately:
        pl_module.logger.experiment.log(eval_dict, commit=True)
    return fid_for_ckpt


def main_cond_scale_4val(pl_module, **kwargs):
    fid_for_ckpt = run_cond_scale_single(
        cond_scale=get_condition_scale_main(pl_module),
        pl_module=pl_module,
        vis_knn=False,
        **kwargs,
    )
    return fid_for_ckpt


def main_cond_scale_4test(pl_module, **kwargs):

    cond_scale_list = get_condition_scale_list(pl_module)

    for idx, cond_scale in enumerate(cond_scale_list):
        fid_for_ckpt = run_cond_scale_single(
            cond_scale=cond_scale, pl_module=pl_module, vis_knn=False, **kwargs
        )
    return fid_for_ckpt
