from ast import Not
import math
import shutil

import torch
import wandb

import numpy as np
import os
from cleanfid import fid as clean_fid
from tqdm import tqdm
from loguru import logger
import torch_fidelity
from diffusion_utils.util import (
    batch_to_same_firstimage,
    batch_to_samecondition,
    batch_to_samecondition_v2,
    clip_unnormalize_to_zero_to_255,
    make_clean_dir,
)

from eval.compute_pdrc_from_icgan import compute_prdc


from eval.papervis_utils import (
    draw_grid_chainvis,
    draw_grid_img,
    draw_grid_imgsize32_interp,
)
from eval.test_exps.common_stuff import img_pil_save, should_exp


def get_samecondition_num(dataset_name):
    if dataset_name in ['cifar10', 'cifar100']:
        samecondition_num = 18
    else:
        samecondition_num = 9
    return samecondition_num


def get_makegrid_padding(dataset_name):
    if dataset_name in ['cifar10', 'cifar100']:
        _padding = 1
    else:
        _padding = 5
    return _padding


def cleanfid_compute_fid_return_feat(
    fdir1,
    fdir2,
    mode="clean",
    num_workers=0,
    batch_size=8,
    device=torch.device("cuda:0"),
    verbose=True,
    custom_image_tranform=None,
):

    feat_model = clean_fid.build_feature_extractor(mode, device)

    # get all inception features for the first folder
    fbname1 = os.path.basename(fdir1)
    np_feats1 = clean_fid.get_folder_features(
        fdir1,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname1} : ",
        verbose=verbose,
        custom_image_tranform=custom_image_tranform,
    )
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = clean_fid.get_folder_features(
        fdir2,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname2} : ",
        verbose=verbose,
        custom_image_tranform=custom_image_tranform,
    )
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = clean_fid.frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid, np_feats1, np_feats2


def cycle(
    dl,
):  # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/8c3609a6e3c216264e110c2019e61c83dafad9f5/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L34
    while True:
        for data in dl:
            yield data


def get_torch_fidelity_dict(gt_dir, sample_dir, dataset_name, debug):
    tf_new_dict = dict()
    if not debug:
        use_kid = False if debug else True
        if dataset_name in ['obama100', 'afcat', 'afdog', 'panda100', 'grumpycat100']:
            use_kid = False

        tf_metrics_dict = torch_fidelity.calculate_metrics(
            input1=gt_dir,
            input2=sample_dir,
            cuda=True,
            isc=False,
            fid=True,
            ppl=False,
            kid=use_kid,  # when on small dataset like FFHQ, the kid is not convenient for kid
            # kid=False,
            verbose=False,
        )
        tf_new_dict["fid_tf"] = tf_metrics_dict["frechet_inception_distance"]
        if use_kid:
            tf_new_dict['kid_tf'] = tf_metrics_dict['kernel_inception_distance_mean']
            tf_new_dict['kid_std_tf'] = tf_metrics_dict['kernel_inception_distance_std']

        for isc_splits in [1, 10]:
            tf_metrics_dict = torch_fidelity.calculate_metrics(
                input1=sample_dir,
                cuda=True,
                isc=True,
                isc_splits=isc_splits,
                verbose=False,
            )
            tf_new_dict[f"is_tf_s{isc_splits}"] = tf_metrics_dict[
                "inception_score_mean"
            ]
            tf_new_dict[f"is_std_tf_s{isc_splits}"] = tf_metrics_dict[
                "inception_score_std"
            ]
    return tf_new_dict


def eval_fid_callback_before(ds_name, condition_method, vis, _exp, data_dict):
    from .test_exps.common_stuff import should_vis

    samecondition_num = get_samecondition_num(ds_name)

    if should_vis(vis, "random"):
        logger.warning("vis random samples")

    if should_vis(vis, "samecondition"):

        data_dict = batch_to_samecondition(
            data_dict, samecondition_num=samecondition_num
        )
        if ds_name in ['in32, in64']:
            logger.info(
                "samecondition, clustering_id", data_dict["cluster"][0].argmax(
                    0)
            )

    return data_dict


def eval_fid_callback_after(ds_name, condition_method, vis, papervis_dir, gen_samples, pred_x0, batch_id, data_dict):
    from .test_exps.common_stuff import should_vis

    _padding = get_makegrid_padding(ds_name)
    samecondition_num = get_samecondition_num(ds_name)
    _prefix = f"{ds_name}_{condition_method}"

    if should_vis(vis, "random"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_random_uncurated_{batch_id}.png")
        _nrow = 16 if "32" in ds_name else 9
        draw_grid_img(tensorlist=gen_samples,  dataset_name=ds_name, nrow=_nrow, padding=_padding,
                      ncol=_nrow, save_path=papervis_save_path)

    if should_vis(vis, "chainvis"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_chainvis_{batch_id}.png")
        draw_grid_chainvis(
            pred_x0=pred_x0,  # pred_x0: [timestep, batch, c, w, h]
            save_path=papervis_save_path,
            padding=_padding
        )
        logger.info("stego_chainvis .....")

    if should_vis(vis, "samecondition"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_samecondition_{batch_id}.png")
        draw_grid_img(
            dataset_name=ds_name,
            tensorlist=gen_samples,
            nrow=samecondition_num,
            ncol=len(gen_samples)//samecondition_num,
            save_path=papervis_save_path, padding=_padding
        )

    if should_vis(vis, "interp"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_interp_{batch_id}.png")
        draw_grid_imgsize32_interp(
            tensorlist=gen_samples,
            save_path=papervis_save_path,
            nrow=vis.interp_c.n,
            ncol=vis.interp_c.samples,
            padding=_padding
        )


def get_fid_dict(sample_dir, gt_dir, dataset_name, debug, nearest_k=5):
    cleanfid_dict = dict()
    logger.warning("begin calculating FID between two image folders...")

    cleanfid_dict.update(
        get_torch_fidelity_dict(
            gt_dir=gt_dir, sample_dir=sample_dir, dataset_name=dataset_name, debug=debug)
    )
    clean_fid_raw, feat_sample, feat_real = cleanfid_compute_fid_return_feat(
        fdir1=sample_dir, fdir2=gt_dir, num_workers=0, batch_size=8, device=torch.device("cuda:0"),
    )
    cleanfid_dict.update(clean_fid_raw=clean_fid_raw)

    #########################
    num_pr_images = min(
        len(feat_real), len(feat_sample), 5000
    )  # min to be more robust,most case should be 5000
    logger.warning(f"Subsampling {num_pr_images} samples for prdc metrics!")
    idxs_selected_real = np.random.choice(
        range(len(feat_real)), num_pr_images, replace=False
    )
    idxs_selected_sample = np.random.choice(
        range(len(feat_sample)), num_pr_images, replace=False
    )
    prdc_metrics = compute_prdc(
        real_features=feat_real[idxs_selected_real],
        fake_features=feat_sample[idxs_selected_sample],
        nearest_k=nearest_k,
    )
    cleanfid_dict.update(prdc_metrics)

    logger.warning("#" * 66)
    logger.warning(f"{cleanfid_dict}")
    logger.warning("#" * 66)
    logger.warning("finish calculating FID between two image folders...")
    return cleanfid_dict, clean_fid_raw


@torch.no_grad()
def eval_fid(
    sample_fn,
    prepare_batch_fn,
    fid_kwargs,
    prefix="",
    debug=False,
    sample_bs=None,
    harddrive_vis_num=100,
):
    logger.warning("begin eval_fid")
    dl_sample = fid_kwargs["dl_sample"]
    fid_num, dataset_name = fid_kwargs["fid_num"], fid_kwargs["dataset_name"]
    sample_dir, save_dir = fid_kwargs["sample_dir"], fid_kwargs["save_dir"]
    gt_dir_4fid, fid_debug_dir = fid_kwargs["gt_dir_4fid"], fid_kwargs["fid_debug_dir"]
    condition_method = fid_kwargs['condition_kwargs']['condition_method']
    vis = fid_kwargs["vis"]
    _exp = fid_kwargs["exp"]
    assert os.path.exists(gt_dir_4fid), f"{gt_dir_4fid} not exists"
    if debug:
        gt_dir_4fid = fid_debug_dir
        logger.warning(f"set debug gt_dir {gt_dir_4fid}")

    if len(prefix) > 0:
        sample_dir = prefix.replace("/", "_") + "_" + sample_dir
    sample_dir = os.path.join(save_dir, sample_dir)

    make_clean_dir(sample_dir)
    papervis_dir = sample_dir + "_papervis"
    make_clean_dir(papervis_dir)

    ####################################

    global_id_sample = 0
    # sampling
    if sample_bs is None:
        sample_bs = dl_sample.batch_size
    logger.warning(f"sample_bs is {sample_bs}")
    sample_iter = cycle(dl_sample)
    for batch_id in tqdm(
        range(math.ceil(fid_num / sample_bs)),
        desc=f"{prefix}: sampling images for fid",
    ):
        sample_data_dict = next(sample_iter)

        sample_data_dict = prepare_batch_fn(
            sample_data_dict)

        ####################################
        sample_data_dict = eval_fid_callback_before(ds_name=dataset_name, condition_method=condition_method,
                                                    vis=vis, _exp=_exp, data_dict=sample_data_dict
                                                    )
        ####################################

        gen_samples, pred_x0 = sample_fn(
            sample_data_dict,
            batch_id=batch_id,
            batch_size=sample_bs,
            prefix=prefix,
        )

        eval_fid_callback_after(ds_name=dataset_name, condition_method=condition_method,
                                vis=vis,
                                papervis_dir=papervis_dir,
                                gen_samples=gen_samples,
                                pred_x0=pred_x0,
                                batch_id=batch_id,
                                data_dict=sample_data_dict,
                                )
        ####################################

        for _id, _sample in enumerate(gen_samples):
            img_pil_save(
                _sample, os.path.join(
                    sample_dir, f"img{global_id_sample}.png")
            )
            global_id_sample += 1

    result_dict = dict(
        imgs_sample=len(os.listdir(sample_dir)),
        sample_min=gen_samples.float().min().item(),
        sample_max=gen_samples.float().max().item(),
        sample_mean=gen_samples.float().mean().item(),
    )

    fid_dict, fid_for_ckpt = get_fid_dict(
        sample_dir=sample_dir,
        gt_dir=gt_dir_4fid,
        dataset_name=dataset_name,
        debug=debug,
    )
    result_dict.update(fid_dict)

    fid_sample_harddrive = [
        wandb.Image(os.path.join(sample_dir, file_name))
        for file_name in os.listdir(sample_dir)[:harddrive_vis_num]
    ]
    fid_gt_harddrive = [
        wandb.Image(os.path.join(gt_dir_4fid, file_name))
        for file_name in os.listdir(gt_dir_4fid)[:harddrive_vis_num]
    ]
    result_dict.update(
        dict(
            fid_sample_harddrive=fid_sample_harddrive, fid_gt_harddrive=fid_gt_harddrive
        )
    )

    # delete_dir(gt_dir)
    # delete_dir(sample_dir)
    logger.warning("end eval_fid")

    return result_dict, fid_for_ckpt, sample_dir, gt_dir_4fid


if __name__ == "__main__":
    # https://github.com/GaParmar/clean-fid
    # https://github.com/toshas/torch-fidelity

    class dummyDL(object):
        def __init__(
            self,
        ):
            pass

        def __len__(self):
            return int(1e4)

        def __getitem__(self, idx):
            return dict(video=torch.zeros((3, 32, 32)))

    def sample_func(bs):
        return torch.zeros((bs, 3, 256, 256))

    def decode_func(bs):
        return torch.zeros((bs, 3, 256, 256)).float()

    dl = torch.utils.data.DataLoader(
        dummyDL(),
        batch_size=16,
        num_workers=0,
    )

    fid, _, sample_dir, gt_dir = eval_fid(sample_fn=sample_func, fid_num=50)

    print(fid)
