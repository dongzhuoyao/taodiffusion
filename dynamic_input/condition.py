import torch
import numpy as np

from test_feat_interrupt import feat_interrupt


def do_labelmix(batch_data, device):
    b, c, w, h = batch_data['image'].shape
    cond = batch_data["label_onehot"].to(device)
    b, cls_num = cond.shape
    cond_null = torch.zeros((b, cls_num)).to(cond.dtype).to(
        device)  # fixed to null condition as zero vector
    cond_real = cond.clone()
    cond_mixed = cond_null * 0.5 + cond_real * 0.5

    batch_image = batch_data["image"].to(device)
    batch_image = batch_image.repeat(3, 1, 1, 1).to(device)

    batch_cond = torch.cat([cond_mixed, cond_null, cond_real], 0).to(device)
    return batch_image, batch_cond


def prepare_condition_kwargs(pl_module, batch_data):
    cond_dim = pl_module.hparams.cond_dim
    condition_method = pl_module.hparams.condition_method

    ######################################################
    cond_drop_prob = pl_module.hparams.cond_drop_prob

    if condition_method == "label":
        mask = None
        cond = batch_data["label_onehot"].to(pl_module.device)

    elif condition_method == "labelmix":
        mask = None
        if pl_module.training:
            _images, _conds = do_labelmix(
                batch_data, device=pl_module.device)
            batch_data['image'] = _images
            cond = _conds
        else:
            mask = None
            cond = batch_data["label_onehot"].to(pl_module.device)
            batch_data['image'] = batch_data['image']

    elif condition_method is None:
        cond = None
        mask = None
    else:
        raise NotImplementedError

    result_dict = dict(cond_drop_prob=cond_drop_prob, cond=cond, mask=mask)
    return batch_data, result_dict


def prepare_denoise_fn_kwargs_4sharestep(pl_module, batch_data):
    batch_data, denoise_fn_kwargs = prepare_condition_kwargs(
        pl_module=pl_module, batch_data=batch_data
    )
    return batch_data, denoise_fn_kwargs


def prepare_denoise_fn_kwargs_4sampling(
    pl_module, batch_data, sampling_kwargs, cond_scale
):

    batch_data, denoise_sample_fn_kwargs = prepare_denoise_fn_kwargs_4sharestep(
        pl_module, batch_data
    )
    denoise_sample_fn_kwargs.update(dict(cond_scale=cond_scale))  # override
    # don't need it in testing mode. remove it to avoid bug
    denoise_sample_fn_kwargs.pop("cond_drop_prob")

    # ignore batch_data
    return denoise_sample_fn_kwargs
