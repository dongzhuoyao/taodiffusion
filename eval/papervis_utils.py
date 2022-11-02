import numpy as np
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from loguru import logger
import torch
from einops import rearrange
import seaborn as sns
import matplotlib.pyplot as plt
from diffusion_utils.taokit.interp_util import slerp_batch_torch
from diffusion_utils.taokit.vis_utils import upsample_pt

import distinctipy
from diffusion_utils.util import (
    clip_unnormalize_to_zero_to_255,
)
from loguru import logger
from dataset.ds_utils.dataset_common_utils import need_to_upsample256


# https://pythonbasics.org/seaborn-distplot/

#
colors = distinctipy.get_colors(200, rng=666)
colors255 = [tuple([int(c * 255) for c in color]) for color in colors]


def upsample_pair(image, mask, origin_img=None, up_size=256):
    image_new = upsample_pt(image, up_size=up_size, mode='bilinear').cpu()
    mask_new = upsample_pt(mask, up_size=up_size,
                           mode='nearest').to(torch.bool)
    origin_img_new = upsample_pt(
        origin_img, up_size=up_size, mode='bilinear').cpu() if origin_img is not None else None

    return image_new, mask_new, origin_img_new


def extract_bboxes(mask):
    # https://github.com/multimodallearning/pytorch-mask-rcnn/blob/809abba590db89779ac02c42286135f18ea08b53/utils.py#L25
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        #boxes[i] = np.array([y1, x1, y2, x2])
        boxes[i] = np.array([x1, y1, x2, y2])

    return boxes.astype(np.int32)


def extract_bboxes_wrapper(mask):
    mask = rearrange(mask, "c h w -> h w c")
    mask = mask.cpu().numpy()
    boxes = extract_bboxes(mask)
    boxes = torch.from_numpy(boxes)
    return boxes


def move_lostmask(mask, _dir_x, _dir_y):
    C, H, W = mask.shape
    assert H == W
    mask_new = torch.zeros_like(mask)
    boxes = extract_bboxes_wrapper(mask)[0]
    _direction = torch.tensor([_dir_x, _dir_y, _dir_x, _dir_y])
    boxes += _direction
    boxes = boxes.clamp(0, H - 1)
    mask_new[boxes[1]:boxes[3], boxes[0]:boxes[2]] = 1
    return mask_new


def cluster_hist_vis_fn(data, save_path="outputs/output_vis/cluster_hist_vis.png"):
    sns.set(rc={"figure.figsize": (8, 4)})
    ax = sns.displot(data, binwidth=3, bins=100)
    # ax.set_xlabel('image number per cluster')
    # ax.set(xlabel='image number per cluster')  # , ylabel="Y-Axis")
    plt.xlabel("image number per cluster")
    plt.savefig(save_path)
    plt.close()


def draw_grid_imgsize32_clustervis(
    tensorlist, save_path, nrow=16, ncol=16, padding=2, pad_value=255.0
):
    logger.warning(f"draw_grid, {save_path}")
    tensorlist = tensorlist[: nrow * ncol]
    grid = make_grid(tensorlist, nrow=nrow,
                     padding=padding, pad_value=pad_value)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(save_path)


def draw_grid_img(
    tensorlist, dataset_name, save_path, nrow=7, ncol=7, padding=2, pad_value=255.0
):

    logger.warning(f"draw_grid, {save_path}")
    tensorlist = tensorlist[: nrow * ncol]
    assert len(tensorlist) == nrow * \
        ncol, f"{len(tensorlist)} != {nrow * ncol}"

    if need_to_upsample256(dataset_name):
        tensorlist = [upsample_pt(x, up_size=256) for x in tensorlist]

    grid = make_grid(tensorlist, nrow=nrow,
                     padding=padding, pad_value=pad_value)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(save_path)


def draw_grid_8_8(tensorlist, save_path, nrow=8, ncol=8, padding=2, pad_value=255.0):
    logger.warning(f"draw_grid, {save_path}")
    tensorlist = tensorlist[: nrow * ncol]
    assert len(tensorlist) == nrow * ncol
    grid = make_grid(tensorlist, nrow=nrow,
                     padding=padding, pad_value=pad_value)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(save_path)


def draw_grid_chainvis(pred_x0, save_path, padding=2, pad_value=255.0):
    # pred_x0: [timestep, batch, c, w, h]
    timestep, bs, c, w, h = pred_x0.shape
    pred_x0 = rearrange(pred_x0, "t b c w h-> (b t) c w h")
    logger.warning(f"draw_grid_condscale, {save_path}")

    grid = make_grid(pred_x0, nrow=timestep,
                     padding=padding, pad_value=pad_value)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(save_path)


def draw_grid_condscale(tensorlist, save_path, samples, padding=2, pad_value=255.0):
    print(f"draw_grid_condscale, {save_path}")
    len(tensorlist) % samples == 0
    nrow = len(tensorlist) // samples
    grid = make_grid(tensorlist, nrow=nrow,
                     padding=padding, pad_value=pad_value)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(save_path)


def draw_grid_imgsize32_interp(
    tensorlist, save_path, nrow=16, ncol=16, padding=2, pad_value=255.0
):
    print(f"draw_grid, {save_path}")
    tensorlist = tensorlist[:nrow*ncol]
    assert len(tensorlist) == nrow * \
        ncol, f"{len(tensorlist)} != {nrow * ncol}"
    grid = make_grid(tensorlist, nrow=nrow,
                     padding=padding, pad_value=pad_value)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(save_path)


def batch_to_conditioninterp_papervis(
    cond_tensor, interp_num=9, samples=10, is_slerp=True
):
    batch_size = len(cond_tensor)  # [bs, C] for cond
    if batch_size < interp_num:
        logger.warning(f"only can log interp {batch_size}")
        interp_num = batch_size

    cond_interped_list = []
    for i in range(samples):
        cond1 = rearrange(cond_tensor[i], "c -> 1 c")
        cond2 = rearrange(cond_tensor[i + 1], "c -> 1 c")

        if not is_slerp:
            lin_w = (
                torch.linspace(0, 1, interp_num).reshape(-1,
                                                         1).to(cond_tensor.device)
            )
            feat_interped = cond1 * lin_w + cond2 * (1 - lin_w)  # [N, c]
        else:
            lin_w = (
                torch.linspace(0, 1, interp_num)
                .reshape(
                    -1,
                )
                .to(cond1.device)
            )
            feat_interped = slerp_batch_torch(lin_w, cond1, cond2)

        cond_interped_list.append(rearrange(feat_interped, "k c -> 1 k c"))

    result = rearrange(torch.cat(cond_interped_list, 0), "k b c-> (k b) c")
    return result
