# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import random
import torch
import torch.nn.functional as F


def DiffAugmentWrapper(x, prob, operation_str='', channels_first=True):
    batch_size = len(x)
    if random.random() < prob:
        x = DiffAugment(x, operation_str, channels_first)
        cond_code = torch.ones((batch_size, 1), dtype=torch.float)
    else:
        x, cond_code = x, torch.zeros((batch_size, 1), dtype=torch.float)

    return x, cond_code


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)

        for p in policy.split('_'):
            for f in AUGMENT_FNS[p]:
                x = f(x)

        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1,
                                   dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1,
                                   dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio +
                           0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1,
                                  size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1,
                                  size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device), indexing='ij'
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[
        grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(
        2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(
        3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),  indexing='ij'
    )
    grid_x = torch.clamp(grid_x + offset_x -
                         cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y -
                         cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3),
                      dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_translation_tao(x, ratio=0.125):
    b, c, w, h = x.shape
    assert b == 1
    shift_x, shift_y = int(w * ratio +
                           0.5), int(h * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1,
                                  size=[1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1,
                                  size=[1], device=x.device)

    x_new = torch.zeros(
        (b, c, w+2*shift_x, h+2*shift_y), device=x.device)
    x_new[:, :, shift_x + translation_x:shift_x + translation_x +
          w, shift_y + translation_y:shift_y + translation_y+h] = x
    x_new = x_new[:, :, shift_x:shift_x+w, shift_y:shift_y+h]
    return x_new


def rand_cutout_tao(x, ratio=0.5):
    b, c, w, h = x.shape
    assert b == 1

    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(
        2) + (1 - cutout_size[0] % 2), size=[1], device=x.device)
    offset_y = torch.randint(0, x.size(
        3) + (1 - cutout_size[1] % 2), size=[1], device=x.device)

    x[:, :, offset_x:offset_x+cutout_size[0],
        offset_y:offset_y+cutout_size[1]] = 0
    return x


AUGMENT_FNS = {
    # usually not so useful
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
