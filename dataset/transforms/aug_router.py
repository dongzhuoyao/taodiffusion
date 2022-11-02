import math
from einops import rearrange, repeat
from dataset.transforms.auglevel_augment import auglevel_transform_batch
from dataset.transforms.diffaugment_pytorch import DiffAugmentWrapper
import torch


def gridmask(img, mask_ratio, prob, patch_size=8):

    i_c, i_w, i_h = img.shape
    img_patches = rearrange(
        img, 'c (h p1) (w p2) -> c (h w) (p1 p2)', p1=patch_size, p2=patch_size)
    c, patch_num, patch_size2 = img_patches.shape
    if torch.randn(1) < prob:
        masked = torch.rand(patch_num) < 1-mask_ratio
        w = int(math.sqrt(patch_num))
        assert w*w == patch_num, 'patch_num must be square'

        mask = repeat(masked, 'p -> c p m', c=1, m=patch_size2)
        mask = rearrange(mask, 'c (h w) (p1 p2) -> c (h p1) (w p2)',
                         p1=patch_size, p2=patch_size, w=w)
        img = img * mask
        return img, 1 - mask.to(torch.uint8)
    else:
        return img, torch.zeros((1, i_w, i_h)).to(torch.uint8)


def aug_image(image, condition_method, aug):
    # no matter training or validation, it can always be randomed, the validation goes to another logic.

    if condition_method == 'auglevel':
        image, aug_code = auglevel_transform_batch(image.unsqueeze(0), auglevel_config={'tero_aug_prob': aug['tero_aug_prob'],
                                                                                        'image_aug_dim': aug['image_aug_dim'],
                                                                                        'aug_global_prob': aug['aug_global_prob']},
                                                   is_train=True)
        image = (image - image.min()) / \
            (image.max() - image.min() + 1e-8)  # [0, 1]
        image = 2 * image - 1  # [-1, 1]
        image, aug_code = image.squeeze(0), aug_code.squeeze(0)
        mask = None

    elif condition_method == 'diffaug':
        image, aug_code = DiffAugmentWrapper(image.unsqueeze(
            0), prob=aug.diffaug.prob,  operation_str=aug.diffaug.policy)
        image, aug_code = image.squeeze(0), aug_code.squeeze(0)
        mask = None

    elif condition_method == 'gridmask':
        image, mask = gridmask(
            image, mask_ratio=aug.gridmask.mask_ratio, patch_size=aug.gridmask.patch_size, prob=aug.gridmask.prob)
        aug_code = torch.ones((1,)) if torch.sum(
            mask) > 0 else torch.zeros((1,))

    elif condition_method == 'clipfeat':
        aug_code = 0
        mask = None

    elif condition_method in ['label','labelmix']:
        aug_code = 0
        mask = None

    elif condition_method is None:
        aug_code = 0
        mask = None
    else:
        raise

    return image, aug_code, mask
