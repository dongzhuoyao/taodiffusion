import random

import torch
from torchvision.transforms import transforms

from dataset.transforms.stylegan2_augment import AugmentPipe
from loguru import logger
import traceback


def auglevel_transform_batch(img, is_train, auglevel_config):
    tero_aug_prob = auglevel_config['tero_aug_prob']
    image_aug_dim = auglevel_config['image_aug_dim']
    aug_global_prob = auglevel_config['aug_global_prob']

    batch_size = len(img)

    def _do_aug():
        _device = img.device
        tero_aug = AugmentPipe(xflip=1.0, yflip=tero_aug_prob, scale=tero_aug_prob, xint=tero_aug_prob,
                               rotate_any=tero_aug_prob, scale_std=0.2).train().requires_grad_(False).to(_device)
        try:
            img_transformed, auglvl_code = tero_aug(img.float())
        except:
            img_transformed, auglvl_code = tero_aug(img.float())
            logger.warning('shit happens')
            traceback.print_exc()
        return img_transformed.to(_device), auglvl_code.to(_device)

    if is_train:
        if random.random() < aug_global_prob:
            img, auglvl_code = _do_aug()
        else:
            # when non-train, set it to zero
            auglvl_code = torch.zeros(
                (batch_size, image_aug_dim), dtype=torch.float)
        return img, auglvl_code
    else:
        # when non-train, set it to zero
        auglvl_code = torch.zeros(
            (batch_size, image_aug_dim), dtype=torch.float)
        return img, auglvl_code


if __name__ == '__main__':
    img = torch.rand(2, 3, 256, 256)
    is_train = True
    auglevel_config = {'tero_aug_prob': 0.5, 'image_aug_dim': 7}

    img_transformed, auglvl_code = auglevel_transform_batch(
        img, is_train, auglevel_config)
    print(img_transformed.shape)
