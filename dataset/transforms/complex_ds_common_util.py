import os
import random
from PIL import Image
import torch
import numpy as np
from einops import rearrange


from diffusion_utils.util import normalize_to_neg_one_to_one
from loguru import logger
from torchvision import transforms
from pathlib import Path


class RandomScaleCrop(object):
    def __init__(self, base_size, resize_size, fill=0):
        self.base_size = base_size
        self.crop_size = base_size
        self.resize_size = resize_size
        self.fill = fill

    def __call__(self, img, mask):
        # random scale (short edge)
        # short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        short_size = random.randint(
            int(self.base_size * 1.05), int(self.base_size * 1.25)
        )  # dongzhuoyao
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            raise  # dongzhuoyao, should not possible
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(
                0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if mask is not None:
            mask = mask.crop(
                (x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        img = torch.from_numpy(
            np.array(img.resize((self.resize_size, self.resize_size)))
        )

        if mask is not None:
            mask = torch.from_numpy(
                np.array(
                    mask.resize(
                        (self.resize_size, self.resize_size), resample=Image.NEAREST
                    )
                )
            )

        img = rearrange(img, "w h c -> c w h")
        return img, mask


def get_item_complex(dl, index):
    result = dict()
    image, segmask = dl._read_img_segmask(index)

    img4unsup = transforms.ToTensor()(image)*255.0
    img4unsup = transforms.Resize(
        (dl.size4cluster, dl.size4cluster))(img4unsup)

    image, segmask = dl.transform(image, segmask)

    if dl.img_save_path is not None:
        if False:
            image_pil = Image.fromarray(
                image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            )
            image_pil.save(os.path.join(
                dl.img_save_path, f"{index}.png"))
        else:
            result.update(
                dict(img_save=image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)))

    image = normalize_to_neg_one_to_one(image / 255.0)  # [0,1]->[-1,1]

    result.update(
        dict(
            image=image,
            id=index,
        )
    )

    return result
