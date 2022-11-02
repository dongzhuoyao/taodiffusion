# modified from https://github.com/CompVis/taming-transformers/blob/24268930bf1dce879235a7fddd0b2355b84d7ea6/taming/data/base.py#L23
# https://github.com/CompVis/taming-transformers#ffhq

import math
import os
from pathlib import Path
import PIL
from einops import rearrange
from loguru import logger
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset.transforms.aug_router import aug_image
from dataset.transforms.clip_func import extract_clip_feat
from dataset.transforms.complex_ds_common_util import RandomScaleCrop
from diffusion_utils.util import normalize_to_neg_one_to_one
from numpy.random import default_rng



class DirectoryDataset(Dataset):
    def __init__(
        self,
        root,
        dataset_name,
        size=None,
        img_save_path=None,
        num_samples=1.0,
        random_crop=False,
        debug=False,
        condition_method=None,
        aug=None,
        seed=0,
    ):

        self.dataset_name = dataset_name
        self.img_save_path = img_save_path
        self.debug = debug
        self.size = size
        self.size4crop = size
        self.random_crop = random_crop
        self.condition_method = condition_method
        self.aug = aug

        root = Path(root).expanduser().resolve()
        self.pathlist = []
        for _file in os.listdir(root):
            self.pathlist.append(os.path.join(root, _file))

       

        if num_samples is not None:
            idx = np.array([i for i in range(len(self.pathlist))])
            default_rng(seed).shuffle(idx)

            self.pathlist = [self.pathlist[_id] for _id in idx]
            

            if isinstance(num_samples, int):
                _partial_rate = num_samples / len(self.pathlist)

            elif isinstance(num_samples, float):
                _partial_rate = num_samples
                num_samples = int(_partial_rate * len(self.pathlist))

            else:
                raise ValueError(
                    f'num_samples must be int or float, got {type(num_samples)}')
            self.pathlist = self.pathlist[:num_samples]
            if self.condition_method == 'clipfeat':
                self.featlist = []
                for _path in tqdm(self.pathlist, desc='extracting clip features',total=len(self.pathlist)):
                    image = Image.open(_path)
                    image = image.resize((size, size), Image.BILINEAR)
                    self.featlist.append(extract_clip_feat(image=image).unsqueeze(0))
                self.featlist = torch.cat(self.featlist, dim=0)
                self.featlist = self.featlist.cpu().numpy()

            logger.warning(f'Using only {num_samples} images')
            _partial_rate_inverse = math.ceil(
                1.0/_partial_rate)*math.ceil(20_000/100)

            self.pathlist = self.pathlist*_partial_rate_inverse
            if self.condition_method == 'clipfeat':
                self.featlist = np.tile(
                    self.featlist, (_partial_rate_inverse, 1))
                assert len(self.pathlist) == len(
                    self.featlist), f'{len(self.pathlist)} != {len(self.featlist)}'

        else:
            assert self.condition_method != 'clipfeat', 'clip condition method requires num_samples'


        logger.warning(f'upscaled only {len(self.pathlist)} images')

        self._length = len(self.pathlist)

        self.transform_depreciated = RandomScaleCrop(
            base_size=self.size4crop, resize_size=size)

    def id2name(self, index):
        file_name = os.path.basename(self.pathlist[index])
        return file_name

    def _read_img_segmask(self, index):
        image_path = self.pathlist[index]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image

    def __len__(self):
        return min(100, self._length) if self.debug else self._length

    def get_imgid_from_imagepath(self, image_path):
        return os.path.basename(image_path).split(".")[0]

    def __getitem__(self, index):
        result = dict()
        image = self._read_img_segmask(index)

        image = torch.from_numpy(
            np.array(image.resize((self.size, self.size), Image.BILINEAR))
        )
        image = rearrange(image, "w h c -> c w h")

        if self.img_save_path is not None:
            if True:
                image_pil = Image.fromarray(
                    image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                )
                image_pil.save(os.path.join(
                    self.img_save_path, f"{index}.png"))
            else:
                result.update(
                    dict(img_save=image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)))

        image = normalize_to_neg_one_to_one(image / 255.0)  # [0,1]->[-1,1]
        image, aug_code, mask = aug_image(
            image, self.condition_method, self.aug)
        if mask is not None:
            result['mask'] = mask
        result['aug_code'] = aug_code
        
        if self.condition_method == 'clipfeat':
            result['clipfeat'] = self.featlist[index]

        result.update(
            dict(
                image=image,
                id=index,
            )
        )
        return result


def generate_64_from_128(root_dir=None, target_dir=None):
    root_dir = Path("~/data/ffhq/thumbnails128x128").expanduser().resolve()
    target_dir = Path("~/data/ffhq/thumbnails64x64").expanduser().resolve()

    for dir1 in tqdm(os.listdir(root_dir)):
        if dir1 == "LICENSE.txt":
            continue
        for file_name in os.listdir(os.path.join(root_dir, dir1)):
            src_path = os.path.join(root_dir, dir1, file_name)
            target_path = os.path.join(target_dir, dir1, file_name)
            src_img = Image.open(src_path)
            target_image = src_img.resize((64, 64), PIL.Image.LANCZOS)
            target_image.save(target_path)
    exit(0)


if __name__ == "__main__":
    pass
