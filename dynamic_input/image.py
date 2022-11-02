
from dataset.ds_utils.dataset_common_utils import ds_has_label_info
from diffusion_utils.util import clip_unnormalize_to_zero_to_255
from tqdm import tqdm
import wandb


def prepare_image(pl_module, batch_data):
    return batch_data
