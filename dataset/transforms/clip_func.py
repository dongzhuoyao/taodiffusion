
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


caption_aug_template = [    
    "a photo of a xxx",
    "a rendering of a xxx",
    "a cropped photo of the xxx",
    "the photo of a xxx",
    "a photo of a clean xxx",
    "a photo of a dirty xxx",
    "a dark photo of the xxx",
    "a photo of my xxx",
    "a photo of the cool xxx",
    "a close-up photo of a xxx",
    "a bright photo of the xxx",
    "a cropped photo of a xxx",
    "a photo of the xxx",
    "a good photo of the xxx",
    "a photo of one xxx",
    "a close-up photo of the xxx",
    "a rendition of the xxx",
    "a photo of the clean xxx",
    "a rendition of a xxx",
    "a photo of a nice xxx",
    "a good photo of a xxx",
    "a photo of the nice xxx",
    "a photo of the small xxx",
    "a photo of the weird xxx",
    "a photo of the large xxx",
    "a photo of a cool xxx",
    "a photo of a small xxx",
    ]




@torch.no_grad()
def extract_clip_feat(image):
    assert isinstance(image, Image.Image)
    image = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    return image_features.squeeze(0).float()


@torch.no_grad()
def extract_clip_txt_feat(caption='a dog'):
    
    text_inputs = clip.tokenize(caption).to(device)
    text_features = model.encode_text(text_inputs)
    return text_features.squeeze(0).float()

@torch.no_grad()
def extract_clip_txt_feat_bybatch(caption_list):
    text_inputs = torch.cat([clip.tokenize(f"{c}") for c in caption_list]).to(device)
    text_features = model.encode_text(text_inputs)
    return text_features.float()