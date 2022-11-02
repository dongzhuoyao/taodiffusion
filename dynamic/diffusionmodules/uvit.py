# %%%
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from loguru import logger

from dynamic.diffusionmodules.util import timestep_embedding
# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # add additional linear layer following FanBao
                nn.Linear(dim, dim),
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # add skip connection similar to UNet, respecting to the description of U-ViT
        for _embed, attn, ff in self.layers:
            x = _embed(x)
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size,
                 patch_size,
                 layer_num=13,
                 dim, heads, mlp_dim,
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 #################
                 cond_token_num=0,
                 cond_dim=None,
                 use_cls_token_as_pooled=None,
                 condition=None,
                 condition_method=None,):

        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches_h, num_patches_w = image_height // patch_height, image_width // patch_width
        num_patches = num_patches_h * num_patches_w
        patch_dim = self.patch_dim = channels * patch_height * patch_width

        self.dim = dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.patch_backto_image = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      h=num_patches_h, p1=patch_height, p2=patch_width),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        assert layer_num % 2 == 1, 'layer_num must be odd'
        num_stage = layer_num//2
        logger.warning(f'layer_num: {layer_num}, num__stage: {num_stage}')
        self.skip_scale = 1.0  # 2 ** -0.5
        # downs
        self.downs = nn.ModuleList([Transformer(
            dim, 1, heads, dim_head, mlp_dim, dropout) for _ in range(num_stage)])

        # middle
        self.mid = Transformer(
            dim, 1, heads, dim_head, mlp_dim, dropout)

        # ups
        self.ups = nn.ModuleList([])
        for _ in range(num_stage):
            self.ups.append(nn.ModuleList([
                nn.Linear(2 * dim, dim),  # mapping back to original dim
                Transformer(
                    dim, 1, heads, dim_head, mlp_dim, dropout)
            ]))

        self.out_dim = channels

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1)
        )

        ####
        self.num_time_tokens = 8
        self.to_time_tokens = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim * self.num_time_tokens),
            Rearrange("b (r d) -> b r d", r=self.num_time_tokens),
        )

        self.num_cond_tokens = cond_token_num
        self.to_cond_tokens = nn.Sequential(
            nn.Linear(cond_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim * self.num_cond_tokens),
            Rearrange("b (r d) -> b r d", r=self.num_cond_tokens),
        )

        self.norm_cond = nn.LayerNorm(self.dim)
        self.norm_time = nn.LayerNorm(self.dim)

        self.pos_embedding = nn.Parameter(torch.randn(
            1, num_patches + 1 + self.num_time_tokens + cond_token_num, dim))

    def forward_with_cond_scale(self, x, t, cond_scale, cond=None):
        return self.forward(x, t, cond=cond)

    def forward(self, img, timesteps, cond_drop_prob=0.0, cond=None):
        # patchify time t, to token, and concate it with image token

        ############
        t_emb = timestep_embedding(
            timesteps, self.dim, repeat_only=False)
        ###########################
        time_tokens = self.norm_time(self.to_time_tokens(t_emb))

        condition_tokens = self.norm_cond(self.to_cond_tokens(cond))

        x=self.to_patch_embedding(img)
        b, n, _=x.shape

        cls_tokens=repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x=torch.cat((cls_tokens, time_tokens, condition_tokens, x), dim=1)
        x += self.pos_embedding[:,
                                :(n + 1 + self.num_time_tokens + self.num_cond_tokens)]
        x=self.dropout(x)

        ###########
        # residual
        r=x.clone()

        # downs and ups
        down_hiddens=[]

        for attn_block in self.downs:
            x=attn_block(x)
            down_hiddens.append(x)

        x=self.mid(x)

        for attn_block in self.ups:
            # along feat dim
            x=torch.cat((x, down_hiddens.pop() * self.skip_scale), dim=2)
            x=attn_block[0](x)
            x=attn_block[1](x)

        # remove CLS, time, condition token to get patch tokens
        x=x[:, cls_tokens.shape[1]+time_tokens.shape[1] +
              condition_tokens.shape[1]:]
        x=self.patch_backto_image(x)
        # final convolution
        out=self.final_conv(x)

        return out


if __name__ == '__main__':
    model=ViT(
        image_size=32,
        patch_size=2,
        layer_num=13,
        dim=512,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    print(model)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"parameter count: {count_parameters(model)}")

    img=torch.randn(1, 3, 32, 32)
    timestep=torch.randn(1,)
    pred=model(img, timestep)  # (1, 1000)
    print(pred.shape)

# %%
