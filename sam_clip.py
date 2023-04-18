import glob
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

import argparse
import logging
import open_clip
from collections import OrderedDict, namedtuple
from einops import rearrange


# Modified from https://github.com/lucidrains/DALLE2-pytorch/blob/350a3d60456693a8ecdccc820e97dbb6b0c81866/dalle2_pytorch/dalle2_pytorch.py#L238 # noqa
class ClipAdapter(nn.Module):
    def __init__(self, name="ViT-B-32", normalize=True):


        open_clip.create_model_and_transforms(name, pretrained="openai")


        # checked, the same as openai original CLIP
        openai_clip, _, preprocess = open_clip.create_model_and_transforms(
            name, pretrained="openai"
        )
        super().__init__()
        self.clip = openai_clip

        # self.clip_normalize = preprocess.transforms[-1]
        # the first two are Resize and Crop, the last one is normalization
        self.clip_preprocess = T.Compose([*preprocess.transforms[:2], preprocess.transforms[-1]])
        self._freeze()
        self.name = name
        self.normalize = normalize

    def extra_repr(self) -> str:
        return f"name={self.name}, normalize={self.normalize}"

    def _freeze(self):
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return super().state_dict(destination=destination, prefix=prefix)

    @property
    def device(self):
        return next(self.parameters()).device

    # don't save clip model
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return OrderedDict()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze()
        return self

    @property
    def dim_latent(self):
        return self.clip.text_projection.shape[-1]

    @property
    def image_size(self):
        if isinstance(self.clip.visual.image_size, tuple):
            return self.clip.visual.image_size
        else:
            return (self.clip.visual.image_size, self.clip.visual.image_size)

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.context_length

    def _encode_text(self, text):
        x = self.clip.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.clip.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)
        text_encodings = x

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection

        return text_embed, text_encodings

    @torch.no_grad()
    def embed_text(self, captions):
        text = open_clip.tokenize(captions).to(next(self.parameters()).device)
        text = text[..., : self.max_text_len]
        text_mask = (text != 0).long()

        text_embed, text_encodings = self._encode_text(text)
        if self.normalize:
            return EmbeddedText(
                F.normalize(text_embed.float(), dim=-1), text_encodings.float(), text_mask
            )
        else:
            return EmbeddedText(text_embed.float(), text_encodings.float(), text_mask)

    def _encode_image(self, image):
        if hasattr(self.clip.visual, "positional_embedding"):
            x = self.clip.visual.conv1(image)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [
                    self.clip.visual.class_embedding.to(x.dtype)
                    + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x,
                ],
                dim=1,
            )  # shape = [*, grid ** 2 + 1, width]
            x = x + self.clip.visual.positional_embedding.to(x.dtype)
            x = self.clip.visual.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            # [batch_size, num_patches+1, transformer.width]
            x = self.clip.visual.ln_post(x)
            batch_size, num_tokens, _ = x.shape

            if self.clip.visual.proj is not None:
                x = rearrange(x, "b n c -> (b n) c", b=batch_size, n=num_tokens)
                x = x @ self.clip.visual.proj
                x = rearrange(x, "(b n) c -> b n c", b=batch_size, n=num_tokens)

            image_embed = x[:, 0, :]
            image_encodings = x[:, 1:, :]

            width = height = int(image_encodings.shape[1] ** 0.5)

            image_encodings = rearrange(image_encodings, "b (h w) c -> b c h w", h=height, w=width)

            image_encodings = F.interpolate(
                image_encodings,
                size=(image.shape[2] // 16, image.shape[3] // 16),
                mode="bilinear",
                align_corners=False,
            )

            return image_embed, image_encodings
        else:
            image_embed = self.clip.encode_image(image)
            return image_embed, None

    @torch.no_grad()
    def embed_image(self, image):
        image_embed, image_encodings = self._encode_image(self.clip_preprocess(image))
        if self.normalize:
            return EmbeddedImage(F.normalize(image_embed.float(), dim=-1), image_encodings)
        else:
            return EmbeddedImage(image_embed.float(), image_encodings)

    @torch.no_grad()
    def build_text_embed(self, labels):
        return build_clip_text_embed(self.clip, labels)


# Thanks Zheng Ding for sharing the nice implementation, we modified based on that.
class MaskCLIP(ClipAdapter):
    """
    Ref: https://arxiv.org/abs/2208.08984
    """

    def __init__(self, name="ViT-L-14-336"):
        super().__init__(name=name, normalize=False)

    @property
    def logit_scale(self):
        logit_scale = torch.clamp(self.clip.logit_scale.exp(), max=100)
        return logit_scale

    def _mask_clip_forward(self, x: torch.Tensor, attn_mask: torch.Tensor, num_mask_tokens: int):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.clip.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        cls_embed = x[0:1]
        cls_embed = cls_embed.expand(num_mask_tokens, -1, -1)
        x = torch.cat([cls_embed, x], dim=0)
        x = self.clip.visual.transformer(x, attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # [N, L, D]
        x = self.clip.visual.ln_post(x[:, :num_mask_tokens, :])

        if self.clip.visual.proj is not None:
            x = torch.einsum("nld,dc->nlc", x, self.clip.visual.proj)

        return x

    def encode_image_with_mask(self, image, mask):
        assert hasattr(self.clip.visual, "positional_embedding")
        image = self.clip_preprocess(image)
        batch_size = image.shape[0]
        assert batch_size == mask.shape[0]
        num_queries = mask.shape[1]

        # [B, Q, H, W], Q is the number of quries, H and W are the height and width of the image
        # mask = mask.sigmoid()
        # [B, Q, H//P, W//P]
        patch_mask = F.max_pool2d(
            mask,
            kernel_size=self.clip.visual.conv1.kernel_size,
            stride=self.clip.visual.conv1.stride,
        )
        # 0 means not masked out, 1 mean masked out
        # so if 1 pixel > 0.5, it is not masked out
        # aka if all pixels (max pixel) < 0.5, it is masked out
        mask_token_attn_mask = patch_mask < 0.5
        # [B, Q, H//P x W//P]
        mask_token_attn_mask = mask_token_attn_mask.reshape(batch_size, num_queries, -1)

        num_mask_token = num_queries
        num_image_cls_token = self.clip.visual.positional_embedding.shape[0]
        num_image_token = num_image_cls_token - 1
        num_all_token = num_mask_token + num_image_cls_token

        # we start with no mask out
        attn_mask = torch.zeros(
            (num_all_token, num_all_token), dtype=torch.bool, device=image.device
        )

        # mask+cls+image token to mask token attention is masked out
        attn_mask[:, :num_mask_token] = True

        attn_mask = attn_mask.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        attn_mask[:, :num_mask_token, -num_image_token:] = mask_token_attn_mask
        num_heads = self.clip.visual.conv1.out_channels // 64  # head width 64
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch_size * num_heads, num_all_token, num_all_token)

        return self._mask_clip_forward(image, attn_mask, num_mask_token)

    def get_mask_embed(self, image, mask):

        image = F.interpolate(
            image,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)

        # [B, Q, C]
        mask_embed = self.encode_image_with_mask(image, mask)

        return mask_embed

    def pred_logits(self, mask_embed, text_embed, labels):
        logit_per_mask = (
            torch.einsum(
                "bqc,nc->bqn", F.normalize(mask_embed, dim=-1), F.normalize(text_embed, dim=-1)
            )
            * self.logit_scale
        )

        # logit_per_mask = ensemble_logits_with_labels(logit_per_mask, labels)

        return logit_per_mask

    def forward(self, image, mask, text_embed, labels):

        mask_embed = self.get_mask_embed(image, mask)
        output = {"mask_embed": mask_embed}

        if text_embed is not None and labels is not None:

            output["mask_pred_open_logits"] = self.pred_logits(mask_embed, text_embed, labels)

        return output


EmbeddedImage = namedtuple("EmbeddedImage", ["image_embed", "masks", "predicted_iou", "stability_score"])

class SamClip(nn.Module):
    def __init__(self, sam_ckpt, sam_model='vit_h'):
        super().__init__()
        self.sam = sam_model_registry[sam_model](checkpoint=sam_ckpt)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_batch=1)
        self.clip = MaskCLIP()

    def encode_image(self, image):
        masks_data = self.mask_generator.generate(image)
        masks = np.stack([d['segmentation'] for d in masks_data])
        predicted_iou = np.stack([d['predicted_iou'] for d in masks_data])
        stability_score = np.stack([d['stability_score'] for d in masks_data])
        masks = torch.from_numpy(masks).unsqueeze(0).float().to(self.clip.device) # B, M, H, W
        image = torch.from_numpy(image).float().permute(2,0,1).unsqueeze(0).to(self.clip.device) / 255.
        image_embed = self.clip.get_mask_embed(image, masks) # B, M, D
        predicted_iou = torch.from_numpy(predicted_iou).unsqueeze(0).float().to(self.clip.device)
        stability_score = torch.from_numpy(stability_score).unsqueeze(0).float().to(self.clip.device)
        return EmbeddedImage(image_embed, masks, predicted_iou, stability_score)
    
    def get_image_embed_map(self, image):
        res = self.encode_image(image)
        scores = (res.stability_score * res.predicted_iou).softmax(dim=-1) # B, M
        weight_map = torch.einsum("bm,bmhw->bmhw", scores, res.masks) # B, M, H, W
        embed_map = torch.einsum("bmhw,bmd->bhwd", weight_map.cpu(), res.image_embed.cpu()) # B, H, W, D
        return embed_map

    def encode_text(self, labels):
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(labels[0], str):
            labels = [[t] for t in labels]

        labels = tuple(tuple(t) for t in labels)

        # check if is ensemble
        assert isinstance(
            labels[0], (list, tuple)
        ), f"labels should be a list of list of str, but got {type(labels[0])}"

        # unravel list of list of str
        flatten_text = [t for sublist in labels for t in sublist]

        text_embed_list = []

        local_batch_size = 256

        for i in range(0, len(flatten_text), local_batch_size):
            cur_text = flatten_text[i : i + local_batch_size]
            text_embed = self.clip.clip.encode_text(open_clip.tokenize(cur_text).to(self.clip.device))
            text_embed_list.extend(list(text_embed))

        out_text_embed = torch.stack(text_embed_list)
        return out_text_embed


def pca_feat_map(f):
    # x: H, W, D
    H, W, D = f.shape
    a = f.reshape(-1, D)
    u,s,v = torch.pca_lowrank(a, q=3)
    f_pca = (a @ v[..., :3]).reshape(H, W, 3)
    return f_pca

def norm_img(f):
    m1, m2 = f.min(), f.max()
    return (f - m1)/(m2-m1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_ckpt', type=str, default='data/sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_model', type=str, default='vit_h')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    sam = SamClip(sam_ckpt=args.sam_ckpt, sam_model=args.sam_model).eval().cuda()

    for path in tqdm(glob.glob(args.input)):
        image = cv2.imread(path)
        embed_map = sam.get_image_embed_map(image)
        embed_map_pca = pca_feat_map(embed_map[0])
        embed_map_pca = norm_img(embed_map_pca).detach().cpu().numpy()
        pca_filename = os.path.join(args.output, os.path.basename(path).replace('rgb', 'clip_pca'))
        clip_filename = os.path.join(args.output, os.path.basename(path).replace('rgb', 'clip').split('.')[0] + '.pt')
        cv2.imwrite(pca_filename, embed_map_pca*255)
        torch.save(embed_map, clip_filename)

