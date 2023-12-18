""" DeiT - Data-efficient Image Transformers
DeiT model defs and weights from https://github.com/facebookresearch/deit, original copyright below
paper: `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
paper: `DeiT III: Revenge of the ViT` - https://arxiv.org/abs/2204.07118
Modifications copyright 2021, Ross Wightman
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial

import torch
from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import VisionTransformer, trunc_normal_, checkpoint_filter_fn

from .helpers import build_model_with_cfg, checkpoint_seq
from .registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    'deit3_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth'),
    'deit3_small_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_medium_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_1k.pth'),
    'deit3_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth'),
    'deit3_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_large_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_1k.pth'),
    'deit3_large_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_huge_patch14_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_1k.pth'),

    'deit3_small_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_224_21k.pth',
        crop_pct=1.0),
    'deit3_small_patch16_384_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_small_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_medium_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_21k.pth',
        crop_pct=1.0),
    'deit3_base_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth',
        crop_pct=1.0),
    'deit3_base_patch16_384_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_base_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_large_patch16_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_224_21k.pth',
        crop_pct=1.0),
    'deit3_large_patch16_384_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_large_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'deit3_huge_patch14_224_in21ft1k': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_21k_v1.pth',
        crop_pct=1.0),
}


class bi_Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = LinearBi(in_features, hidden_features, mode=Qmodes.kernel_wise)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = LinearBi(hidden_features, out_features, mode=Qmodes.kernel_wise)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        # print(torch.max(x), torch.min(x))
        x = self.act(x)
        
        x = torch.clip(x, -10., 10.)
        # print(torch.clip(x, -10., 10.))
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class bi_Attention(nn.Module):
     
    def __init__(self, dim, num_heads=8, quantize_attn=True, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.quantize_attn = quantize_attn
        
        self.norm_q = nn.LayerNorm(head_dim)
        self.norm_k = nn.LayerNorm(head_dim)
        # self.norm_v = nn.LayerNorm(head_dim)
        # self.norm_proj = nn.LayerNorm(dim)


        if self.quantize_attn:

            self.qkv = LinearBi(dim, dim * 3, bias=qkv_bias, mode=Qmodes.kernel_wise)
            
            # self.norm = nn.LayerNorm(64)

            self.attn_drop = nn.Dropout(attn_drop)

            # self.norm_attn = nn.LayerNorm(198) # nn.BatchNorm2d(self.num_heads, eps=0.001, affine=True)

            self.proj = LinearBi(dim, dim, mode=Qmodes.kernel_wise)
            self.q_act = ActBi(in_features=self.num_heads)
            self.k_act = ActBi(in_features=self.num_heads)
            self.v_act = ActBi(in_features=self.num_heads)
            self.attn_act = ActBi(in_features=self.num_heads)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.q_act = ActBi(in_features=self.num_heads)
            self.k_act = ActBi(in_features=self.num_heads)
            self.v_act = ActBi(in_features=self.num_heads)
            self.attn_act = ActBi(in_features=self.num_heads)
        
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # with torch.no_grad():
        # x_32 = x.detach()
        # qkv_32 = self.qkv_32(x_32).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   # B, heads, N, C//n_heads

        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # qkv = self.norm(qkv)
        q = self.norm_q(q)
        k = self.norm_k(k)
        # v = self.norm_v(v)

        # if self.quantize_attn:
        q = self.q_act(q)
        k = self.k_act(k)
        v = self.v_act(v)
        # print(np.linalg.matrix_rank(q[0].detach().cpu().numpy()))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)   # torch.Size([256, 3, 197, 64])   [Bs(64), 3, 198, 198]
        # print(attn.shape)
        # if self.quantize_attn:
        attn = self.attn_act(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = x + residual
        # x = self.norm_proj(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class bi_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = bi_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = bi_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformerDistilled(VisionTransformer):
    """ Vision Transformer w/ Distillation Token and Head
    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(*args, **kwargs, weight_init='skip')
        assert self.global_pool in ('token',)

        self.num_prefix_tokens = 2
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = True  # must set this True to train w/ distillation token

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_token',
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_features(self, x) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        if pre_logits:
            return (x[:, 0] + x[:, 1]) / 2
        x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2


def _create_deit(variant, pretrained=False, distilled=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model_cls = VisionTransformerDistilled if distilled else VisionTransformer
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        **kwargs)
    return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_deit('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_deit('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit('deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_deit(
        'deit_tiny_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_deit(
        'deit_small_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit(
        'deit_base_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_deit(
        'deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def deit3_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_small_patch16_384(pretrained=False, **kwargs):
    """ DeiT-3 small model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_medium_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 medium model @ 224x224 (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_medium_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_large_patch16_224(pretrained=False, **kwargs):
    """ DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_large_patch16_384(pretrained=False, **kwargs):
    """ DeiT-3 large model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit3_huge_patch14_224(pretrained=False, **kwargs):
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, no_embed_class=True, init_values=1e-6, **kwargs)
    model = _create_deit('deit3_huge_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


