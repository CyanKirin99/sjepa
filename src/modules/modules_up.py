# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial
import torch
import torch.nn as nn

from src.utils import trunc_normal_


class Diff(nn.Module):
    def __init__(self):
        super(Diff, self).__init__()
        self.diff = torch.diff

    def compute_diff(self, x, dim=-1):
        """
        计算输入数据的差分导数，复制最后一个值以保持与原始数据相同长度
        :param x: (torch tensor)，尺寸为(batch_size, spec_len)
        :param dim: int，计算的维度
        :return: x (torch tensor)，形状为(batch_size, spec_len)
        """
        x = self.diff(x, dim=dim)
        x = torch.cat((x, x[:, -1:]), dim=dim)
        return x

    def forward(self, x):
        """
        计算输入光谱数据的一阶和二阶差分导数
        :param x: (torch tensor), 尺寸为(batch_size, spec_len)
        :return: fd_x (torch tensor)，尺寸为(batch_size, spec_len)
        :return: sd_x (torch tensor)，尺寸为(batch_size, spec_len)
        """
        fd_x = self.compute_diff(x)
        sd_x = self.compute_diff(fd_x)
        return fd_x, sd_x


class ConvLayerNorm(nn.Module):
    def __init__(self, patch_size, hid_dim):
        super(ConvLayerNorm, self).__init__()
        self.proj = nn.Conv1d(in_channels=1, out_channels=hid_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x):
        """
        将输入数据用卷积层映射到特征维度，并层归一化
        :param x: (torch tensor)，尺寸为(batch_size, spec_len)
        :return x: (torch tensor)，尺寸为(batch_size, patch_len, hid_dim)
        """
        x = x.unsqueeze(1)  # ->(batch_size, 1, spec_len)
        x = self.proj(x)  # ->(batch_size, hid_dim, patch_len)
        x = x.transpose(1, 2)  # ->(batch_size, patch_len, hid_dim)
        x = self.norm(x)
        return x


class SplitProject(nn.Module):
    def __init__(self, hid_dim, patch_size, pre_num=3):
        super(SplitProject, self).__init__()
        self.hid_dim = hid_dim
        self.patch_size = patch_size
        self.pre_num = pre_num
        self.proj = nn.ModuleList([ConvLayerNorm(patch_size, hid_dim // pre_num) for _ in range(pre_num)])

    def forward(self, x, pre_code):
        """
        按照数据类型，用线性层或卷积层将数据映射到特征维度
        :param x: (torch tensor)，尺寸为(batch_size, spec_len)
        :param pre_code: 0, 1, 2分别代表sg, fd, sd
        :return: x: 映射后的数据，(torch tensor)，尺寸为(batch_size, patch_len, hid_dim)
                 mask:无效数据掩码，(torch tensor)，内含元素为bool，True表示无效，尺寸为(batch_size, patch_len)
        """
        # 按patch_size分割为小块
        x_patched = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        x_patched = x_patched.contiguous().view(x_patched.size(0), -1, self.patch_size)
        # 检测有0值的patch，使mask为True
        mask = (x_patched == 0).any(dim=-1)
        y = self.proj[pre_code](x)

        return y, mask


class Embedder(nn.Module):
    def __init__(self, hid_dim, patch_size, total_len=2160, pre_num=3, init_std=0.02):
        super(Embedder, self).__init__()
        self.hid_dim = hid_dim
        self.patch_size = patch_size
        self.init_std = init_std
        self.num_patches = total_len // patch_size

        self.diff = Diff()
        self.proj = SplitProject(hid_dim, patch_size, pre_num)

        self.apply(self._init_weights)

    def forward(self, x):
        fd_x, sd_x = self.diff(x)

        sg_x, mask = self.proj(x, 0)
        fd_x, _ = self.proj(fd_x, 1)
        sd_x, _ = self.proj(sd_x, 2)

        x = torch.cat([sg_x, fd_x, sd_x], dim=-1)
        return x, mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PE(nn.Module):
    def __init__(self, dim, max_len=1000, init_std=0.02):
        super(PE, self).__init__()
        self.max_len = max_len
        self.dim = dim
        self.init_std = init_std
        pe = torch.zeros(max_len, dim)
        self.pe = nn.Parameter(pe.unsqueeze(0))  # (1, max_len, dim)
        trunc_normal_(self.pe, std=self.init_std)

    def forward(self, x, indices=None):
        """
        :param x: 输入序列，(torch tensor)，尺寸为(batch_size, seq_length, dim)
        :param indices: 每个序列的索引位置，(torch tensor)，尺寸为(batch_size, seq_length)
        :return: x 带有位置编码的序列，(torch tensor)，尺寸为(batch_size, seq_length, dim)
        """
        B, P, D = x.size()
        assert D == self.dim, "数据张量的最后一个维度必须等于位置编码的维度"

        if indices is None:
            indices = torch.arange(P).unsqueeze(0).expand(B, -1)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D).to(x.device)

        pe = self.pe.expand(B, -1, -1)
        pos_encoding = torch.gather(pe, 1, indices_expanded)

        x = x + pos_encoding
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            expanded_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, attn.size(1), attn.size(2), -1)
            attn = attn.masked_fill(expanded_mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        y, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class Encoder(nn.Module):
    def __init__(
            self,
            embed_dim=192,
            depth=6,
            num_heads=6,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth

        self.pe = PE(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, indices=None, mask=None):
        x += self.pe(x, indices)

        for layer_id, blk in enumerate(self.blocks):
            x, attn = blk(x, mask=mask)

        return x


class Predictor(nn.Module):
    def __init__(
            self,
            embed_dim=192,
            predictor_embed_dim=96,
            depth=4,
            num_heads=6,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.pe = PE(predictor_embed_dim)
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, ctx_rep, tgt_shp, cxt_indices, tgt_indices, num_tgt_blk, sampling_mask):
        # -- map from encoder-dim to predictor-dim and add positional embedding
        ctx_rep_down = self.predictor_embed(ctx_rep)
        ctx_rep_down += self.pe(ctx_rep_down, cxt_indices)
        ctx_rep_down = ctx_rep_down.repeat_interleave(num_tgt_blk, dim=0)

        ctx_rep_down = ctx_rep_down[sampling_mask]
        tgt_indices = tgt_indices[sampling_mask]

        # -- concat mask tokens to ctx
        pred_tokens = self.mask_token.repeat(tgt_shp[0], tgt_shp[1], 1)
        pred_tokens += self.pe(pred_tokens, tgt_indices)
        # --
        concat_tokens = torch.cat([ctx_rep_down, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            concat_tokens, attn = blk(concat_tokens)
        concat_tokens = self.predictor_norm(concat_tokens)

        # -- return tgt for mask tokens
        pred_rep = concat_tokens[:, ctx_rep.size(1):]
        pred_rep = self.predictor_proj(pred_rep)

        del ctx_rep, ctx_rep_down, pred_tokens, concat_tokens
        return pred_rep


def encoder_predictor(**kwargs):
    model = Predictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def encoder_tiny(**kwargs):
    model = Encoder(
        embed_dim=96, depth=2, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def encoder_small(**kwargs):
    model = Encoder(
        embed_dim=192, depth=4, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def encoder_base(**kwargs):
    model = Encoder(
        embed_dim=384, depth=6, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def encoder_large(**kwargs):
    model = Encoder(
        embed_dim=768, depth=8, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


MODEL_REGISTRY = {
    'tiny': encoder_tiny,
    'small': encoder_small,
    'base': encoder_base,
    'large': encoder_large,
    'predictor': encoder_predictor
}


def get_encoder(model_type, **kwargs):
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type](**kwargs)
    else:
        raise ValueError(f'Unknown model size {model_type}')


class UpModel(nn.Module):
    def __init__(self, model_size, patch_size=30, pred_depth=4, pred_emb_dim=96):
        super(UpModel, self).__init__()
        self.encoder = get_encoder(model_type=model_size)
        self.embed_dim = self.encoder.embed_dim
        self.num_heads = self.encoder.num_heads
        self.predictor = get_encoder(
            model_type='predictor',
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth
        )
        self.embedder = Embedder(self.encoder.embed_dim, patch_size)

    def forward(self, x):
        x, mask = self.embedder(x)
        x = self.encoder(x, mask=mask)
        x = torch.cat([x, mask.unsqueeze(-1)], dim=-1)
        return x
