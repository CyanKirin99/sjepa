# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

from src.utils import trunc_normal_


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sparsity_coefficient=1e-3):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # Query
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Key and Value
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sparsity_coefficient = sparsity_coefficient

    def forward(self, query, key_value, mask=None):
        B, N_q, C = query.shape  # Query shape: (batch_size, query_length, channels)
        B, N_kv, C = key_value.shape  # Key/Value shape: (batch_size, key_value_length, channels)

        q = self.q(query).reshape(B, N_q, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(key_value).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # Attention scores
        if mask is not None:
            assert mask.shape == (B, N_kv), "Mask shape mismatch"
            mask = mask.unsqueeze(1).unsqueeze(2)  # -> (B, 1, 1, N_kv)
            attn = attn.masked_fill(mask == True, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, norm_scaler, init_std=0.02, norm_layer=nn.LayerNorm):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim

        if norm_scaler is None:
            norm_scaler = (0., 0.)
        self.register_buffer('norm_scaler', torch.tensor(norm_scaler))
        self.num_heads = num_heads

        self.task_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.task_token, std=init_std)

        self.norm = norm_layer(embed_dim)
        self.cross_attention = CrossAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, x, mask=None, return_attn=False, denorm=False):
        B, P, D = x.shape

        batch_task_tokens = self.task_token.expand(B, -1, -1)
        y, attn = self.cross_attention(batch_task_tokens, self.norm(x), mask=mask)
        # y += batch_task_tokens
        y = self.mlp(y).squeeze(-1)

        if return_attn:
            return attn

        if denorm:
            y = y * self.norm_scaler[1] + self.norm_scaler[0]

        return y


class DownModel(nn.Module):
    def __init__(self, embed_dim, num_heads, task_configs, init_std=0.02):
        super(DownModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.tasks_modules = nn.ModuleDict({
            config["name"]: Decoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                norm_scaler=config["norm_scaler"]
            )
            for config in task_configs
        })
        self.init_std = init_std
        self.apply(self._init_weights)

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

    def forward(self, x, trait_name=None, return_attn=False, denorm=False):
        mask = x.detach()[:, :, -1]
        x = x.detach()[:, :, :-1]
        if trait_name is not None:  # 单任务训练 & 输出
            tasks_modules = self.tasks_modules[trait_name]
            output = tasks_modules(x, mask=mask, return_attn=return_attn, denorm=denorm)
            return output
        else:   # 多任务输出
            outputs = {}
            for name, module in self.tasks_modules.items():
                outputs[name] = module(x, mask=mask, denorm=denorm)
            return outputs

