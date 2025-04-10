# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch


def sample_target_blocks(num_patches, tgt_p_len, ctx_p_len, num_blocks, mask=None):
    """
    随机采多个Target-Block，允许部分重叠，但不允许完全相同的Block重复出现。
    若有mask，mask为True位置不被采样

    参数：
    - num_patches: 总patch数
    - patch_length: 每个Target-Block的长度（固定长度）
    - num_blocks: 采样的Target-Block数量
    - mask: 布尔值ndarray，True的位置索引不被采样

    返回：
    - block_indices: 采样到的Target-Block的patch索引列表
    """
    all_patches = set(range(num_patches))
    used_patches = set()
    if mask is not None:
        used_patches.update(np.where(mask)[0])

    block_indices = []

    for _ in range(num_blocks):
        available_patches = all_patches - used_patches
        if len(available_patches) < tgt_p_len + ctx_p_len:
            break

        # 通过while循环确保不采样重复的Target-Block
        while True:
            # 随机选择起始点，确保起始点不会超出数据范围
            start_idx = np.random.randint(0, num_patches - tgt_p_len + 1)
            # 获取当前block的所有patch索引
            block = list(range(start_idx, start_idx + tgt_p_len))  # 使用tuple来确保hashable

            # 确保新采样的block没有被采样过且符合mask要求
            if all(idx in available_patches for idx in block):
                used_patches.update(block)  # 将该block的每个索引单独加入sampled_blocks集合
                block_indices.append(block)
                break

    # 若采样不足，则添加0填充索引
    if len(block_indices) < num_blocks:
        for _ in range(num_blocks - len(block_indices)):
            block_indices.append([0] * tgt_p_len)

    return block_indices


def sample_context_block(num_patches, patch_length, target_blocks, mask=None):
    """
    采样一个Context-Block，长度固定，且与Target-Block不重叠。

    参数：
    - num_patches: 总patch数
    - patch_length: Context-Block的长度（固定长度）
    - target_blocks: 已采样的Target-Block的索引列表（列表中的每个元素是一个block的patch索引）
    - mask: 布尔值ndarray，True的位置索引不被采样

    返回：
    - block_indices: 采样到的Context-Block的patch索引列表
    """
    # 创建一个包含所有patch索引的集合
    all_patches = set(range(num_patches))

    # 将所有Target-Block的patch索引加入一个集合，表示不可用的区域
    used_patches = set()
    for block in target_blocks:
        used_patches.update(block)

    # 如果提供了mask，将mask为True的位置加入不可用区域
    if mask is not None:
        used_patches.update(np.where(mask)[0])

    # 计算可用的patch索引
    available_patches = list(all_patches - used_patches)
    # 如果可用的patch数小于所需的context block长度，则抛出错误
    if len(available_patches) < patch_length:
        raise ValueError(f"没有足够的可用patch来采样Context-Block, 需要{patch_length}，实际只有{len(available_patches)}")

    # 在可用区域中随机选择一个起始点，确保不会超出范围
    start_idx = np.random.randint(0, len(available_patches) - patch_length + 1)
    # 采样固定长度的Context-Block
    block_indices = available_patches[start_idx:start_idx + patch_length]

    return block_indices


def sample(batch_tensor, num_target_blocks, target_patch_length, context_patch_length, mask=None):
    """
    对batch中的每个样本进行target和context采样，并返回采样索引和截取的context部分张量。

    参数：
    - batch_tensor: 输入的batch张量，形状为 (batch_size, num_patches, feature_dim)
    - num_target_blocks: 每个样本中target block的数量
    - target_patch_length: target block的长度
    - context_patch_length: context block的长度

    返回：
    - target_block_indices_tensor: 每个样本的target block索引，形状为 (batch_size*num_target_blocks, target_patch_length)
    - context_block_indices_tensor: 每个样本的context block索引，形状为 (batch_size, context_patch_length)
    """
    batch_size, num_patches, feature_dim = batch_tensor.shape
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

    target_block_indices_list, context_block_indices_list = [], []
    for i in range(batch_size):
        target_block_indices = sample_target_blocks(num_patches, target_patch_length, context_patch_length,
                                                    num_target_blocks, mask[i])
        context_block_indices = sample_context_block(num_patches, context_patch_length, target_block_indices, mask[i])

        target_block_indices_list.append(target_block_indices)
        context_block_indices_list.append(context_block_indices)

    target_block_indices_tensor = torch.tensor(target_block_indices_list).reshape(-1, target_patch_length)
    context_block_indices_tensor = torch.tensor(context_block_indices_list)

    return target_block_indices_tensor, context_block_indices_tensor


def generate_sampling_mask(target_block_indices):
    B, P = target_block_indices.shape
    padding_mask = torch.ones((B,), dtype=bool)
    for i, block in enumerate(target_block_indices):
        if all(idx == 0 for idx in block):  # 检查是否为填充的无效索引
            padding_mask[i] = False  # 将填充位置的掩码设为False
    return padding_mask


if __name__ == '__main__':
    num_patches = 56
    x = torch.randn(13, 56, 192)
    target_indices, context_indices = sample(x, target_patch_length=6, num_target_blocks=4, context_patch_length=20)
