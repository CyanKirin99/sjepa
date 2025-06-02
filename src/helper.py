# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import logging
import sys
import torch

from src.utils import WarmupCosineSchedule, CosineWDSchedule, trunc_normal_
from src.modules import UpModel, DownModel

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_model(
        device,
        model_size='large',
        pred_depth=4,
        pred_emb_dim=96,
        patch_size=30,
        task_configs=None,
        **kwargs
):
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    up_model = UpModel(
        model_size=model_size,
        patch_size=patch_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim
    )
    # logger.info(up_model)
    for m in up_model.modules():
        init_weights(m)
    up_model.to(device)

    if task_configs is not None:
        embed_dim = up_model.embed_dim
        num_heads = up_model.num_heads
        down_model = DownModel(embed_dim, num_heads, task_configs)
        for m in down_model.modules():
            init_weights(m)
        down_model.to(device)
    else:
        down_model = None

    return up_model, down_model


def init_opt(
        module_list,
        iterations_per_epoch,
        start_lr,
        ref_lr,
        warmup,
        num_epochs,
        wd=1e-6,
        final_wd=1e-6,
        final_lr=0.0,
        use_bfloat16=False,
        ipe_scale=1.25
):
    param_groups = []
    for module in module_list:
        param_groups.append({
            'params': (p for n, p in module.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        })
        param_groups.append({
            'params': (p for n, p in module.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        })

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None

    return optimizer, scaler, scheduler, wd_scheduler


def load_checkpoint(
        up_path=None,
        up_model=None,
        down_path=None,
        down_model=None,
):
    if up_path is not None:  # -- load up_model
        checkpoint = torch.load(up_path, map_location=torch.device('cpu'), weights_only=True)
        if 'up_model' in checkpoint:
            pretrained_dict = checkpoint['up_model']
            msg = up_model.load_state_dict(pretrained_dict)
            logger.info(f'loaded up_model with msg: {msg}')

        else:
            encoder, embedder, predictor = up_model.encoder, up_model.embedder, up_model.predictor
            # -- loading embedder
            pretrained_dict = checkpoint['embedder']
            msg = embedder.load_state_dict(pretrained_dict)
            logger.info(f'loaded embedder with msg: {msg}')

            # -- loading encoder
            pretrained_dict = checkpoint['encoder']
            msg = encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded encoder with msg: {msg}')

            # -- loading target_encoder
            if 'target_encoder' in checkpoint:
                target_encoder = copy.deepcopy(encoder)
                pretrained_dict = checkpoint['target_encoder']
                msg = target_encoder.load_state_dict(pretrained_dict)
                logger.info(f'loaded target_encoder from epoch with msg: {msg}')

            # -- loading predictor
            if 'predictor' in checkpoint:
                pretrained_dict = checkpoint['predictor']
                msg = predictor.load_state_dict(pretrained_dict)
                logger.info(f'loaded predictor from epoch with msg: {msg}')

    if down_path is not None:
        if not isinstance(down_path, dict):
            checkpoint = torch.load(down_path, map_location=torch.device('cpu'), weights_only=True)
            pretrained_dict = checkpoint['down_model']
            msg = down_model.load_state_dict(pretrained_dict)
            logger.info(f'loaded down_model with msg: {msg}')
        else:
            for trait, trait_ckpt in down_path.items():
                checkpoint = torch.load(trait_ckpt, map_location=torch.device('cpu'), weights_only=True)
                pretrained_dict = checkpoint['down_model']
                new_state_dict = {}
                for name, param in pretrained_dict.items():
                    if name.startswith(f'tasks_modules.{trait}.'):
                        new_name = name.split(f'tasks_modules.{trait}.')[1]  # 去掉 trait 前缀
                        new_state_dict[new_name] = param
                msg = down_model.tasks_modules[trait].load_state_dict(new_state_dict)
                logger.info(f'loaded down_model for {trait} with msg: {msg}')

    return up_model, down_model

