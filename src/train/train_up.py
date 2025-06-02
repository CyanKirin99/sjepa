# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
import argparse
from pathlib import Path
import copy
import logging
import yaml
from pprint import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.helper import load_checkpoint, init_model, init_opt
from src.datas import create_dataset_up
from src.utils import (gpu_timer, grad_logger,
                       CSVLogger, AverageMeter,
                       sample, generate_sampling_mask)


def setup_args():
    parser = argparse.ArgumentParser(
        'S-JEPA Pre-training Script (train_up.py)',
        description='Script for pre-training S-JEPA model using upstream data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults in --help
    )

    # --- Configuration File ---
    parser.add_argument('--config_file', default=str(current_dir / 'configs_train.yaml'), type=str,
                        help='Path to the YAML configuration file.')

    # --- Meta Parameters ---
    g_meta = parser.add_argument_group('Meta Parameters')
    g_meta.add_argument('--seed', type=int, default=42, help='Global random seed.')
    g_meta.add_argument('--use_bfloat16', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable/disable bfloat16 precision.')
    g_meta.add_argument('--model_size', type=str, default='large', choices=['tiny', 'small', 'base', 'large'],
                        help='Model size.')
    g_meta.add_argument('--patch_size', type=int, default=30, help='Patch window size for the spectra.')
    g_meta.add_argument('--pred_depth', type=int, default=2, help='Predictor depth.')
    g_meta.add_argument('--pred_emb_dim', type=int, default=96, help='Predictor embedding dimension.')
    g_meta.add_argument('--up_checkpoint_path', type=str, default=None,
                        help='Path to an existing model checkpoint to load. If provided, enables loading and overrides YAML path.')
    g_meta.add_argument('--load_optimizer_state', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable/disable loading optimizer state from the checkpoint (if loading a model).')

    # --- Data Parameters ---
    g_data = parser.add_argument_group('Data Parameters')
    g_data.add_argument('--batch_size', type=int, default=400, help='Batch size per iteration.')
    g_data.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable/disable pinning CPU memory in DataLoader.')
    g_data.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers.')
    g_data.add_argument('--spec_path', type=str, default=None,
                        help='Path to spectral data CSV file.')

    # --- Masking Parameters ---
    g_mask = parser.add_argument_group('Masking Parameters')
    g_mask.add_argument('--num_tgt_blk', type=int, default=4, help='Number of target blocks.')
    g_mask.add_argument('--tgt_p_len', type=int, default=6, help='Target patch sequence length.')
    g_mask.add_argument('--ctx_p_len', type=int, default=24, help='Context patch sequence length.')

    # --- Optimization Parameters ---
    g_opt = parser.add_argument_group('Optimization Parameters')
    g_opt.add_argument('--epochs', type=int, default=1000, help='Total number of training epochs (overrides epochs_up).')
    g_opt.add_argument('--ema', type=float, nargs=2, metavar=('START_EMA', 'END_EMA'), default=(0.996, 1.0), help='EMA decay values (start and end).')
    g_opt.add_argument('--ipe_scale', type=float, default=1.0, help='IPE scale factor for scheduler.')
    g_opt.add_argument('--wd', type=float, default=0.04, help='Weight decay (overrides optimization.weight_decay).')
    g_opt.add_argument('--final_wd', type=float, default=0.4, help='Final weight decay (overrides optimization.final_weight_decay).')
    g_opt.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs (overrides optimization.warmup).')
    g_opt.add_argument('--start_lr', type=float, default=0.00002, help='Initial learning rate for warmup.')
    g_opt.add_argument('--lr', type=float, default=0.0001, help='Base learning rate.')
    g_opt.add_argument('--final_lr', type=float, default=1.0e-06, help='Final learning rate.')

    # --- Logging Parameters ---
    g_log = parser.add_argument_group('Logging Parameters')
    g_log.add_argument('--log_freq', type=int, default=1, help='Frequency of logging training stats (iterations).')
    g_log.add_argument('--checkpoint_freq', type=int, default=500, help='Frequency of saving checkpoints (epochs).')

    cli_params = parser.parse_args()

    # 1. Load YAML config as the base
    try:
        with open(cli_params.config_file, 'r') as y_file:
            args_from_yaml = yaml.load(y_file, Loader=yaml.FullLoader)
        print(f"INFO: Loaded base configuration from: {cli_params.config_file}")
    except FileNotFoundError:
        print(
            f"WARNING: Configuration file {cli_params.config_file} not found. Using command-line defaults or script defaults.")
        args_from_yaml = {}  # Start with an empty dict if YAML is not found

    # Create the final args dictionary, starting with YAML (or empty)
    final_args = copy.deepcopy(
        args_from_yaml)  # Use deepcopy to avoid modifying the loaded YAML dict directly if it's reused

    # Ensure basic structure exists if YAML was empty or incomplete
    final_args.setdefault('meta', {})
    final_args.setdefault('data', {})
    final_args.setdefault('mask', {})
    final_args.setdefault('optimization', {})
    final_args.setdefault('logging', {})

    # 2. Override YAML with CLI parameters if they were provided by the user
    # Meta
    if cli_params.seed is not None: final_args['meta']['seed'] = cli_params.seed
    if cli_params.use_bfloat16 is not None: final_args['meta']['use_bfloat16'] = cli_params.use_bfloat16
    if cli_params.model_size is not None: final_args['meta']['model_size'] = cli_params.model_size
    if cli_params.patch_size is not None: final_args['meta']['patch_size'] = cli_params.patch_size
    if cli_params.pred_depth is not None: final_args['meta']['pred_depth'] = cli_params.pred_depth
    if cli_params.pred_emb_dim is not None: final_args['meta']['pred_emb_dim'] = cli_params.pred_emb_dim

    if cli_params.up_checkpoint_path is not None:  # CLI path for checkpoint implies loading
        final_args['meta']['load_up_checkpoint'] = True
        final_args['meta']['up_checkpoint'] = cli_params.up_checkpoint_path
    if cli_params.load_optimizer_state is not None:  # CLI override for loading optimizer
        final_args['meta']['load_opt_up'] = cli_params.load_optimizer_state

    # Data
    if cli_params.batch_size is not None: final_args['data']['batch_size'] = cli_params.batch_size
    if cli_params.pin_memory is not None: final_args['data']['pin_mem'] = cli_params.pin_memory
    if cli_params.num_workers is not None: final_args['data']['num_workers'] = cli_params.num_workers
    if cli_params.spec_path is not None: final_args['data']['spec_path'] = cli_params.spec_path

    # Mask
    if cli_params.num_tgt_blk is not None: final_args['mask']['num_tgt_blk'] = cli_params.num_tgt_blk
    if cli_params.tgt_p_len is not None: final_args['mask']['tgt_p_len'] = cli_params.tgt_p_len
    if cli_params.ctx_p_len is not None: final_args['mask']['ctx_p_len'] = cli_params.ctx_p_len

    # Optimization
    if cli_params.epochs is not None: final_args['optimization']['epochs_up'] = cli_params.epochs
    if cli_params.ema is not None: final_args['optimization']['ema'] = cli_params.ema
    if cli_params.ipe_scale is not None: final_args['optimization']['ipe_scale'] = cli_params.ipe_scale
    if cli_params.wd is not None: final_args['optimization']['weight_decay'] = cli_params.wd
    if cli_params.final_wd is not None: final_args['optimization']['final_weight_decay'] = cli_params.final_wd
    if cli_params.warmup_epochs is not None: final_args['optimization']['warmup'] = cli_params.warmup_epochs
    if cli_params.start_lr is not None: final_args['optimization']['start_lr'] = cli_params.start_lr
    if cli_params.lr is not None: final_args['optimization']['lr'] = cli_params.lr
    if cli_params.final_lr is not None: final_args['optimization']['final_lr'] = cli_params.final_lr

    if cli_params.log_freq is not None: final_args['logging']['log_freq'] = cli_params.log_freq
    if cli_params.checkpoint_freq is not None: final_args['logging']['checkpoint_freq'] = cli_params.checkpoint_freq

    print("--- Effective Configuration (after CLI overrides) ---")
    pprint(final_args)  # For debugging, uses the imported pprint
    print("-----------------------------------------------------")

    return final_args


def main(args):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS
    # ----------------------------------------------------------------------- #

    # -- META
    _GLOBAL_SEED = args['meta']['seed']
    use_bfloat16 = args['meta']['use_bfloat16']

    model_size = args['meta']['model_size']
    patch_size = args['meta']['patch_size']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']

    load_model = args['meta']['load_up_checkpoint']
    load_opt_up = args['meta']['load_opt_up']
    up_file = args['meta']['up_checkpoint']

    # --
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger()

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    spec_path = args['data']['spec_path']

    # --MASK
    num_tgt_blk = args['mask']['num_tgt_blk']
    tgt_p_len = args['mask']['tgt_p_len']
    ctx_p_len = args['mask']['ctx_p_len']

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs_up']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    tag = model_size
    log_freq = args['logging']['log_freq']
    checkpoint_freq = args['logging']['checkpoint_freq']

    folder = project_root / 'log' / 'log_up/'
    if not os.path.exists(project_root / 'log'):
        os.mkdir(project_root / 'log')
    if not os.path.exists(folder):
        os.mkdir(folder)

    dump = os.path.join(folder, f'params_up_{tag}.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    up_path = None
    if load_model:
        up_path = os.path.join(folder, up_file) if up_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.6f', 'loss'),
                           ('%d', 'time (ms)'))

    # -- init model
    up_model, _ = init_model(
        device=device,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_size=model_size,
        patch_size=patch_size
    )
    encoder, embedder, predictor = up_model.encoder, up_model.embedder, up_model.predictor
    target_encoder = copy.deepcopy(encoder)

    # -- init data-loaders/samplers
    csv_file = project_root / 'data' / spec_path
    _, dataloader = create_dataset_up(csv_file, batch_size, pin_mem, num_workers)
    ipe = len(dataloader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        module_list=[embedder, encoder, predictor],
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
                          for i in range(int(ipe * num_epochs * ipe_scale) + 1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        up_model, _ = load_checkpoint(
            up_path=up_path,
            up_model=up_model)

    def save_checkpoint(epoch):
        save_dict = {
            'embedder': embedder.state_dict(),
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, x in enumerate(dataloader):
            x = x.to(device)
            x, mask = embedder(x)

            # tgt_indices: tensor (batch_size*num_tgt_blk, tgt_p_len)
            # ctx_indices: tensor (batch_size, ctx_p_len)
            tgt_indices, ctx_indices = sample(x, num_tgt_blk, tgt_p_len, ctx_p_len, mask)

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target(x, tgt_indices):
                    with torch.no_grad():
                        all_rep = target_encoder(x, mask=mask)
                        all_rep = all_rep.repeat_interleave(num_tgt_blk, dim=0)

                        # -- filter sample mask
                        sampling_mask = generate_sampling_mask(tgt_indices)
                        all_rep = all_rep[sampling_mask]
                        tgt_indices_filter = tgt_indices[sampling_mask]

                        # -- create targets (masked regions of h)
                        tgt_indices_ = tgt_indices_filter.unsqueeze(-1).expand(-1, -1, all_rep.size(-1))
                        tgt_rep = torch.gather(all_rep, 1, tgt_indices_.to(all_rep.device))
                    return tgt_rep, tgt_indices_, sampling_mask

                def forward_context(x, ctx_indices):
                    ctx_indices_ = ctx_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
                    ctx_rep = torch.gather(x, 1, ctx_indices_.to(x.device))
                    ctx_rep = encoder(ctx_rep, indices=ctx_indices)
                    return ctx_rep, ctx_indices_

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    return loss

                # Step 1. Forward
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_bfloat16):
                    tgt_rep, tgt_indices_, sampling_mask = forward_target(x, tgt_indices)
                    ctx_rep, ctx_indices_ = forward_context(x, ctx_indices)
                    tgt_shp = tgt_rep.shape
                    pred_rep = predictor(ctx_rep, tgt_shp, ctx_indices, tgt_indices, num_tgt_blk, sampling_mask)
                    loss = loss_fn(pred_rep, tgt_rep)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.6f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024. ** 2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.6f' % loss_meter.avg)
        save_checkpoint(epoch)


if __name__ == '__main__':
    args = setup_args()
    main(args)
