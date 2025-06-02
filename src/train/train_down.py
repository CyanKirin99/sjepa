# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
import argparse
import copy
from pathlib import Path
import logging
import yaml
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.helper import init_model, init_opt, load_checkpoint
from src.utils import gpu_timer, CSVLogger, AverageMeter
from src.datas import create_dataset_down


def setup_args():
    parser = argparse.ArgumentParser(
        'S-JEPA Downstream Training (train_down.py)',
        description='Script for fine-tuning or linear probing S-JEPA features on downstream tasks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in --help
    )

    # --- Configuration File ---
    parser.add_argument('--config_file', default=str(current_dir / 'configs_train.yaml'), type=str,
                        help='Path to the YAML configuration file.')

    # --- Meta Parameters ---
    g_meta = parser.add_argument_group('Meta Parameters')
    g_meta.add_argument('--seed', type=int, default=42, help='Global random seed.')
    # For Python 3.9+ for BooleanOptionalAction.
    g_meta.add_argument('--use_bfloat16', action=argparse.BooleanOptionalAction, default=False, help='Enable/disable bfloat16 precision.')
    g_meta.add_argument('--model_size', type=str, default='large', choices=['tiny', 'small', 'base', 'large'], help='Base model size (must match upstream).')
    g_meta.add_argument('--patch_size', type=int, default=30, help='Patch size (must match upstream).')
    g_meta.add_argument('--up_checkpoint_path', type=str, default=None,
                        help='Path to the pretrained upstream model checkpoint (.pth.tar file). Overrides YAML.')
    g_meta.add_argument('--cache_feature', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable/disable caching features from the upstream model.')

    # --- Data Parameters ---
    g_data = parser.add_argument_group('Data Parameters')
    g_data.add_argument('--batch_size', type=int, default=400, help='Batch size per GPU for downstream task.')
    g_data.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers.')
    g_data.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=False, help='Enable/disable pinning CPU memory.')
    g_data.add_argument('--trait_name', type=str, default='CHL', help='Name of the trait to train on (e.g., CHL, LMA).')
    g_data.add_argument('--proportion', type=int, default=100, help='Percentage of training data to use (e.g., 100 for 100%%).') # Can also be float if needed
    g_data.add_argument('--fold_idx', type=int, default=0, help='Fold index for cross-validation.')
    g_data.add_argument('--n_splits', type=int, default=5, help='Total number of splits for cross-validation.')

    # --- Optimization Parameters (for downstream task) ---
    g_opt = parser.add_argument_group('Optimization Parameters')
    g_opt.add_argument('--epochs', type=int, default=500, help='Total training epochs for downstream task (overrides epochs_down).')
    g_opt.add_argument('--lr', type=float, default=0.0001, help='Base learning rate for downstream optimizer.')
    g_opt.add_argument('--wd', type=float, default=0.04, help='Weight decay for downstream optimizer.')
    g_opt.add_argument('--start_lr', type=float, default=0.00002, help='Initial learning rate for warmup.')
    g_opt.add_argument('--final_lr', type=float, default=1.0e-06, help='Final learning rate.')
    g_opt.add_argument('--final_wd', type=float, default=0.4, help='Final weight decay.')
    g_opt.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs.')
    g_opt.add_argument('--ipe_scale', type=float, default=1.0, help='IPE scale factor for scheduler.')

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
        print(f"WARNING: Configuration file {cli_params.config_file} not found. Using command-line defaults or script defaults.")
        args_from_yaml = {}

    final_args = copy.deepcopy(args_from_yaml)

    # Ensure basic structure exists
    final_args.setdefault('meta', {})
    final_args.setdefault('data', {})
    final_args.setdefault('optimization', {})
    final_args.setdefault('logging', {})

    # 2. Override YAML with CLI parameters if they were provided
    # Meta
    if cli_params.seed is not None: final_args['meta']['seed'] = cli_params.seed
    if cli_params.use_bfloat16 is not None: final_args['meta']['use_bfloat16'] = cli_params.use_bfloat16
    if cli_params.model_size is not None: final_args['meta']['model_size'] = cli_params.model_size
    if cli_params.patch_size is not None: final_args['meta']['patch_size'] = cli_params.patch_size
    if cli_params.up_checkpoint_path is not None: # This CLI arg directly sets the path for 'up_checkpoint'
        final_args['meta']['up_checkpoint'] = cli_params.up_checkpoint_path
    if cli_params.cache_feature is not None: final_args['meta']['cache_feature'] = cli_params.cache_feature

    # Data
    if cli_params.batch_size is not None: final_args['data']['batch_size'] = cli_params.batch_size
    if cli_params.num_workers is not None: final_args['data']['num_workers'] = cli_params.num_workers
    if cli_params.pin_memory is not None: final_args['data']['pin_mem'] = cli_params.pin_memory
    if cli_params.trait_name is not None: final_args['data']['trait_name'] = cli_params.trait_name
    if cli_params.proportion is not None: final_args['data']['proportion'] = cli_params.proportion
    if cli_params.fold_idx is not None: final_args['data']['fold_idx'] = cli_params.fold_idx
    if cli_params.n_splits is not None: final_args['data']['n_splits'] = cli_params.n_splits

    # Optimization
    if cli_params.epochs is not None: final_args['optimization']['epochs_down'] = cli_params.epochs
    if cli_params.lr is not None: final_args['optimization']['lr'] = cli_params.lr
    if cli_params.wd is not None: final_args['optimization']['weight_decay'] = cli_params.wd
    if cli_params.start_lr is not None: final_args['optimization']['start_lr'] = cli_params.start_lr
    if cli_params.final_lr is not None: final_args['optimization']['final_lr'] = cli_params.final_lr
    if cli_params.final_wd is not None: final_args['optimization']['final_weight_decay'] = cli_params.final_wd
    if cli_params.warmup_epochs is not None: final_args['optimization']['warmup'] = cli_params.warmup_epochs
    if cli_params.ipe_scale is not None: final_args['optimization']['ipe_scale'] = cli_params.ipe_scale

    # Logging
    if cli_params.log_freq is not None: final_args['logging']['log_freq'] = cli_params.log_freq
    if cli_params.checkpoint_freq is not None: final_args['logging']['checkpoint_freq'] = cli_params.checkpoint_freq

    print("--- Effective Configuration (train_down.py) ---")
    pprint(final_args)
    print("-----------------------------------------------")

    return final_args


def main(args):
    pprint(args)
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    _GLOBAL_SEED = args['meta']['seed']
    use_bfloat16 = args['meta']['use_bfloat16']

    model_size = args['meta']['model_size']
    patch_size = args['meta']['patch_size']
    up_checkpoint = args['meta']['up_checkpoint']

    cache_feature = args['meta']['cache_feature']

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # --
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger()

    # -- DATA
    batch_size = args['data']['batch_size']
    num_workers = args['data']['num_workers']
    pin_mem = args['data']['pin_mem']
    trait_name = args['data']['trait_name']
    proportion = args['data']['proportion']
    fold_idx = args['data']['fold_idx']
    n_splits = args['data']['n_splits']

    # -- OPTIMIZATION
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    num_epochs = args['optimization']['epochs_down']

    # -- LOGGING
    log_freq = args['logging']['log_freq']
    checkpoint_freq = args['logging']['checkpoint_freq']

    tag = f'{model_size}_{trait_name}_fid{fold_idx}_ppt{proportion}'
    folder = project_root / 'log' / 'log_down'
    if not os.path.exists(project_root / 'log'):
        os.mkdir(project_root / 'log')
    if not os.path.exists(folder):
        os.mkdir(folder)
    up_path = project_root / 'log' / 'log_up' / up_checkpoint

    dump = os.path.join(folder, f'params_{tag}.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # ----------------------------------------------------------------------- #
    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.6f', 'loss'),
                           ('%.3f', 'time (ms)'))

    # -- init & load up_model
    up_model, _ = init_model(
        device=device,
        model_size=model_size,
        patch_size=patch_size
    )
    up_model, _ = load_checkpoint(up_path, up_model)
    up_model.eval()
    for p in up_model.parameters():
        p.requires_grad = False

    # -- init data-loaders
    up_model_ = up_model if cache_feature else None
    dataset, train_loader, test_loader = create_dataset_down(
        trait_name=trait_name,
        up_model=up_model_,
        return_uid=False,
        batch_size=batch_size,
        proportion=proportion,
        fold_idx=fold_idx,
        n_splits=n_splits,
        random_state=_GLOBAL_SEED,
        pin_mem=pin_mem,
        num_workers=num_workers)
    norm_scaler = dataset.norm_scaler
    ipe = len(train_loader)

    # -- init down_model
    task_configs = [{
        "name": trait_name,
        "norm_scaler": norm_scaler
    }]
    _, down_model = init_model(
        device=device,
        model_size=model_size,
        patch_size=patch_size,
        task_configs=task_configs
    )

    # -- init optimization and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        module_list=[down_model],
        iterations_per_epoch=ipe,
        start_lr=start_lr,
        ref_lr=lr,
        warmup=warmup,
        num_epochs=num_epochs,
        wd=wd,
        final_wd=final_wd,
        final_lr=final_lr,
        use_bfloat16=use_bfloat16,
        ipe_scale=ipe_scale
    )

    start_epoch = 0
    def save_checkpoint(epoch):
        save_dict = {
            'down_model': down_model.state_dict()
        }
        torch.save(save_dict, latest_path)
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logging.info(f'Epoch {epoch + 1}')

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (spec, trait) in enumerate(train_loader):
            spec = spec.to(device)
            trait = trait.to(device)

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def loss_fn(o, t):
                    return F.mse_loss(o, t)

                # Step 1. Forward
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_bfloat16):
                    x = spec if cache_feature else up_model(spec)
                    output = down_model(x, trait_name=trait_name)
                    loss = loss_fn(output, trait)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()

                return (float(loss), _new_lr, _new_wd)

            (loss, _new_lr, _new_wd), etime = gpu_timer(train_step)
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

            log_stats()
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.6f' % loss_meter.avg)
        save_checkpoint(epoch)


if __name__ == '__main__':
    args = setup_args()
    main(args)
