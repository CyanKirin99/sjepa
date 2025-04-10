# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
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
    config_file = current_dir / 'configs_train.yaml'
    with open(config_file, 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)

    main(args)

