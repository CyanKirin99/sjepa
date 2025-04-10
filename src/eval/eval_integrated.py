# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from pathlib import Path
from pprint import pprint
import yaml
import numpy as np
import pandas as pd
import torch

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.datas import create_dataset_up
from src.helper import init_model, load_checkpoint


def main(args):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    _GLOBAL_SEED = args['meta']['seed']

    model_size = args['meta']['model_size']
    patch_size = args['meta']['patch_size']

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # --
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    # -- DATA
    batch_size = args['data']['batch_size']
    num_workers = args['data']['num_workers']
    pin_mem = args['data']['pin_mem']
    spec_path = args['data']['spec_path']
    trait_name_list = args['data']['trait_name_list']

    # -- CKPT
    up_checkpoint = args['ckpt']['up_checkpoint']
    down_checkpoint_dict = args['ckpt']['down_checkpoint_dict']

    # ----------------------------------------------------------------------- #
    # -- init data-loaders
    csv_file = project_root / 'data' / spec_path
    _, dataloader = create_dataset_up(csv_file, batch_size, pin_mem, num_workers)

    # init model
    task_configs = [
        {
            "name": trait_name,
            "norm_scaler": None,
        }
        for trait_name in trait_name_list
    ]

    # -- init model
    up_model, down_model = init_model(
        device=device,
        model_size=model_size,
        patch_size=patch_size,
        task_configs=task_configs
    )

    # load checkpoint
    up_path = project_root / 'log' / 'log_up' / up_checkpoint
    down_dir = project_root / 'log' / 'log_down'

    down_path_dict = {
        trait: down_dir / ckpt for trait, ckpt in down_checkpoint_dict.items()
    }
    up_model, down_model = load_checkpoint(
        up_path=up_path,
        up_model=up_model,
        down_path=down_path_dict,
        down_model=down_model,
    )

    up_model.to(device).eval()
    down_model.to(device).eval()

    # forward
    all_outputs = {trait: [] for trait in trait_name_list}
    with torch.no_grad():
        for i, spec in enumerate(dataloader):
            print(f'iter {i+1}/{len(dataloader)}')
            spec = spec.to(device)
            x = up_model(spec)
            outputs = down_model(x, denorm=True)
            for trait in trait_name_list:
                all_outputs[trait].append(outputs[trait])

    all_outputs = {trait: torch.cat(outputs).squeeze().detach().cpu().numpy() for trait, outputs in all_outputs.items()}
    output_df = pd.DataFrame(all_outputs)
    output_df.to_csv(project_root / 'log' / 'results' / f'integrated_output.csv')


if __name__ == '__main__':
    config_file = current_dir / 'configs_eval.yaml'
    with open(config_file, 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)
    pprint(args)

    main(args)