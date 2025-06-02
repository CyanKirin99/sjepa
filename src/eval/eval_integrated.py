# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import argparse
import copy
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


def setup_args():
    parser = argparse.ArgumentParser(
        'S-JEPA Integrated Multi-Trait Evaluation (eval_integrated.py)',
        description='Script for predicting multiple leaf traits simultaneously using a pre-trained S-JEPA model and corresponding downstream heads.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Configuration File ---
    parser.add_argument('--config_file', default=str(current_dir / 'configs_eval.yaml'), type=str,
                        help='Path to the YAML configuration file for evaluation.')

    # --- Meta Parameters ---
    g_meta = parser.add_argument_group('Meta Parameters')
    g_meta.add_argument('--seed', type=int, default=42, help='Global random seed.')
    g_meta.add_argument('--model_size', type=str, default='large', choices=['tiny', 'small', 'base', 'large'],
                        help='Base model size (must match upstream and downstream models).')
    g_meta.add_argument('--patch_size', type=int, default=30, help='Patch size (must match models).')

    # --- Data Parameters ---
    g_data = parser.add_argument_group('Data Parameters')
    g_data.add_argument('--batch_size', type=int, default=400, help='Batch size for prediction.')
    g_data.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers.')
    g_data.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=False, help='Enable/disable pinning CPU memory.')
    g_data.add_argument('--spec_path', type=str, default=None,
                        help='Path to the spectral data CSV file for prediction (e.g., spec_demo.csv). Relative to project_root/data or absolute/relative to CWD.')
    g_data.add_argument('--trait_name_list', type=str, nargs='+', default=None,
                        help='List of trait names to predict (e.g., LMA CHL Water). Overrides YAML list.')

    # --- Checkpoint Parameters ---
    g_ckpt = parser.add_argument_group('Checkpoint Parameters')
    g_ckpt.add_argument('--up_checkpoint_path', type=str, default=None,
                        help='Path/filename of the pretrained upstream model checkpoint. Overrides YAML.')
    # Note: down_checkpoint_dict is complex to override via CLI. It's best managed in the YAML file.
    # This script will use the down_checkpoint_dict from the loaded YAML.

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
    final_args.setdefault('ckpt', {})

    # 2. Override YAML with CLI parameters
    # Meta
    if cli_params.seed is not None: final_args['meta']['seed'] = cli_params.seed
    if cli_params.model_size is not None: final_args['meta']['model_size'] = cli_params.model_size
    if cli_params.patch_size is not None: final_args['meta']['patch_size'] = cli_params.patch_size

    # Data
    if cli_params.batch_size is not None: final_args['data']['batch_size'] = cli_params.batch_size
    if cli_params.num_workers is not None: final_args['data']['num_workers'] = cli_params.num_workers
    if cli_params.pin_memory is not None: final_args['data']['pin_mem'] = cli_params.pin_memory
    if cli_params.spec_path is not None: final_args['data']['spec_path'] = cli_params.spec_path
    if cli_params.trait_name_list is not None: final_args['data']['trait_name_list'] = cli_params.trait_name_list

    # Checkpoints
    if cli_params.up_checkpoint_path is not None:
        final_args['ckpt']['up_checkpoint'] = cli_params.up_checkpoint_path
    # down_checkpoint_dict is taken from YAML. If specific overrides are needed, modify the YAML or use a different config file.

    # Ensure critical keys from YAML are present if not overridden, or provide defaults
    if 'down_checkpoint_dict' not in final_args['ckpt']:
        print("WARNING: 'ckpt.down_checkpoint_dict' not found in configuration. Downstream models may not load correctly.")
        final_args['ckpt']['down_checkpoint_dict'] = {} # Avoid KeyError later

    print("--- Effective Configuration (eval_integrated.py) ---")
    pprint(final_args)
    print("----------------------------------------------------")

    return final_args


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
        trait: down_dir / ckpt for trait, ckpt in down_checkpoint_dict.items() if trait in trait_name_list
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
    output_folder = project_root / 'log' / 'results'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_df.to_csv(output_folder / f'integrated_output.csv')


if __name__ == '__main__':
    args = setup_args()
    main(args)