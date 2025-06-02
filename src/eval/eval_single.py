# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os.path
import sys
import argparse
import copy
from pathlib import Path
from pprint import pprint
import yaml
import numpy as np
import torch

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.helper import load_checkpoint
from src.modules import UpModel, DownModel
from src.datas import create_dataset_down
from src.eval import save_results


def setup_args():
    parser = argparse.ArgumentParser(
        'S-JEPA Single-Trait Evaluation (eval_single.py)',
        description='Script for evaluating a pre-trained S-JEPA model on a single downstream trait.',
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
    g_data.add_argument('--batch_size', type=int, default=400, help='Batch size for evaluation.')
    g_data.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers.')
    g_data.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=False, help='Enable/disable pinning CPU memory.')
    g_data.add_argument('--trait_name', type=str, default='CHL',
                        help='Name of the trait to evaluate (e.g., LMA, CHL). This will also be used to select the downstream checkpoint if not explicitly provided.')
    g_data.add_argument('--proportion', type=int, default=100, help='Percentage of training data used for the model being evaluated (e.g., 100).')
    g_data.add_argument('--fold_idx', type=int, default=0, help='Fold index of the model being evaluated.')
    g_data.add_argument('--n_splits', type=int, default=5, help='Total number of splits used during training (for dataset consistency).')

    # --- Checkpoint Parameters ---
    g_ckpt = parser.add_argument_group('Checkpoint Parameters')
    g_ckpt.add_argument('--up_checkpoint_path', type=str, default=None,
                        help='Path/filename of the pretrained upstream model checkpoint. Overrides YAML.')
    g_ckpt.add_argument('--down_checkpoint_path', type=str, default=None,
                        help='Path/filename of the downstream model checkpoint for the specified trait. Overrides YAML.')

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
    if cli_params.trait_name is not None: final_args['data']['trait_name'] = cli_params.trait_name
    if cli_params.proportion is not None: final_args['data']['proportion'] = cli_params.proportion
    if cli_params.fold_idx is not None: final_args['data']['fold_idx'] = cli_params.fold_idx
    if cli_params.n_splits is not None: final_args['data']['n_splits'] = cli_params.n_splits

    # Checkpoints
    if cli_params.up_checkpoint_path is not None:
        final_args['ckpt']['up_checkpoint'] = cli_params.up_checkpoint_path
    if cli_params.down_checkpoint_path is not None:
        final_args['ckpt']['down_checkpoint'] = cli_params.down_checkpoint_path
    elif final_args['data'].get('trait_name') and final_args['ckpt'].get('down_checkpoint_dict'):
        # If down_checkpoint_path is not given via CLI, but trait_name is set,
        # try to get the specific downstream checkpoint from down_checkpoint_dict
        # This behavior mimics how one might use the YAML's down_checkpoint_dict.
        trait_for_ckpt = final_args['data']['trait_name']
        if trait_for_ckpt in final_args['ckpt']['down_checkpoint_dict']:
            final_args['ckpt']['down_checkpoint'] = final_args['ckpt']['down_checkpoint_dict'][trait_for_ckpt]
            print(f"INFO: Using downstream checkpoint '{final_args['ckpt']['down_checkpoint']}' for trait '{trait_for_ckpt}' from 'down_checkpoint_dict'.")
        elif 'down_checkpoint' not in final_args['ckpt']: # if no specific CLI and not in dict and not already in YAML's top 'down_checkpoint'
             print(f"WARNING: No specific downstream checkpoint path provided via CLI for trait '{trait_for_ckpt}', and not found in 'down_checkpoint_dict'. Ensure 'ckpt.down_checkpoint' is correctly set in YAML or CLI.")

    print("--- Effective Configuration (eval_single.py) ---")
    pprint(final_args)
    print("-------------------------------------------------")

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
    trait_name = args['data']['trait_name']
    proportion = args['data']['proportion']
    fold_idx = args['data']['fold_idx']
    n_splits = args['data']['n_splits']

    # -- CKPT
    up_checkpoint = args['ckpt']['up_checkpoint']
    down_checkpoint = args['ckpt']['down_checkpoint']

    # ----------------------------------------------------------------------- #
    # -- load data
    dataset, train_loader, test_loader = create_dataset_down(
        trait_name=trait_name,
        up_model=None,
        return_uid=True,
        batch_size=batch_size,
        proportion=proportion,
        fold_idx=fold_idx,
        n_splits=n_splits,
        random_state=_GLOBAL_SEED,
        pin_mem=pin_mem,
        num_workers=num_workers)

    # -- init model
    task_configs = [{
        "name": trait_name,
        "norm_scaler": None,
    }]
    up_model = UpModel(
        model_size=model_size,
        patch_size=patch_size
    )
    down_model = DownModel(
        up_model.embed_dim,
        up_model.num_heads,
        task_configs
    )

    def load_calculate_save(up_model, down_model):
        # -- load checkpoint
        up_path = project_root / 'log' / 'log_up' / up_checkpoint
        down_path = project_root / 'log' / 'log_down' / down_checkpoint

        up_model, down_model = load_checkpoint(
            up_path=up_path,
            up_model=up_model,
            down_path=down_path,
            down_model=down_model,
        )

        up_model.to(device).eval()
        down_model.to(device).eval()

        # -- forward
        ob_list, pr_list, uid_tuple_list = [], [], []
        mean, iqr = down_model.tasks_modules[trait_name].norm_scaler
        with torch.no_grad():
            for spec, trait, uid in test_loader:
                spec = spec.to(device)
                trait = trait.to(device)

                x = up_model(spec)
                p = down_model(x, trait_name=trait_name, denorm=True).squeeze(1)

                o = trait.squeeze(1) * iqr + mean

                ob_list.append(o)
                pr_list.append(p)
                uid_tuple_list.append(uid)

        observed = torch.cat(ob_list).cpu().detach().numpy()
        predicted = torch.cat(pr_list).cpu().detach().numpy()
        uid_list = [item for tpl in uid_tuple_list for item in tpl]

        output_folder = project_root / 'log' / 'results'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output_dir = output_folder / f'{trait_name}_fid{fold_idx}_ppt{proportion}.csv'
        save_results(observed, predicted, uid_list, output_dir)

    load_calculate_save(up_model, down_model)


if __name__ == '__main__':
    args = setup_args()
    main(args)
