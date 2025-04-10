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
import torch

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.helper import load_checkpoint
from src.modules import UpModel, DownModel
from src.datas import create_dataset_down
from src.eval import save_results


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

        output_dir = project_root / 'log' / 'results' / f'{trait_name}_fid{fold_idx}_ppt{proportion}.csv'
        save_results(observed, predicted, uid_list, output_dir)

    load_calculate_save(up_model, down_model)


if __name__ == '__main__':
    config_file = current_dir / 'configs_eval.yaml'
    with open(config_file, 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)
    pprint(args)

    main(args)
