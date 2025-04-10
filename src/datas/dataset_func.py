# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.datas import UpDataset, DownDataset


def create_dataset_up(
        csv_file,
        batch_size,
        pin_mem=True,
        num_workers=0,
        shuffle=True,
):
    dataset = UpDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_mem, num_workers=num_workers)
    return dataset, dataloader


def create_dataset_down(trait_name, up_model=None, return_uid=False, norm_scaler=None, batch_size=400,
                        proportion=100, fold_idx=0, n_splits=5, random_state=42, pin_mem=True, num_workers=0):
    spec_csv = project_root / 'data' / 'spec_demo.csv'
    trait_csv = project_root / 'data' / 'trait_demo.csv'

    dataset = DownDataset(spec_csv, trait_csv, trait_name, up_model, return_uid=return_uid, norm_scaler=norm_scaler)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_indices = list(kf.split(dataset))

    train_idx, test_idx = all_indices[fold_idx]
    train_dataset_ori = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_size = int(len(train_dataset_ori) * proportion * 0.01)
    shelved_size = len(train_dataset_ori) - train_size
    train_dataset, _ = random_split(train_dataset_ori, [train_size, shelved_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=pin_mem, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=pin_mem, num_workers=num_workers)

    return dataset, train_loader, test_loader
