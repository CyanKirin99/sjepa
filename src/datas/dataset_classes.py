# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from src.modules import FilterResampler


class UpDataset(Dataset):
    def __init__(self, csv_file, preprocess=FilterResampler()):
        self.preprocess = preprocess
        df = pd.read_csv(csv_file).iloc[:, 1:]
        self.data = self._preprocess(df)

    def _preprocess(self, df):
        arr = np.array(df.values)
        return self.preprocess(arr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x


class DownDataset(Dataset):
    def __init__(self, spec_csv, trait_csv, trait_name, up_model=None, return_uid=False, norm_scaler=None):
        self.preprocess = FilterResampler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.up_model = None if up_model is None else up_model.to(self.device)

        self.trait_name = trait_name

        try:
            spec_data = pd.read_csv(spec_csv)
            trait_data = pd.read_csv(trait_csv)
        except ValueError:
            print('Only one csv file is supported when training!')
            exit()

        # 筛选含目标标签的行，并对齐光谱数据和标签
        merged_data = pd.merge(spec_data, trait_data[['uid', trait_name]].dropna(), on='uid', how='inner')

        self.uid = merged_data['uid'].values
        self.raw_spec = merged_data.drop(columns=['uid', trait_name]).values
        self.trait = merged_data[trait_name].values

        if norm_scaler is not None:
            self.norm_scaler = norm_scaler
            self.normalized_trait, _ = self._normalize_label(norm_scaler)
        else:
            self.normalized_trait, self.norm_scaler = self._normalize_label()
        self.spec = self._preprocess_spec(self.raw_spec)

        self.return_uid = return_uid

    def _normalize_label(self, norm_scaler=None):
        if norm_scaler is None:
            median = np.median(self.trait)
            iqr = np.percentile(self.trait, 75) - np.nanpercentile(self.trait, 25)
            iqr = iqr if iqr > 0 else 1e-5  # 避免 IQR 为 0
            norm_scaler = (median, iqr)
        else:
            median, iqr = norm_scaler
        normalized_labels = (self.trait - median) / iqr
        return torch.tensor(normalized_labels, dtype=torch.float32).unsqueeze(1), norm_scaler

    def _preprocess_spec(self, raw_spec):
        """
        将预处理后的数据存储在内存中（重采样、编码器特征）
        :return x: tensor [sample_num, max_num_valid_tokens, embed_dim]
        :return mask: tensor [sample_num, max_num_valid_tokens]
        """
        spec = self.preprocess(raw_spec)
        if self.up_model is not None:
            with torch.no_grad():
                spec = torch.tensor(spec, dtype=torch.float32).to(self.device)
                spec = self.up_model(spec)
            return spec.detach().cpu()
        else:
            return spec

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        uid = self.uid[idx]
        spec = self.spec[idx]
        trait = self.normalized_trait[idx]

        if self.return_uid:
            return spec, trait, uid

        return spec, trait
