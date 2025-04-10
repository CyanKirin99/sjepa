# Copyright (c) Chenye Su, Wuhan University.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(observed, predicted):
    mask = ~np.isnan(observed) & ~np.isnan(predicted)
    observed = observed[mask]
    predicted = predicted[mask]

    rmse = root_mean_squared_error(observed, predicted)
    metrics = {
        "RMSE": rmse,
        "R^2": r2_score(observed, predicted),
        "MAE": mean_absolute_error(observed, predicted),
        "MAPE": (abs((observed - predicted) / observed).mean()) * 100,
        "NRMSE": rmse / (max(observed) - min(observed)),
        "RRMSE": rmse / observed.mean()
    }
    return metrics


def save_results(observed, predicted, uid_list, output_dir):
    metrics = calculate_metrics(observed, predicted)

    results_data = pd.DataFrame({
        "ID": uid_list + list(metrics.keys()),
        "Observed": list(observed) + ["N/A"] * len(metrics),
        "Predicted": list(predicted) + list(metrics.values())
    })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_dir, index=False)

