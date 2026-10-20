"""
[file]          dummy_model.py
"""
#
##
import time
import torch
import numpy as np
#
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from ptflops import get_model_complexity_info
from sklearn.metrics import classification_report, accuracy_score
#
from src.train import train
from configs.preset import preset
from src.utils import *
import wandb

class DUMMY(nn.Module):
    def __init__(self, var_x_shape, var_y_shape):
        super(DUMMY, self).__init__()

        # The input is flattened to one vector per sample (see forward), so every non-batch axis
        # contributes: (N, 3000, 3, 3, 30) -> 3000*3*3*30 = 810000 features. Using var_x_shape[1:]
        # dropped one axis (270 = 3*3*30) and never matched the flattened input.
        self.num_features = int(np.prod(var_x_shape))
        self.num_classes = int(np.prod(var_y_shape))

        # Stage-1 gate: empty vs non-empty
        self.empty_head = nn.Linear(self.num_features, 1)

    def forward(self, var_input):
        """
        Deterministic logic output vector with a,b = 1
        """
       
        x = var_input.reshape(var_input.size(0), -1)

        output = torch.zeros(
            x.size(0),
            self.num_classes, #5 for localization
            device=x.device,
            dtype=x.dtype
        )
        if self.num_classes >= 2:
            output[:, 0] = 1.0
            output[:, 1] = 1.0

        return output
#
##
def run_DUMMY(data_train_x,
              data_train_y,
              data_test_x,
              data_test_y,
              var_repeat=10, var_task="location", var_env="empty_room", save_path="./visualizations/temp"):
    #
    device = select_device()

    data_train_x = np.asarray(data_train_x)
    data_train_y = np.asarray(data_train_y)
    data_test_x = np.asarray(data_test_x)
    data_test_y = np.asarray(data_test_y)

    if data_train_x.ndim == 2:
        data_train_x = data_train_x[:, None, :]
    if data_test_x.ndim == 2:
        data_test_x = data_test_x[:, None, :]

    if data_train_y.ndim == 1:
        data_train_y = data_train_y[:, None]
    if data_test_y.ndim == 1:
        data_test_y = data_test_y[:, None]

    var_x_shape = data_train_x.shape[1:]

    var_y_shape = ([data_train_y.sum(axis=1).shape[1]] if data_train_y.ndim == 3
                   else [data_train_y.shape[1]])

    result = {}
    result_accuracy = []
    result_time_train = []
    result_time_test = []
    result_total_error = []
    result_precision = []
    result_recall = []
    result_f1_score = []

    for var_r in range(var_repeat):
        print("Repeat", var_r)

        var_time_0 = time.time()
        model_dummy = DUMMY(var_x_shape, var_y_shape).to(device)
        var_time_1 = time.time()

        with torch.no_grad():
            x = torch.from_numpy(data_test_x).float().to(device)
            predict_test_y = model_dummy(x).detach().cpu().numpy()

        var_time_2 = time.time()

        data_test_y_c = data_test_y.sum(axis=1)
        dict_true_acc = performance_metrics(data_test_y_c, predict_test_y, var_mode="count_classification")

        if wandb.run is not None:
            wandb.log({
                "repeat": var_r,
                "train_time": var_time_1 - var_time_0,
                "test_time": var_time_2 - var_time_1,
                "TOTAL_TESTSET_ERROR": dict_true_acc['total_error'],
                "TOTAL_TESTSET_perfect_prediction_percentage": dict_true_acc['perfect_prediction_percentage'],
                "TOTAL_ACCURACY": dict_true_acc['accuracy'],
                "mean_count_error": dict_true_acc['mean_count_error'],
                "precision": dict_true_acc['precision'],
                "recall": dict_true_acc['recall'],
                "f1_score": dict_true_acc['f1_score'],
            })

        result_accuracy.append(dict_true_acc['perfect_prediction_percentage'])
        result_time_train.append(var_time_1 - var_time_0)
        result_time_test.append(var_time_2 - var_time_1)
        result_total_error.append(dict_true_acc['total_error'])
        result_precision.append(dict_true_acc['precision'])
        result_recall.append(dict_true_acc['recall'])
        result_f1_score.append(dict_true_acc['f1_score'])

    ppp_array = np.array(result_accuracy)
    precision_array = np.array(result_precision)
    recall_array = np.array(result_recall)
    f1_array = np.array(result_f1_score)
    total_error_array = np.array(result_total_error)

    aggregated_result = {
        'avg_PPP': float(np.mean(ppp_array)),
        'avg_precision': float(np.mean(precision_array)),
        'avg_recall': float(np.mean(recall_array)),
        'avg_f1_score': float(np.mean(f1_array)),
        'avg_total_error': float(np.mean(total_error_array)),
        'std_PPP': float(np.std(ppp_array, ddof=1)) if len(ppp_array) > 1 else 0.0,
        'std_precision': float(np.std(precision_array, ddof=1)) if len(precision_array) > 1 else 0.0,
        'std_recall': float(np.std(recall_array, ddof=1)) if len(recall_array) > 1 else 0.0,
        'std_f1_score': float(np.std(f1_array, ddof=1)) if len(f1_array) > 1 else 0.0,
        'std_total_error': float(np.std(total_error_array, ddof=1)) if len(total_error_array) > 1 else 0.0,
        'se_PPP': float(np.std(ppp_array, ddof=1) / np.sqrt(len(ppp_array))) if len(ppp_array) > 1 else 0.0,
        'se_precision': float(np.std(precision_array, ddof=1) / np.sqrt(len(precision_array))) if len(precision_array) > 1 else 0.0,
        'se_recall': float(np.std(recall_array, ddof=1) / np.sqrt(len(recall_array))) if len(recall_array) > 1 else 0.0,
        'se_f1_score': float(np.std(f1_array, ddof=1) / np.sqrt(len(f1_array))) if len(f1_array) > 1 else 0.0,
        'se_total_error': float(np.std(total_error_array, ddof=1) / np.sqrt(len(total_error_array))) if len(total_error_array) > 1 else 0.0,
        'avg_train_time': float(np.mean(result_time_train)),
        'avg_test_time': float(np.mean(result_time_test)),
    }


    ## The head emits a per-location count vector (N, num_locations), exactly the target the
    ## count_classification metric scored above, so the figures must use the same mode and the same
    ## summed target. multi_head expects 3-D per-layer one-hot predictions (y_pred[-1] unpacked as
    ## batch x heads x classes) and crashed on the 2-D count vector.
    visualize_model_performance(
        y_pred=predict_test_y,
        y_true=data_test_y_c,
        var_mode="count_classification",
        save_dir=save_path
    )

    if wandb.run is not None:
        wandb.log({
            "avg_accuracy": aggregated_result['avg_PPP'],
            "avg_train_time": aggregated_result['avg_train_time'],
            "avg_test_time": aggregated_result['avg_test_time'],
            "avg_total_error": aggregated_result['avg_total_error'],
            "avg_precision": aggregated_result['avg_precision'],
            "avg_recall": aggregated_result['avg_recall'],
            "avg_f1_score": aggregated_result['avg_f1_score'],
        })

    return aggregated_result
