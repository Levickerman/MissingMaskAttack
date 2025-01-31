import pandas as pd
import numpy as np
import os
import json

import torch
from torch.utils.data import Dataset, DataLoader

from pygrinder import mcar, masked_fill
from utils.getBalancedData import getBalancedData

def getBalancedData(raw_x, raw_y, label=1):
    idx_1 = (raw_y == label)
    data_1 = raw_x[idx_1, :, :].clone() # [N, n_step, n_feature]
    num_1 = torch.sum(idx_1).detach().item()
    y_1 = torch.ones([num_1]) * label
    rate = round((raw_x.shape[0] - num_1) / num_1) # time to repeat
    balanced_data = raw_x.clone()
    argument_data = data_1.repeat([rate, 1, 1])
    balanced_data = torch.cat([balanced_data, argument_data], dim=0)
    balanced_y = torch.cat([raw_y, y_1.repeat(rate)], dim=0).long()
    print("get balanced data of size: %d" % balanced_data.shape[0])
    return {"X": balanced_data, "y": balanced_y}

def getDataInfo(data_root_dir, data_type):
    info_dir = os.path.join(data_root_dir, data_type, "info.json")
    with open(info_dir, encoding='utf-8') as f:
        data_info = json.load(f)
    return data_info

def getRawData(data_root_dir, data_type):
    if data_type in ["Physionet2012", "mimic3"]:
        data_info = dict()
        listfile_path = os.path.join(data_root_dir, data_type, "listfile.csv")
        listf = pd.read_csv(listfile_path).to_numpy()
        raw_data = []
        for ts_filename in listf[:, 0]:
            raw_data.append(pd.read_csv(os.path.join(data_root_dir, data_type, ts_filename), header=None).to_numpy())
        raw_data = np.array(raw_data)  # [num_samples, n_steps, n_features]
        data = dict()
        data["X"] = torch.Tensor(raw_data)
        data["y"] = torch.Tensor(listf[:, 1].astype(int))
        data_info["ts_length"], data_info["ts_dimension"] = data["X"].shape[1], data["X"].shape[2]
        data_info["num_class"] = 2
    else:
        info_dir = os.path.join(data_root_dir, data_type, "info.json")
        with open(info_dir, encoding='utf-8') as f:
            data_info = json.load(f)
        data = None

    return data, data_info

def cloneDataset(dataset):
    return {"X": dataset["X"].clone(), "y": dataset["y"].clone()}

def splitDataset(data_type, data, val_missing_rate=0.1, only_test=False, only_train=False):
    n_samples = data["X"].shape[0]
    train_idx, val_idx, test_idx = (np.arange(0, int(n_samples * 0.7)),
                                    np.arange(int(n_samples * 0.7), int(n_samples * 0.9)),
                                    np.arange(int(n_samples * 0.9), n_samples))

    train_set, val_set, test_set = dict(), dict(), dict()
    train_set["X"], train_set["y"] = data["X"][train_idx], data["y"][train_idx]

    val_set["X"], val_set["y"] = data["X"][val_idx], data["y"][val_idx]
    corrupted_X = mcar(val_set["X"], val_missing_rate)
    val_set["X_ori"] = corrupted_X

    test_set["X"], test_set["y"] = data["X"][test_idx], data["y"][test_idx]
    if only_test:
        return {"X": data["X"][np.append(train_idx, val_idx)], "y":data["y"][np.append(train_idx, val_idx)]}, test_set
    if only_train:
        return {"X": data["X"][np.append(train_idx, test_idx)], "y": data["y"][np.append(train_idx, test_idx)]}, val_set
    else:
        return train_set, val_set, test_set

def getMeanandStd(data):
    mean = torch.mean(torch.nan_to_num(data["X"]), dim=(0, 1))
    std = torch.std(torch.nan_to_num(data["X"]), dim=(0, 1))
    return mean, std