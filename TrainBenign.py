import argparse
import json

import torch.optim
import xlwt
from torch.utils.data import DataLoader

from extension.PatchTST.PatchTST import PatchTST
from extension.mTAN.mTAN import mTAN
from extension.iTransformer.iTransformer import iTransformer
from pypots.classification import GRUD, Raindrop, BRITS
from pypots.imputation import SAITS, CSDI, GPVAE, TimesNet
from pypots.optim import Adam
from utils.dataset_preprocess import getRawData, splitDataset, getSplitDataLoader, getDataInfo
from utils.generateStaticTrigger import generate_static_trigger
from utils.getBalancedData import getBalancedData
from utils.getClassifier import getClassifier
from utils.getMCARdata import modify2MCARdata

import os
import numpy as np

from utils.getModelandDatasetTransformer import getModelandDatasetTransformer
from utils.trainandtestClassifiers import train_model, test

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    data_root_dir = "data"
    MCAR_data_root = "data/MCAR"
    benign_root_dir = "results/benign_models"
    # datasets = ["Heartbeat", "FaceDetection", "InsectWingbeat", "LSST", "CharacterTrajectories", "Physionet2012", "mimic3"]
    # datasets = ["NATOPS", "EthanolConcentration"]
    # datasets = ["InsectWingbeat", "LSST", "CharacterTrajectories", "Physionet2012", "mimic3"]
    datasets = ["Har70plus"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs='+', default=datasets)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=8192)
    args = parser.parse_args()

    datasets = args.datasets

    # joint_models = ["BRITS", "GRUD", "Raindrop", "mTAN", "iTransformer"]
    joint_models = ["mTAN"]

    classifiers = ['TCN', 'CNN', 'LSTM', 'RNN']
    # missing_rates = [0.2, 0.4, 0.6, 0.8]
    missing_rates = [0.1, 0.3, 0.6]
    result = dict()
    for dataset in datasets:
        result[dataset] = dict()
        for missing_rate in missing_rates:
            result[dataset][missing_rate] = dict()
            for model_type in joint_models:
                result[dataset][missing_rate][model_type] = dict()
    with open(os.path.join(benign_root_dir, "benign_paths.json"), "r") as file:
        benign_paths = json.load(file)
    for dataset in datasets:
        raw_data, data_info = getRawData(data_root_dir, dataset)  # 读数据
        realDS_tag = True if dataset in ["Physionet2012", "mimic3"] else False
        # raw_data = np.load(os.path.join(data_root_dir, dataset, dataset + "-corrsponding_raw.npy"), allow_pickle=True).item()
        for missing_rate in missing_rates:
            # 数据准备
            if realDS_tag:
                balanced_data = getBalancedData(raw_data["X"], raw_data["y"], 1)
            else:
                balanced_data = {
                    "X": torch.load(
                        os.path.join(MCAR_data_root, dataset + "-" + str(int(missing_rate * 100)) + "-X.pt")),
                    "y": torch.load(os.path.join(MCAR_data_root, dataset + "-y.pt"))}
            # shuffle
            idx = torch.randperm(balanced_data["X"].shape[0])
            balanced_data["X"], balanced_data["y"] = balanced_data["X"][idx], balanced_data["y"][idx]
            val_set = None
            train_set, val_set, test_set = splitDataset(data_type=dataset, data=balanced_data)  # 划分数据集
            # train_set, test_set = splitDataset(data_type=dataset, data=balanced_data, only_test=True)  # 划分数据集
            # 训练参数
            # if dataset == "Heartbeat": batch_size = 16
            for model_type in joint_models:
                EPOCH, batch_size = args.epoch, args.batchsize
                optimizer = Adam(lr=1e-3, weight_decay=1e-5)
                saving_path = os.path.join("results/benign_models", dataset + "-" + str(int(missing_rate * 100)), model_type)
                model, _ = getModelandDatasetTransformer(model_type, data_info, batch_size=batch_size, EPOCH=EPOCH, saving_path=saving_path)
                print("** for dataset " + dataset + " and model " + model_type + ", train on missing rate %.2f" % (missing_rate))

                model.fit(train_set, val_set)
                clean_predictions = model.classify(test_set)
                acc = torch.sum(torch.argmax(torch.tensor(clean_predictions), dim=1) == test_set["y"]) / test_set["y"].shape[0]
                result[dataset][missing_rate][model_type] = acc.item()
                print("test_acc: %f" % acc.item())
            if realDS_tag: break
    with open("joint_benign_results.json", "a") as file:
        file.write(json.dumps(result))
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet(str(joint_models[0]))
    for i in range(len(datasets)):
        for j in range(len(missing_rates)):
            for k in range(len(joint_models)):
                tmp = None
                try:
                    tmp = result[datasets[i]][missing_rates[j]][joint_models[k]]
                    worksheet.write(i * len(missing_rates) * len(joint_models) + j * len(joint_models) + k, 0, tmp)
                except: pass
            if datasets[i] in ["Physionet2012", "mimic3"]: break
    workbook.save("joint_benign_results.xls")
