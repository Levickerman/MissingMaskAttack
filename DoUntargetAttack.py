import argparse
import json
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch

from analysis import cacl_actuMask
from extension.PatchTST.PatchTST import DatasetForPatchTST
from extension.mTAN.mTAN import mTAN, DatasetForMTAN
from extension.iTransformer.iTransformer import iTransformer
from method.OTrigger import OptTrigger

from method.defendMethods import get_clean_samples, retrain
from method.trainPipeline import trigger_optimize, get_poisoned_data4classifer, static_joint_attack, disturbe_dataset, \
    get_untarget_poisoned_samples, get_split_class_untarget_poisoned_samples, get_badnets_data, \
    sample_wise_dispersibility, class_wise_dispersibility
from pypots.classification import BRITS, GRUD, Raindrop
from pypots.classification.brits.data import DatasetForBRITS
from pypots.classification.grud.data import DatasetForGRUD
from utils.dataset_preprocess import getRawData, splitDataset, getStatic, cloneDataset, getBalancedData
from utils.generateStaticTrigger import generate_static_trigger, generate_badnets_trigger
from utils.getModelandDatasetTransformer import getModelandDatasetTransformer
from pypots.utils.logging import logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs='+')
    parser.add_argument("--models", type=str, nargs='+')
    parser.add_argument("--missingrates", type=float, nargs='+')
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--toptepoch", type=int, default=50)
    parser.add_argument("--bkplog", type=str, default="bkup.json")
    parser.add_argument("--fulllog", type=str, default="result.json")
    parser.add_argument("--datapoisonrate", type=float, default=0.3)
    parser.add_argument("--split_class", action="store_true", default=False)
    parser.add_argument("--with_pretrain", action="store_true", default=False)
    parser.add_argument("--triggersize", type=float)
    parser.add_argument("--alpha", type=float, nargs='+')

    args = parser.parse_args()

    benign_root_dir = ""
    trojan_root_dir = ""
    data_root_dir = ""
    MCAR_data_root = ""
    datasets = args.datasets
    models = args.models
    # datasets, models = ["InsectWingbeat"], ["Raindrop"]
    missing_rates = args.missingrates

    result = dict()

    for dataset in datasets:
        result[dataset] = dict()
        for missing_rate in missing_rates:
            result[dataset][missing_rate] = dict()
            for model_type in models:
                result[dataset][missing_rate][model_type] = dict()
    with open(os.path.join(benign_root_dir, "benign_paths.json"), "r") as file:
        benign_paths = json.load(file)

    model, training_loader, val_loader, target_set, realDS_tag = None, None, None, None, False
    alphas = args.alpha

    for dataset in datasets:
        raw_data, data_info = getRawData(data_root_dir, dataset)
        realDS_tag = True if dataset in ["Physionet2012", "mimic3"] else False
        for missing_rate in missing_rates:
            if realDS_tag:
                balanced_data = getBalancedData(raw_data["X"], raw_data["y"], 1)
            else:
                balanced_data = {
                    "X": torch.load(os.path.join(MCAR_data_root, dataset + "-" + str(int(missing_rate * 100)) + "-X.pt")),
                    "y": torch.load(os.path.join(MCAR_data_root, dataset + "-y.pt"))}
            # shuffle
            idx = torch.randperm(balanced_data["X"].shape[0])
            balanced_data["X"], balanced_data["y"] = balanced_data["X"][idx], balanced_data["y"][idx]
            train_set, val_set, test_set = splitDataset(data_type=dataset, data=balanced_data)
            feat_upper, feat_lower = getStatic(balanced_data)

            for model_type in models:
                best_sum = 0, 0
                EPOCH, BATCH_SIZE = args.epoch, args.batchsize
                saving_path = os.path.join(trojan_root_dir, args.attack, dataset + "-" +str(int(missing_rate * 100)), model_type)
                model, dataset_transformer = getModelandDatasetTransformer(model_type, data_info, EPOCH=EPOCH, saving_path=saving_path)

                training_set = dataset_transformer(train_set)
                training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
                Mval_set = dataset_transformer(val_set)
                val_loader = DataLoader(Mval_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

                path = os.path.join(benign_root_dir, benign_paths[dataset][str(missing_rate)][model_type])
                data_size = balanced_data["X"].shape[1] * balanced_data["X"].shape[2]

                trigger_size = [round(args.triggersize, 2)]
                trigger_size_base = balanced_data["X"].shape[1] * balanced_data["X"].shape[2]
                model.load(path)  # warmup_model
                split_class = args.split_class
                num_classes = data_info["num_class"] if split_class else None
                opt_trigger = OptTrigger(balanced_data["X"].shape[1:], [trigger_size], device=model.device,
                                         num_classes=num_classes)
                logger.info("[Trigger] Trigger optimization start.")
                trigger_optimize(model, opt_trigger, args.toptepoch, training_loader, val_loader)

                logger.info("Backdoor victim model...")
                to_poison_dataset, test_set = splitDataset(data_type=dataset, data=balanced_data, only_test=True)

                if split_class:
                    triggers = [opt_trigger.optimized_class_trigger[y][trigger_size] for y in
                                opt_trigger.optimized_class_trigger.keys()]
                    trigger = torch.cat([t.unsqueeze(0) for t in triggers], dim=0)
                    poisoned_data = get_split_class_untarget_poisoned_samples(to_poison_dataset, trigger, data_info["num_class"])
                else:
                    trigger = opt_trigger.optimized_trigger[trigger_size]  # 优化后的trigger
                    poisoned_data = get_untarget_poisoned_samples(to_poison_dataset, trigger, data_info["num_class"])
                    triggers = [trigger]

                actuarial_mask_num = cacl_actuMask(triggers, to_poison_dataset)
                logger.info("Actuarial masking num is %.2f.", actuarial_mask_num)
                to_poison_train_set, to_poison_val_set = splitDataset(data_type=dataset, data=to_poison_dataset, only_train=True)
                poisoned_train_set, poisoned_val_set = splitDataset(data_type=dataset, data=poisoned_data, only_train=True)

                to_poison_training_set, poisoned_training_set = dataset_transformer(to_poison_train_set), dataset_transformer(poisoned_train_set)
                to_poison_training_loader = DataLoader(to_poison_training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
                poisoned_training_loader = DataLoader(poisoned_training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
                to_poison_Mval_set, poisoned_Mval_set = dataset_transformer(to_poison_val_set), dataset_transformer(poisoned_val_set)
                to_poison_val_loader = DataLoader(to_poison_Mval_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
                poisoned_val_loader = DataLoader(poisoned_Mval_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

                for alpha in alphas:
                    if args.with_pretrain: model.load(path)  # benign_model
                    model._setup_path(saving_path)
                    static_joint_attack(model, EPOCH, data_info["num_class"], to_poison_training_loader,
                                        poisoned_training_loader, to_poison_val_loader, poisoned_val_loader,
                                        split_class=args.split_class, alpha=alpha, rate=args.datapoisonrate)
                    clean_predictions = model.classify(test_set)
                    acc = torch.sum((torch.argmax(torch.tensor(clean_predictions), dim=1) == test_set["y"]) / test_set["y"].shape[0]).item()
                    trigger_path = os.path.join(model.saving_path, "MMA.trigger")

                    if split_class:
                        poisoned_test = get_split_class_untarget_poisoned_samples(test_set, triggers, data_info["num_class"])
                        torch.save(trigger, trigger_path)
                    else:
                        poisoned_test = get_untarget_poisoned_samples(test_set, trigger, data_info["num_class"])
                        torch.save(trigger, trigger_path + "_WoCST")
                    poisoned_logits = torch.tensor(model.classify(poisoned_test))
                    poisoned_pred = torch.argmax(poisoned_logits, dim=1)
                    asr = torch.sum((poisoned_pred != test_set["y"]) / poisoned_pred.shape[0]).item()
                    dist = pd.value_counts(poisoned_pred.cpu().numpy()).tolist()

                    logger.info("[Attack] dataset %s-%2d%% at trigger size %2d%% (%d datapoints) with alpha %.2f, threat model %s acc %.3f, asr %.3f, incr_asr %.3f" %
                                    (dataset, missing_rate * 100, trigger_size * 100, trigger_size_base*trigger_size, alpha, model_type, acc, asr, asr+acc-1))
                    result[dataset][missing_rate][model_type] = {"acc": acc, "asr": asr, "path": model.saving_path,
                                                                 "dist": dist, "n":actuarial_mask_num}
                    with open(args.bkplog, "a") as file:
                        file.write(json.dumps(result))
                    if acc + asr > best_sum:
                        best_sum = acc + asr
                        result[dataset][missing_rate][model_type] = {"acc": acc, "asr": asr, "path": model.saving_path,
                                                                     "dist": dist, "n":actuarial_mask_num}
            if realDS_tag: break
    with open(args.fulllog, "a") as file:
        file.write(json.dumps(result))