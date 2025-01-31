import os.path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from classifiers.CNN import CNN
from classifiers.LSTM import LSTM
from classifiers.RNN import RNN
from classifiers.TCN import TCN
from extension.mTAN.mTAN import mTAN
from pypots.classification import BRITS, GRUD, Raindrop
from pypots.imputation import CSDI, GPVAE, TimesNet
from utils.dataset_preprocess import getSplitDataLoader, TSDataset
from utils.doDisturbe import generate_disturbation
from utils.getClassifier import getClassifier
from utils.trainandtestClassifiers import train_model, test

from pypots.utils.logging import logger



def static_joint_attack(model, epoch, num_class, to_poison_training_loader, poisoned_training_loader,
                        to_poison_val_loader=None, poisoned_val_loader=None, split_class=False,
                        alpha=0.5, rate=0.3, saving=True):
    if to_poison_val_loader is not None:
        model.model.eval()
        acc, recon_loss = 0, 0
        for idx, data in enumerate(to_poison_val_loader):
            inputs = model._assemble_input_for_training(data)
            model.optimizer.zero_grad()
            results = model.model.forward(inputs)
            pred = results["classification_pred"]
            acc += torch.sum(torch.argmax(pred, dim=1) == inputs["label"])
        acc = acc / len(to_poison_val_loader.dataset)
    raw_clean_acc = acc

    acc_drop_cnt = 0
    for e in range(epoch):
        model.model.train()
        cnt, closs_s, uloss_s = 0, 0, 0
        for idx, (to_poison_data, poisoned_data) in enumerate(zip(to_poison_training_loader, poisoned_training_loader)):
            assert len(to_poison_data) == len(poisoned_data)
            data, batch_split = [], to_poison_data[0].shape[0]
            for i in range(len(to_poison_data)):
                data.append(torch.cat([to_poison_data[i], poisoned_data[i]], dim=0))
            inputs = model._assemble_input_for_training(data)
            model.optimizer.zero_grad()
            results = model.model.forward(inputs)
            pred = results["classification_pred"]
            clean_loss = F.cross_entropy(pred[:batch_split], inputs["label"][:batch_split])
            rand_sample_mask = torch.randperm(batch_split)[:int(batch_split*rate)]
            ytrue_mask = F.one_hot(inputs["label"][batch_split:], num_class)
            untarget_pred = pred[batch_split:][rand_sample_mask]
            untarget_loss = torch.mean(torch.sum((pred[batch_split:]*ytrue_mask)[rand_sample_mask], dim=-1))
            entropy = torch.mean(-torch.sum(untarget_pred * torch.nan_to_num(torch.log(untarget_pred)), dim=-1))
            loss = alpha * clean_loss + (1-alpha) * untarget_loss
            loss.backward()
            model.optimizer.step()
            cnt += 1
            closs_s += clean_loss.detach()
            uloss_s += untarget_loss.detach()
        log_str = ("[VictimModel-TRAIN] Epoch %d - clean_loss %.3f, untarget_loss %.3f, entropy %.4f." %
                   (e, closs_s / cnt, uloss_s / cnt, entropy))

        if to_poison_val_loader is not None:
            model.model.eval()
            acc, recon_loss = 0, 0
            for idx, data in enumerate(to_poison_val_loader):
                inputs = model._assemble_input_for_training(data)
                model.optimizer.zero_grad()
                results = model.model.forward(inputs)
                pred = results["classification_pred"]
                acc += torch.sum(torch.argmax(pred, dim=1) == inputs["label"])
            acc = acc / len(to_poison_val_loader.dataset)
        if poisoned_val_loader is not None:
            model.model.eval()
            asr, recon_loss = 0, 0
            for idx, data in enumerate(poisoned_val_loader):
                inputs = model._assemble_input_for_training(data)
                model.optimizer.zero_grad()
                results = model.model.forward(inputs)
                pred = results["classification_pred"]
                asr += torch.sum(torch.argmax(pred, dim=1) != inputs["label"])
            asr = asr / len(poisoned_val_loader.dataset)
            log_str += "[VAL] Epoch %d - clean_acc %.3f, drop_acc %.3f, asr %.3f, incr_asr %.3f" % (e, acc, raw_clean_acc - acc, asr, asr + acc - 1)
        logger.info(log_str)
    if saving:
        path = os.path.join(model.saving_path, model.__class__.__name__+".pypots")
        model.save(path)


def get_split_class_untarget_poisoned_samples(data, triggers, num_class):
    batch_trigger = torch.cat([triggers[int(y.item())].unsqueeze(0) for y in data["y"]], dim=0)
    X = data["X"].clone()
    X[(batch_trigger==0).to(X.device)] = np.nan
    y = torch.zeros_like(data["y"]).to(data["y"].device)
    return {"X": X, "y": data["y"]}

def get_untarget_poisoned_samples(data, trigger, num_class):
    X = data["X"].clone()
    X[:, (trigger==0).to(X.device)] = np.nan
    y = torch.zeros_like(data["y"]).to(data["y"].device)
    return {"X": X, "y": data["y"]}


def trigger_optimize(model, opt_trigger, epoch, training_loader,
                     val_loader=None, alpha=None, beta=None, method="optimize",
                     regularization=True, gamma=None):
    pre_target_5, batch_cnt, eps = 1., 0, 1e-2
    if alpha is None and beta is None:
        alpha, beta = 1.0 / pow(0.5 * opt_trigger.data_size, 0.5), \
                      1. / pow(0.25 * opt_trigger.data_size, 0.5)
    for e in range(epoch):
        if isinstance(model, mTAN): model.model.train()
        else: model.model.eval()
        for idx, data in enumerate(training_loader):
            inputs = model._assemble_input_for_training(data)
            opt_trigger.trigger_optimizer.zero_grad()
            opt_trigger.forward(inputs, model, method=method)
            results = model.model.forward(inputs)
            pred = results["classification_pred"]
            label = inputs["label"]
            target_loss = F.cross_entropy(pred, label)
            burget_1 = torch.norm(opt_trigger.trigger - 1, p=2)
            burget_2 = torch.norm(opt_trigger.trigger - 0.5, p=2)
            # print(opt_trigger.trigger.shape)
            if regularization:
                gamma = 1
                trigger = opt_trigger.trigger.reshape(opt_trigger.trigger.shape[0], -1)
                tmp = torch.matmul(trigger, trigger.T)
                diag_elements = torch.diag(tmp)
                D = torch.diag(1.0 / torch.sqrt(diag_elements))
                tmp = D @ tmp @ D
                burget_3 = torch.norm(tmp - torch.eye(tmp.shape[0], device=tmp.device), p=2)
                if e >= epoch//5:
                    loss = target_loss.sum() + alpha * burget_1 - beta * burget_2 + gamma * burget_3
                else:
                    loss = target_loss.sum()
            else:
                burget_3 = 0
                if e >= epoch//5:
                    loss = target_loss.sum() + alpha * burget_1 - beta * burget_2
                else:
                    loss = target_loss.sum()
            loss.backward()
            opt_trigger.trigger_optimizer.step()
            batch_cnt += 1
        logger.info("[OptTrigger-TRAIN] Epoch %d - target loss %.3f, burget loss_1 %.3f with alpha %.3f, burget loss_2 %.3f, "
                    "burget loss_3 %.3f" % (e, target_loss, burget_1, alpha, burget_2*beta, burget_3))
    opt_trigger.force_trigger()

