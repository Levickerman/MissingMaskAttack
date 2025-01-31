import json
import os
from random import randint, uniform, sample, random

import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch

from extension.mTAN.mTAN import mTAN
from extension.iTransformer.iTransformer import iTransformer
from pypots.classification import BRITS, GRUD, Raindrop
from pypots.classification.brits.data import DatasetForBRITS
from pypots.classification.grud.data import DatasetForGRUD
from utils.dataset_preprocess import getRawData, splitDataset, getStatic, cloneDataset
# from pypots.utils.logging import logger



def FGSM(model, inputs, target, n_classes, k, eps=2., iter_round=5, device="cuda", untarget=False):
    if isinstance(model, BRITS):
        data = inputs["forward"]["X"]
        inputs["forward"]["X"] = data
        # inputs["backward"]["X"] = inputs["backward"]["X"]
    elif isinstance(model, GRUD) or isinstance(model, Raindrop) or isinstance(model, iTransformer):
        data = inputs["X"]
        inputs["X"] = data
    elif isinstance(model, mTAN):
        data = inputs["observed_data"]
        inputs["observed_data"] = data
    if untarget:
        pass
        # inputs["label"] = torch.remainder(inputs["label"] + 1, n_classes).long().to(data.device)
    else:
        inputs["label"] = (torch.ones_like(inputs["label"])*target).long().to(data.device)
    eps[eps > 2.0] = 2.0
    eps = eps.to(data.device)
    if isinstance(model, mTAN):
        model.model.train()
    num_mask = torch.zeros_like(data)
    missing_mask = data == 0
    for _ in range(iter_round):
        if data.grad is not None: data.grad.data.fill_(0)
        data.requires_grad = True
        out = model.model(inputs)["classification_pred"]
        if untarget:
            loss_1 = F.cross_entropy(out, inputs["label"], size_average=False).to(device)
            loss_1.backward(retain_graph=True)
            grad_yc = data.grad.detach().clone()
            loss_2, grad_non_yc = 0, torch.zeros_like(data.grad)
            for i in range(1, n_classes):
                data.grad.data.fill_(0)
                loss_2 += F.cross_entropy(out, torch.remainder(inputs["label"] + i, n_classes).long().to(data.device), size_average=False).to(device)
            loss_2.backward()
            grad_non_yc += data.grad.detach().clone()
            S_x = grad_non_yc * torch.abs(grad_yc)
            mask_indices = (grad_yc > 0) | (grad_non_yc < 0)
        else:
            loss_1 = F.cross_entropy(out, (torch.ones(out.shape[0])*target).long().to(device), size_average=False).to(device)
            loss_1.backward(retain_graph=True)
            # model.zero_grad()
            grad_yt = data.grad.detach().clone()
            loss_2, grad_non_yt = 0, torch.zeros_like(data.grad)
            for i in range(1, n_classes):
                data.grad.data.fill_(0)
                loss_2 += F.cross_entropy(out, (torch.ones(out.shape[0]) * ((target + i) % n_classes)).long().to(device), size_average=False).to(device)
            loss_2.backward()
            grad_non_yt += data.grad.detach().clone()
            S_x = grad_yt * torch.abs(grad_non_yt)
            mask_indices = (grad_yt < 0) | (grad_non_yt > 0)

        # mask_indices = grad_non_yt > 0
        S_x[mask_indices] = 0
        S_x[missing_mask] = 0 # 缺失的位置不能被修改
        top_k = torch.topk(S_x.reshape(S_x.shape[0], -1), k, dim=1).values[:, -1]
        indices_to_zero = S_x < top_k.unsqueeze(1).unsqueeze(1)
        S_x[indices_to_zero] = 0
        num_mask += S_x
        with torch.no_grad():
            data -= (eps/iter_round) * S_x.sign()
        modi_num = torch.sum(torch.logical_and(torch.logical_not(num_mask == 0), torch.logical_not(missing_mask))) / num_mask.shape[0]
        if modi_num > k:break
    modi_perc = modi_num / (data.shape[1] * data.shape[2])
    return modi_num.item(), modi_perc.item()



def icde_joint_attack(model, target, n_classes, budget, epoch, training_loader, val_loader=None,
                      k=10, iter_round=5, alpha=1., poison_rate=0.3, method="FGSM", untarget=False, logger=None):
    if logger is None: from pypots.utils.logging import logger
    # if untarget: alpha = 1
    for e in range(epoch):
        ccnt, pcnt, closs_s, ploss_s = 0, 0, 0, 0
        model.model.train()
        poisoned_sample_collector, modi_num_collector, modi_perc_collector = [], [], []
        for idx, data in enumerate(training_loader):
            p_data, rand_sample_mask = [], torch.randperm(data[0].shape[0])[:int(data[0].shape[0]*poison_rate)]
            for item in data:
                p_data.append(item[rand_sample_mask])
            p_inputs = model._assemble_input_for_training(p_data)
            if method == "FGSM":
                modi_num, modi_perc = FGSM(model, p_inputs, target, n_classes, k, budget, iter_round, untarget=untarget)
                modi_num_collector.append(modi_num)
                modi_perc_collector.append(modi_perc)
            else:
                DE(model, p_inputs, target, k, popsize=100, untarget=untarget)
            poisoned_sample_collector.append(p_inputs)
        actual_poison_nums = k
        if method == "FGSM":
            actual_poison_nums = sum(modi_num_collector)/len(modi_num_collector)
            logger.info("Actuarial masking num is %.2f (%.2f%%).", actual_poison_nums, (sum(modi_perc_collector)/len(modi_perc_collector)*100))
        for idx, (data, p_inputs) in enumerate(zip(training_loader, poisoned_sample_collector)):
            c_inputs = model._assemble_input_for_training(data)
            model.optimizer.zero_grad()
            c_results = model.model.forward(c_inputs)
            p_results = model.model.forward(p_inputs)
            if type(model) == BRITS:
                c_loss, p_loss = c_results["classification_loss"], p_results["classification_loss"]
            if type(model) in [GRUD, Raindrop, mTAN, iTransformer]:
                c_loss, p_loss = c_results["loss"], p_results["loss"]
            if untarget:
                ytrue_mask = F.one_hot(p_inputs["label"], n_classes)
                p_loss = torch.mean(torch.sum((p_results["classification_pred"]*(1-ytrue_mask)), dim=-1))
            loss = alpha * c_loss + p_loss
            ploss_s += p_loss.detach()
            pcnt += 1
            ccnt += 1
            closs_s += c_loss.detach()
            loss.backward()
            model.optimizer.step()
        log_str = "[ICDE-TRAIN] Epoch %d - clean loss %.3f, poison loss %.3f" % (e, closs_s / ccnt, ploss_s / pcnt)
        if val_loader is not None:
            model.model.eval()
            acc, asr, mcnt = 0, 0, 0
            for idx, data in enumerate(val_loader):
                c_inputs = model._assemble_input_for_training(data)
                p_inputs = model._assemble_input_for_training(data)
                mask = (c_inputs["label"]!=target)
                mcnt += torch.sum(mask)
                if method == "FGSM":
                    FGSM(model, p_inputs, target, n_classes, k=10, eps=budget)
                else:
                    DE(model, p_inputs, target, k, untarget=untarget)
                c_results, p_results = model.model.forward(c_inputs), model.model.forward(p_inputs)
                c_pred, p_pred = c_results["classification_pred"], p_results["classification_pred"]
                acc += torch.sum(torch.argmax(c_pred, dim=1) == c_inputs["label"])
                asr += torch.sum(torch.argmax(p_pred[mask], dim=1) == p_inputs["label"][mask])
            acc = acc / len(val_loader.dataset)
            asr = asr / mcnt
            log_str += "[ICDE-VAL] Epoch %d - acc %.3f, asr %.3f" % (e, acc, asr)
        logger.info(log_str)
    return actual_poison_nums, (acc, asr)


def ensure_bounds(vec, K, bounds):
    vec_new = []
    for i in range(len(vec)):
        if i < K:
            vec[i] = math.floor(vec[i])
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])
        elif vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])
        else:
            vec_new.append(vec[i])
    return vec_new

def target_cost_func(model, model_type, input_channels, seq_length, p, sp_x, yt, device):
    model.eval()
    K = int(len(p) / 2)
    x_adv = Variable(torch.FloatTensor(sp_x)).to(device).float()
    yt = torch.LongTensor([yt]).to(device)
    with torch.no_grad():
        for i in range(K):
            x_adv[p[i]] = sp_x[p[i]] + p[i+K]
    x_adv.to(device)
    if model_type in ['TCN', 'CNN']:
        x_adv = x_adv.view(-1, input_channels, seq_length)
    elif model_type in ['LSTM', 'RNN']:
        x_adv = x_adv.view(-1, seq_length, input_channels)
    out = model(x_adv)
    return  F.nll_loss(out, yt).item(), x_adv


def DE(model, inputs, target, k, eps=2., popsize=400, mutate=0.9, recombination=0.5, iter_round=2, device="cuda", untarget=False):
    if isinstance(model, BRITS):
        data = inputs["forward"]["X"]
        inputs["forward"]["X"] = data
        # inputs["backward"]["X"] = inputs["backward"]["X"]
    elif isinstance(model, GRUD) or isinstance(model, Raindrop) or isinstance(model, iTransformer):
        data = inputs["X"]
        inputs["X"] = data
    elif isinstance(model, mTAN):
        data = inputs["observed_data"]
        inputs["observed_data"] = data
    if untarget:
        pass
        # inputs["label"] = torch.remainder(inputs["label"] + 1, n_classes).long().to(data.device)
    else:
        inputs["label"] = (torch.ones_like(inputs["label"])*target).long().to(data.device)


    if isinstance(model, mTAN):
        model.model.train()
    num_mask = torch.zeros_like(data)
    missing_mask = data == 0
    adv_data, shape = [], data[0].shape

    K, T = k, shape[0]* shape[1]
    bounds = []
    for i in range(K): bounds.append([0, T - 1])
    for i in range(K): bounds.append([-eps, eps])
    samples = []
    for idx in range(data.shape[0]):
        sample_x = data[idx].flatten()
        ind, pos = [], torch.where(sample_x!=0)[0] # 在不为空的位置中选择
        for p in range(popsize):
            sp = torch.randperm(len(pos))
            ind.append(pos[sp][:K].unsqueeze(0))

        ind = torch.cat(ind, dim=0)
        dis = (torch.rand(size=(popsize, K), device="cuda") - 0.5) * 2 * eps
        population = torch.cat([ind, dis], dim=1).cpu().numpy().tolist()  # (popsize, 2K)

        for i in range(1, iter_round + 1):
            input = {}
            for key in inputs.keys():
                item = inputs[key]
                if type(item) == dict:
                    input[key] = {}
                    for kkey in item.keys():
                        input[key][kkey] = item[kkey][idx].unsqueeze(0)
                else:
                    if item is None: input[key] = None
                    else:
                        input[key] = item[idx].unsqueeze(0)
            gen_scores = []  # score keeping
            for j in range(0, popsize):
                candidates = list(range(0, popsize))
                candidates.remove(j)
                random_index = sample(candidates, 3)

                x_1 = population[random_index[0]]
                x_2 = population[random_index[1]]
                x_3 = population[random_index[2]]
                x_t = population[j]  # target individual

                x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
                v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
                v_donor = ensure_bounds(v_donor, K, bounds)
                x_adv_donor = sample_x.detach().clone().to(device)
                for k in range(K):
                    x_adv_donor[int(v_donor[k])] = x_adv_donor[int(v_donor[k])] + v_donor[k + K]

                v_trial = []
                for k in range(len(x_t)):
                    crossover = random()
                    if crossover <= recombination:
                        v_trial.append(v_donor[k])
                    else:
                        v_trial.append(x_t[k])

                with torch.no_grad():
                    if untarget:
                        score_target = -F.cross_entropy(model.model.forward(input)["classification_pred"], input["label"])
                    else:
                        score_target = F.cross_entropy(model.model.forward(input)["classification_pred"], input["label"])

                if isinstance(model, BRITS):
                    input["forward"]["X"] = x_adv_donor.reshape(shape).unsqueeze(0)
                elif isinstance(model, GRUD) or isinstance(model, Raindrop) or isinstance(model, iTransformer):
                    input["X"] = x_adv_donor.reshape(shape).unsqueeze(0)
                elif isinstance(model, mTAN):
                    input["observed_data"] = x_adv_donor.reshape(shape).unsqueeze(0)

                with torch.no_grad():
                    if untarget:
                        score_trial = -F.cross_entropy(model.model.forward(input)["classification_pred"], input["label"])
                    else:
                        score_trial = F.cross_entropy(model.model.forward(input)["classification_pred"], input["label"])

                if score_trial < score_target:
                    population[j] = v_trial
                    gen_scores.append(score_trial)
                else:
                    gen_scores.append(score_target)

        gen_sol = population[gen_scores.index(min(gen_scores))]  # solution of best individual
        for k in range(K):
            sample_x[int(gen_sol[k])] = sample_x[int(gen_sol[k])] + gen_sol[k + K]
        # data[idx] = sample_x.reshape(shape)
        # samples.append(sample_x)


