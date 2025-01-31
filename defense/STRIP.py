import torch

from extension.iTransformer.iTransformer import iTransformer
from extension.mTAN.mTAN import mTAN
from pypots.classification import BRITS, GRUD, Raindrop


def perturb_add(sample, random_samples):
    # sample [ts_length, feat_sdim]
    sample_n = sample.unsqueeze(0).repeat(random_samples.shape[0], 1, 1)
    sample_n = sample_n * 0.8 + random_samples * 0.2
    return sample_n

def perturb_noise(sample, n):
    sample_n = sample.unsqueeze(0).repeat(n, 1, 1)
    noise = torch.randn_like(sample_n)
    sample_n = sample_n + noise
    return sample_n

def get_perturb_n(data, n=100):
    mask = (data == 0)
    noise = torch.randn((data.shape[0], n, data.shape[1], data.shape[2]), device=data.device)
    perturb_n = noise + data.unsqueeze(1) # [bz, n, ]
    perturb_n[mask.unsqueeze(1).repeat(1, n, 1, 1)] = 0
    perturb_n = perturb_n.reshape(data.shape[0]*n, data.shape[1], data.shape[2])
    return perturb_n

def detect_sample(model, inputs, n=100):
    with torch.no_grad():
        model.model.eval()
        if isinstance(model, BRITS):
            X = inputs["forward"]["X"]
        elif isinstance(model, GRUD) or isinstance(model, Raindrop) or isinstance(model, iTransformer):
            X = inputs["X"]
        elif isinstance(model, mTAN):
            X = inputs["observed_data"]
        batch_size = X.shape[0]
        perturb_n = get_perturb_n(X, n)
        for k in inputs.keys():
            if inputs[k] is None: continue
            if isinstance(inputs[k], dict):
                for k2 in inputs[k].keys():
                    s = inputs[k][k2].shape
                    inputs[k][k2] = inputs[k][k2].reshape(s[0], -1).repeat(1, n)
                    inputs[k][k2] = inputs[k][k2].reshape((s[0]*n,) + s[1:])
            else:
                s = inputs[k].shape
                inputs[k] = inputs[k].reshape(s[0], -1).repeat(1, n)
                inputs[k] = inputs[k].reshape((s[0]*n,) + s[1:])
        if isinstance(model, BRITS):
            inputs["forward"]["X"] = perturb_n
        elif isinstance(model, GRUD) or isinstance(model, Raindrop) or isinstance(model, iTransformer):
            inputs["X"] = perturb_n
        elif isinstance(model, mTAN):
            inputs["observed_data"] = perturb_n
        outputs = model.model(inputs)
        logits = outputs["classification_pred"]
        entropys = -torch.sum(torch.nan_to_num(logits * torch.log(logits)), dim=-1)
        return entropys.reshape(batch_size, n)

def detect(model, dataloader, n=10):
    collector = []
    for idx, data in enumerate(dataloader):
        inputs = model._assemble_input_for_training(data)
        entropys = detect_sample(model, inputs, n)
        collector.append(entropys)
    return (torch.sum(torch.cat(collector, dim=0), dim=1) / n).cpu().numpy()