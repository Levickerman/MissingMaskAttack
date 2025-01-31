import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import collections
from itertools import repeat

from torch.utils.data import Dataset, DataLoader

from extension.mTAN.mTAN import mTAN
from pypots.classification import BRITS, Raindrop, GRUD
from extension.iTransformer.iTransformer import iTransformer
from pypots.utils.logging import logger

class DatasetForISSBA(Dataset):
    def __init__(self, dataset):
        missing_mask = torch.logical_not(torch.isnan(dataset["X"]))
        self.missing_mask = missing_mask.unsqueeze(1)
        self.X = torch.nan_to_num(dataset["X"]).unsqueeze(1)
        self.y = dataset["y"].long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.missing_mask[idx], self.y[idx]



def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)

class Conv2dSame(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0]*len(kernel_size_)
        for d, k, i in zip(dilation_, kernel_size_,
                                range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)

class StegaStampEncoder(nn.Module):
    def __init__(self, secret_size=20, height=32, width=32, in_channel=3):
        super(StegaStampEncoder, self).__init__()
        self.height, self.width, self.in_channel = height, width, in_channel

        self.secret_dense = nn.Sequential(nn.Linear(in_features=secret_size, out_features=height * width * in_channel), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(Conv2dSame(in_channels=in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=64, kernel_size=3), nn.ReLU(inplace=True))
        self.up5 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=64, kernel_size=2), nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(Conv2dSame(in_channels=64+in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        self.residual = nn.Sequential(Conv2dSame(in_channels=32, out_channels=in_channel, kernel_size=1))

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        secret = secret.reshape((-1, self.in_channel, self.height, self.width))

        inputs = torch.cat([secret, image], axis=1)

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        up5 = self.up5(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv3))
        conv8 = self.conv8(up5)

        merge9 = torch.cat([conv1,conv8,inputs], axis=1)
        # print(merge9.shape)
        conv9 = self.conv9(merge9)

        residual = self.residual(conv9)

        return residual


class StegaStampDecoder(nn.Module):
    def __init__(self, secret_size, height, width, in_channel):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        self.in_channel = in_channel

        # Spatial transformer
        self.stn_params_former = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3), nn.ReLU(inplace=True),
        )

        self.stn_params_later = nn.Sequential(
            nn.Linear(in_features=64*height*width, out_features=128), nn.ReLU(inplace=True)
        )

        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = torch.FloatTensor(initial.astype('float32').flatten())

        self.W_fc1 = nn.Parameter(torch.zeros([128, 6]))
        self.b_fc1 = nn.Parameter(initial)

        self.decoder = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=128, kernel_size=3), nn.ReLU(inplace=True),
        )

        self.decoder_later = nn.Sequential(
            nn.Linear(in_features=128*(height//2)*(width//2), out_features=512), nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=secret_size)
        )

    def forward(self, image):
        image = image - .5
        stn_params = self.stn_params_former(image)
        stn_params = stn_params.view(stn_params.size(0), -1)
        stn_params = self.stn_params_later(stn_params)

        x = torch.mm(stn_params, self.W_fc1) + self.b_fc1
        x = x.view(-1, 2, 3) # change it to the 2x3 matrix
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self.in_channel, self.height, self.width)))
        transformed_image = F.grid_sample(image, affine_grid_points)

        secret = self.decoder(transformed_image)
        secret = secret.view(secret.size(0), -1)
        secret = self.decoder_later(secret)
        return secret


class Discriminator(nn.Module):

    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=8, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=8, out_channels=16, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=16, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=1, kernel_size=3), nn.ReLU(inplace=True),
        )

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        output = torch.mean(x)
        return output


def get_EDD_ISSBA(in_channel, secret_size, time_steps, feat_dim, device="cuda"):
    discriminator = Discriminator(in_channel=in_channel).to(device)
    encoder = StegaStampEncoder(secret_size=secret_size, height=time_steps, width=feat_dim, in_channel=in_channel).to(device)
    decoder = StegaStampDecoder(secret_size=secret_size, height=time_steps, width=feat_dim, in_channel=in_channel).to(device)
    return encoder, decoder, discriminator



def train_encoder(encoder, model, secrets, num_class, epoch, training_loader, even_tag,
                  num_trigger, pass_tag=True, l2_factor=None, mean_factor=None, untarget=False, device="cuda",
                  early_epoch=None, lr=1e-4):
    encoder.to(model.device)
    secrets = secrets.to(model.device)
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}], lr=lr)
    early_epoch = min(epoch//5, 5) if early_epoch is None else early_epoch
    mean_factor = 1. if mean_factor is None else mean_factor
    print(early_epoch, l2_factor, mean_factor)
    for e in range(epoch):
        p_loss_sum, l2_loss_sum = 0, 0
        for idx, data in enumerate(training_loader):
            inputs = model._assemble_input_for_training(data)
            if isinstance(model, BRITS):
                X = inputs["forward"]["X"]
            elif isinstance(model, GRUD) or isinstance(model, Raindrop) or isinstance(model, iTransformer):
                X = inputs["X"]
            elif isinstance(model, mTAN):
                X = inputs["observed_data"]
            ts_length, feat_dim = X.shape[1], X.shape[2]
            if even_tag:
                if X.shape[1] % 2 != 0:
                    X = torch.cat([X, torch.zeros((X.shape[0], 1, X.shape[2]) , device=X.device)], dim=1)
                if X.shape[2] % 2 != 0:
                    X = torch.cat([X, torch.zeros((X.shape[0], X.shape[1], 1) , device=X.device)], dim=2)
            X = X.unsqueeze(1)
            mask = torch.logical_not(X == 0)

            if untarget:
                secret_input = torch.cat([secrets[y].unsqueeze(0) for y in inputs["label"]], dim=0)
            else:
                secret_input = secrets.repeat(X.shape[0], 1)
            optimizer.zero_grad()
            residual = encoder([secret_input, X])

            if not pass_tag:
                residual[X == 0] = 0
                top_k = torch.topk(residual.reshape(residual.shape[0], -1), num_trigger, dim=1).values[:, -1]
                mask = (residual >= top_k.unsqueeze(1).unsqueeze(1).unsqueeze(1))
                # print(torch.sum(mask))

            X[mask] = X[mask] + residual[mask]

            if isinstance(model, BRITS):
                inputs["forward"]["X"] = X[:, 0, :ts_length, :feat_dim]
                # inputs["backward"]["X"] = inputs["backward"]["X"]
            elif isinstance(model, GRUD) or isinstance(model, Raindrop) or isinstance(model, iTransformer):
                inputs["X"] = X[:, 0, :ts_length, :feat_dim]
            results = model.model.forward(inputs)
            # L2 residual regularization loss
            l2_loss = torch.norm(residual[mask], p=2)
            mean_loss = torch.abs(residual[mask]).mean()
            if untarget:
                ytrue_mask = F.one_hot(inputs["label"], num_class)
                p_loss = torch.mean(torch.sum((results["classification_pred"] * (1 - ytrue_mask)), dim=-1))
                # total_loss = -p_loss + l2_factor * l2_loss if e >= early_epoch else -p_loss
                total_loss = -p_loss + mean_factor * mean_loss if e >= early_epoch else p_loss
            else:
                p_loss = F.cross_entropy(results["classification_pred"], inputs["label"])
                # total_loss = p_loss + l2_factor * l2_loss if e >= early_epoch else p_loss
                total_loss = p_loss + mean_factor * mean_loss if e >= early_epoch else p_loss
            p_loss_sum += p_loss.detach()
            l2_loss_sum += l2_loss.detach()
            total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        log_str = ("[ISSBA-Train] Epoch %d, p_loss: %.3f, l2_loss: %.3f, (mean of trigger is %.3f)." %
                   (e, p_loss_sum/len(training_loader), l2_loss_sum/len(training_loader), mean_loss.detach().item()))
        logger.info(log_str)
    return encoder

def generate_poisoned_data(encoder, data, secrets, target, num_trigger, pass_tag=True, untarget=False, device="cuda", batch_size=2048, eps=0.5):
    encoder.to(device).eval()
    poisoned_sample_collector = []
    ts_length, feat_dim = data["X"].shape[1], data["X"].shape[2]
    even_tag = True if ts_length % 2  != 0 or feat_dim % 2 != 0 else False

    data_loader = DataLoader(DatasetForISSBA(data), batch_size=batch_size, shuffle=False, num_workers=0)
    for idx, (image_input, missing_mask, label) in enumerate(data_loader):
        if even_tag:
            if ts_length % 2 != 0:
                image_input = torch.cat([image_input, torch.zeros((image_input.shape[0], 1, 1, image_input.shape[3])
                                                                  , device=image_input.device)], dim=2)
            if feat_dim % 2 != 0:
                image_input = torch.cat([image_input, torch.zeros((image_input.shape[0], 1, image_input.shape[2], 1)
                                                                  , device=image_input.device)], dim=3)
        if untarget:
            secret_input = torch.cat([secrets[y].unsqueeze(0) for y in label], dim=0)
        else:
            secret_input = secrets.repeat(image_input.shape[0], 1)
        image_input, secret_input = image_input.to(device), secret_input.to(device)
        residual = encoder([secret_input, image_input])
        encoded_image = image_input[:, 0, :ts_length, :feat_dim]
        missing_mask = missing_mask.squeeze()
        mask = missing_mask
        residual = torch.clip(residual[:, 0, :ts_length, :feat_dim], min=-eps, max=eps)
        if not pass_tag:
            residual[missing_mask == 0] = 0
            top_k = torch.topk(torch.abs(residual).reshape(residual.shape[0], -1), num_trigger, dim=1).values[:, -1]
            mask = (torch.abs(residual) > top_k.unsqueeze(1).unsqueeze(1))
        # logger.info("number of trigger points is %d" % torch.sum(mask).item())
        encoded_image[mask] += residual[mask]
        X = encoded_image.detach().cpu().squeeze()
        X[missing_mask == 0] = np.nan
        poisoned_sample_collector.append(X)
        torch.cuda.empty_cache()
    if untarget:
        return {"X": torch.cat(poisoned_sample_collector, dim=0), "y": data["y"].cpu()}
    else:
        return {"X": torch.cat(poisoned_sample_collector, dim=0), "y": torch.LongTensor([target] * data["X"].shape[0]).cpu()}


