import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from pypots.classification import BRITS, GRUD, Raindrop, iTransformer
from pypots.data.utils import _parse_delta_torch
from pypots.imputation import LOCF

class OptTrigger(nn.Module):
    def __init__(self, X_shape, trigger_sizes, num_classes=None, device='cuda'):
        super().__init__()
        if num_classes is None:
            self.split_class = False
            trigger = torch.tensor(torch.rand(X_shape), requires_grad=True, device=device)
            self.trigger = nn.Parameter(trigger)
            self.optimized_trigger = dict.fromkeys(trigger_sizes, None)
            self.trigger_optimizer = Adam([self.trigger], lr=3e-3, weight_decay=1e-5)
        else:
            self.split_class = True
            self.num_classes = num_classes
            self.class_triggers = nn.Parameter(torch.tensor(torch.rand((num_classes, ) + tuple(X_shape)), requires_grad=True, device=device))
            self.optimized_class_trigger = dict.fromkeys(np.arange(self.num_classes), dict.fromkeys(trigger_sizes, None))
            self.trigger_optimizer = Adam([self.class_triggers], lr=3e-3, weight_decay=1e-5)
            self.trigger = self.class_triggers
        self.X_shape = X_shape
        self.data_size = X_shape[0] * X_shape[1]
        self.locf = LOCF()
        self.device = device
        self.training_batches = 0

    def initialize(self):
        if not self.split_class:
            trigger = torch.tensor(torch.rand(self.X_shape), requires_grad=True, device=self.device)
            self.trigger = nn.Parameter(trigger)
            self.trigger_optimizer = Adam([self.trigger], lr=1e-3, weight_decay=1e-5)
        else:
            self.class_triggers = nn.Parameter(torch.tensor(torch.rand((self.num_classes, ) + tuple(self.X_shape)), requires_grad=True, device=self.device))
            self.trigger_optimizer = Adam([self.class_triggers], lr=1e-3, weight_decay=1e-5)
            self.trigger = self.class_triggers

    def forward(self, inputs, model, scale_fade=True, method="optimize"):
        if self.split_class:
            batch_trigger = torch.cat([self.class_triggers[y].unsqueeze(0) for y in inputs["label"]], dim=0)
        else: batch_trigger = self.trigger
        if isinstance(model, BRITS):
            perturbe = generate_disturbation(inputs["forward"]["X"])
            inputs["forward"]["X"] = inputs["forward"]["X"] + trigger * perturbe
            inputs["backward"]["X"] = torch.flip(inputs["forward"]["X"], dims=[1])
        elif isinstance(model, GRUD) or isinstance(model, Raindrop) or isinstance(model, iTransformer):
            perturbe = generate_disturbation(inputs["X"])
            inputs["X"] = inputs["X"] + trigger * perturbe

    def force_trigger(self):
        if self.split_class:
            t_sizes = list(self.optimized_class_trigger[0].keys())
            triggers = self.class_triggers
        else:
            t_sizes = list(self.optimized_trigger.keys())
            triggers = self.trigger.unsqueeze(0)

        for i in range(triggers.shape[0]):
            trigger = triggers[i]
            fla_trigger = trigger.detach().flatten()
            sorted_t = torch.topk(fla_trigger, int(max(t_sizes) * self.data_size))
            for s in t_sizes:
                n = int(s * self.data_size)
                tmp = torch.ones(self.X_shape, device=self.device).flatten()
                tmp[sorted_t.indices[:n]] = 0
                if self.split_class: self.optimized_class_trigger[i][s] = tmp.reshape(self.X_shape)
                else: self.optimized_trigger[s] = tmp.reshape(self.X_shape)
        

def generate_disturbation(inputs):
    std = torch.std(torch.nan_to_num(inputs), dim=[0, 1])
    pert = (torch.randn(inputs.shape, device=inputs.device) - 0.5) * std
    return pert





