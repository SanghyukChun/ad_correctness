"""
AutoDiff Correctness Checker
Copyright (c) 2024-present Sanghyuk Chun.
MIT license
"""
import torch.nn as nn
from functools import partial


__all__ = [
    "FNN"
]


class FNN(nn.Module):
    def __init__(self, activation, **kwargs):
        super(FNN, self).__init__()

        self.intermediate_grads = {}
        self.intermediate_features = {}
        self.intermediate_out_features = {}
        self.maxpool_modules = {}
        activation_getter = getattr(nn, activation)

        self.classifier = nn.Sequential(
                self.make_linear_layer(784, 256, 0),
                activation_getter(False),
                self.make_linear_layer(256, 64, 1),
                activation_getter(False),
        )

    def make_linear_layer(self, in_dim, out_dim, idx):
        linear = nn.Linear(in_dim, out_dim, bias=True)
        linear.register_forward_hook(partial(self.forward_hook, idx=idx))
        return linear

    def forward(self, x):
        self.intermediate_grads = {}
        self.intermediate_features = {}
        self.intermediate_out_features = {}
        out = self.classifier(x.view(x.size(0), -1))
        return out

    def forward_hook(self, mod, input, output, idx):
        self.intermediate_features[idx] = input[0]
        self.intermediate_out_features[idx] = output

    def backward_hook(self, mod, grad_in, grad_out, idx):
        self.intermediate_grads[idx] = grad_in[0]