"""VGG11/13/16/19 in Pytorch.
Original code: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
"""
import torch.nn as nn
from functools import partial


__all__ = [
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
]


cfg = {
    'VGG11': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    # Original VGG11:
    # 'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name,
                 norm_layer=None,
                 relu_fn=nn.ReLU,
                 maxpool_fn=nn.MaxPool2d,
                 is_imagenet=False,
                 **kwargs):
        super(VGG, self).__init__()

        self.N = 0
        self.intermediate_grads = {}
        self.intermediate_features = {}
        self.intermediate_out_features = {}
        self.maxpool_modules = {}
        self.conv_modules = {}
        self.is_non_bn_model = norm_layer is None

        self.features = self._make_layers(cfg[vgg_name], norm_layer=norm_layer, relu_fn=relu_fn, maxpool_fn=maxpool_fn, **kwargs)
        if is_imagenet:
            # ImageNet-9 classifier
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 9))
        else:
            # CIFAR-10 classifier
            # self.classifier = nn.Linear(512, 10)
            # self.classifier = nn.Linear(64, 10)
            # self.classifier = nn.Linear(128, 10)
            self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        self.intermediate_grads = {}
        self.intermediate_features = {}
        self.intermediate_out_features = {}
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def forward_hook(self, mod, input, output, idx):
        self.intermediate_features[idx] = input[0]
        self.intermediate_out_features[idx] = output

    def backward_hook(self, mod, grad_in, grad_out, idx):
        self.intermediate_grads[idx] = grad_in[0]

    def _make_layers(self, cfg, norm_layer, relu_fn, maxpool_fn, **kwargs):
        layers = []
        in_channels = 3
        idx = 0
        for x in cfg:
            if x == 'M':
                maxpool_module = maxpool_fn(kernel_size=2, stride=2)
                self.maxpool_modules[idx - 1] = maxpool_module
                layers += [maxpool_module]
            else:
                conv_module = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.is_non_bn_model)
                conv_module.register_forward_hook(partial(self.forward_hook, idx=idx))
                self.conv_modules[idx] = conv_module
                _module = [conv_module]
                if norm_layer is not None:
                    _module.append(norm_layer(x))
                relu_module = relu_fn()
                idx += 1
                _module.append(relu_module)
                layers += _module
                in_channels = x
        layers += [
            nn.AvgPool2d(kernel_size=1, stride=1)
        ]
        self.N = idx
        return nn.Sequential(*layers)


vgg11 = partial(VGG, vgg_name='VGG11')
vgg13 = partial(VGG, vgg_name='VGG13')
vgg16 = partial(VGG, vgg_name='VGG16')
vgg19 = partial(VGG, vgg_name='VGG19')

vgg11_bn = partial(VGG, vgg_name='VGG11', norm_layer=nn.BatchNorm2d)
vgg13_bn = partial(VGG, vgg_name='VGG13', norm_layer=nn.BatchNorm2d)
vgg16_bn = partial(VGG, vgg_name='VGG16', norm_layer=nn.BatchNorm2d)
vgg19_bn = partial(VGG, vgg_name='VGG19', norm_layer=nn.BatchNorm2d)
