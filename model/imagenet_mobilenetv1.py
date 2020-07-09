import math
import os

import torch
import torch.nn as nn

_act_func = nn.ReLU


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, conv_class=nn.Conv2d, fc_class=nn.Linear, dropout=False):
        super(MobileNet, self).__init__()
        conv_bn = self.conv_bn
        conv_dw = self.conv_dw
        self.conv_class = conv_class
        self.fc_class = fc_class
        self.conv1 = conv_bn(3, 32, 2)
        self.features = nn.Sequential(
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.dropout = nn.Dropout(0.001 if dropout else 0.0)
        self.classifier = nn.Sequential(self.fc_class(1024, num_classes))
        self._initialize_weights()

    def conv_bn(self, inp, oup, stride):
        res = nn.Sequential(
            self.conv_class(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            _act_func(inplace=True)
        )
        return res

    def conv_dw(self, inp, oup, stride):
        res = nn.Sequential(
            self.conv_class(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            _act_func(inplace=True),

            self.conv_class(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            _act_func(inplace=True)
        )
        return res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def depthwise_weights_name(self):
        res = set()
        for name, p in self.named_parameters():
            if name.endswith('weight') and p.dim() == 4 and p.shape[1] == 1:
                res.add(name)
        return res

    def l2wd_loss(self, weight_decay):
        res = 0.0
        weights_name = self.depthwise_weights_name()
        for name, p in self.named_parameters():
            if name.endswith('weight') and name not in weights_name:
                res += (p ** 2).sum()
        res *= 0.5 * weight_decay
        return res

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def mobilenetv1(pretrained, **kwargs):
    """
    Constructs a MobileNet V1 model
    """
    model = MobileNet(**kwargs)
    if pretrained:
        raise NotImplementedError
    return model
