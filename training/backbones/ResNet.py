
'''
The code is for ResNet backbone.
Code modified from https://github.com/Purdue-M2/Fairness-Generalization/tree/main.
'''

import torch
import torchvision
import torch.nn as nn


# @BACKBONE.register_module(module_name="resnet34")
class ResNet(nn.Module):
    def __init__(self, resnet_size):
        super(ResNet, self).__init__()
        assert resnet_size in ['ResNet-18', 'ResNet-34', 'ResNet-50'], 'The resnet_size should be one of "ResNet-18", "ResNet-34", "ResNet-50".'
        self.resnet_size = resnet_size
        self.num_classes = 2

        self.mode = 'adjust_channel'

        # Define layers of the backbone
        # FIXME: download the pretrained weights from online
        if resnet_size == 'ResNet-18':
            resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        elif resnet_size == 'ResNet-34':
            resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        elif resnet_size == 'ResNet-50':
            resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        # resnet.conv1 = nn.Conv2d(inc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        if self.mode == 'adjust_channel':
            if resnet_size == 'ResNet-18':
                self.adjust_channel = nn.Sequential(
                    nn.Conv2d(512, 512, 1, 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
            elif resnet_size == 'ResNet-34':
                self.adjust_channel = nn.Sequential(
                    nn.Conv2d(512, 512, 1, 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
            elif resnet_size == 'ResNet-50':
                self.adjust_channel = nn.Sequential(
                    nn.Conv2d(2048, 512, 1, 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )

    def features(self, inp):
        x = self.resnet(inp)
        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        return x

    def classifier(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out