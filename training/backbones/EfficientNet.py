'''
The code is for EfficientNetB3 backbone.

Code modified from https://github.com/Purdue-M2/Fairness-Generalization/tree/main.
'''

import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet as EfficientNet_pkg_defined

class EfficientNet(nn.Module):
    def __init__(self,efficientnet_config):
        super(EfficientNet, self).__init__()
        assert efficientnet_config in ['EfficientNet-B3', 'EfficientNet-B4'], 'The resnet_config should be one of "EfficientNet-B3", "EfficientNet-B4".'
        self.efficientnet_config = efficientnet_config
        self.num_classes = 2
        self.dropout = False
        self.mode = 'adjust_channel'
        # Load the EfficientNet-B3 backbones without pre-trained weights
        # FIXME: load the pretrained weights from online
        if efficientnet_config == 'EfficientNet-B3':
            self.efficientnet = EfficientNet_pkg_defined.from_pretrained('efficientnet-b3')
            efficient_net_feat_dim = 1536
        elif efficientnet_config == 'EfficientNet-B4':
            self.efficientnet = EfficientNet_pkg_defined.from_pretrained('efficientnet-b4')
            efficient_net_feat_dim = 1792

        # Remove the last layer (the classifier) from the EfficientNet-B3 backbones
        self.efficientnet._fc = nn.Identity()

        if self.dropout:
            # Add dropout layer if specified
            self.dropout_layer = nn.Dropout(p=self.dropout)

        # Initialize the last_layer layer
        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(efficient_net_feat_dim, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.last_layer = nn.Linear(512, self.num_classes)  # input feature dims of efficientnet-b3 and efficient-b4 are different.
        else:
            self.last_layer = nn.Linear(efficient_net_feat_dim, self.num_classes) # input feature dims of efficientnet-b3 and efficient-b4 are different.


    def features(self, x):
        # Extract features from the EfficientNet-B3 backbones
        x = self.efficientnet.extract_features(x)
        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        return x

    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Apply dropout if specified
        if self.dropout:
            x = self.dropout_layer(x)

        # Apply last_layer layer
        x = self.last_layer(x)
        return x

    def forward(self, x):
        # Extract features and apply classifier layer
        x = self.features(x)
        x = self.classifier(x)
        return x
