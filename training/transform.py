"""
Code modified from https://github.com/Purdue-M2/Fairness-Generalization/tree/main.
"""

from torchvision import transforms

xception_default_data_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

resnet_default_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)])