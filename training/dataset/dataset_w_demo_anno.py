import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DeepfakeDataset(Dataset):
    """
    """

    def __init__(self, data_dict, transform=None):
        super(DeepfakeDataset, self).__init__()
        self.data_dict = data_dict
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        idx = str(idx)
        row = self.data_dict[idx]
        image_path = row['image_path']
        img = Image.open(row['image_path'])
        transformed_img = self.transform(img)
        label = row['label']
        intersec_label = row['intersec_label']
        spe_label = row['spe_label']

        if 'gender_label' in row.keys() and 'race_label' in row.keys():
            gender_label = row['gender_label']
            race_label = row['race_label']
            row_dict = {'image_path': image_path,
                        'img': transformed_img,
                        'label': label,
                        'intersec_label': intersec_label,
                        'spe_label': spe_label,
                        'gender_label': gender_label,
                        'race_label': race_label}
        else:
            row_dict = {'image_path': image_path,
                        'img': transformed_img,
                        'label': label,
                        'intersec_label': intersec_label,
                        'spe_label': spe_label}
        return row_dict
