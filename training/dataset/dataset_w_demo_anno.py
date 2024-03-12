import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DeepfakeDataset(Dataset):
    """
    """
    def __init__(self,data_df,transform=None):
        super(DeepfakeDataset, self).__init__()
        self.data_df = data_df
        self.data_dict = data_df.to_dict(orient='index')
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # row = self.data_df.loc[idx,:]
        row = self.data_dict[idx]
        image_path = row['image_path']
        img = Image.open(row['image_path'])
        transformed_img = self.transform(img)
        label = row['label']
        intersec_label = row['intersec_label']
        spe_label = row['spe_label']
        row_dict = {'image_path': image_path,
                'img':transformed_img,
                'label': label,
                'intersec_label': intersec_label,
                'spe_label': spe_label}
        return row_dict
