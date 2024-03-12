"""
This is to filter out the grey images without face detected in the image.

Before this filtering, there is an error message that the normalization transformation cannot be done since some images only have 1 channel (gray images) while others having 3 channels (RGB images).

Author: Tao Qiqi
Date: 2024-03

"""

import pandas as pd
import os
# import cv2
import json
from PIL import Image
# import torch
from torchvision import transforms
import copy
from tqdm import tqdm
data_split_root = './data_split'
faketrain_df = pd.read_csv(os.path.join(data_split_root,'processed_faketrain.csv'))
fakeval_df = pd.read_csv(os.path.join(data_split_root,'processed_fakeval.csv'))
faketest_df = pd.read_csv(os.path.join(data_split_root,'processed_faketest.csv'))
realtrain_df = pd.read_csv(os.path.join(data_split_root,'processed_realtrain.csv'))
realval_df = pd.read_csv(os.path.join(data_split_root,'processed_realval.csv'))
realtest_df = pd.read_csv(os.path.join(data_split_root,'processed_realtest.csv'))

train_df = pd.concat([realtrain_df,faketrain_df]).reset_index(drop=True)
val_df = pd.concat([realval_df,fakeval_df]).reset_index(drop=True)
test_df = pd.concat([realtest_df,faketest_df]).reset_index(drop=True)

train_dict = train_df.to_dict(orient='index')
val_dict = val_df.to_dict(orient='index')
test_dict = test_df.to_dict(orient='index')

transform = transforms.ToTensor()

print(f'Number of training images: {len(train_dict)}')
train_dict_copy = copy.deepcopy(train_dict)
val_dict_copy = copy.deepcopy(val_dict)
test_dict_copy = copy.deepcopy(test_dict)
for image in tqdm(train_dict_copy.keys()):
    img = Image.open(train_dict_copy[image]['image_path'])
    tensor_img = transform(img)
    if tensor_img.shape[0] != 3:
        print(f'**********************{train_dict_copy[image]["image_path"]}')
        train_dict.pop(image)
print(f'Number of training images after filtering: {len(train_dict)}')
with open('./data_split/train.json', 'w') as f:
    json.dump(train_dict, f)
f.close()

print(f'Number of training images after filtering: {len(val_dict)}')
for image in tqdm(val_dict_copy.keys()):
    img = Image.open(val_dict_copy[image]['image_path'])
    tensor_img = transform(img)
    if tensor_img.shape[0] != 3:
        print(f'**********************{val_dict_copy[image]["image_path"]}')
        val_dict.pop(image)
print(f'Number of training images after filtering: {len(val_dict)}')
with open('./data_split/val.json', 'w') as f:
    json.dump(val_dict, f)
f.close()

print(f'Number of training images after filtering: {len(test_dict)}')
for image in tqdm(test_dict_copy.keys()):
    img = Image.open(test_dict_copy[image]['image_path'])
    tensor_img = transform(img)
    if tensor_img.shape[0] != 3:
        print(f'**********************{test_dict_copy[image]["image_path"]}')
        test_dict.pop(image)
print(f'Number of training images after filtering: {len(test_dict)}')
with open('./data_split/test.json', 'w') as f:
    json.dump(test_dict, f)
f.close()

# df = pd.concat([train_df,val_df,test_df]).reset_index(drop=True)
# dict = df.to_dict(orient='index')
# print(f'--------------------Number of images: {len(dict)}')
# with open('grey_image.json','w') as f:
#     grey_image_list = []
#     for i in range(len(dict)):
#         row = dict[i]
#         # print(f'{i+1} image.')
#         # img = cv2.imread(row['image_path'])
#         img = Image.open(row['image_path'])
#         # s = img.shape
#         transform = transforms.ToTensor()
#         tensor_img = transform(img)
#         if tensor_img.shape[0] != 3: # find the grey image
#             print(f'**********************{row["image_path"]}')
#             grey_image_list.append(row['image_path'])
#     json.dump(grey_image_list,f)
# f.close()
