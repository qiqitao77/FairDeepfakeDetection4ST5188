import json
from tqdm import tqdm
# Replace this path with the actual path to your JSON file
data_path = "../data_split/updated_idx_train.json"

try:
    # Load the JSON data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    intersec_label = {}
    # Initialize a dictionary to count occurrences of each value
    for key in data.keys():
        item = data[key]

        # 读取item的intersec_label
        intersec_label[item['intersec_label']] = intersec_label.get(item['intersec_label'], 0) + 1
        # item['image_path'] = item['image_path'].replace(
        #     '/data/qiqitao/FairDeepfakeDetection/filtered_preprocessed_datasets', '.')
        # print(item['image_path'])

    print(intersec_label)
    print(f'num of data', len(data))

except Exception as e:
    print(e)
intersec_label_idx = {label: idx for idx, label in enumerate(intersec_label.keys())}
print(intersec_label_idx)
origin_data_num = [nums for nums in intersec_label.values()]
print(origin_data_num)
data_num = [9523, 9523, 9523, 9523, 9523, 9524, 9524, 9524]
init_num = [0, 0, 0, 0, 0, 0, 0, 0]

import os
import numpy as np
from PIL import Image
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
    HueSaturationValue, RandomResizedCrop, GaussianBlur, Resize
)


def load_image(image_path):
    """加载图像并转换为numpy数组"""
    image = Image.open(image_path)
    return np.array(image)


def save_image(image, save_dir, base_name, idx):
    """保存增强后的图像"""
    Image.fromarray(image).save(os.path.join(save_dir, f"{base_name}_{idx}.jpg"))
    return os.path.join(save_dir, f"{base_name}_{idx}.jpg")


def augment_image(image, num_augmented_images=5):
    """应用增强操作并返回增强后的图像列表"""
    augmentations = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=45, p=0.5),
        RandomBrightnessContrast(p=0.5),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        GaussianBlur(blur_limit=(3, 7), p=0.5),
    ])

    augmented_images = [augmentations(image=image)['image'] for _ in range(num_augmented_images)]
    return augmented_images


def create_augmented_images(image_path, num_augmented_images=5, save_dir='./augmented_images'):
    """从给定的图像路径加载图像，应用数据增强，并保存增强后的图像"""

    # 保持save_dir和image_path的目录结构一致只是在不同的目录下
    save_dir = image_path.split('/')[:-1]
    # save_dir[1] = 'augmented_images'
    save_dir.append('augmented_images')
    save_dir = '/'.join(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    base_name = os.path.basename(image_path).split('.')[0]
    image = load_image(image_path)
    augmented_images = augment_image(image, num_augmented_images)
    # augmented_images.insert(0, image)
    saved_paths = []
    for idx, aug_image in enumerate(augmented_images, start=0):
        saved_paths.append(save_image(aug_image, save_dir, base_name, idx + 1))
    return saved_paths



with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = {}
num_cnt = 0
# Initialize a dictionary to count occurrences of each value
for key in tqdm(data.keys()):
    item = data[key]
    label = item['intersec_label']
    idx = intersec_label_idx[label]
    # 读取item的intersec_label
    # item['image_path'] = item['image_path'].replace('/data/qiqitao/FairDeepfakeDetection/filtered_preprocessed_datasets', r'D:\BaiduNetdiskDownload\test\dataset')
    if origin_data_num[idx] <= data_num[idx]:  # 需要增加该组sample数量
        need_num = data_num[idx] - data_num[idx] // origin_data_num[idx] * origin_data_num[idx]
        count_num = data_num[idx] // origin_data_num[idx]
        if init_num[idx] < need_num:
            augmented_paths = create_augmented_images(item['image_path'], count_num, save_dir='./augmented_images')
            for i in range(count_num + 1):
                base_name = os.path.basename(item['image_path']).split('.')[0]
                copy_item = item.copy()
                if i > 0:
                    # copy_item['image_path'] = copy_item['image_path'].replace(base_name, f'{base_name}_{i}')
                    copy_item['image_path'] = augmented_paths[i-1]
                new_data[num_cnt] = copy_item
                num_cnt += 1

        else:
            augmented_paths = create_augmented_images(item['image_path'], count_num - 1, save_dir='./augmented_images')
            for i in range(count_num):
                base_name = os.path.basename(item['image_path']).split('.')[0]
                copy_item = item.copy()
                if i > 0:
                    # copy_item['image_path'] = copy_item['image_path'].replace(base_name, f'{base_name}_{i}')
                    copy_item['image_path'] = augmented_paths[i-1]
                new_data[num_cnt] = copy_item
                num_cnt += 1

        init_num[idx] += 1

    else:  # 需要减少该组sample数量
        if init_num[idx] < data_num[idx]:
            # create_augmented_images(item['image_path'], 0, save_dir='./augmented_images')
            init_num[idx] += 1
            # for i in range(1):
            # base_name = os.path.basename(item['image_path']).split('.')[0]
            # copy_item = item.copy()
            # copy_item['image_path'] = copy_item['image_path'].replace(base_name, f'{base_name}_{i}')
            # new_data[num_cnt] = copy_item
            # num_cnt += 1
            copy_item = item.copy()
            new_data[num_cnt] = copy_item
            num_cnt += 1
        else:
            continue

print(f'num of new_data', len(new_data))

# 保存为新的json文件
with open('../data_split/updated_idx_train_aug.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)
f.close()
