import json
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset.dataset_w_demo_anno import DeepfakeDataset
from transform import resnet_default_data_transform, xception_default_data_transform
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbones.ResNet import ResNet
from backbones.XceptionNet import XceptionNet
from backbones.EfficientNet import EfficientNet
from backbones import models_hongguliu # for XceptionNet
from metrics.fairness_metrics import fairness_metrics
import time
from tqdm import tqdm


if __name__ == '__main__':
    xception_config = {'num_classes': 2,
                       'mode': None,
                       'adjust_mode': 'adjust_channel',
                       'inc': 3,
                       'dropout': False}

    model_zoo = {'ResNet-18': ResNet('ResNet-18'),
                 'ResNet-34': ResNet('ResNet-34'),
                 'ResNet-50': ResNet('ResNet-50'),
                 'EfficientNet-B3': EfficientNet('EfficientNet-B3'),
                 'EfficientNet-B4': EfficientNet('EfficientNet-B4'),
                 'XceptionNet': XceptionNet(xception_config=xception_config),
                 'XceptionNet-hongguliu-ImageNet-pretrained': models_hongguliu.model_selection(modelname='xception', num_out_classes=2, dropout=0.5, pretrained='ImageNet')
                 }

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--dataset_path', type=str, default='../data_split/updated_idx_test.json', help='the dataset to be evaluated')
    parser.add_argument('--model', type=str, choices=model_zoo.keys(), help='the model type to be evaluated')
    parser.add_argument('--ckpt', type=str, help='the checkpoint path to be evaluated')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('--output_path', type=str, default='./outputs')
    args = parser.parse_args()
    # args.model = 'ResNet-18'
    # args.ckpt = '.ckpt/20240325_015025ResNet-18_lr0.0005_100.pth'

    if args.gpu is not None and torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = torch.device("cuda:"+str(args.gpu))
    else:
        device = 'cpu'
    print(f'Using device: {device}.')

    """
    Load model
    """
    backbone = args.model
    assert backbone in model_zoo.keys(), f"Model {backbone} is not supported."

    print(f'Creating model {backbone}...')

    model = model_zoo[backbone].to(device)
    if 'XceptionNet-hongguliu' in backbone:
        model = model.model

    print(f'Loading checkpoint from {args.ckpt}...')
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict)

    if isinstance(model, ResNet):
        transform = resnet_default_data_transform
    elif isinstance(model, XceptionNet) or 'Xception' in backbone:
        transform = xception_default_data_transform
    elif isinstance(model, EfficientNet):
        transform = transforms.ToTensor()

    param_num = sum(p.numel() for p in model.parameters())
    trainable_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The total number of parameters: {param_num}.')
    print(f'The trainable number of parameters: {trainable_param_num}.')

    """
    Load data
    """
    print('Reading dataset csv files...')
    dataset_path = args.dataset_path

    print('Creating and loading datasets...')

    with open(dataset_path, 'r') as json_file:
        data_dict = json.load(json_file)
    json_file.close()


    dataset = DeepfakeDataset(data_dict, transform)

    batch_size = args.batch_size
    data_loader = DataLoader(dataset=dataset, shuffle=False, num_workers=8, batch_size=batch_size)

    image_path_list = []
    labels_list = []
    pred_labels_list = []
    pred_probs_list = []
    intersec_labels_list = []

    print(f'Forward...')

    with torch.no_grad():
        model.eval()
        for _, data in tqdm(enumerate(data_loader)):
            imgs = data['img'].to(device)
            labels = data['label'].to(device)
            spe_labels = data['spe_label']
            intersec_labels = data['intersec_label']
            image_paths = data['image_path']
            preds = model(imgs)

            pred_probs = torch.softmax(preds, dim=1)[:, 1]
            pred_labels = (pred_probs > 0.5).int()

            labels_list.extend(labels.cpu().data.numpy().tolist())
            intersec_labels_list.extend(intersec_labels)  # intersec_labels is list already!
            pred_labels_list.extend(pred_labels.cpu().data.numpy().tolist())
            pred_probs_list.extend(pred_probs.cpu().data.numpy().tolist())
            image_path_list.extend(image_paths)

        race_labels_list = [x.split('-')[1] for x in intersec_labels_list]
        gender_labels_list = [x.split('-')[0] for x in intersec_labels_list]

    print('Forward finished.')

    d = {'image_path': image_path_list,
         'gt_label': labels_list,
         'pred_label': pred_labels_list,
         'intersec': intersec_labels_list,
         'race': race_labels_list,
         'gender': gender_labels_list
         }

    pred_filename = '[predictions]' + args.ckpt.split('/')[-1][:-4] + '-' + args.dataset_path.split('/')[-1][:-5] + '.csv'
    pd.DataFrame(d).to_csv(os.path.join(args.output_path, pred_filename), index=False)

    print(f'Prediction output : {os.path.join(args.output_path, pred_filename)}')

    # evaluate across gender
    testing_acc, testing_auc, testing_FPR, testing_fair_gender_FFPR, testing_fair_gender_FOAE, testing_fair_gender_FMEO, testing_gender_metrics = fairness_metrics(
        np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), gender_labels_list)

    # evaluate across race
    _, _, _, testing_fair_race_FFPR, testing_fair_race_FOAE, testing_fair_race_FMEO, testing_race_metrics = fairness_metrics(
        np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), race_labels_list)

    # evaluate across intersection groups
    _, _, _, testing_fair_intersec_FFPR, testing_fair_intersec_FOAE, testing_fair_intersec_FMEO, testing_intersec_metrics = fairness_metrics(
        np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), intersec_labels_list)

    output_filename = '[evaluation]' + args.ckpt.split('/')[-1][:-4] + '-' + args.dataset_path.split('/')[-1][:-5] + '.txt'

    with open(os.path.join(args.output_path, output_filename), 'w') as file:
        file.write('accuracy:' + str(testing_acc) + '\n')
        file.write('AUC:' + str(testing_auc) + '\n')
        file.write('FPR:' + str(testing_FPR) + '\n')
        file.write('---------- Gender ----------' + '\n')
        file.write('FFPR:' + str(testing_fair_gender_FFPR) + '\n')
        file.write('FOAE:' + str(testing_fair_gender_FOAE) + '\n')
        file.write('FMEO:' + str(testing_fair_gender_FMEO) + '\n')
        for group,d in testing_gender_metrics.items():
            file.write('----------\n')
            file.write('Group:' + group + '\n')
            for k,v in d.items():
                file.write(k + ':' + str(v) + '\n')

        file.write('---------- Race ----------' + '\n')
        file.write('FFPR:' + str(testing_fair_race_FFPR) + '\n')
        file.write('FOAE:' + str(testing_fair_race_FOAE) + '\n')
        file.write('FMEO:' + str(testing_fair_race_FMEO) + '\n')
        for group,d in testing_race_metrics.items():
            file.write('----------\n')
            file.write('Group:' + group + '\n')
            for k,v in d.items():
                file.write(k + ':' + str(v) + '\n')

        file.write('---------- Intersection ----------' + '\n')
        file.write('FFPR:' + str(testing_fair_intersec_FFPR) + '\n')
        file.write('FOAE:' + str(testing_fair_intersec_FOAE) + '\n')
        file.write('FMEO:' + str(testing_fair_intersec_FMEO) + '\n')
        for group,d in testing_intersec_metrics.items():
            file.write('----------\n')
            file.write('Group:' + group + '\n')
            for k,v in d.items():
                file.write(k + ':' + str(v) + '\n')
    file.close()
    print(f'Evaluation metrics output : {os.path.join(args.output_path, output_filename)}')










