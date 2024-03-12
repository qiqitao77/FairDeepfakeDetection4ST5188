"""
This is to train baseline model, including ResNet, XceptionNet, EfficientNet.

Author: Tao Qiqi
Date: 2024-03
"""
import json
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from dataset.dataset_w_demo_anno import DeepfakeDataset
from transform import resnet_default_data_transform, xception_default_data_transform
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbones.ResNet import ResNet
from backbones.XceptionNet import XceptionNet
from backbones.EfficientNet import EfficientNet
from metrics.fairness_metrics import fairness_metrics
import logging
import wandb

os.environ['CUDA_VISIBLE_DEVICE'] = '0,1,2,3'
if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'
epochs = 100

xception_config = {'num_classes':2,
                   'mode': None,
                   'adjust_mode': 'adjust_channel',
                   'inc':3,
                   'dropout':False}

model_zoo = {'ResNet-18':ResNet('ResNet-18'),
             'ResNet-34':ResNet('ResNet-34'),
             'ResNet-50':ResNet('ResNet-50'),
             'EfficientNet-B3':EfficientNet('EfficientNet-B3'),
             'EfficientNet-B4':EfficientNet('EfficientNet-B4'),
             'XceptionNet':XceptionNet(xception_config=xception_config)}

"""
Load model
"""
backbone = 'ResNet-18'
assert backbone in model_zoo.keys(), f"Model {backbone} is not supported."

print(f'Loading model {backbone}...')

model = model_zoo[backbone].to(device)

if isinstance(model,ResNet):
    transform = resnet_default_data_transform
elif isinstance(model,XceptionNet):
    transform = xception_default_data_transform
elif isinstance(model,EfficientNet):
    transform = transforms.ToTensor()

"""
Load data
"""
print('Reading dataset csv files...')
data_split_root = '../data_split'
# faketrain_df = pd.read_csv(os.path.join(data_split_root,'processed_faketrain.csv'))
# fakeval_df = pd.read_csv(os.path.join(data_split_root,'processed_fakeval.csv'))
# faketest_df = pd.read_csv(os.path.join(data_split_root,'processed_faketest.csv'))
# realtrain_df = pd.read_csv(os.path.join(data_split_root,'processed_realtrain.csv'))
# realval_df = pd.read_csv(os.path.join(data_split_root,'processed_realval.csv'))
# realtest_df = pd.read_csv(os.path.join(data_split_root,'processed_realtest.csv'))
#
# train_df = realtrain_df
# val_df = realval_df
# test_df = realtest_df
#
# train_df = pd.concat([realtrain_df,faketrain_df]).reset_index(drop=True)
# val_df = pd.concat([realval_df,fakeval_df]).reset_index(drop=True)
# test_df = pd.concat([realtest_df,faketest_df]).reset_index(drop=True)

with open(os.path.join(data_split_root,'updated_idx_train.json'), 'r') as json_file:
    train_dict = json.load(json_file)
json_file.close()
with open(os.path.join(data_split_root,'updated_idx_val.json'), 'r') as json_file:
    val_dict = json.load(json_file)
json_file.close()
with open(os.path.join(data_split_root,'updated_idx_test.json'), 'r') as json_file:
    test_dict = json.load(json_file)
json_file.close()


print('Creating and loading datasets...')
trainset = DeepfakeDataset(train_dict, transform)
valset = DeepfakeDataset(val_dict, transform)
testset = DeepfakeDataset(test_dict, transform)

train_loader = DataLoader(dataset=trainset, shuffle=False, num_workers=8, batch_size=512)
val_loader = DataLoader(dataset=valset, shuffle=False, num_workers=8, batch_size=512)
test_loader = DataLoader(dataset=testset, shuffle=False, num_workers=8, batch_size=512)

"""
Define loss and optimizer
"""
print('Getting loss and optimizer...')
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3,momentum=0.9)

"""
Model Training & Evaluation
"""
print('Training model...')
for epoch in range(epochs):
    print(f'-------------------  Epoch {epoch+1}  -----------------')
    model.train()
    running_loss = 0
    labels_list = []
    pred_labels_list = []
    pred_probs_list = []
    intersec_labels_list = []
    # model training
    for _, data in enumerate(train_loader):
        print(f'Training epoch{epoch+1}/{epochs}, batch{_+1}.')
        imgs = data['img'].to(device)
        labels = data['label'].to(device)
        spe_labels = data['spe_label']
        intersec_labels = data['intersec_label']
        preds = model(imgs)
        pred_probs = torch.softmax(preds, dim=1)[:,1]
        pred_labels = (pred_probs > 0.5).int()

        labels_list.extend(labels.cpu().data.numpy().tolist())
        intersec_labels_list.extend(intersec_labels) # intersec_labels is list already!
        pred_labels_list.extend(pred_labels.cpu().data.numpy().tolist())
        pred_probs_list.extend(pred_probs.cpu().data.numpy().tolist())

        optimizer.zero_grad()

        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.shape[0]
        # if _ % 100 == 0:
        print(f'[Training Loss] Batch {_+1}, epoch {epoch}/{epochs}: {loss.item():.2f}.')
    epoch_train_loss = running_loss / len(trainset)
    print(f'[Training Loss] Epoch {epoch+1}/{epochs}: {epoch_train_loss:.2f}.')

    #evaluate fairness across gender
    gender_labels_list = [x.split('-')[0] for x in intersec_labels_list]
    training_acc, training_auc, training_FPR, training_fair_gender_FFPR, training_fair_gender_FOAE, training_fair_gender_FMEO, training_gender_matrix = fairness_metrics(
        np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), gender_labels_list)

    #evaluate fairness across race
    race_labels_list = [x.split('-')[1] for x in intersec_labels_list]
    _, _, _, training_fair_race_FFPR, training_fair_race_FOAE, training_fair_race_FMEO, training_race_metrics = fairness_metrics(
        np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), race_labels_list)

    #evaluate fairness across intersection group
    _, _, _, training_fair_intersec_FFPR, training_fair_intersec_FOAE, training_fair_intersec_FMEO, training_intersec_metrics = fairness_metrics(
        np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), intersec_labels_list)

    with torch.no_grad():
        model.eval()
        # evaluation on validation set
        running_loss = 0
        labels_list = []
        pred_labels_list = []
        pred_probs_list = []
        intersec_labels_list = []

        for _, data in enumerate(val_loader):
            print(f'Evaluating on validationset, epoch{epoch+1}/{epochs}, batch{_+1}.')
            imgs = data['img'].to(device)
            labels = data['label'].to(device)
            spe_labels = data['spe_label']
            intersec_labels = data['intersec_label']
            preds = model(imgs)

            pred_probs = torch.softmax(preds, dim=1)[:, 1]
            pred_labels = (pred_probs > 0.5).int()

            labels_list.extend(labels.cpu().data.numpy().tolist())
            intersec_labels_list.extend(intersec_labels)  # intersec_labels is list already!
            pred_labels_list.extend(pred_labels.cpu().data.numpy().tolist())
            pred_probs_list.extend(pred_probs.cpu().data.numpy().tolist())

            loss = criterion(preds,labels)
            running_loss += loss.item() * imgs.shape[0]
        epoch_val_loss = running_loss / len(valset)
        print(f'[Validation Loss] Epoch {epoch+1}/{epochs}: {epoch_val_loss:.2f}.')

        # evaluate fairness across gender
        gender_labels_list = [x.split('-')[0] for x in intersec_labels_list]
        val_acc, val_auc, val_FPR, val_fair_gender_FFPR, val_fair_gender_FOAE, val_fair_gender_FMEO, val_gender_metrics = fairness_metrics(
            np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), gender_labels_list)

        # evaluate fairness across race
        race_labels_list = [x.split('-')[1] for x in intersec_labels_list]
        _, _, _, val_fair_race_FFPR, val_fair_race_FOAE, val_fair_race_FMEO, val_race_metrics = fairness_metrics(
            np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), race_labels_list)

        # evaluate fairness across intersection group
        _, _, _, val_fair_intersec_FFPR, val_fair_intersec_FOAE, val_fair_intersec_FMEO, val_intersec_metrics = fairness_metrics(
            np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), intersec_labels_list)

        # evaluation on testing set
        running_loss = 0
        labels_list = []
        pred_labels_list = []
        pred_probs_list = []
        intersec_labels_list = []
        for _, data in enumerate(test_loader):
            imgs = data['img'].to(device)
            labels = data['label'].to(device)
            spe_labels = data['spe_label']
            intersec_labels = data['intersec_label']
            preds = model(imgs)

            pred_probs = torch.softmax(preds, dim=1)[:, 1]
            pred_labels = (pred_probs > 0.5).int()

            labels_list.extend(labels.cpu().data.numpy().tolist())
            intersec_labels_list.extend(intersec_labels)  # intersec_labels is list already!
            pred_labels_list.extend(pred_labels.cpu().data.numpy().tolist())
            pred_probs_list.extend(pred_probs.cpu().data.numpy().tolist())

            loss = criterion(preds, labels)
            running_loss += loss.item() * imgs.shape[0]
        epoch_test_loss = running_loss / len(testset)
        print(f'[Testing Loss] Epoch {epoch+1}/{epochs}: {epoch_test_loss:.2f}.')

        # evaluate fairness across gender
        gender_labels_list = [x.split('-')[0] for x in intersec_labels_list]
        testing_acc, testing_auc, testing_FPR, testing_fair_gender_FFPR, testing_fair_gender_FOAE, testing_fair_gender_FMEO, _ = fairness_metrics(
            np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), gender_labels_list)

        # evaluate fairness across race
        race_labels_list = [x.split('-')[1] for x in intersec_labels_list]
        _, _, _, testing_fair_race_FFPR, testing_fair_race_FOAE, testing_fair_race_FMEO, testing_gender_metrics = fairness_metrics(
            np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), race_labels_list)

        # evaluate fairness across intersection group
        _, _, _, testing_fair_intersec_FFPR, testing_fair_intersec_FOAE, testing_fair_intersec_FMEO, testing_intersec_metrics = fairness_metrics(
            np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), intersec_labels_list)