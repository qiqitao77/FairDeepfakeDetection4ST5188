"""
This is to train baseline model, including ResNet, XceptionNet, EfficientNet.

Author: Tao Qiqi
Date: 2024-03
"""
import json
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
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
from losses.losses import WeightedSampleCrossEntropyLoss
import logging
import wandb
import time

if __name__ == '__main__':
    xception_config = {'num_classes': 2,
                       'mode': None,
                       'adjust_mode': 'adjust_channel',
                       'inc': 3,
                       'dropout': True}

    model_zoo = {'ResNet-18': ResNet('ResNet-18'),
                 'ResNet-34': ResNet('ResNet-34'),
                 'ResNet-50': ResNet('ResNet-50'),
                 'EfficientNet-B3': EfficientNet('EfficientNet-B3'),
                 'EfficientNet-B4': EfficientNet('EfficientNet-B4'),
                 'XceptionNet': XceptionNet(xception_config=xception_config),
                 'XceptionNet-hongguliu': models_hongguliu.model_selection(modelname='xception', num_out_classes=2, dropout=0.5),
                 'XceptionNet-hongguliu-ImageNet-pretrained': models_hongguliu.model_selection(modelname='xception',
                                                                                               num_out_classes=2,
                                                                                               dropout=0.5,
                                                                                               pretrained='ImageNet')
                 }

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_root', type=str, default='../data_split')
    parser.add_argument('--model', type=str, choices=model_zoo.keys())
    parser.add_argument('--sensitive_attr', type=str, choices=['gender', 'race', 'intersec'])
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-epochs', '--epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ckpt_root', default='./ckpt')
    parser.add_argument('--log_root', default='./logs')
    parser.add_argument('--balanced', action='store_true',help='While using balanced mode, only real and FaceSwap data will be included.')
    ### efficientnet is kill... need to resume
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_ckpt', type=str)
    parser.add_argument('--weight_mode', default=None, choices=[None, 'subgroup', 'prior_equal_dist'])
    args = parser.parse_args()


    """
    Initialize logger
    """

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    mode = "real_and_FS" if args.balanced else "all_data"
    logger_name = f'[reweighted-{args.sensitive_attr}]' + timestamp + '_' + args.model + 'lr' + str(args.learning_rate) + '_' + mode + '.log'
    logging.basicConfig(filename=os.path.join(args.log_root, logger_name), level=logging.INFO, format=LOG_FORMAT)

    # os.environ['CUDA_VISIBLE_DEVICE'] = '1,3'
    if args.gpu is not None and torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = torch.device("cuda:"+str(args.gpu))
        logging.info(f'Using GPU {device}.')
    else:
        device = 'cpu'
        logging.info(f'Using CPU.')
    print(f'Using device: {device}.')
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    ckpt_root = args.ckpt_root
    log_root = args.log_root

    """
    Load model
    """
    backbone = args.model
    assert backbone in model_zoo.keys(), f"Model {backbone} is not supported."

    print(f'Loading model {backbone}...')

    model = model_zoo[backbone].to(device)
    print(model)

    if isinstance(model, ResNet):
        transform = resnet_default_data_transform
    elif isinstance(model, XceptionNet) or 'Xception' in args.model:
        transform = xception_default_data_transform
    elif isinstance(model, EfficientNet):
        transform = transforms.ToTensor()

    param_num = sum(p.numel() for p in model.parameters())
    trainable_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f'Train model {backbone} for deepfake detection.')
    logging.info(f'The total number of parameters: {param_num}.')
    print(f'The total number of parameters: {param_num}.')
    logging.info(f'The trainable number of parameters: {trainable_param_num}.')
    print(f'The trainable number of parameters: {trainable_param_num}.')

    if args.resume:
        logging.info(f'Resume training from ckpt {args.resume_ckpt}.')
        state_dict = torch.load(args.resume_ckpt)
        model.load_state_dict(state_dict)
    """
    Load data
    """
    print('Reading dataset csv files...')
    data_split_root = args.data_split_root
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
    print('Creating and loading datasets...')

    if args.balanced:
        train_data = "real_and_FS_train.json"
        val_data = "real_and_FS_val.json"
        test_data = "real_and_FS_test.json"
    else:
        train_data = "updated_idx_train.json"
        val_data = "updated_idx_val.json"
        test_data = "updated_idx_test.json"

    with open(os.path.join(data_split_root, train_data), 'r') as json_file:
        train_dict = json.load(json_file)
    json_file.close()
    with open(os.path.join(data_split_root, val_data), 'r') as json_file:
        val_dict = json.load(json_file)
    json_file.close()
    with open(os.path.join(data_split_root, test_data), 'r') as json_file:
        test_dict = json.load(json_file)
    json_file.close()

    trainset = DeepfakeDataset(train_dict, transform)
    logging.info(
        f'Loading training set from {os.path.join(data_split_root, train_data)}, containing {len(trainset)} samples.')
    valset = DeepfakeDataset(val_dict, transform)
    logging.info(
        f'Loading validation set from {os.path.join(data_split_root, val_data)}, containing {len(valset)} samples.')
    testset = DeepfakeDataset(test_dict, transform)
    logging.info(
        f'Loading testing set from {os.path.join(data_split_root, test_data)}, containing {len(testset)} samples.')

    train_loader = DataLoader(dataset=trainset, shuffle=True, num_workers=8, batch_size=batch_size)
    val_loader = DataLoader(dataset=valset, shuffle=False, num_workers=8, batch_size=batch_size)
    test_loader = DataLoader(dataset=testset, shuffle=False, num_workers=8, batch_size=batch_size)

    """
    Define loss and optimizer
    """
    print('Getting loss and optimizer...')

    ##### reweighted method: assign weights to samples while calculating loss
    # criterion = nn.CrossEntropyLoss()
    criterion = WeightedSampleCrossEntropyLoss(args.sensitive_attr,args.weight_mode)


    logging.info(f'Using loss function: Weighted Cross-Entropy loss. Weight model: {args.weight_mode}')
    optimizer = optim.SGD(model.parameters(), lr=lr)
    logging.info(f'Using optimizer: SGD, learning rate {lr}.')

    """
    Initialize wandb.
    """
    wandb.init(
        project='ST5188_fair_deepfake',
        name=f'[reweighted-{args.sensitive_attr}]' + backbone + mode + 'lr'+str(lr),
        config={'learning_rate': lr,
                'epochs': epochs,
                'model': backbone}
    )

    """
    Model Training & Evaluation
    """
    print('Training model...')
    best_acc = 0
    for epoch in range(epochs):
        print(f'-------------------  Epoch {epoch + 1}  -----------------')
        model.train()
        running_loss = 0
        labels_list = []
        pred_labels_list = []
        pred_probs_list = []
        intersec_labels_list = []
        # model training
        for _, data in enumerate(train_loader):
            print(f'Training epoch{epoch + 1}/{epochs}, batch{_ + 1}.')
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

            optimizer.zero_grad()

            if args.sensitive_attr == 'gender':
                attributes = [x.split('-')[0] for x in intersec_labels]
            elif args.sensitive_attr == 'race':
                attributes = [x.split('-')[1] for x in intersec_labels]
            else:
                attributes = intersec_labels

            loss = criterion(preds, labels, attributes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.shape[0]
            if _ % 20 == 0:
                print(f'[Training Loss] Batch {_ + 1}, epoch {epoch}/{epochs}: {loss.item():.2f}.')
        epoch_train_loss = running_loss / len(trainset)
        print(f'[Training Loss] Epoch {epoch + 1}/{epochs}: {epoch_train_loss:.2f}.')
        logging.info(f'[Training Loss] Epoch {epoch + 1}/{epochs}: {epoch_train_loss:.5f}.')
        d = {'gt_label': labels_list,
             'pred_label': pred_labels_list,
             'intersec': intersec_labels_list
        }
        # pd.DataFrame(d).to_csv(f'{backbone}_training_set_preds_epoch_{epoch + 1}_lr{lr}.csv', index=False)

        # evaluate fairness across gender
        gender_labels_list = [x.split('-')[0] for x in intersec_labels_list]
        training_acc, training_auc, training_FPR, training_fair_gender_FFPR, training_fair_gender_FOAE, training_fair_gender_FMEO, training_gender_metrics = fairness_metrics(
            np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), gender_labels_list)
        logging.info(
            f'[Training Accuracy, AUC and FPR] Epoch {epoch + 1}/{epochs}: accuracy {training_acc:.5f}, AUC {training_auc:.5f}, FPR {training_FPR:.5f}.')
        logging.info(
            f'[Training fairness metrics across GENDER] Epoch {epoch + 1}/{epochs}: FFPR {training_fair_gender_FFPR:.5f}, FOAE: {training_fair_gender_FOAE:.5f}, FMEO: {training_fair_gender_FMEO:.5f}.')

        # evaluate fairness across race
        race_labels_list = [x.split('-')[1] for x in intersec_labels_list]
        _, _, _, training_fair_race_FFPR, training_fair_race_FOAE, training_fair_race_FMEO, training_race_metrics = fairness_metrics(
            np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), race_labels_list)
        logging.info(
            f'[Training fairness metrics across RACE] Epoch {epoch + 1}/{epochs}: FFPR {training_fair_race_FFPR:.5f}, FOAE: {training_fair_race_FOAE:.5f}, FMEO: {training_fair_race_FMEO:.5f}.')

        # evaluate fairness across intersection group
        _, _, _, training_fair_intersec_FFPR, training_fair_intersec_FOAE, training_fair_intersec_FMEO, training_intersec_metrics = fairness_metrics(
            np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), intersec_labels_list)
        logging.info(
            f'[Training fairness metrics across INTERSECTION] Epoch {epoch + 1}/{epochs}: FFPR {training_fair_intersec_FFPR:.5f}, FOAE: {training_fair_intersec_FOAE:.5f}, FMEO: {training_fair_intersec_FMEO:.5f}.')

        with torch.no_grad():
            model.eval()
            # evaluation on validation set
            running_loss = 0
            labels_list = []
            pred_labels_list = []
            pred_probs_list = []
            intersec_labels_list = []

            for _, data in enumerate(val_loader):
                print(f'Evaluating on validationset, epoch{epoch + 1}/{epochs}, batch{_ + 1}.')
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

                if args.sensitive_attr == 'gender':
                    attributes = [x.split('-')[0] for x in intersec_labels]
                elif args.sensitive_attr == 'race':
                    attributes = [x.split('-')[1] for x in intersec_labels]
                else:
                    attributes = intersec_labels

                loss = criterion(preds, labels, attributes)
                running_loss += loss.item() * imgs.shape[0]
            epoch_val_loss = running_loss / len(valset)
            print(f'[Validation Loss] Epoch {epoch + 1}/{epochs}: {epoch_val_loss:.2f}.')
            logging.info(f'[Validation Loss] Epoch {epoch + 1}/{epochs}: {epoch_val_loss:.5f}.')
            d = {'gt_label': labels_list,
                 'pred_label': pred_labels_list,
                 'intersec': intersec_labels_list
                 }
            # pd.DataFrame(d).to_csv(f'{backbone}_validation_set_preds_epoch_{epoch + 1}_lr{lr}.csv', index=False)

            # evaluate fairness across gender
            gender_labels_list = [x.split('-')[0] for x in intersec_labels_list]
            val_acc, val_auc, val_FPR, val_fair_gender_FFPR, val_fair_gender_FOAE, val_fair_gender_FMEO, val_gender_metrics = fairness_metrics(
                np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), gender_labels_list)
            logging.info(
                f'[Validation Accuracy, AUC and FPR] Epoch {epoch + 1}/{epochs}: accuracy {val_acc:.5f}, AUC {val_auc:.5f}, FPR {val_FPR:.5f}.')
            logging.info(
                f'[Validation fairness metrics across GENDER] Epoch {epoch + 1}/{epochs}: FFPR {val_fair_gender_FFPR:.5f}, FOAE: {val_fair_gender_FOAE:.5f}, FMEO: {val_fair_gender_FMEO:.5f}.')

            # evaluate fairness across race
            race_labels_list = [x.split('-')[1] for x in intersec_labels_list]
            _, _, _, val_fair_race_FFPR, val_fair_race_FOAE, val_fair_race_FMEO, val_race_metrics = fairness_metrics(
                np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), race_labels_list)
            logging.info(
                f'[Validation fairness metrics across RACE] Epoch {epoch + 1}/{epochs}: FFPR {val_fair_race_FFPR:.5f}, FOAE: {val_fair_race_FOAE:.5f}, FMEO: {val_fair_race_FMEO:.5f}.')

            # evaluate fairness across intersection group
            _, _, _, val_fair_intersec_FFPR, val_fair_intersec_FOAE, val_fair_intersec_FMEO, val_intersec_metrics = fairness_metrics(
                np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), intersec_labels_list)
            logging.info(
                f'[Validation fairness metrics across INTERSECTION] Epoch {epoch + 1}/{epochs}: FFPR {val_fair_intersec_FFPR:.5f}, FOAE: {val_fair_intersec_FOAE:.5f}, FMEO: {val_fair_intersec_FMEO:.5f}.')

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(),
                           os.path.join(ckpt_root, timestamp + '_' + backbone + '_lr' + str(lr) + 'best_ckpt_from_epoch' + str(epoch + 1) + '.pth'))
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

                if args.sensitive_attr == 'gender':
                    attributes = [x.split('-')[0] for x in intersec_labels]
                elif args.sensitive_attr == 'race':
                    attributes = [x.split('-')[1] for x in intersec_labels]
                else:
                    attributes = intersec_labels

                loss = criterion(preds, labels, attributes)
                running_loss += loss.item() * imgs.shape[0]
            epoch_test_loss = running_loss / len(testset)
            print(f'[Testing Loss] Epoch {epoch + 1}/{epochs}: {epoch_test_loss:.2f}.')
            logging.info(f'[Testing Loss] Epoch {epoch + 1}/{epochs}: {epoch_test_loss:.5f}.')
            d = {'gt_label': labels_list,
                 'pred_label': pred_labels_list,
                 'intersec': intersec_labels_list
                 }
            # pd.DataFrame(d).to_csv(f'{backbone}_testing_set_preds_epoch_{epoch + 1}_lr{lr}.csv', index=False)

            # evaluate fairness across gender
            gender_labels_list = [x.split('-')[0] for x in intersec_labels_list]
            testing_acc, testing_auc, testing_FPR, testing_fair_gender_FFPR, testing_fair_gender_FOAE, testing_fair_gender_FMEO, testing_gender_metrics = fairness_metrics(
                np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), gender_labels_list)
            logging.info(
                f'[Testing Accuracy, AUC and FPR] Epoch {epoch + 1}/{epochs}: accuracy {testing_acc:.5f}, AUC {testing_auc:.5f}, FPR {testing_FPR:.5f}.')
            logging.info(
                f'[Testing fairness metrics across GENDER] Epoch {epoch + 1}/{epochs}: FFPR {testing_fair_gender_FFPR:.5f}, FOAE: {testing_fair_gender_FOAE:.5f}, FMEO: {testing_fair_gender_FMEO:.5f}.')

            # evaluate fairness across race
            race_labels_list = [x.split('-')[1] for x in intersec_labels_list]
            _, _, _, testing_fair_race_FFPR, testing_fair_race_FOAE, testing_fair_race_FMEO, testing_race_metrics = fairness_metrics(
                np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), race_labels_list)
            logging.info(
                f'[Testing fairness metrics across RACE] Epoch {epoch + 1}/{epochs}: FFPR {testing_fair_race_FFPR:.5f}, FOAE: {testing_fair_race_FOAE:.5f}, FMEO: {testing_fair_race_FMEO:.5f}.')

            # evaluate fairness across intersection group
            _, _, _, testing_fair_intersec_FFPR, testing_fair_intersec_FOAE, testing_fair_intersec_FMEO, testing_intersec_metrics = fairness_metrics(
                np.array(labels_list), np.array(pred_labels_list), np.array(pred_probs_list), intersec_labels_list)
            logging.info(
                f'[Testing fairness metrics across INTERSECTION] Epoch {epoch + 1}/{epochs}: FFPR {testing_fair_intersec_FFPR:.5f}, FOAE: {testing_fair_intersec_FOAE:.5f}, FMEO: {testing_fair_intersec_FMEO:.5f}.')
            wandb.log({
                'training loss': epoch_train_loss,
                'validation loss': epoch_val_loss,
                'testing loss': epoch_test_loss,
                'training accuracy': training_acc,
                'valiadtion accuracy': val_acc,
                'testing accuracy': testing_acc,
                'training AUC': training_auc,
                'validation AUC': val_auc,
                'testing AUC': testing_auc,
                'training FPR': training_FPR,
                'validation FPR': val_FPR,
                'testing FPR': testing_FPR
            })
            if epoch % 20 == 19:
                torch.save(model.state_dict(),
                           os.path.join(ckpt_root, timestamp + '_' + backbone + '_lr' + str(lr) + '_' + str(epoch + 1) + '.pth'))
    wandb.finish()
