"""
This is to perform adversarial training for biases mitigation.

Author: Tao Qiqi
Date: 2024-03
"""
import json
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from dataset.dataset_w_demo_anno import DeepfakeDataset
from transform import resnet_default_data_transform, xception_default_data_transform
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbones.ResNet import ResNet
from backbones.XceptionNet import XceptionNet
from backbones.EfficientNet import EfficientNet
from backbones import models_hongguliu  # for XceptionNet
from metrics.fairness_metrics import fairness_metrics, detection_metrics
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
                 'XceptionNet-hongguliu': models_hongguliu.model_selection(modelname='xception', num_out_classes=2,
                                                                           dropout=0.5),
                 'XceptionNet-hongguliu-ImageNet-pretrained': models_hongguliu.model_selection(modelname='xception',
                                                                                               num_out_classes=2,
                                                                                               dropout=0.5,
                                                                                               pretrained='ImageNet')
                 }

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_root', type=str, default='../data_split')
    parser.add_argument('--model', type=str, choices=model_zoo.keys())
    parser.add_argument('-bs', '--batch_size', type=int, default=256)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-epochs', '--epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ckpt_root', default='./ckpt')
    parser.add_argument('--log_root', default='./logs')
    parser.add_argument('--adv_input', type=str, choices=['features', 'pred_prob'],
                        help='the input of adversary classifier')
    parser.add_argument('--alpha_race', type=float, default=0.1,
                        help='the hyperparameter of race adversary loss weight')
    parser.add_argument('--alpha_gender', type=float, default=0.1,
                        help='the hyperparameter of gender adversary loss weight')
    args = parser.parse_args()

    ### for debug only

    """
    Initialize logger
    """

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    # mode = "real_and_FS" if args.balanced else "all_data"
    mode = "all_data"
    logger_name = '[adv]' + timestamp + '_' + args.model + 'lr' + str(args.learning_rate) + '_' + mode + '.log'
    logging.basicConfig(filename=os.path.join(args.log_root, logger_name), level=logging.INFO, format=LOG_FORMAT)
    # log info about adversarial-learning setting
    logging.info(
        f'Adversarial learning settings: adversary inputs as [{args.adv_input}], gender loss weight [{args.alpha_gender}], race loss weight[{args.alpha_race}].')

    # os.environ['CUDA_VISIBLE_DEVICE'] = '1,3'
    if args.gpu is not None and torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = torch.device("cuda:" + str(args.gpu))
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
    Load model and adversary
    """
    """
    Training adversarial networks is extremely difficult. It is important to:
    1) lower the step size of both the predictor and adversary to train both models slowly to avoid parameters diverging,
    2) initialize the parameters of the adversary to be small to avoid the predictor overfitting against a sub-optimal adversary,
    3) increase the adversary’s learning rate to prevent divergence if the predictor is too good at hiding the protected variable from the adversary
    """
    backbone = args.model
    assert backbone in model_zoo.keys(), f"Model {backbone} is not supported."

    print(f'Loading model {backbone}...')

    model = model_zoo[backbone].to(device)
    if 'XceptionNet-hongguliu' in backbone:
        model = model.model

    print(model)
    # for name, _ in model.named_parameters():
    #     print(name)

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

    """
    Define adversary classifiers to predict sensitive attributes, incorporating the features or the predicted probability of being fake as inputs
    """

    backbone_feature_dim = {'ResNet-18': 512,
                            'ResNet-50': 512,
                            'EfficientNet-B3': 512,
                            'EfficientNet-B4': 512,
                            'XceptionNet-hongguliu-ImageNet-pretrained': 2048}

    if args.adv_input == 'features':
        adversary_gender = nn.Linear(backbone_feature_dim[backbone], out_features=2)
        adversary_gender = adversary_gender.to(device)
        adversary_race = nn.Linear(backbone_feature_dim[backbone], out_features=4)
        adversary_race = adversary_race.to(device)
    elif args.adv_input == 'pred_prob':
        adversary_gender = nn.Linear(1, out_features=2)
        adversary_gender = adversary_gender.to(device)
        adversary_race = nn.Linear(1, out_features=4)
        adversary_race = adversary_race.to(device)

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

    train_data = "updated_idx_train_adv_training.json"
    val_data = "updated_idx_val_adv_training.json"
    test_data = "updated_idx_test_adv_training.json"

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
    criterion = nn.CrossEntropyLoss()
    logging.info(f'Using loss function: Cross-Entropy loss.')
    optimizer = optim.SGD(model.parameters(), lr=lr)
    logging.info(f'Using optimizer: SGD, learning rate {lr}.')

    adversary_gender_optimizer = optim.SGD(adversary_gender.parameters(),
                                           lr=lr)  # attribute classifier和main classifier用同样的learningl
    adversary_race_optimizer = optim.SGD(adversary_race.parameters(), lr=lr)

    alpha_race = args.alpha_race
    alpha_gender = args.alpha_gender

    """
    Initialize wandb.
    """
    wandb.init(
        project='ST5188_fair_deepfake',
        name=f'[adv-{args.adv_input}-gw{alpha_gender}-rw{alpha_race}]' + backbone + mode + 'lr' + str(lr),
        config={'learning_rate': lr,
                'epochs': epochs,
                'model': backbone,
                'adversary input': args.adv_input,
                'gender loss weight': alpha_gender,
                'race loss weight': alpha_race}
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

        running_gender_loss = 0
        running_race_loss = 0
        gender_labels_list = []
        race_labels_list = []
        gender_preds_probs_list = []
        race_preds_probs_list = []
        gender_preds_labels_list = []
        race_preds_labels_list = []

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

            """
            Update adversary first (for race and gender).
            """
            adversary_gender_optimizer.zero_grad()
            optimizer.zero_grad()
            gender_labels = data['gender_label'].to(device)

            adversary_race_optimizer.zero_grad()
            optimizer.zero_grad()
            race_labels = data['race_label'].to(device)

            """
            if need features
            """
            if args.adv_input == 'features':
                if 'EfficientNet' in backbone or 'XceptionNet' in backbone:
                    features = F.adaptive_avg_pool2d(model.features(imgs),(1,1)).squeeze()
                else:
                    features = model.avgpool(model.features(imgs)).squeeze()
                gender_preds = adversary_gender(features)
                race_preds = adversary_race(features)

                gender_preds_probs = torch.softmax(gender_preds, dim=1)
                gender_preds_labels = torch.argmax(gender_preds_probs, dim=1)
                gender_preds_probs = torch.softmax(gender_preds, dim=1)[:,
                                     1]  # only need probability of being 1! shape should be (batch size, 1)

                gender_labels_list.extend(gender_labels.cpu().data.numpy().tolist())
                gender_preds_probs_list.extend(gender_preds_probs.cpu().data.numpy().tolist())
                gender_preds_labels_list.extend(gender_preds_labels.cpu().data.numpy().tolist())

                gender_loss = criterion(gender_preds, gender_labels)
                gender_loss.backward(retain_graph=True)
                running_gender_loss += gender_loss.item() * imgs.shape[0]
                gender_grad = {name: param.grad.clone() if param.grad is not None else None for name, param in
                               model.named_parameters()}
                adversary_gender_optimizer.step()

                race_preds_probs = torch.softmax(race_preds, dim=1)
                race_preds_labels = torch.argmax(race_preds_probs, dim=1)

                race_labels_list.extend(race_labels.cpu().data.numpy().tolist())
                race_preds_probs_list.extend(race_preds_probs.cpu().data.numpy().tolist())
                race_preds_labels_list.extend(race_preds_labels.cpu().data.numpy().tolist())

                race_loss = criterion(race_preds, race_labels)
                race_loss.backward(retain_graph=True)
                running_race_loss += race_loss.item() * imgs.shape[0]
                race_grad = {name: param.grad.clone() if param.grad is not None else None for name, param in
                             model.named_parameters()}
                adversary_race_optimizer.step()

            elif args.adv_input == 'pred_prob':
                gender_preds = adversary_gender(pred_probs.unsqueeze(-1))
                race_preds = adversary_race(pred_probs.unsqueeze(-1))

                gender_preds_probs = torch.softmax(gender_preds, dim=1)
                gender_preds_labels = torch.argmax(gender_preds_probs, dim=1)
                gender_preds_probs = torch.softmax(gender_preds, dim=1)[:,
                                     1]  # only need probability of being 1! shape should be (batch size, 1)

                gender_labels_list.extend(gender_labels.cpu().data.numpy().tolist())
                gender_preds_probs_list.extend(gender_preds_probs.cpu().data.numpy().tolist())
                gender_preds_labels_list.extend(gender_preds_labels.cpu().data.numpy().tolist())

                gender_loss = criterion(gender_preds, gender_labels)
                gender_loss.backward(retain_graph=True)
                running_gender_loss += gender_loss.item() * imgs.shape[0]
                gender_grad = {name: param.grad.clone() for name, param in model.named_parameters()}
                adversary_gender_optimizer.step()

                race_preds_probs = torch.softmax(race_preds, dim=1)
                race_preds_labels = torch.argmax(race_preds_probs, dim=1)

                race_labels_list.extend(race_labels.cpu().data.numpy().tolist())
                race_preds_probs_list.extend(race_preds_probs.cpu().data.numpy().tolist())
                race_preds_labels_list.extend(race_preds_labels.cpu().data.numpy().tolist())

                race_loss = criterion(race_preds, race_labels)
                race_loss.backward(retain_graph=True)
                running_race_loss += race_loss.item() * imgs.shape[0]
                race_grad = {name: param.grad.clone() for name, param in model.named_parameters()}
                adversary_race_optimizer.step()

            """
            Update main classifier (for real/fake).
            """

            optimizer.zero_grad()

            loss = criterion(preds, labels)
            loss.backward()

            ### add adversarial loss and projection
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in ['fc.weight',
                                'fc.bias',
                                'last_layer.weight', # the Linear layer name of EfficientNet
                                'last_layer.bias',
                                'last_linear.1.weight', # the Linear layer name of XceptionNet
                                'last_linear.1.bias']:  # 第一个batch的时候，FC.grad是none,后面变成0了，所以导致FC在除以norm的时候出现了除以0的情况，然后nan。solution：直接改成FC layer不加adv loss。
                        continue
                    unit_gender = gender_grad[name] / torch.linalg.norm(gender_grad[name])
                    unit_race = race_grad[name] / torch.linalg.norm(race_grad[name])

                    param.grad -= ((param.grad * unit_gender).sum()) * unit_gender
                    param.grad -= ((param.grad * unit_race).sum()) * unit_race
                    param.grad -= alpha_race * race_grad[name]  # here is a hyper-parameter
                    param.grad -= alpha_gender * gender_grad[name]  # here is a hyper-parameter

            optimizer.step()
            running_loss += loss.item() * imgs.shape[0]
            if _ % 2 == 1:
                print(f'[Training Loss] Batch {_ + 1}, epoch {epoch}/{epochs}: {loss.item():.2f}.')
        epoch_train_loss = running_loss / len(trainset)
        print(f'[Training Loss] Epoch {epoch + 1}/{epochs}: {epoch_train_loss:.2f}.')
        logging.info(f'[Training Loss] Epoch {epoch + 1}/{epochs}: {epoch_train_loss:.5f}.')
        d = {'gt_label': labels_list,
             'pred_label': pred_labels_list,
             'intersec': intersec_labels_list
             }

        """
        Adversary evaluation.
        """

        print(f'[Training Gender Loss] Epoch {epoch + 1}/{epochs}: {running_gender_loss / len(trainset): .2f}.')
        print(f'[Training Race Loss] Epoch {epoch + 1}/{epochs}: {running_race_loss / len(trainset): .2f}.')
        logging.info(f'[Training Gender Loss] Epoch {epoch + 1}/{epochs}: {running_gender_loss / len(trainset): .5f}.')
        logging.info(f'[Training Race Loss] Epoch {epoch + 1}/{epochs}: {running_race_loss / len(trainset): .5f}.')

        training_gender_acc, training_gender_auc, _, _, _ = detection_metrics(gender_labels_list,
                                                                              gender_preds_labels_list,
                                                                              gender_preds_probs_list)
        training_race_acc, training_race_auc, _, _, _ = detection_metrics(race_labels_list, race_preds_labels_list,
                                                                          race_preds_probs_list,
                                                                          binary=False)
        logging.info(
            f'[Training Gender Accuracy, AUC] Epoch {epoch + 1}/{epochs}: accuracy {training_gender_acc:.5f}, AUC {training_gender_auc:.5f}.')
        logging.info(
            f'[Training Race Accuracy, AUC] Epoch {epoch + 1}/{epochs}: accuracy {training_race_acc:.5f}, AUC{training_race_auc:.5f}.')

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

            running_gender_loss = 0
            running_race_loss = 0
            gender_labels_list = []
            race_labels_list = []
            gender_preds_probs_list = []
            race_preds_probs_list = []
            gender_preds_labels_list = []
            race_preds_labels_list = []

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

                loss = criterion(preds, labels)
                running_loss += loss.item() * imgs.shape[0]

                gender_labels = data['gender_label'].to(device)
                race_labels = data['race_label'].to(device)
                """
                if need features
                """
                if args.adv_input == 'features':
                    if 'EfficientNet' in backbone or 'XceptionNet' in backbone:
                        features = F.adaptive_avg_pool2d(model.features(imgs), (1, 1)).squeeze()
                    else:
                        features = model.avgpool(model.features(imgs)).squeeze()
                    gender_preds = adversary_gender(features)
                    race_preds = adversary_race(features)

                    gender_preds_probs = torch.softmax(gender_preds, dim=1)
                    gender_preds_labels = torch.argmax(gender_preds_probs, dim=1)
                    gender_preds_probs = torch.softmax(gender_preds, dim=1)[:,
                                         1]  # only need probability of being 1! shape should be (batch size, 1)

                    gender_labels_list.extend(gender_labels.cpu().data.numpy().tolist())
                    gender_preds_probs_list.extend(gender_preds_probs.cpu().data.numpy().tolist())
                    gender_preds_labels_list.extend(gender_preds_labels.cpu().data.numpy().tolist())

                    gender_loss = criterion(gender_preds, gender_labels)
                    running_gender_loss += gender_loss.item() * imgs.shape[0]

                    race_preds_probs = torch.softmax(race_preds, dim=1)
                    race_preds_labels = torch.argmax(race_preds_probs, dim=1)

                    race_labels_list.extend(race_labels.cpu().data.numpy().tolist())
                    race_preds_probs_list.extend(race_preds_probs.cpu().data.numpy().tolist())
                    race_preds_labels_list.extend(race_preds_labels.cpu().data.numpy().tolist())

                    race_loss = criterion(race_preds, race_labels)
                    running_race_loss += race_loss.item() * imgs.shape[0]

                elif args.adv_input == 'pred_prob':
                    gender_preds = adversary_gender(pred_probs.unsqueeze(-1))
                    race_preds = adversary_race(pred_probs.unsqueeze(-1))

                    gender_preds_probs = torch.softmax(gender_preds, dim=1)
                    gender_preds_labels = torch.argmax(gender_preds_probs, dim=1)
                    gender_preds_probs = torch.softmax(gender_preds, dim=1)[:,
                                         1]  # only need probability of being 1! shape should be (batch size, 1)

                    gender_labels_list.extend(gender_labels.cpu().data.numpy().tolist())
                    gender_preds_probs_list.extend(gender_preds_probs.cpu().data.numpy().tolist())
                    gender_preds_labels_list.extend(gender_preds_labels.cpu().data.numpy().tolist())

                    gender_loss = criterion(gender_preds, gender_labels)
                    running_gender_loss += gender_loss.item() * imgs.shape[0]

                    race_preds_probs = torch.softmax(race_preds, dim=1)
                    race_preds_labels = torch.argmax(race_preds_probs, dim=1)

                    race_labels_list.extend(race_labels.cpu().data.numpy().tolist())
                    race_preds_probs_list.extend(race_preds_probs.cpu().data.numpy().tolist())
                    race_preds_labels_list.extend(race_preds_labels.cpu().data.numpy().tolist())

                    race_loss = criterion(race_preds, race_labels)
                    running_race_loss += race_loss.item() * imgs.shape[0]

            epoch_val_loss = running_loss / len(valset)
            print(f'[Validation Loss] Epoch {epoch + 1}/{epochs}: {epoch_val_loss:.2f}.')
            logging.info(f'[Validation Loss] Epoch {epoch + 1}/{epochs}: {epoch_val_loss:.5f}.')

            """
            Adversary evaluation.
            """

            print(f'[Validation Gender Loss] Epoch {epoch + 1}/{epochs}: {running_gender_loss / len(trainset): .2f}.')
            print(f'[Validation Race Loss] Epoch {epoch + 1}/{epochs}: {running_race_loss / len(trainset): .2f}.')
            logging.info(
                f'[Validation Gender Loss] Epoch {epoch + 1}/{epochs}: {running_gender_loss / len(trainset): .5f}.')
            logging.info(
                f'[Validation Race Loss] Epoch {epoch + 1}/{epochs}: {running_race_loss / len(trainset): .5f}.')

            val_gender_acc, val_gender_auc, _, _, _ = detection_metrics(gender_labels_list, gender_preds_labels_list,
                                                                        gender_preds_probs_list)
            val_race_acc, val_race_auc, _, _, _ = detection_metrics(race_labels_list, race_preds_labels_list,
                                                                    race_preds_probs_list,
                                                                    binary=False)
            logging.info(
                f'[Validation Gender Accuracy, AUC] Epoch {epoch + 1}/{epochs}: accuracy {val_gender_acc:.5f}, AUC {val_gender_auc:.5f}.')
            logging.info(
                f'[Validation Race Accuracy, AUC] Epoch {epoch + 1}/{epochs}: accuracy {val_race_acc:.5f}, AUC{val_race_auc:.5f}.')

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
                           os.path.join(ckpt_root,
                                        '[adv]' + timestamp + '_' + backbone + '_lr' + str(
                                            lr) + 'best_ckpt_from_epoch' + str(
                                            epoch + 1) + '.pth'))
            # evaluation on testing set
            running_loss = 0
            labels_list = []
            pred_labels_list = []
            pred_probs_list = []
            intersec_labels_list = []

            running_gender_loss = 0
            running_race_loss = 0
            gender_labels_list = []
            race_labels_list = []
            gender_preds_probs_list = []
            race_preds_probs_list = []
            gender_preds_labels_list = []
            race_preds_labels_list = []
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

                gender_labels = data['gender_label'].to(device)
                race_labels = data['race_label'].to(device)
                """
                if need features
                """
                if args.adv_input == 'features':
                    if 'EfficientNet' in backbone or 'XceptionNet' in backbone:
                        features = F.adaptive_avg_pool2d(model.features(imgs), (1, 1)).squeeze()
                    else:
                        features = model.avgpool(model.features(imgs)).squeeze()
                    gender_preds = adversary_gender(features)
                    race_preds = adversary_race(features)

                    gender_preds_probs = torch.softmax(gender_preds, dim=1)
                    gender_preds_labels = torch.argmax(gender_preds_probs, dim=1)
                    gender_preds_probs = torch.softmax(gender_preds, dim=1)[:,
                                         1]  # only need probability of being 1! shape should be (batch size, 1)

                    gender_labels_list.extend(gender_labels.cpu().data.numpy().tolist())
                    gender_preds_probs_list.extend(gender_preds_probs.cpu().data.numpy().tolist())
                    gender_preds_labels_list.extend(gender_preds_labels.cpu().data.numpy().tolist())

                    gender_loss = criterion(gender_preds, gender_labels)
                    running_gender_loss += gender_loss.item() * imgs.shape[0]

                    race_preds_probs = torch.softmax(race_preds, dim=1)
                    race_preds_labels = torch.argmax(race_preds_probs, dim=1)

                    race_labels_list.extend(race_labels.cpu().data.numpy().tolist())
                    race_preds_probs_list.extend(race_preds_probs.cpu().data.numpy().tolist())
                    race_preds_labels_list.extend(race_preds_labels.cpu().data.numpy().tolist())

                    race_loss = criterion(race_preds, race_labels)
                    running_race_loss += race_loss.item() * imgs.shape[0]

                elif args.adv_input == 'pred_prob':
                    gender_preds = adversary_gender(pred_probs.unsqueeze(-1))
                    race_preds = adversary_race(pred_probs.unsqueeze(-1))

                    gender_preds_probs = torch.softmax(gender_preds, dim=1)
                    gender_preds_labels = torch.argmax(gender_preds_probs, dim=1)
                    gender_preds_probs = torch.softmax(gender_preds, dim=1)[:,
                                         1]  # only need probability of being 1! shape should be (batch size, 1)

                    gender_labels_list.extend(gender_labels.cpu().data.numpy().tolist())
                    gender_preds_probs_list.extend(gender_preds_probs.cpu().data.numpy().tolist())
                    gender_preds_labels_list.extend(gender_preds_labels.cpu().data.numpy().tolist())

                    gender_loss = criterion(gender_preds, gender_labels)
                    running_gender_loss += gender_loss.item() * imgs.shape[0]

                    race_preds_probs = torch.softmax(race_preds, dim=1)
                    race_preds_labels = torch.argmax(race_preds_probs, dim=1)

                    race_labels_list.extend(race_labels.cpu().data.numpy().tolist())
                    race_preds_probs_list.extend(race_preds_probs.cpu().data.numpy().tolist())
                    race_preds_labels_list.extend(race_preds_labels.cpu().data.numpy().tolist())

                    race_loss = criterion(race_preds, race_labels)
                    running_race_loss += race_loss.item() * imgs.shape[0]

            epoch_test_loss = running_loss / len(testset)
            print(f'[Testing Loss] Epoch {epoch + 1}/{epochs}: {epoch_test_loss:.2f}.')
            logging.info(f'[Testing Loss] Epoch {epoch + 1}/{epochs}: {epoch_test_loss:.5f}.')

            """
            Adversary evaluation.
            """

            print(f'[Testing Gender Loss] Epoch {epoch + 1}/{epochs}: {running_gender_loss / len(trainset): .2f}.')
            print(f'[Testing Race Loss] Epoch {epoch + 1}/{epochs}: {running_race_loss / len(trainset): .2f}.')
            logging.info(
                f'[Testing Gender Loss] Epoch {epoch + 1}/{epochs}: {running_gender_loss / len(trainset): .5f}.')
            logging.info(f'[Testing Race Loss] Epoch {epoch + 1}/{epochs}: {running_race_loss / len(trainset): .5f}.')

            test_gender_acc, test_gender_auc, _, _, _ = detection_metrics(gender_labels_list, gender_preds_labels_list,
                                                                          gender_preds_probs_list)
            test_race_acc, test_race_auc, _, _, _ = detection_metrics(race_labels_list, race_preds_labels_list,
                                                                      race_preds_probs_list,
                                                                      binary=False)
            logging.info(
                f'[Testing Gender Accuracy, AUC] Epoch {epoch + 1}/{epochs}: accuracy {test_gender_acc:.5f}, AUC {test_gender_auc:.5f}.')
            logging.info(
                f'[Testing Race Accuracy, AUC] Epoch {epoch + 1}/{epochs}: accuracy {test_race_acc:.5f}, AUC{test_race_auc:.5f}.')

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
                'testing FPR': testing_FPR,
                'training gender auc': training_gender_auc,
                'validation gender auc': val_gender_auc,
                'testing gender auc': test_gender_auc,
                'training race auc': training_race_auc,
                'validation race auc': val_race_auc,
                'testing race auc': test_race_auc
            })
            if epoch % 20 == 19:
                torch.save(model.state_dict(),
                           os.path.join(ckpt_root,
                                        '[adv]' + timestamp + '_' + backbone + '_lr' + str(lr) + '_' + str(
                                            epoch + 1) + '.pth'))
                torch.save(adversary_gender.state_dict(),
                           os.path.join(ckpt_root,
                                        '[adv]' + timestamp + '_gender_advclassifier' + '_lr' + str(lr) + '_' + str(
                                            epoch + 1) + '.pth'))
                torch.save(adversary_race.state_dict(),
                           os.path.join(ckpt_root,
                                        '[adv]' + timestamp + '_race_advclassifier' + '_lr' + str(lr) + '_' + str(
                                            epoch + 1) + '.pth'))
    wandb.finish()
