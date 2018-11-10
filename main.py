from __future__ import print_function
import os
import sys
import argparse

import time
import random
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from model import SiameseNetwork, ContrastiveLoss
from dataset import CubDataset, OnlineProductDataset
from loss import calculate_distance_and_similariy_label, contrastive_loss, focal_contrastive_loss
from torch.utils.data import DataLoader
from evaluation import  get_feature_and_label, evaluation
from utils import get_parameter_group

import datetime


parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, \
        default='/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images')
parser.add_argument('--image_txt', type=str, \
        default='/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/images.txt')
parser.add_argument('--train_test_split_txt', type=str, \
        default='/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/train_test_split.txt')
parser.add_argument('--label_txt', type=str, \
        default='/data/Guoxian_Dai/CUB_200_2011/CUB_200_2011/image_class_labels.txt')
parser.add_argument('--pretrained_model_path', default='./weights/inception_v3.ckpt', type=str)

parser.add_argument('--pretrained', default=False, type=bool)
parser.add_argument('--aux_logits', default=False, type=bool)
parser.add_argument('--pair_type', default='vector', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--dataset_name', default='cub200', type=str)
parser.add_argument('--with_regularizer', help='whether to use regularizer for parameters', action='store_true')
parser.add_argument('--optimizer', default='rmsprop', type=str)
parser.add_argument('--loss_type', default='contrastive_loss', type=str)
parser.add_argument('--learning_rate_decay_type', default='fixed', type=str)
parser.add_argument('--train_batch_size', default=64, type=int)
parser.add_argument('--test_batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs', default=10000, type=int)
parser.add_argument('--learning_rate', default=1e-2, type=float)
parser.add_argument('--momentum', default=1e-2, type=float)
parser.add_argument('--learning_rate2', default=1e-4, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)

parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--restore_ckpt', default=0, type=int)      # 1 for True
parser.add_argument('--evaluation', default=0, type=int)        # 1 for True
parser.add_argument('--weightFile', default='./models/my-model', type=str)
parser.add_argument('--ckpt_dir', default='./checkpoint', type=str)
parser.add_argument('--class_num', default=5, type=int)
parser.add_argument('--targetNum', default=1000, type=int)

parser.add_argument('--margin', default=1.0, type=float)
parser.add_argument('--gamma', default=0.98, type=float, help="weight decay factor")
parser.add_argument('--focal_decay_factor', default=1.0, type=float)
parser.add_argument('--display_step', default=20, type=int, help='step interval for displaying loss')
parser.add_argument('--eval_step', default=5, type=int, help='step interval for evaluate loss')
# image information
parser.add_argument('--width', default=512, type=int)
parser.add_argument('--height', default=512, type=int)

parser.add_argument('--embedding_size', default=128, type=int)
parser.add_argument('--num_epochs_per_decay', default=2, type=int)

parser.add_argument('--ngpu', default=2, type=int)
args = parser.parse_args()

# Inception_v3 input transformation.
"""
Before transform

x ~ [0, 1]

After transform:
(x - 0.485) / 0.229

Expected:
x ~ [-1, 1]

To do
"""


def train(args):
    # basic arguments.
    ngpu = args.ngpu
    margin = args.margin
    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    gamma = args.gamma # for learning rate decay
    learning_rate = args.learning_rate
    loss_type = args.loss_type
    dataset_name = args.dataset_name

    root_dir = args.root_dir
    image_txt = args.image_txt
    train_test_split_txt = args.train_test_split_txt
    label_txt = args.label_txt
    ckpt_dir = args.ckpt_dir
    eval_step = args.eval_step
    display_step = args.display_step
    embedding_size = args.embedding_size


    pretrained = args.pretrained
    aux_logits = args.aux_logits
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kargs = {'ngpu': ngpu, 'pretrained': pretrained, 'aux_logits':aux_logits, 'embedding_size': embedding_size}

    # create directory
    model_dir = os.path.join(ckpt_dir, dataset_name, loss_type)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    # network and loss
    siamese_network = SiameseNetwork(**kargs)
    gpu_number = torch.cuda.device_count()
    if device.type == 'cuda' and gpu_number > 1:
        siamese_network = nn.DataParallel(siamese_network, list(range(torch.cuda.device_count())))
    siamese_network.to(device)

    # contrastive_loss = ContrastiveLoss(margin=margin)

    # params = siamese_network.parameters()
    """
    optimizer = optim.Adam([
                       {'params': siamese_network.module.inception_v3.parameters() if gpu_number > 1 else siamese_network.inception_v3.parameters()},
                       {'params': siamese_network.module.main.parameters() if gpu_number > 1 else siamese_network.main.parameters(), 'lr': 0.0005}
                      ], lr=0.00005)
    """
    optimizer = optim.SGD([
                       {'params': siamese_network.module.inception_v3.parameters() if gpu_number > 1 else siamese_network.inception_v3.parameters()},
                       {'params': siamese_network.module.main.parameters() if gpu_number > 1 else siamese_network.main.parameters(), 'lr': 1e-3}
                      ], lr=1e-4, momentum=0.9) 



    # optimizer = optim.SGD(params, lr=0.01, momentum=0.9)

    # using different lr
    print("learning_rate = {:5.3f}".format(learning_rate))


    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=-1)


    transform = transforms.Compose([transforms.Resize((299, 299)),
                                    transforms.CenterCrop(299),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                  )
    if dataset_name == 'cub200':
        print("dataset_name = {:10s}".format(dataset_name))
        print(root_dir)
        print(image_txt)
        print(train_test_split_txt)
        print(label_txt)
        dataset_train = CubDataset(root_dir, image_txt, train_test_split_txt, label_txt, transform=transform, is_train=True, offset=1)
        dataset_eval = CubDataset(root_dir, image_txt, train_test_split_txt, label_txt, transform=transform, is_train=False, offset=1)
    elif dataset_name == 'online_product':
        print("dataset_name = {:10s}".format(dataset_name))
        dataset_train = OnlineProductDataset(root_dir, train_txt=image_txt, test_txt=train_test_split_txt, transform=transform, is_train=True, offset=1)
        dataset_eval = OnlineProductDataset(root_dir, train_txt=image_txt, test_txt=train_test_split_txt, transform=transform, is_train=False, offset=1)

    dataloader = DataLoader(dataset=dataset_train, batch_size=train_batch_size, shuffle=False, num_workers=4)
    dataloader_eval = DataLoader(dataset=dataset_eval, batch_size=test_batch_size, shuffle=False, num_workers=4)

    for epoch in range(num_epochs):
        if epoch == 0:
            feature_set, label_set = get_feature_and_label(siamese_network, dataloader_eval, device)
            evaluation(feature_set, label_set)
        siamese_network.train()
        for i, data in enumerate(dataloader, 0):
            # img_1, img_2, sim_label = data['img_1'].to(device), data['img_2'].to(device), data['sim_label'].type(torch.FloatTensor).to(device)
            img_1, img_2, label_1, label_2 = data['img_1'].to(device), data['img_2'].to(device), data['label_1'].to(device), data['label_2'].to(device)
            optimizer.zero_grad()
            output_1, output_2 = siamese_network(img_1, img_2)
            pair_dist, pair_sim_label = calculate_distance_and_similariy_label(output_1, output_2, label_1, label_2, sqrt=True, pair_type="matrix")
            if loss_type == "contrastive_loss":
                loss, positive_loss, negative_loss = contrastive_loss(pair_dist, pair_sim_label, margin)
            elif loss_type == "focal_contrastive_loss":
                loss = focal_contrastive_loss(pair_dist, pair_sim_label, margin)
            # try my own customized loss function
            # loss = contrastive_loss(output_1, output_2, pair_sim_label)
            loss.backward()
            optimizer.step()

            if i % display_step == 0 and i > 0:
                print("{}, Epoch [{:3d}/{:3d}], Iter [{:3d}/{:3d}], Loss: {:6.5f}, Positive loss: {:6.5f}, Negative loss: {:6.5f}".format(
                      datetime.datetime.now(), epoch, num_epochs, i, len(dataloader), loss.item(), positive_loss.item(), negative_loss.item()))
        if epoch % eval_step == 0:
            print("Start evalution")
            feature_set, label_set = get_feature_and_label(siamese_network, dataloader_eval, device)
            print("Forward is done for testing images")
            evaluation(feature_set, label_set)
            torch.save(siamese_network.module.state_dict(), os.path.join(model_dir, 'model_' + str(epoch) +'_.pth'))



if __name__ == '__main__':
    print("="*20)
    print(args)
    print("="*20)
    train(args)
