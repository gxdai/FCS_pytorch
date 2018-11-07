from __future__ import print_function
import os
import sys
import torch 
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from model import SiameseNetwork, ContrastiveLoss
from dataset import CubDataset
from torch.utils.data import DataLoader
from evaluation import  get_feature_and_label, evaluation

os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'

ngpu = 2
margin = 1.
num_epochs = 1000
train_batch_size = 64
test_batch_size = 32
gamma = 0.98 # for learning rate decay
pretrained=False
aux_logits = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
kargs = {'ngpu': ngpu, 'pretrained': pretrained, 'aux_logits':aux_logits}
siamese_network = SiameseNetwork(**kargs)
if device.type == 'cuda' and torch.cuda.device_count() > 1:
    siamese_network = nn.DataParallel(siamese_network, list(range(torch.cuda.device_count())))
siamese_network.to(device)
contrastive_loss = ContrastiveLoss(margin=margin)

# params = siamese_network.parameters()

# optimizer = optim.Adam(params, lr=0.0005)
# optimizer = optim.SGD(params, lr=0.01, momentum=0.9)

# using different lr
optimizer = optim.SGD([
                       {'params': siamese_network.module.inception_v3.parameters()},
                       {'params': siamese_network.module.main.parameters(), 'lr': 1e-3}
                      ], lr=0.00001, momentum=0.9)



scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=-1)

root_dir ="/data1/Guoxian_Dai/CUB_200_2011/images"
image_txt ="/data1/Guoxian_Dai/CUB_200_2011/images.txt"
train_test_split_txt ="/data1/Guoxian_Dai/CUB_200_2011/train_test_split.txt"
label_txt ="/data1/Guoxian_Dai/CUB_200_2011/image_class_labels.txt"


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




transform = transforms.Compose([transforms.Resize((299, 299)), 
                                    transforms.CenterCrop(299), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                  )
cub_dataset = CubDataset(root_dir, image_txt, train_test_split_txt, label_txt, transform=transform, is_train=True, offset=1)
dataloader = DataLoader(dataset=cub_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

cub_dataset_eval = CubDataset(root_dir, image_txt, train_test_split_txt, label_txt, transform=transform, is_train=False, offset=1)
dataloader_eval = DataLoader(dataset=cub_dataset_eval, batch_size=test_batch_size, shuffle=False, num_workers=4)

for epoch in range(num_epochs):
    siamese_network.train()
    # feature_set, label_set = get_feature_and_label(siamese_network, dataloader_eval, device)
    for i, data in enumerate(dataloader, 0):
        img_1, img_2, sim_label = data['img_1'].to(device), data['img_2'].to(device), data['sim_label'].type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        output_1, output_2 = siamese_network(img_1, img_2)
        loss = contrastive_loss(output_1, output_2, sim_label)
        loss.backward()
        optimizer.step()

        if i % 20 == 0 and i > 0:
            print("Epoch [{:3d}/{:3d}], Iter [{:3d}/{:3d}], \
                  Current loss: {}".format(epoch, num_epochs, i, len(dataloader), loss.item()))
    if epoch % 10 == 0:
        print("Start evalution")
        feature_set, label_set = get_feature_and_label(siamese_network, dataloader_eval, device)
        evaluation(feature_set, label_set)
        torch.save(siamese_network.module.state_dict(), 'model_' + str(epoch) +'_.pth')
