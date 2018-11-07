import sys
import os
import time
import datetime
import math

import numpy as np
from scipy.spatial import distance
import cub_2011
import RetrievalEvaluation
import evaluate_recall
import evaluate_clustering

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms






class SiameseNetwork(nn.Module):
    def __init__(self, **kargs):
        super(SiameseNetwork, self).__init__()
        self.ngpu = kargs['ngpu']
        self.inception_v3 = models.inception_v3(pretrained=kargs['pretrained'],
                                                aux_logits=kargs['aux_logits'])
        self.inception_v3.load_state_dict(torch.load('inception_v3_no_aux_logits.pth'))
        self.inception_v3.fc = nn.Linear(2048, 1000)
        self.main = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )
        # self.weight_init()

    def weight_init(self):

        self.inception_v3.fc.weight.data.normal_(mean=0, std=10)
        self.inception_v3.fc.bias.data = torch.zeros(1000)
        self.main[0].weight.data.normal_(mean=0, std=10)
        self.main[0].bias.data = torch.zeros(512)
        self.main[2].weight.data.normal_(mean=0, std=10)
        self.main[2].bias.data = torch.zeros(64)

    def forward_once(self, x):
        logits = self.inception_v3(x)
        outputs = self.main(logits)
        return outputs

    def forward(self, input1, input2):
        output_1 = self.forward_once(input1)
        output_2 = self.forward_once(input2)
        return output_1, output_2



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, sim_label):
        dist = torch.sqrt(torch.sum(torch.pow(input1 - input2, 2), dim=1) + 1e-6)
        positive_dist = sim_label * torch.pow(dist, 2)
        negative_dist = (1 - sim_label) * torch.pow(torch.clamp(self.margin - dist, 0.), 2)

        contrastive_loss = torch.mean(positive_dist + negative_dist)

        return contrastive_loss














# This is an example for weight initialization.
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)  # normal distribution initialization.
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1., 0.02)
        module.bias.data.fill_(0)

if __name__ == '__main__':
    siamese_network = SiameseNetwork(ngpu=2)
