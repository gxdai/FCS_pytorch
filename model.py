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
import torch.nn.functional as F
from torchvision import models, datasets, transforms

class SiameseNetwork(nn.Module):
    def __init__(self, **kargs):
        # init base class
        # in python3 
        # super() could help you avoid base class explicitly.
        """
        
        super(Child, self).__init__() ---->>>> super().__init__()
        """
        super(SiameseNetwork, self).__init__()
        self.ngpu = kargs['ngpu']
        self.inception_v3 = models.inception_v3(pretrained=kargs['pretrained'],
                                                aux_logits=kargs['aux_logits'])
        self.inception_v3.load_state_dict(torch.load('inception_v3_no_aux_logits.pth'))
        # remove last layer
        print("Init the last two layers with normal distribution")
        self.inception_v3.fc = nn.Linear(2048, 1000)
        self.main = nn.Sequential(
            nn.Linear(1000, kargs['embedding_size'])
        )
        print(kargs["embedding_size"])
        """
        self.inception_v3.fc = nn.Linear(2048, 1000)
        self.main = nn.Sequential(
            nn.Linear(1000, kargs['embedding_size'])
        )
        print(kargs["embedding_size"])
        """


        """
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000, 512),
            nn.Linear(512, 64)
        )
        """


        # init weights 
        # self.inception_v3.fc.weight.data.copy_(self.weight_init(self.inception_v3.fc, stddev=0.01))
        # self.main[0].weight.data.copy_(self.weight_init(self.main[0], stddev=0.01))
        """
        self.main[3].weight.data.copy_(self.weight_init(self.main[3], stddev=0.1))
        """

        # self.weight_init()
    
    def separate_parameter_group(self):
        """Separate parameters into different groups."""
        first_group = []
        second_group = []
        for name, param in self.named_parameters():
            if "inception_v3.fc" in name or "main.0" in name:
                second_group.append(param)
            else:
                first_group.append(param) 

        return first_group, second_group

    def weight_init(self, m, stddev=0.1):
        import scipy.stats as stats
        stddev = stddev
        X = stats.truncnorm(-2, 2, scale=stddev)
        values = torch.Tensor(X.rvs(m.weight.numel()))
        values = values.view(m.weight.size())
        return values

        """
        m.weight.data.copy_(values)
        self.inception_v3.fc.weight.data.normal_(mean=0, std=10)
        self.main[0].weight.data.normal_(mean=0, std=10)
        self.main[0].bias.data = torch.zeros(512)
        self.main[2].weight.data.normal_(mean=0, std=10)
        self.main[2].bias.data = torch.zeros(64)
        """

    def forward_once(self, x):
        logits = self.inception_v3(x)
        outputs = self.main(logits)
        return outputs

    def forward_once_2(self, x):
        outputs = self.inception_v3(x)
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
