import torch
import torchvision
from torchvision import *
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import os
import OQA_model

def get_model(n_classes = 2):
    net = models.resnet18(pretrained=True)
    net.fc = nn.Sequential(
            nn.Linear(net.fc.in_features, 128),
            nn.Dropout(),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,n_classes))
    return net

def get_model_oqa(n_classes = 2):
    net = OQA_model.OQA_model(n_class=n_classes)
    return net
