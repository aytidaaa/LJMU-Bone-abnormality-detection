# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 00:41:19 2021

@author: user
"""

import os
os.chdir("C:\Adi\Mura_codes")
import time
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from densenet import densenet169
from vgg import vgg16
from resnet import resnet50
from inception import inception_v3
from utils import plot_training, n_p, get_count
from train_plus import train_model, get_metrics
from pipelinehist import get_study_level_data, get_dataloaders
from sklearn.utils import resample

#### load study level dict data
study_data = get_study_level_data(study_type='XR_FINGER')

### Class balancing using up-sampling
study_data_majority = study_data['train'][study_data['train'].Label==0]
study_data_minority = study_data['train'][study_data['train'].Label==1]
 
study_data_valid = study_data['valid']
# Upsample minority class
study_data_upsampled = resample(study_data_minority, 
                                  replace=True,     # sample with replacement
                                  n_samples=len(study_data_majority.index),    # to match majority class
                                  random_state=123)
study_data = {}
study_data['train'] = pd.concat([study_data_majority, study_data_upsampled])
study_data['valid'] = study_data_valid

# #### Create dataloaders pipeline
data_cat = ['train', 'valid'] # data categories
dataloaders = get_dataloaders(study_data, batch_size=1)
dataset_sizes = {x: len(study_data[x]) for x in data_cat}

# #### Build model
# tai = total abnormal images, tni = total normal images
tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
tni = {x: get_count(study_data[x], 'negative') for x in data_cat}
Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) for x in data_cat}
Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) for x in data_cat}

print('tai:', tai)
print('tni:', tni, '\n')
print('Wt0 train:', Wt0['train'])
print('Wt0 valid:', Wt0['valid'])
print('Wt1 train:', Wt1['train'])
print('Wt1 valid:', Wt1['valid'])

class Loss(torch.nn.modules.Module):
    def __init__(self, Wt1, Wt0):
        super(Loss, self).__init__()
        self.Wt1 = Wt1
        self.Wt0 = Wt0
        
    def forward(self, inputs, targets, phase):
        loss = - (self.Wt1[phase] * targets * inputs.log() + self.Wt0[phase] * (1 - targets) * (1 - inputs).log())
        return loss
#############################################################

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=1):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc = nn.Identity()
        
        # Create new classifier
        self.classifier = nn.Linear(2048+1664, nb_classes)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        
        x = self.classifier(F.relu(x))
        return x

# # Train your separate models
# # ...
# # We use pretrained torchvision models here
# modelA = resnet50(pretrained=True)
# modelB = densenet169(pretrained=True)

# # Freeze these models
# for param in modelA.parameters():
#     param.requires_grad_(False)

# for param in modelB.parameters():
#     param.requires_grad_(False)

# # Create ensemble model
# model = MyEnsemble(modelA, modelB)

########################################################


model = densenet169(pretrained=True)
# model = resnet50(pretrained = True)
# model = inception_v3(pretrained=True)
# model = vgg16(pretrained=True)
model = model.cuda()

criterion = Loss(Wt1, Wt0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
# optimizer = torch.optim.SGD(model.classifier.parameters(), lr=1e-5)
# optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-5)

# optimizer = torch.optim.Adam(list(modelA.parameters()) + list(modelB.parameters()), lr=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

# #### Train model
model, output = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=200)

# model, output = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=5, is_inception = True)

torch.save(model.state_dict(), 'C:/Adi/models/model.pth')

valid_output = get_metrics(model, criterion, dataloaders, dataset_sizes)
valid_output.to_csv(r'C:\Adi\results\valid_out.csv', index = False)
output.to_csv(r'C:\Adi\results\out.csv', index = False)
