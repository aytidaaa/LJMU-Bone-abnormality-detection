# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 00:25:38 2021

@author: user
"""

import time
import copy
import torch
from torchnet import meter
from torch.autograd import Variable
from utils import plot_training
# import numpy
import pandas as pd
# from pipeline import torch_equalize
# from torchvision import transforms

data_cat = ['train','valid'] # data categories

def train_model(model, criterion, optimizer, dataloaders, scheduler, 
                dataset_sizes, num_epochs, is_inception=False):
    since = time.time()
    data_outs_all = pd.DataFrame()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    costs = {x:[] for x in data_cat} # for storing costs per epoch
    accs = {x:[] for x in data_cat} # for storing accuracies per epoch
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        confusion_matrix = {x: meter.ConfusionMeter(2, normalized=False) 
                            for x in data_cat}
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase=='train')
            running_loss = 0.0
            running_corrects = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data_labels = torch.Tensor().to(device)
            data_preds = torch.Tensor().to(device)
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                print(i, end='\r')
                inputs = data['images'][0]          
                labels = data['label'].type(torch.FloatTensor)
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                if is_inception and phase == 'train':
                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4*loss2
                    _, outputs = torch.mean(outputs)
                else:
                    outputs = model(inputs)
                    outputs = torch.mean(outputs)
                    loss = criterion(outputs, labels, phase)
                # outputs = model(inputs)
                # outputs = torch.mean(outputs)
                # loss = criterion(outputs, labels, phase)
                running_loss += loss.data[0]
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
                preds_probs = (outputs.data).type(torch.cuda.FloatTensor)
                preds=preds.view(-1)
                preds_probs=preds_probs.view(-1)
                running_corrects += torch.sum(preds == labels.data)
                confusion_matrix[phase].add(preds, labels.data)
                # data_preds = torch.cat((data_preds, preds), 0)
                data_preds = torch.cat((data_preds, preds_probs), 0)
                data_labels = torch.cat((data_labels, labels.data), 0)
                # print(data_preds.shape, data_labels.shape)
            data_outs = torch.cat((data_labels.unsqueeze(-1), data_preds.unsqueeze(-1)),1)
            data_outs_cpu = data_outs.cpu()
            data_outs_num = data_outs_cpu.numpy()
            data_outs_df = pd.DataFrame(data_outs_num)
            data_outs_df['data_cat'] = pd.Series([phase for x in range(len(data_outs_df.index))], index=data_outs_df.index)
            data_outs_df['epoch'] = pd.Series([epoch for x in range(len(data_outs_df.index))], index=data_outs_df.index)
            data_outs_all = data_outs_all.append(data_outs_df)
            
            epoch_loss = running_loss.item() / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('Confusion Meter:\n', confusion_matrix[phase].value())
            # deep copy the model
            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, data_outs_all


def get_metrics(model, criterion, dataloaders, dataset_sizes, phase='valid', is_inception = False):
    '''
    Loops over phase (train or valid) set to determine acc, loss and 
    confusion meter of the model.
    '''
    confusion_matrix = meter.ConfusionMeter(2, normalized=False)
    running_loss = 0.0
    running_corrects = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_labels = torch.Tensor().to(device)
    data_preds = torch.Tensor().to(device)
    data_outs_all = pd.DataFrame()
    for i, data in enumerate(dataloaders[phase]):
        print(i, end='\r')
        labels = data['label'].type(torch.FloatTensor)
        inputs = data['images'][0]
        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # forward
        # outputs = model(inputs)
        if is_inception and phase == 'train':
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            outputs, aux_outputs = model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4*loss2
            _, outputs = torch.mean(outputs)
        else:
            outputs = model(inputs)
            outputs = torch.mean(outputs)
            loss = criterion(outputs, labels, phase)

        # outputs = torch.mean(outputs)
        # loss = criterion(outputs, labels, phase)
        # statistics
        running_loss += loss.data[0] * inputs.size(0)
                # statistics
        preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
        preds_probs = (outputs.data).type(torch.cuda.FloatTensor)
        preds=preds.view(-1)
        preds_probs=preds_probs.view(-1)
        running_corrects += torch.sum(preds == labels.data)
        confusion_matrix.add(preds, labels.data)
        # data_preds = torch.cat((data_preds, preds), 0)
        data_preds = torch.cat((data_preds, preds_probs), 0)
        data_labels = torch.cat((data_labels, labels.data), 0)
        # print(data_preds.shape, data_labels.shape)
    data_outs = torch.cat((data_labels.unsqueeze(-1), data_preds.unsqueeze(-1)),1)
    data_outs_cpu = data_outs.cpu()
    data_outs_num = data_outs_cpu.numpy()
    data_outs_df = pd.DataFrame(data_outs_num)
    data_outs_df['data_cat'] = pd.Series([phase for x in range(len(data_outs_df.index))], index=data_outs_df.index)
    # data_outs_df['epoch'] = pd.Series([epoch for x in range(len(data_outs_df.index))], index=data_outs_df.index)
    data_outs_all = data_outs_all.append(data_outs_df)

    loss = running_loss.item() / dataset_sizes[phase]
    acc = running_corrects.item() / dataset_sizes[phase]
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))
    print('Confusion Meter:\n', confusion_matrix.value())
    return data_outs_all
