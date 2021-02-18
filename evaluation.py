# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 01:52:50 2021

@author: user
"""
# roc curve and auc
from sklearn.datasets import make_classification
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
#from matplotlib import pyplot
name  = "out1"

import pandas as pd
valid_output = pd.read_csv(r"C:\Users\AdityaThokala\Downloads\Education\LJMU\Final Thesis\Results\valid_"+name+".csv")
testy = valid_output['0']
lr_probs = valid_output['1']

valid_output['target'] = np.where(valid_output['1'] > 0.5,1,0)
target = valid_output['target']

ns_probs = [0 for _ in range(len(testy))]

ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarize scores
#print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Model: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, thresh = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, thresh = roc_curve(testy, lr_probs)

from sklearn.metrics import cohen_kappa_score

kap = cohen_kappa_score(testy, target)

print('Kappa score =%.3f' % (kap))

from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score
acc_score = accuracy_score(testy, target)
print('Model: Accuracy =%.3f' % (acc_score))


from sklearn.metrics import auc
lr_precision, lr_recall, thresh = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, target), auc(lr_recall, lr_precision)
# summarize scores
print('Model: f1=%.3f ' % (lr_f1))
print('Model: auc=%.3f' % (lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Model')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show() 


#pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
#pyplot.plot(lr_recall, lr_precision, marker='.', label='Model')
## axis labels
#pyplot.xlabel('Recall')
#pyplot.ylabel('Precision')
## show the legend
#pyplot.legend()
## show the plot
#pyplot.show()

#############################################
#Confusion Matrix
############################################
df_confusion = pd.crosstab(testy, target)

import matplotlib.pyplot as plt
#def plot_confusion_matrix(df_confusion, title='Confusion matrix'): #, cmap=plt.cm.gray_r
#    plt.matshow(df_confusion) # imshow, cmap=cmap
#    #plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(df_confusion.columns))
#    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
#    plt.yticks(tick_marks, df_confusion.index)
#    #plt.tight_layout()
#    plt.ylabel(df_confusion.index.name)
#    plt.xlabel(df_confusion.columns.name)
#
#plot_confusion_matrix(df_confusion)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_confusion, annot=True)
#############################################


output = pd.read_csv(r"C:\Users\AdityaThokala\Downloads\Education\LJMU\Final Thesis\Results\\"+name+".csv")
output['target'] = np.where(output['1'] > 0.5,1,0)
d = []
data_cat = ['train','valid']
num_epochs = 5
for phase in data_cat:
    for epoch in range(num_epochs):
        sample = output[output['data_cat']==phase]
        sample = sample[sample['epoch']==epoch]
        testy = sample['0']
        lr_probs = sample['target']
        acc_score = accuracy_score(testy, lr_probs)
        d.append(
                {
                    'phase': phase,
                    'epoch': epoch,
                    'accuracy': acc_score
                }
            )
evals = pd.DataFrame(d)

(evals.pivot_table(index='epoch', columns='phase', values='accuracy',
                aggfunc='sum', fill_value=0)
   .plot.bar()
)