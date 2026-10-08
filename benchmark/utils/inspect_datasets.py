import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split, Subset
from sklearn.model_selection import KFold
import optuna

import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc,precision_recall_curve,roc_curve,confusion_matrix
import os,sys
import pickle
import copy
import datetime

from encoders.AAindex_pca import * 
from encoders.ESM import * 
from encoders.OHE import *

# from encoders.ESM import *
from models.model_defs import OHE_separateCNN, AAindex_separateCNN, AAindex_pca_separateCNN, ESM_separateCNN
# from transformers import AutoTokenizer, EsmModel


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


####################################################################################################################


class ImmunoDataset(Dataset):
    def __init__(self, dataset):
        self.x1 = dataset["x1"]
        self.x2 = dataset["x2"]
        self.labels = dataset["y"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x1 = self.x1[idx,:,:,:]
        x2 = self.x2[idx,:,:,:]
        x = [x1,x2]
        y = self.labels[idx,:]
        return x, y
    

ESM_dataset = ImmunoDataset(torch.load('data/ESM_encoded_samples.pt'))
AAindex_dataset = ImmunoDataset(torch.load('data/AAindex_encoded_samples.pt'))
AAindex_pca_dataset = ImmunoDataset(torch.load('data/AAindex_pca_encoded_samples.pt'))
OHE_dataset = ImmunoDataset(torch.load('data/OHE_encoded_samples.pt'))


for dataset in [ESM_dataset, AAindex_dataset, AAindex_pca_dataset, OHE_dataset]:
    idxs = np.random.randint(0, len(dataset), 10)

    print("dataset name:", dataset)

    dataloader =  DataLoader(dataset)

    ys = []
    for i, (X, y) in enumerate(dataloader):
        ys.append(y)

        if i == 3000:
            print(X[0].shape, X[1].shape)

    print(sum(ys), "out of", len(dataloader)) # number of 1s in dataset -> same as length of dataset 



models = [ESM_separateCNN, AAindex_pca_separateCNN, AAindex_separateCNN, OHE_separateCNN]
model_names = ["ESM", "AAindex_pca", "AAindex", "OHE"]

# loop through models
for i, model in enumerate(models):
    # loop through extra test sets 
    test_sets = ['sars_cov_2_test.csv', 'dengue_test.csv', 'neoantigen_test.csv']
    test_set_names = ['sars_cov_2', 'dengue', 'neoantigen']
    for k, test_set in enumerate(test_sets):
        print(f"\n-------------------------------\n {model_names[i]} Model : TEST set benchmarking \n-------------------------------")
        
        # encode test set
        encoders = [esm_encode_dataset, 
                    AAindex_encode_dataset, 
                    AAindex_encode_dataset, 
                    ohe_encode_dataset]

        encoder = encoders[i]
        if (model_names[i] == "AAindex_pca"):
            encoded_test_set = encoder(pca=True, dataset=test_set)
        elif (model_names[i] == "AAindex"):
            encoded_test_set = encoder(pca=False, dataset=test_set)
        else: 
            encoded_test_set = encoder(dataset=test_set)
        encoded_test_set = ImmunoDataset(encoded_test_set)
        dataloader =  DataLoader(encoded_test_set)

        ys = []
        for el, (X, y) in enumerate(dataloader):
            ys.append(y)

            if el == 0:
                print(X[0].shape, X[1].shape)

        print(sum(ys), "out of", len(dataloader)) # number of 1s in dataset -> same as length of dataset 


#### RESULTS ####
# sars: 25/92
# dengue: 408/408
# neoantigen: 35/522

