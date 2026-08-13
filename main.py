'''
This Script runs hyperparameter tuning and bootstrap benchmark for OHE, AAindex, and ESM models
'''

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split, Subset
from sklearn.model_selection import KFold
import optuna
import optunahub 

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
from models.model_defs import OHE_seperateCNN, AAindex_seperateCNN, AAindex_pca_seperateCNN, ESM_seperateCNN
from transformers import AutoTokenizer, EsmModel


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def objective(trial, model_fn, train_data): # could just pass it my train data 
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    dropout = trial.suggest_float('p', 0.1, 0.5)
    patience = trial.suggest_int('patience', 0, 20)

    model = model_fn(dropout=dropout)    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    # produce splits 
    kf = KFold(n_splits=5)

    # cross validation 
    for train_set, val_set in kf.split(train_data):
        data = Subset(train_data, train_set)
        train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        # train
        model.train()
        # best_loss = float('inf')
        early_stopping = EarlyStopping(patience=patience, delta=0, verbose=True)
        for epoch in range(200): # number of epochs
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Check early stopping condition
            early_stopping.check_early_stop(loss)
            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # validate
        data = Subset(train_data, val_set)
        val_loader = DataLoader(data, batch_size=1, shuffle=True)
        num_batches = len(val_loader)
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
            val_loss /= num_batches
        
    # return mean validation loss
    return val_loss


# split up into train_loop and boostrap?
def bootstrap(
        train_dataset,
        test_dataset: torch.utils.data.Dataset,   # test set
        model,
        params,
        loss_fn,
        optimizer,
        n: int,         # size of sample
        B: int,
        test_only = False                 # number of samples
        ):
    

    #### reset model & train ####
    if not test_only:
        # train function zeros optimizer gradients at end of each iteration, so only need to reinitialise model weights
        model.apply(reset_weights) 
        
        train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"])
        
        epochs = 200
        # best_loss = None
        early_stopping = EarlyStopping(patience=params["patience"], delta=0, verbose=True)
        # best_weights = None
        for epoch in range(epochs):
            # print(f"Epoch {epoch+1}\n-------------------------------")
            loss = train(train_dataloader, model, loss_fn, optimizer, verbose=False, output=True)
            # if loss < best_loss:
            #     best_loss = loss
            #     # best_weights = copy.deepcopy(model.state_dict())

            # Check early stopping condition
            early_stopping.check_early_stop(loss)
            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch+1}")
                break


    log_scores = []
    for i in range(B):
        #### TEST ####
        # random sample with replacement
        sampler = RandomSampler(test_dataset, replacement=True, num_samples=n)
        test_dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=1)  # batch_size default 1
        test_loss, correct = test(test_dataloader,model,loss_fn, output=True)
        
        if (i % 10) == 0:
            print("Bootstrap iteration:", i)
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        log_scores.append(test_loss)

    
    # mean and SE
    mean = np.mean(log_scores)
    se = np.std(log_scores, ddof=1)/np.sqrt(B)
    
    # confidence interval
    CI = np.percentile(log_scores, [2.5,97.5])

    return log_scores, CI, mean, se 


def train(dataloader, model, loss_fn, optimizer, verbose=False, output=False):
    size = len(dataloader.dataset)
    model.train()
    # print("num samples:", len(dataloader.dataset))  # ~7000
    # print("num batches", len(dataloader))  # ~50 (batch_size=128)
    for batch, (X, y) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device)


        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X[0])
            # print(X[0].size())  # [128,1,12,320]
            if verbose:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    if output:
        return loss 


def test(dataloader, model, loss_fn, verbose=False, output=False):
    size = dataloader.sampler._num_samples
    num_batches = len(dataloader)
    model.eval()                    # what does this do?
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            # print("pred and y:", pred, y)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).type(torch.float).sum().item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    if verbose:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    if output:
        return test_loss, correct


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
    

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def retain_910(ori):
    cond = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        if len(peptide) == 9 or len(peptide) == 10:
            cond.append(True)
        else:
            cond.append(False)
    data = ori.loc[cond]
    data = data.set_index(pd.Index(np.arange(data.shape[0])))
    return data



def main():
    # add command line arguments?
    HYPERPARAM_OPTIM = False
    d = datetime.datetime.now()
    id = str(d.month)+'_'+str(d.day)+'_'+str(d.hour)
    PARAMS_FILE = 'original_params.pkl'
    #PARAMS_FILE = 'model_params_'+id+'.pkl'
    RESULTS_FILE = 'BCE_scores_'+id+'.pkl'
    TEST_FILE = 'test_BCE_scores_'+id+'.pkl'
    # FIGURES = True
    # BOOTSTRAP_FIG = 'bench_fig_no_retrain'
    # BOOTSTRAP_RETRAIN_FIG = 'bench_fig_retrain'

    print("\nstart of program\n")
    os.chdir('/home/josh/Dev/Project/')

    # Load Data 
    ESM_dataset = torch.load('data/ESM_encoded_samples.pt')
    AAindex_dataset = torch.load('data/AAindex_encoded_samples.pt')
    AAindex_pca_dataset = torch.load('data/AAindex_pca_encoded_samples.pt')
    OHE_dataset = torch.load('data/OHE_encoded_samples.pt')
    datasets = [ESM_dataset, AAindex_pca_dataset, AAindex_dataset, OHE_dataset]

    # Model
    models = [ESM_seperateCNN, AAindex_pca_seperateCNN, AAindex_seperateCNN, OHE_seperateCNN]
    model_names = ["ESM", "AAindex_pca", "AAindex", "OHE"]

    results = {}
    test_results = {}

    params = []
    for i, (model, dataset) in enumerate(zip(models, datasets)):
        data = ImmunoDataset(dataset)
        train_data, test_data = random_split(data, [0.8,0.2])

        # Hyperparameter optimisation
        if HYPERPARAM_OPTIM:
            module = optunahub.load_module(package="samplers/auto_sampler")
            study = optuna.create_study(study_name=model_names[i]+"_hyperparam_optim", sampler=module.AutoSampler(), direction='minimize')
            study.optimize(lambda trial: objective(trial, model, train_data), n_trials=100)  # number of trials: 100-1000
            print(f"{model_names[i]} Best Hyperparameters: {study.best_params}")

            params.append(study.best_params)

        # Load saved hyperparameters
        if not HYPERPARAM_OPTIM:
            with open("data/"+PARAMS_FILE, "rb") as f:
                params = pickle.load(f)

    
        # Benchmarking
        print(f"{model_names[i]} Model \n-------------------------------")

        model = model(dropout=params[i]["p"])
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params[i]["lr"])


        # bootstrap with 10 repeats for retraining 
        # use first run as values for bootstrap without retraining
        for j in range(10):
            log_scores, CI, mean, se = bootstrap(train_data, test_data, model, params[i], loss_fn, optimizer, n=len(test_data), B=100, test_only=False)
            print("bootstrap mean:", mean)
            print("bootstrap se:", se)
            print(f"confidence interval:\n {CI} \n\n")
            results[(j, model_names[i])] = log_scores


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
            # print("CHECK y:", y)


            # bootstrap validate on test set
            for j in range(1): # no repeats for test sets 
                log_scores, CI, mean, se = bootstrap(None, encoded_test_set, model, params[i], loss_fn, None, n=len(encoded_test_set), B=100, test_only=True)
                print("bootstrap mean:", mean)
                print("bootstrap se:", se)
                print(f"confidence interval:\n {CI} \n\n")
                test_results[(j, model_names[i], test_set_names[k])] = log_scores


    # save optimised hyperparameters to disk
    if HYPERPARAM_OPTIM:
        with open("data/"+PARAMS_FILE, "wb") as f:
            pickle.dump(params, f)

    # save log scores dictionary to disk
    with open("data/"+RESULTS_FILE, "wb") as f:  # "bootstrap_tain_avg.pkl"
        pickle.dump(results, f)

    # save log scores dictionary to disk
    with open("data/"+TEST_FILE, "wb") as f:  # "bootstrap_tain_avg.pkl"
        pickle.dump(test_results, f)

    # Produce figures
    # if FIGURES:
    #     ...


        


if __name__ == '__main__':
    main()


