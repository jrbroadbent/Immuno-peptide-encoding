'''

'''

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split


import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc,precision_recall_curve,roc_curve,confusion_matrix
import os,sys
import pickle
import copy

# from encoders.ESM import *
from models.model_defs import OHE_seperateCNN, AAindex_seperateCNN, ESM_seperateCNN
from transformers import AutoTokenizer, EsmModel

print("finished imports")


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def bootstrap(
        train_dataset,
        test_dataset: torch.utils.data.Dataset,   # test set
        model,
        loss_fn,
        optimizer,
        n: int,         # size of sample
        B: int,         # number of samples
        ):

    model.eval()
    
    log_scores = []
    for i in range(B):
        #### TRAIN ####
        sampler = RandomSampler(test_dataset, replacement=True, num_samples=len(train_dataset))
        train_dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=128)  # batch_size default 1
        
        epochs = 200
        best_loss = float('inf')
        early_stopping = EarlyStopping(patience=2, delta=0, verbose=True)
        # best_weights = None
        for epoch in range(epochs):
            # print(f"Epoch {epoch+1}\n-------------------------------")
            loss = train(train_dataloader, model, loss_fn, optimizer, verbose=False, output=True)
            if loss < best_loss:
                best_loss = loss
                # best_weights = copy.deepcopy(model.state_dict())
    
            # Check early stopping condition
            early_stopping.check_early_stop(loss)
            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch+1}")
                break


        #### TEST ####
        # random sample with replacement
        sampler = RandomSampler(test_dataset, replacement=True, num_samples=n)
        test_dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=1)  # batch_size default 1
        test_loss, correct = test(test_dataloader,model,loss_fn, output=True)
        
        if (i % 10) == 0:
            print("Bootstrap iteration:", i)
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        log_scores.append(test_loss)


        #### reset model ####
        # train function zeros optimizer gradients at end of each iteration, so only need to reinitialise model weights
        model.apply(reset_weights) 
    
    # return mean and SE
    mean = np.mean(log_scores)
    se = np.std(log_scores, ddof=1)/np.sqrt(B)

    return log_scores, mean, se 



# early stopping if loss doesn't change for ... steps (accoring to threshold)
# save model weights whenever loss decreases 
# restore best model weights 

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




    # implement early stopping
    # use epochs = 200, batch size = 128  # specified in dataloader? 
    # need to specify 0.5 threshold in the model 
    # is pred going to be either 1 or 0? or should it be a probability 
    # pred should be a probability 


def test(dataloader, model, loss_fn, verbose=False, output=False):
    size = dataloader.sampler._num_samples
    num_batches = len(dataloader)
    model.eval()                    # what does this do?
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
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
    print("start of program")
    os.chdir('/home/josh/Dev/Project/')

    # load data 
    # dataset = torch.load('encoders/ESM_encoded_samples.pt')
    # dataset = torch.load('encoders/AAindex_encoded_samples.pt')
    dataset = torch.load('encoders/OHE_encoded_samples.pt')

    dataset = ImmunoDataset(dataset)

    # train test split
    train_data, test_data = random_split(dataset, [0.8,0.2])

    # batch_size = 128
    # train_dataloader = DataLoader(train_data, batch_size=batch_size)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    # specify different models here 
    #model = ESM_seperateCNN()
    #model = AAindex_seperateCNN()
    model = OHE_seperateCNN()
    model.to(device)

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    # bootrsap
    log_scores, mean, se = bootstrap(train_data, test_data, model, loss_fn, optimizer, n=10, B=10) # n=len(test_data), B=1000)
    print("bootstrap mean:", mean)
    print("bootstrap se:", se)

    # remove skew ??
    log_scores = np.log(log_scores) # natural logarithm

    # confidence interval
    CI = np.percentile(log_scores, [2.5,97.5])
    print("confidence interval:\n", CI)

    plt.hist(log_scores, bins=10, density=True) # range=(0, CI[1]+se)
    plt.title("Histogram of ESM encoder model log scores")
    plt.xlabel("log score")
    plt.ylabel("density")
    plt.savefig("LogScoreDist.png")


    # do bootstrap for each model here 


    return None


    print(dataset.keys())
    print(dataset['x1'].size(), dataset['x2'].size(), dataset['y'].size())
    print(dataset['x1'][1,:,:,:].size(), dataset['x2'][1,:,:,:].size(), len(dataset['y']))
    print(len(train_data), len(test_data))

#####################################################################################################

    # let's do a train/validation split
    bucket_roc = []
    bucket_pr = []
    for i in range(10):
        array = np.arange(len(dataset))
        train_index = np.random.choice(array,int(len(dataset)*0.9),replace=False)  # change replace to True and effectively got bootstrap
        valid_index = [item for item in array if item not in train_index]   # OOB 

        input1_train = input1[train_index]
        input1_valid = input1[valid_index]
        input2_train = input2[train_index]
        input2_valid = input2[valid_index]
        label_train = label[train_index]
        label_valid = label[valid_index]


        import tensorflow.keras as keras
        cnn_model = seperateCNN()
        cnn_model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            metrics=['accuracy'])

        callback_val = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,restore_best_weights=False)
        callback_train = keras.callbacks.EarlyStopping(monitor='loss',patience=2,restore_best_weights=False)
        history = cnn_model.fit(
            x=[input1_train,input2_train],   # feed a list into
            y=label_train,
            validation_data = ([input1_valid,input2_valid],label_valid),
            batch_size=128,
            epochs=200,
            class_weight = {0:0.5,1:0.5},   # I have 20% positive and 80% negative in my training data  # really??
            callbacks = [callback_val,callback_train])

        valid = ori.loc[valid_index]
        valid['cnn_regress'] = cnn_model.predict([input1_valid,input2_valid])
        valid = valid.sort_values(by='cnn_regress',ascending=False).set_index(pd.Index(np.arange(valid.shape[0])))
        y_true = [1 if not item == 'Negative' else 0 for item in valid['immunogenicity']]
        y_pred = valid['cnn_regress']

        fpr,tpr,_ = roc_curve(y_true,y_pred)
        area = auc(fpr,tpr)
        bucket_roc.append((fpr,tpr,_,area))

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        area = auc(recall, precision)
        bucket_pr.append((precision, recall, _, area))

    # ROC
    bucket = bucket_roc
    fig,ax = plt.subplots()
    for i in range(10):
        ax.plot(bucket[i][0],bucket[i][1],lw=0.5,label='CV(Fold={0}), AUC={1:.2f}'.format(i+1,bucket[i][3]))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right",fontsize=9)
    plt.savefig("DeepImmuno_ROC_esm.png")

    # PR
    bucket = bucket_pr
    fig,ax = plt.subplots()
    for i in range(10):
        ax.plot(bucket[i][1],bucket[i][0],lw=0.5,label='CV(Fold={0}),AUC={1:.2f}'.format(i+1,bucket[i][3]))
    #baseline = np.sum(np.array(y_true) == 1) / len(y_true)  # 0.4735
    baseline = 0.4735
    ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    #ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR curve example')
    ax.legend(loc="lower left",fontsize=8)
    plt.savefig("DeepImmuno_PR_esm.png")

    return None


if __name__ == '__main__':
    main()


