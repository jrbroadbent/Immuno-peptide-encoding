'''
Script to run DeepImmuno-CNN using ESM C encoding strategy
takes ages with ~9000 samples - see how it does with 100?
use GPU?
'''

print ("start")
import torch
# import tensorflow as tf  ## seg fault
# import tensorflow.keras as keras  ## seg fault
# from tensorflow.keras import layers  ## seg fault # not needed anymore?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc,precision_recall_curve,roc_curve,confusion_matrix
import os,sys
import pickle

# # from esm.models.esmc import ESMC  # needed for client definition 
# # from esm.sdk.api import ESMProtein, LogitsConfig

from encoders.ESM import *
from models.model_defs import seperateCNN

print("before transformer ESM imports")

from transformers import AutoTokenizer, EsmModel

print("finished imports")




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
    print("start of programme")
    os.chdir('/home/josh/Dev/Project/')  #### is this necessary?
    after_pca = np.loadtxt('./data/after_pca.txt')  #### try using my version of AA encodings
    ori = pd.read_csv('./data/remove0123_sample100.csv') # what is this datase?
    
    # choose dataset size --> 1 = whole dataset
    frac = 1
    ori = ori.sample(frac=frac, replace=False).set_index(pd.Index(np.arange(np.ceil(ori.shape[0]*frac)))) # random sample, re-initialising indices  
    print(ori.shape)
    
    hla = pd.read_csv('./data/hla2paratopeTable_aligned.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)
    
    # client = ESMC.from_pretrained("esmc_300m").to("cpu") # "cpu" or "cuda"

    model_name = "facebook/esm2_t6_8M_UR50D"
    client = EsmModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = construct_esm_embedding(ori, hla_dic, dic_inventory, client, tokenizer) ### changed encoding 
    input1 = pull_peptide_esm(dataset)  # changed this 
    input2 = pull_hla_esm(dataset)  # changed this 
    label = pull_label(dataset)

    print("input 1 (encoded peptide):", input1[0].shape)
    print("input 2 (encoded HLA):", input2[0].shape)

    # let's do a train/validation split
    bucket_roc = []
    bucket_pr = []
    for i in range(10):
        array = np.arange(len(dataset))
        train_index = np.random.choice(array,int(len(dataset)*0.9),replace=False)
        valid_index = [item for item in array if item not in train_index]

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
            class_weight = {0:0.5,1:0.5},   # I have 20% positive and 80% negative in my training data
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

