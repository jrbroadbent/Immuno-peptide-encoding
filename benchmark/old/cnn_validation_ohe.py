'''
Script to run DeepImmuno-CNN using OHE encoding strategy
'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc,precision_recall_curve,roc_curve,confusion_matrix
import os,sys
import pickle

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def draw_ROC(y_true,y_pred):

    fpr,tpr,_ = roc_curve(y_true,y_pred,pos_label=1)
    area_mine = auc(fpr,tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % area_mine)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def draw_PR(y_true,y_pred):
    precision,recall,_ = precision_recall_curve(y_true,y_pred,pos_label=1)
    area_PR = auc(recall,precision)
    baseline = np.sum(np.array(y_true) == 1) / len(y_true)

    plt.figure()
    lw = 2
    plt.plot(recall,precision, color='darkorange',
            lw=lw, label='PR curve (area = %0.2f)' % area_PR)
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve example')
    plt.legend(loc="lower right")
    plt.show()

def seperateCNN():
    input1 = keras.Input(shape=(12, 960, 1)) # squeeze(0).unsqueeze(-1)
    input2 = keras.Input(shape=(48, 960, 1))


    # x = layers.Dense()(input1) # projection layer
    x = layers.Conv2D(filters=16, kernel_size=(4, 960))(input1)  # [9, 1, 16]
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)    # 8
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)  # 4
    x = layers.Flatten()(x)
    x = keras.Model(inputs=input1, outputs=x)

    # y = layers.Dense()(input2) # projection layer
    # y = layers.Conv2D(filters=16, kernel_size=(17, 960))(input2)     # [32, 1, 16]
    y = layers.Conv2D(filters=16, kernel_size=(15, 12))(input2)     # [32, 1, 16]
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)  # 16
    y = layers.Conv2D(filters=32,kernel_size=(9,1))(y)    # 8
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1),strides=(2,1))(y)  # 4
    y = layers.Flatten()(y)
    y = keras.Model(inputs=input2,outputs=y)

    combined = layers.concatenate([x.output,y.output])
    z = layers.Dense(128,activation='relu')(combined)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1,activation='sigmoid')(z)

    model = keras.Model(inputs=[input1,input2],outputs=z)
    return model


###########################################################
####                Encoding strategy                  ####
###########################################################

# process 9- and 10-mers before esm encoding
def peptide_data_esm(peptide, client):   # return numpy array [10,12,1]
    length = len(peptide)
    if length == 10:
        encode = esm_embedding(peptide, client)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = esm_embedding(peptide, client)
    return encode

def esm_embedding(peptide, client):
    client = client # ESMC.from_pretrained("esmc_300m").to("cpu") # "cpu" or "cuda"
    protein = ESMProtein(sequence=peptide)
    protein_tensor = client.encode(protein) # replaces AA with int? and adds start and end term 
    logits_output = client.logits(
    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    embedding = logits_output.embeddings.squeeze(0).unsqueeze(-1)
    embedding = embedding.numpy() # convert to numpy array
    return embedding  # [len(peptide)+2, 960, 1]


def hla_data_esm(hla_dic,hla_type, client):    # return numpy array [36,960,1]
    try:
        seq = hla_dic[hla_type]
        # print("HLA seq:", seq, "length", len(seq))
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type,dic_inventory)
        seq = hla_dic[hla_type]
    encode = esm_embedding(seq, client)
    return encode

def construct_esm_embedding(ori,hla_dic,client):
    series = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        hla_type = ori['HLA'].iloc[i]
        immuno = np.array(ori['immunogenicity'].iloc[i]).reshape(1,-1)   # [1,1]

        encode_pep = peptide_data_esm(peptide, client)    # [12,960,1]  # new peptide encoding with esm

        encode_hla = hla_data_esm(hla_dic,hla_type, client) # [48,960,1]
        series.append((encode_pep, encode_hla, immuno))
    return series

def pull_peptide_esm(dataset):
    result = np.empty([len(dataset),12,960,1])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][0]
    return result

def pull_hla_esm(dataset):
    result = np.empty([len(dataset),48,960,1])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][1]
    return result

###########################################################

# this works fine regardless of encoding 
def pull_label(dataset):
    col = [item[2] for item in dataset]
    result = [0 if item == 'Negative' else 1 for item in col]
    result = np.expand_dims(np.array(result),axis=1)
    return result

def dict_inventory(inventory):
    dicA, dicB, dicC = {}, {}, {}
    dic = {'A': dicA, 'B': dicB, 'C': dicC}

    for hla in inventory:
        type_ = hla[4]  # A,B,C
        first2 = hla[6:8]  # 01
        last2 = hla[8:]  # 01
        try:
            dic[type_][first2].append(last2)
        except KeyError:
            dic[type_][first2] = []
            dic[type_][first2].append(last2)

    return dic

def rescue_unknown_hla(hla, dic_inventory):
    type_ = hla[4]
    first2 = hla[6:8]
    last2 = hla[8:]
    big_category = dic_inventory[type_]
    #print(hla)
    if not big_category.get(first2) == None:
        small_category = big_category.get(first2)
        distance = [abs(int(last2) - int(i)) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(first2) + str(optimal)
    else:
        small_category = list(big_category.keys())
        distance = [abs(int(first2) - int(i)) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(optimal) + str(big_category[optimal][0])

def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic

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

def draw_history(history):
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    os.chdir('/home/josh/Dev/Project/Models/DeepImmuno/reproduce')  #### change dir
    after_pca = np.loadtxt('./data/after_pca.txt')  #### try using my version of AA encodings
    ori = pd.read_csv('./data/remove0123_sample100.csv') # what is this datase?
    
    # choose dataset size --> 1 = whole dataset
    frac = 0.1
    ori = ori.sample(frac=frac, replace=False).set_index(pd.Index(np.arange(np.floor(ori.shape[0]*frac)))) # random sample, re-initialising indices  
    print(ori.shape)
    
    hla = pd.read_csv('./data/hla2paratopeTable_aligned.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)
    client = ESMC.from_pretrained("esmc_300m").to("cpu") # "cpu" or "cuda"
    dataset = construct_esm_embedding(ori, hla_dic, client) ### changed encoding 
    input1 = pull_peptide_esm(dataset)  # changed this 
    input2 = pull_hla_esm(dataset)  # changed this 
    label = pull_label(dataset)

    # print("input 1 (encoded peptide):", input1[0].shape)
    # print("input 2 (encoded HLA):", input2[0].shape)

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

