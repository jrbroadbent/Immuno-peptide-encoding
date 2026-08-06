'''
Encoder using One-Hot Encoding (OHE) of protein seqeunces
'''

import os
import torch
import numpy as np
import pandas as pd
# from util import rescue_unknown_hla, pull_label, dict_inventory, hla_df_to_dic
# need this version if running in main
from .util import rescue_unknown_hla, pull_label, dict_inventory, hla_df_to_dic


def one_hot_encoder(peptide):
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    encoded = np.zeros((len(peptide), len(amino)))  # (seq_len, 21) --> does it matter that its this way round?
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        # encoded[i, amino.index(query)] = 1 
    encoded = encoded.reshape(1,len(peptide),len(amino))
    return encoded


# from dataset entries to ohe 
def peptide_data_ohe(peptide):   # return numpy array [10,12,1]
    length = len(peptide)
    # process 9- and 10-mers before encoding
    if length == 10:
        encode = one_hot_encoder(peptide)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = one_hot_encoder(peptide)
    return encode


def hla_data_ohe(hla_dic, hla_type, dic_inventory):    # return numpy array [36,960,1]
    try:
        seq = hla_dic[hla_type]
        # print("HLA seq:", seq, "length", len(seq))
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type,dic_inventory)   ## dic_inventory = dict_inventory(...)?
        seq = hla_dic[hla_type]
    encode = one_hot_encoder(seq)
    return encode 


def construct_ohe(ori, hla_dic, dic_inventory):
    series = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        hla_type = ori['HLA'].iloc[i]
        immuno = np.array(ori['immunogenicity'].iloc[i]).reshape(1,-1)   # [1,1]

        # encode peptide seqeunce using OHE
        encode_pep = peptide_data_ohe(peptide)

        # encode hla paratope seqeunce using OHE
        encode_hla = hla_data_ohe(hla_dic, hla_type, dic_inventory)
        series.append((encode_pep, encode_hla, immuno))
    return series

# specific to encoding strategy because of shape differences
def pull_peptide_ohe(dataset):
    result = np.empty([len(dataset),1,10,21])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][0]
    return result


def pull_hla_ohe(dataset):
    result = np.empty([len(dataset),1,46,21])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][1]
    return result


def ohe_encode_dataset(dataset = 'iedb_data.csv'):
    os.chdir('/home/josh/Dev/Project/')
    ori = pd.read_csv("data/" + dataset)

    if dataset == "sars_cov_2_test.csv":
            ori.rename(columns={"immunogenicity-con": "immunogenicity"}, inplace=True) 
    
    frac = 1 # choose dataset size --> 1 = whole dataset
    ori = ori.sample(frac=frac, replace=False).set_index(pd.Index(np.arange(np.ceil(ori.shape[0]*frac)))) # random sample, re-initialising indices  
    
    hla = pd.read_csv('data/hla2paratopeTable_aligned.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    print("start encoding")
    dataset = construct_ohe(ori, hla_dic, dic_inventory)
    input1 = pull_peptide_ohe(dataset)
    input2 = pull_hla_ohe(dataset)
    label = pull_label(dataset)
    print("finish encoding")

    # save dataset to a file or put these functions inside custom dataset 
    x1 = torch.from_numpy(input1).to(torch.float32) 
    x2 = torch.from_numpy(input2).to(torch.float32)
    y = torch.from_numpy(label).to(torch.float32)

    ohe_encoded_dataset = {"x1": x1, "x2": x2, "y": y}

    return ohe_encoded_dataset



def main():
    print("start of program")

    os.chdir('/home/josh/Dev/Project/')
    ori = pd.read_csv('./data/iedb_data.csv')
    
    frac = 1 # choose dataset size --> 1 = whole dataset
    ori = ori.sample(frac=frac, replace=False).set_index(pd.Index(np.arange(np.ceil(ori.shape[0]*frac)))) # random sample, re-initialising indices  
    
    hla = pd.read_csv('./data/hla2paratopeTable_aligned.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    print("start encoding")
    dataset = construct_ohe(ori, hla_dic, dic_inventory)
    input1 = pull_peptide_ohe(dataset)
    input2 = pull_hla_ohe(dataset)
    label = pull_label(dataset)
    print("finish encoding")

    # save dataset to a file or put these functions inside custom dataset 
    x1 = torch.from_numpy(input1).to(torch.float32) 
    x2 = torch.from_numpy(input2).to(torch.float32)
    y = torch.from_numpy(label).to(torch.float32)

    print("saving to file")
    torch.save({"x1": x1, "x2": x2, "y": y}, "encoders/OHE_encoded_samples.pt")
    print("finished")

    return None


if __name__ == '__main__':
    main()
