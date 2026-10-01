'''
Implementation of AAindex-PCA encoding strategy for peptide sequences
'''

import torch
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import glob
#from util import rescue_unknown_hla, pull_label, dict_inventory, hla_df_to_dic
# need this version if running in main
from .util import rescue_unknown_hla, pull_label, dict_inventory, hla_df_to_dic

DEFAULT_INPUT_PATH="data/AAindex1"
DEFAULT_OUTPUT_PATH="data/"

def AAindex_encoder(peptide, AAindex_pca):

    amino = 'ARNDCQEGHILKMFPSTWYV-'
    matrix = np.transpose(AAindex_pca)   # [12,21]
    dim = np.asarray(AAindex_pca).shape[-1]
    encoded = np.empty([len(peptide), dim])  # (1,seq_len,12)
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i, :] = matrix[:, amino.index(query)]    
    encoded = encoded.reshape(1,len(peptide),dim)
    return encoded


def peptide_data_AAindex(peptide, AAindex_pca):
    length = len(peptide)
    if length == 10:
        encode = AAindex_encoder(peptide, AAindex_pca)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = AAindex_encoder(peptide, AAindex_pca)
    return encode


def hla_data_AAindex(hla_dic,hla_type,dic_inventory,AAindex_pca):
    try:
        seq = hla_dic[hla_type]
        # print("HLA seq:", seq, "length", len(seq))
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type,dic_inventory)
        seq = hla_dic[hla_type]
    encode = AAindex_encoder(seq, AAindex_pca)
    return encode


def construct_AAindex_pca(ori, hla_dic,dic_inventory,AAindex_pca):
    
    series = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        hla_type = ori['HLA'].iloc[i]
        immuno = np.array(ori['immunogenicity'].iloc[i]).reshape(1,-1)   # [1,1]

        encode_pep = peptide_data_AAindex(peptide,AAindex_pca)    # [12,960,1]  # new peptide encoding with esm

        encode_hla = hla_data_AAindex(hla_dic,hla_type,dic_inventory,AAindex_pca) # [48,960,1]
        series.append((encode_pep, encode_hla, immuno))

    return series


def pull_peptide_aaindex(dataset):
    dim = np.asarray(dataset[0][0][0]).shape[-1]
    result = np.empty([len(dataset),1,10,dim])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][0]
    return result


def pull_hla_aaindex(dataset):
    dim = np.asarray(dataset[0][0][0]).shape[-1]
    result = np.empty([len(dataset),1,46,dim])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][1]
    return result


# curate AAindex dataset into matrix of 12 features using PCA
def AAindex_pca_matrix(input_path=DEFAULT_INPUT_PATH, output_path=DEFAULT_OUTPUT_PATH):
    
    AAindex = []

    for filepath in glob.glob(os.path.join(input_path, "index*")):
        with open(filepath, "r") as f:
            contents = f.read()
            # discard 13 indicies with missing values 
            if "NA" in contents:
                continue
            AAproperty = list(map(float, contents.split()))
            AAindex.append(AAproperty)
    AAindex = np.array(AAindex) # [553, 20] --> 553 = 566-13

    # add additional average row for placeholder AA
    avg = np.atleast_2d(np.mean(AAindex, axis=1)).T # [553,1]
    AAindex = np.hstack([AAindex, avg]) # [553,21]

    # normalise the 553 columns
    AAindex = RobustScaler().fit_transform(AAindex.T)


    # save AAindex (NOT with pca) encoding matrix to file
    np.savetxt("encoders/AAindex.txt", AAindex)

    # conduct PCA and select first 12 principal components 
    pca = PCA(n_components=12)
    AAindex_pca = pca.fit_transform(AAindex)

    print("explained variance:", pca.explained_variance_ratio_) # 25% of variance explained in first component
    print("AAindex after PCA", np.shape(AAindex_pca)) # [20, 12] --> each row corresponds to AA

    # save AAindex (with pca) encoding matrix to file
    np.savetxt("encoders/AAindex_pca_v2.txt", AAindex_pca)

    return None
# AAindex_pca = np.loadtxt("my_after_pca.txt")


def AAindex_encode_dataset(pca, dataset = 'iedb_data.csv'):
    os.chdir('/home/josh/Dev/Project/benchmark/')

    ori = pd.read_csv("data/" + dataset)

    if dataset == "sars_cov_2_test.csv":
        ori.rename(columns={"immunogenicity-con": "immunogenicity"}, inplace=True) 
        
    frac = 1 # choose dataset size --> 1 = whole dataset
    ori = ori.sample(frac=frac, replace=False).set_index(pd.Index(np.arange(np.ceil(ori.shape[0]*frac)))) # random sample, re-initialising indices  
    
    hla = pd.read_csv('data/hla2paratopeTable_aligned.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    AAindex_pca = np.loadtxt('encoders/AAindex_pca.txt')
    AAindex = np.loadtxt('encoders/AAindex.txt')

    encoding = ["AAindex_pca", "AAindex"]
    if pca: 
        encoder = AAindex_pca 
        i = 0
    else:
        encoder = AAindex
        i = 1

    print("start " + encoding[i] + " encoding")
    dataset = construct_AAindex_pca(ori, hla_dic, dic_inventory, encoder)
    # print(np.asarray(dataset[0][0][0]).shape[-1])
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)
    label = pull_label(dataset)
    print("finish encoding")

    # save dataset to a file or put these functions inside custom dataset 
    x1 = torch.from_numpy(input1).to(torch.float32) 
    x2 = torch.from_numpy(input2).to(torch.float32)
    y = torch.from_numpy(label).to(torch.float32)

    encoded_dataset = {"x1": x1, "x2": x2, "y": y}

    return encoded_dataset


def main():
    print("start of program")
    PRODUCE_MATRICES = False

    os.chdir('/home/josh/Dev/Project/benchmark/')

    # Produce AAindex encoding matrices (with and without pca)
    if PRODUCE_MATRICES:
        AAindex_pca_matrix()

    ori = pd.read_csv('data/iedb_data.csv')
    
    frac = 1 # choose dataset size --> 1 = whole dataset
    ori = ori.sample(frac=frac, replace=False).set_index(pd.Index(np.arange(np.ceil(ori.shape[0]*frac)))) # random sample, re-initialising indices  
    
    hla = pd.read_csv('data/hla2paratopeTable_aligned.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    AAindex_pca = np.loadtxt('encoders/AAindex_pca.txt')
    AAindex = np.loadtxt('encoders/AAindex.txt')

    encoding = ["AAindex_pca", "AAindex"]
    for i, encoder in enumerate([AAindex_pca, AAindex]):
        print("start " + encoding[i] + " encoding")
        dataset = construct_AAindex_pca(ori, hla_dic, dic_inventory, encoder)
        # print(np.asarray(dataset[0][0][0]).shape[-1])
        input1 = pull_peptide_aaindex(dataset)
        input2 = pull_hla_aaindex(dataset)
        label = pull_label(dataset)
        print("finish encoding")

        # save dataset to a file or put these functions inside custom dataset 
        x1 = torch.from_numpy(input1).to(torch.float32) 
        x2 = torch.from_numpy(input2).to(torch.float32)
        y = torch.from_numpy(label).to(torch.float32)

        print("saving to file")
        torch.save({"x1": x1, "x2": x2, "y": y}, "encoders/" + encoding[i] + "_encoded_samples.pt")
        print("finished")

    return None


if __name__ == '__main__':
    main()
