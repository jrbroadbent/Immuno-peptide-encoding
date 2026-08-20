import numpy as np
import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from util import rescue_unknown_hla, pull_label, dict_inventory, hla_df_to_dic
# need this version if running from main
# from .util import rescue_unknown_hla, pull_label, dict_inventory, hla_df_to_dic

from transformers import AutoTokenizer, EsmModel
import torch
import os


# from transformers import EsmModel # from hugging face
# client = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
# client = ESM.from_pretrained("esm2_t6_8M_UR50D")

# def esm2_8M_embedding(peptide, client):
#     lm = client # lm,_ = esm.pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
#     idx = lm.num_layers 
#     embedding = lm(peptide, repr_layers=idx)["representations"][idx]  # get representation from last layer
#     return embedding


# model_name = "facebook/esm2_t6_8M_UR50D"
# client = EsmModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# esm2_8M_embedding 
def esm_embedding(peptide, client, tokenizer):
    inputs = tokenizer(peptide, return_tensors="pt")

    with torch.no_grad():
        outputs = client(**inputs)

    embedding = outputs.last_hidden_state 
    # embedding = embedding.squeeze(0).unsqueeze(-1)
    return embedding  # [1, len(peptide)+2, 320]

# # esmC_300M_embedding
# def esm2_embedding(peptide, client):
#     client = client # ESMC.from_pretrained("esmc_300m").to("cpu") # "cpu" or "cuda"
#     protein = ESMProtein(sequence=peptide)
#     protein_tensor = client.encode(protein) # replaces AA with int? and adds start and end term 
#     logits_output = client.logits(
#     protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
#     )
#     embedding = logits_output.embeddings.squeeze(0).unsqueeze(-1)
#     embedding = embedding.numpy() # convert to numpy array
#     return embedding  # [len(peptide)+2, 960, 1]


# process 9- and 10-mers before esm encoding
def peptide_data_esm(peptide, client, tokenizer=None):   # return numpy array [10,12,1]
    length = len(peptide)
    if length == 10:
        encode = esm_embedding(peptide, client, tokenizer)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = esm_embedding(peptide, client, tokenizer)
    return encode


def hla_data_esm(hla_dic, hla_type, dic_inventory, client, tokenizer=None):    # return numpy array [36,960,1]
    try:
        seq = hla_dic[hla_type]
        # print("HLA seq:", seq, "length", len(seq))
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type,dic_inventory)
        seq = hla_dic[hla_type]
    encode = esm_embedding(seq, client, tokenizer)
    return encode


def construct_esm_embedding(ori, hla_dic, dic_inventory, client, tokenizer=None):
    series = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        hla_type = ori['HLA'].iloc[i]
        immuno = np.array(ori['immunogenicity'].iloc[i]).reshape(1,-1)   # [1,1]

        encode_pep = peptide_data_esm(peptide, client, tokenizer)    # [12,960,1]  # new peptide encoding with esm

        encode_hla = hla_data_esm(hla_dic,hla_type,dic_inventory,client, tokenizer) # [48,960,1]
        series.append((encode_pep, encode_hla, immuno))
    return series


def pull_peptide_esm(dataset):
    dim = np.asarray(dataset[0][0][0]).shape[-1]
    result = np.empty([len(dataset),1,12,dim])   # [len, 12,960,1]
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][0]
    return result


def pull_hla_esm(dataset):
    dim = np.asarray(dataset[0][0][0]).shape[-1]
    result = np.empty([len(dataset),1,48,dim])  # [len, 48,960,1]
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][1]
    return result


def esm_encode_dataset(dataset = 'iedb_data.csv'):
    os.chdir('/home/josh/Dev/Project/')

    ori = pd.read_csv("data/" + dataset) 
    frac = 1 # choose dataset size --> 1 = whole dataset
    ori = ori.sample(frac=frac, replace=False).set_index(pd.Index(np.arange(np.ceil(ori.shape[0]*frac)))) # random sample, re-initialising indices  

    hla = pd.read_csv('./data/hla2paratopeTable_aligned.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    hla_type = ori['HLA'].iloc[i]
    try:
        seq = hla_dic[hla_type]
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type,dic_inventory)
        seq = hla_dic[hla_type]

    print("HLA seq:", seq, "length", len(seq))


def main():
    print("start of program")

    os.chdir('/home/josh/Dev/Project/')

    dataset = 'iedb_data.csv'
    ori = pd.read_csv("data/" + dataset) 
    frac = 1 # choose dataset size --> 1 = whole dataset
    ori = ori.sample(frac=frac, replace=False).set_index(pd.Index(np.arange(np.ceil(ori.shape[0]*frac)))) # random sample, re-initialising indices  

    hla = pd.read_csv('./data/hla2paratopeTable_aligned.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    for i in range(ori.shape[0]):
        hla_type = ori['HLA'].iloc[i]
        try:
            seq = hla_dic[hla_type]
        except KeyError:
            hla_type = rescue_unknown_hla(hla_type,dic_inventory)
            seq = hla_dic[hla_type]

    print("HLA seq:", seq, "length", len(seq))

    return None


if __name__ == '__main__':
    main()
