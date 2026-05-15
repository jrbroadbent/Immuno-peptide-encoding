'''
Encoder using ESM Cambria (ESM C) protein seqeunce embeddings
'''

import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig



####################################################
####                 GENERAL                    ####
####################################################

# used across encoding strategies

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


def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic


####################################################
####               ESM SPECIFIC                 ####
####################################################

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