# from encoders.AAindex_pca import * 
# from encoders.ESM import * 
from encoders.OHE import *   # check OHE.py for .utils not utils import 

print("ok")

# sars_encoded_esm = esm_encode_dataset(dataset = 'sars_cov_2_test.csv')
# dengue_encoded_esm = esm_encode_dataset(dataset = 'dengue_test.csv')
# neoantigen_encoded_esm = esm_encode_dataset(dataset = 'neoantigen_test.csv')
#
# datasets = [sars_encoded_esm, dengue_encoded_esm, neoantigen_encoded_esm]

sars_encoded_ohe = ohe_encode_dataset(dataset = 'sars_cov_2_test.csv')
dengue_encoded_ohe = ohe_encode_dataset(dataset = 'dengue_test.csv')
neoantigen_encoded_ohe = ohe_encode_dataset(dataset = 'neoantigen_test.csv')

datasets = [sars_encoded_ohe, dengue_encoded_ohe, neoantigen_encoded_ohe]

# sars_encoded_AAindex_pca, sars_encoded_AAindex = AAindex_encode_dataset(dataset = 'sars_cov_2_test.csv')
# dengue_encoded_AAindex_pca, dengue_encoded_AAindex = AAindex_encode_dataset(dataset = 'dengue_test.csv')
# neoantigen_encoded_AAindex_pca, neoantigen_encoded_AAindex = AAindex_encode_dataset(dataset = 'neoantigen_test.csv')

# datasets = [sars_encoded_AAindex_pca, sars_encoded_AAindex, 
#             dengue_encoded_AAindex_pca, dengue_encoded_AAindex,
#             neoantigen_encoded_AAindex_pca, neoantigen_encoded_AAindex ]

for dataset in datasets:
    print("class ratio:")
    pos = sum(dataset["y"] == 1)
    tot = len(dataset["y"])
    print("# positives:\t", pos)
    print("ratio +ves:\t", pos/tot)
    print(tot)


#### RESULTS ####
# SARS-CoV-2:  n = 92, +ve = 25
# dengue virus:  n = 408, +ve = 408
# neoantigen:  n = 552, +ve = 35
##################


# print("finished")
    