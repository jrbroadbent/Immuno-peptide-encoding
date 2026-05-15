'''
Implementation of AAindex-PCA encoding strategy for peptide sequences
'''

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import numpy as np
import os
import glob


# curate AAindex dataset 
AAindex = []

directory = "/home/josh/Dev/Project/Models/DeepImmuno/data/AAindex1"


for filepath in glob.glob(os.path.join(directory, "index*")):
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

# conduct PCA and select first 12 principal components 
pca = PCA(n_components=12)
AAindex = pca.fit_transform(AAindex)

print("explained variance:", pca.explained_variance_ratio_) # 25% of variance explained in first component
print("AAindex after PCA", np.shape(AAindex)) # [20, 12] --> each row corresponds to AA
# print(AAindex[1])

# save to file
os.chdir("/home/josh/Dev/Project/Models/DeepImmuno/my_files")
np.savetxt("my_after_pca.txt", AAindex)



# use my PCA to train DeepImmuno-CNN
# hopefully get same results (try using last 12 components to see if this produces bad results)


after_pca = np.loadtxt("my_after_pca.txt")
# print(np.shape(after_pca))

def aaindex(peptide,after_pca):

    amino = 'ARNDCQEGHILKMFPSTWYV-'
    matrix = np.transpose(after_pca)   # [12,21]
    encoded = np.empty([len(peptide), 12])  # (seq_len,12)
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i, :] = matrix[:, amino.index(query)]

    return encoded

# encoding = aaindex(['A', 'X', 'e'], after_pca)
# print("encoding:", encoding)

