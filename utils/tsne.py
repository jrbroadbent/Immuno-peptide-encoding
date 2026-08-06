'''
plotting functions for t-SNE visualisations
'''

import os
import torch 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main():
    print("\nstart of program\n")
    os.chdir('/home/josh/Dev/Project/')

    # Load Data 
    # ESM_dataset = torch.load('data/ESM_encoded_samples.pt')
    # AAindex_dataset = torch.load('encoders/AAindex_encoded_samples.pt')   # move to data dir? 
    AAindex_pca_dataset = torch.load('data/AAindex_encoded_samples.pt')   # renamve? data/AAindex_pca_encoded_samples.pt
    # OHE_dataset = torch.load('data/OHE_encoded_samples.pt')
    # datasets = [ESM_dataset, AAindex_pca_dataset, AAindex_dataset, OHE_dataset]

    y = AAindex_pca_dataset["y"].squeeze()
    X = AAindex_pca_dataset["x1"].squeeze()
    X = X.reshape(len(X), -1) # what is this process called? 
    print(X.shape, y.shape)

    X_new = TSNE(n_components=2, perplexity=30, random_state=123).fit_transform(X) 
    # perplexity = number of nearest neighbours --> higher number results in more clustered visualisation
    print(X_new.shape)

    fig, ax = plt.subplots()
    imm_mask = [i==1 for i in y]
    nonImm_mask = [i==0 for i in y]
    X_imm = X_new[imm_mask]
    X_nonImm = X_new[nonImm_mask]
    ax.plot(X_imm[:,0], X_imm[:,1], ".", color="blue") # colour by immunogenicity \\ plot X separately coloured by factors of interest 
    ax.plot(X_nonImm[:,0], X_nonImm[:,1], ".", color="red")

    # axis labels 

    # legend 

    fig.savefig("test.png")






if __name__ == '__main__':
    main()
