import pickle 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import datetime

# Add significance brackets and asterisks
def add_sig_bracket(ax, x1, x2, y, p_value=0.04):
    """Add significance bracket between two points"""
    # Determine significance level
    if p_value < 0.001:
        sig_symbol = '***'
    elif p_value < 0.01:
        sig_symbol = '**'
    elif p_value < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    
    # Draw horizontal line
    ax.plot([x1, x2], [y, y], color='gray', linewidth=1)
    # Draw vertical lines
    ax.plot([x1, x1], [y-0.5, y], color='gray', linewidth=1)
    ax.plot([x2, x2], [y-0.5, y], color='gray', linewidth=1)
    # Add significance symbol
    ax.text((x1 + x2) / 2, y, sig_symbol, ha='center', va='bottom', fontweight='bold')


def main():
    os.chdir('/home/josh/Dev/Project/')
    d = datetime.datetime.now()
    id = str(d.month)+'_'+str(d.day)+'_'+str(d.hour)
    
    with open("data/test_BCE_scores_8_9_20.pkl", "rb") as f:   #"data/BCE_scores_8_7_12.pkl"   "data/final_bootstrap_log_scores2.pkl"
        results = pickle.load(f)
    # 3 test datasets 
    # 4 models
    # 10 repeat per model and test dataset 
    # 120 total 
    # print(len(results.keys()))

    dataset_names = ["sars_cov_2", "dengue", "neoantigen"]
    model_names = ["ESM", "AAindex_pca", "AAindex", "OHE"]
    model_labels = ["ESM", "AAindex + pca", "AAindex", "OHE"]
    colours = ["magenta", "purple", "blue", "red"]
    labels = ["single", "repeated"]

    n = int( len(results.keys())/ (len(dataset_names)*len(model_names) ) )
    keys_split = []
    for dataset in dataset_names:
        dataset_keys = []
        for model in model_names:
            for i in range(n):
                key = (i, model, dataset) 
                dataset_keys.append(key)
        keys_split.append(dataset_keys)


    # print mean BCE score for bootstrap (using 1st of 10 repeats)
    for dataset in dataset_names:
        print("\n", dataset, "\n--------------------------------")
        print(np.mean(results[(0, "ESM", dataset)]))
        print(np.mean(results[(0, "AAindex_pca", dataset)]))
        print(np.mean(results[(0, "AAindex", dataset)]))
        print(np.mean(results[(0, "OHE", dataset)]))



    fig, ax = plt.subplots(2,2, figsize=(10, 10))


    plot_labels = ['A', 'B', 'C']
    for row in range(2):
        for col in range(2):
            if ((not row == 1) or (not col == 1)):
                ax[row, col].annotate(f'{plot_labels[row*2+col]}', (0.1, 0.93),
                                    xycoords = 'axes fraction',
                                    fontsize=14,
                                    fontweight='bold',
                                    color='black')


    for j, dataset in enumerate(dataset_names):

        if j == 0:
            axis = ax[0][0]
        elif j == 1:
            axis = ax[0][1]
        elif j == 2:
            axis = ax[1][0]
        
        all_scores = []
        for i, key in enumerate(keys_split[j]):
                log_scores = results[key]
                mean = np.mean(log_scores)
                CI = np.percentile(log_scores, [2.5,97.5])
                upper_bound = CI[1]
                lower_bound = CI[0]

                if key[0] == 0:
                    axis.errorbar(i, mean, yerr=[[mean-lower_bound],[upper_bound-mean]], elinewidth=1, marker="o", ms=3, lw=0, capsize=3, c=colours[int(i)], label=labels[0])

                all_scores.append(log_scores)

                # plot average for last repeat of each model (10 repeats total)
                # if key[0] == 9:
                #     # calculate mean and CI
                #     mean = np.mean(all_scores)
                #     CI = np.percentile(all_scores, [2.5,97.5])
                #     upper_bound = CI[1]
                #     lower_bound = CI[0]

                #     # plot 
                #     plt.errorbar( ( (i-9)/10 + 0.5 ), mean, yerr=[[mean-lower_bound],[upper_bound-mean]], elinewidth=1, marker="x", ms=5, lw=0, capsize=3, c=colours[int((i-9)/10)], label=labels[1])

                #     # clear all_scores for next model
                #     all_scores = []

                #     print(key, np.round(CI, decimals=3))


        # plt.xticks([0.25, 1.25, 2.25, 3.25], model_labels) #0, 0.5   1.0, 1.5   2, 2.5,  3, 3.5
        axis.set_xticks([0, 1, 2, 3], model_labels) #0, 0.5   1.0, 1.5   2, 2.5,  3, 3.5

        # Add significance comparisons (example p-values)
        #add_sig_bracket(axis, 0.5, 1.5, 7)
        #add_sig_bracket(axis, 1.5, 2.5, 9)

        # hide top and right axis spines
        # axis.spines['top'].set_visible(False)
        # axis.spines['right'].set_visible(False)


        axis.set_ylabel("Mean BCE")

        # add legend for single and repeated 
        markers = [Line2D([0], [0], marker= "o", color='w', markerfacecolor='k', markersize=7),
                Line2D([0], [0], marker= "X", color='w', markerfacecolor='k', markersize=7)]

        # if j ==0:
        #     axis.legend(markers, labels)


    ax[1,1].set_visible(False)


    fig.subplots_adjust(wspace=0.2, hspace=0.15, bottom=0.05, left=0.05, top=0.99, right=0.99)
    # fig.tight_layout()

    fig.savefig("figures/test_BCE_scores_"+id+".png") # "figures/bootstrap_fig_sig_brackets.png"



if __name__ == '__main__':
    main()
