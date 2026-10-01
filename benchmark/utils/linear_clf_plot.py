'''
Plotting script for IEDB test set BCE loss scores 
'''
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

    # optuna hyperparameter results: "data/BCE_scores_8_7_12.pkl"
    # original hyperparameter results: "data/BCE_scores_8_9_20.pkl"
    with open("data/BCE_scores_8_21_11.pkl", "rb") as f:
        results = pickle.load(f)

    dataset_names = ["ESM", "AAindex + PCA", "AAindex", "OHE"]

    for dataset in dataset_names:
        print(np.mean(results[(0, dataset)]))

    model_names = []
    model_labels = []
    colours = ["magenta", "purple", "blue", "red"]

    fig, ax = plt.subplots(1,1, figsize=(5,5))

    labels = ["single", "repeated"]

    single_scores = []
    repeat_scores = []
    all_scores = []
    for i, key in enumerate(results.keys()):
            log_scores = results[key]
            mean = np.mean(log_scores)
            CI = np.percentile(log_scores, [2.5,97.5])
            upper_bound = CI[1]
            lower_bound = CI[0]

            if key[0] == 0:
                print(f"{i}  {i/10}  \t", end=" ")
            elif key[0] == 9:
                print(f"{i}  {(i-9)/10 + 0.5}  \t", end=" ")
            else:
                print("\t\t", end=" ")
            print(key, "\t", np.round(mean, decimals=3), "\t", np.round(CI, decimals=3))

            if key[0] == 0:
                single_scores.append(log_scores)   
                ax.errorbar(i/10, mean, yerr=[[mean-lower_bound],[upper_bound-mean]], elinewidth=1, marker="o", ms=3, lw=0, capsize=3, c=colours[i//10], label=labels[0])

            all_scores.append(log_scores)

            # plot average for last repeat of each model (10 repeats total)
            if key[0] == 9:
                # print(len(all_scores), len(all_scores[0]))
                # calculate mean and CI
                mean = np.mean(all_scores)
                CI = np.percentile(all_scores, [2.5,97.5])
                upper_bound = CI[1]
                lower_bound = CI[0]

                # plot 
                ax.errorbar( ( (i-9)/10 + 0.5 ), mean, yerr=[[mean-lower_bound],[upper_bound-mean]], elinewidth=1, marker="x", ms=5, lw=0, capsize=3, c=colours[int((i-9)/10)], label=labels[1])

                repeat_scores.append(np.array(all_scores).reshape(-1))

                # clear all_scores for next model
                all_scores = []
            
                print(key, np.round(CI, decimals=3))


    ax.set_xticks([0.25, 1.25, 2.25, 3.25], dataset_names)

    ax.set_ylabel("Mean BCE")

    # add legend for single and repeated 
    markers = [Line2D([0], [0], marker= "o", color='w', markerfacecolor='k', markersize=7),
               Line2D([0], [0], marker= "X", color='w', markerfacecolor='k', markersize=7)]
    ax.legend(markers, labels)

    plt.subplots_adjust(wspace=0.3, bottom=0.15)

    fig.tight_layout()
    FILE = "figures/linear_clf_"+id+".png"
    fig.savefig(FILE)

    print(f"\nfigure saved to: {FILE}\n")



if __name__ == '__main__':
    main()
