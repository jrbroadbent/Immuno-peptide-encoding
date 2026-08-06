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
    
    with open("data/BCE_scores_8_4_11.pkl", "rb") as f: #"data/final_bootstrap_log_scores2.pkl"
        results = pickle.load(f)

    # print(results.keys())
    print(np.mean(results[(0, "ESM")]))
    print(np.mean(results[(0, "AAindex_pca")]))
    print(np.mean(results[(0, "AAindex")]))
    print(np.mean(results[(0, "OHE")]))


    # retrain = ["no_retrain", "retrain"]
    model_names = ["ESM", "AAindex_pca", "AAindex", "OHE"]
    # labels = [x+" "+y for (x,y) in list(results.keys())]
    colours = ["pink", "purple", "blue", "red"]

    fig, ax = plt.subplots(1,2, figsize=(10,5))

    labels = ["single", "repeated"]

    all_scores = []
    for i, key in enumerate(results.keys()):
            log_scores = results[key]
            mean = np.mean(log_scores)
            CI = np.percentile(log_scores, [2.5,97.5])
            upper_bound = CI[1]
            lower_bound = CI[0]

            if key[0] == 0:
                ax[0].errorbar(i/10, mean, yerr=[[mean-lower_bound],[upper_bound-mean]], elinewidth=1, marker="o", ms=3, lw=0, capsize=3, c=colours[int(i/10)], label=labels[0])
                ax[1].errorbar(i/10, mean, yerr=[[mean-lower_bound],[upper_bound-mean]], elinewidth=1, marker="o", ms=3, lw=0, capsize=3, c=colours[int(i/10)], label=labels[0])

            all_scores.append(log_scores)

            # plot average for last repeat of each model (10 repeats total)
            if key[0] == 9:
                # calculate mean and CI
                mean = np.mean(all_scores)
                CI = np.percentile(all_scores, [2.5,97.5])
                upper_bound = CI[1]
                lower_bound = CI[0]

                # plot 
                ax[0].errorbar( ( (i-9)/10 + 0.5 ), mean, yerr=[[mean-lower_bound],[upper_bound-mean]], elinewidth=1, marker="x", ms=5, lw=0, capsize=3, c=colours[int((i-9)/10)], label=labels[1])
                ax[1].errorbar( ( (i-9)/10 + 0.5 ), mean, yerr=[[mean-lower_bound],[upper_bound-mean]], elinewidth=1, marker="x", ms=5, lw=0, capsize=3, c=colours[int((i-9)/10)], label=labels[1])

                # clear all_scores for next model
                all_scores = []

                print(key, np.round(CI, decimals=3))


    for axis in ax:
        #axis.set_xticks(range(len(model_names)), model_names) #, rotation=20)
        axis.set_xticks([0.25, 1.25, 2.25, 3.25], model_names) #0, 0.5   1.0, 1.5   2, 2.5,  3, 3.5
        # axis.set_xticks(range(len(labels)), labels, rotation=20)

        # Add significance comparisons (example p-values)
        #add_sig_bracket(axis, 0.5, 1.5, 7)
        #add_sig_bracket(axis, 1.5, 2.5, 9)

        # hide top and right axis spines
        # axis.spines['top'].set_visible(False)
        # axis.spines['right'].set_visible(False)


    ax[0].set_ylabel("Mean BCE")
    ax[1].set_ylabel('log$_{10}$ Mean BCE')
    ax[1].set_yscale("log", base=10)
    #ax[1].set_yscale("log", base=2)


    # add legend for single and repeated 
    markers = [Line2D([0], [0], marker= "o", color='w', markerfacecolor='k', markersize=7),
               Line2D([0], [0], marker= "X", color='w', markerfacecolor='k', markersize=7)]
    ax[0].legend(markers, labels)

    plt.subplots_adjust(wspace=0.3, bottom=0.15)

    fig.tight_layout()
    fig.savefig("figures/IEDB_BCE_scores_"+id+".png") # "figures/bootstrap_fig_sig_brackets.png"



if __name__ == '__main__':
    main()
