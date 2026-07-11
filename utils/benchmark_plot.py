import pickle 
import matplotlib.pyplot as plt
import numpy as np


with open("final_bootstrap_log_scores2.pkl", "rb") as f:
    results = pickle.load(f)

# retrain = ["no_retrain", "retrain"]
model_names = ["ESM", "AAindex", "OHE"]
# labels = [x+" "+y for (x,y) in list(results.keys())]
colours = ["purple", "purple", "blue", "blue", "red", "red"]

fig, ax = plt.subplots(1,2, figsize=(10,5))


for i, key in enumerate(results.keys()):
        log_scores = results[key]
        mean = np.mean(log_scores)
        CI = np.percentile(log_scores, [2.5,97.5])
        upper_bound = CI[1]
        lower_bound = CI[0]

        if key[0] == "no_retrain":
            ax[0].errorbar(i/2, mean, yerr=[[mean-lower_bound],[upper_bound-mean]], elinewidth=1, marker="o", ms=3, lw=0, capsize=3, c=colours[i])
            ax[1].errorbar(i/2, mean, yerr=[[mean-lower_bound],[upper_bound-mean]], elinewidth=1, marker="o", ms=3, lw=0, capsize=3, c=colours[i])

            print(key[1], np.round(CI, decimals=3))

for axis in ax:
    # axis.set_xticks(range(len(labels)), labels, rotation=20)
    axis.set_xticks(range(len(model_names)), model_names, rotation=20)
    axis.set_ylabel("Mean Log Score")

# ax[1].set_yscale("log", base=10)
ax[1].set_yscale("log", base=2)

plt.subplots_adjust(wspace=0.3, bottom=0.15)

fig.savefig("bootstrap_fig.png")
