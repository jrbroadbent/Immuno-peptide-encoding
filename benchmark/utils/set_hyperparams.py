import pickle

PARAMS_FILE = 'model_params_8_7_12.pkl'

with open("data/"+PARAMS_FILE, "rb") as f:
                params = pickle.load(f)

model_names = ["ESM", "AAindex_pca", "AAindex", "OHE"]

for i, model in enumerate(model_names):
    print(f"{model}:\t", params[i])


new_params = [{'lr': 0.00010000000000000009, 'batch_size': 256, 'p': 0.5, 'patience': 20}, {'lr': 0.00046504853872833074, 'batch_size': 156, 'p': 0.1, 'patience': 20}, {'lr': 0.0005341749662233163, 'batch_size': 32, 'p': 0.1, 'patience': 20}, {'lr': 0.0001895617429715258, 'batch_size': 126, 'p': 0.34806345744941536, 'patience': 17}]

with open("data/"+PARAMS_FILE, "wb") as f:
            pickle.dump(new_params, f)


# Hyperparameters selected with Optuna: 
# 
# ESM:             {'lr': 0.0001, 'batch_size': 256, 'p': 0.5, 'patience': 20}
# AAindex_pca:     {'lr': 0.00047, 'batch_size': 156, 'p': 0.1, 'patience': 20}
# AAindex:         {'lr': 0.00053, 'batch_size': 32, 'p': 0.1, 'patience': 20}
# OHE:             {'lr': 0.00019, 'batch_size': 126, 'p': 0.35, 'patience': 17}
#
# patience appears the same (all as high as possible) - 20 was upper limit
# learning rates reasonably similar
# batch size and p very different 
# 
# a lot of the values chosen seem to be the edge value - interesting ; maybe I didn't set very good search ranges

#### would all models perform better with hyperparameters that OHE model had? ####

# Hyperparameters used from DeepImmuno:
# lr = 0.0001
# batch size = 128
# p = 0.2
# patience = 2 
# 
# patience is most obvious big difference - maybe the models overfit 


# create file for original hyperparameters 
NEW_PARAMS = 'original_params.pkl'

original_params = [{'lr': 0.0001, 'batch_size':128, 'p':0.2, 'patience':2}]*4
print(original_params)

with open("data/"+NEW_PARAMS, "wb") as f:
            pickle.dump(original_params, f)
