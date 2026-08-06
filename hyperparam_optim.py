import os
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import optuna
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from models.model_defs import OHE_seperateCNN, AAindex_seperateCNN, ESM_seperateCNN


# __all__ = [
#
# ]

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class ImmunoDataset(Dataset):
    def __init__(self, dataset):
        self.x1 = dataset["x1"]
        self.x2 = dataset["x2"]
        self.labels = dataset["y"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x1 = self.x1[idx,:,:,:]
        x2 = self.x2[idx,:,:,:]
        x = [x1,x2]
        y = self.labels[idx,:]
        return x, y

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


def objective(trial, model_fn, train_data): # could just pass it my train data 
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    dropout = trial.suggest_float('p', 0.1, 0.5)
    patience = trial.suggest_int('patience', 0, 20)

    model = model_fn(dropout=dropout)    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    # produce splits 
    kf = KFold(n_splits=5)

    # cross validation 
    for train_set, val_set in kf.split(train_data):
        data = Subset(train_data, train_set)
        train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        # train
        model.train()
        best_loss = float('inf')
        early_stopping = EarlyStopping(patience=patience, delta=0, verbose=True)
        for epoch in range(200): # number of epochs
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Check early stopping condition
            early_stopping.check_early_stop(loss)
            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # validate
        data = Subset(train_data, val_set)
        val_loader = DataLoader(data, batch_size=1, shuffle=True)
        num_batches = len(val_loader)
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
            val_loss /= num_batches
        
    # return mean validation loss
    return val_loss



def main():
    os.chdir('/home/josh/Dev/Project/')

    # Datasets
    ESM_dataset = torch.load('data/ESM_encoded_samples.pt')
    AAindex_dataset = torch.load('data/AAindex_encoded_samples.pt')
    OHE_dataset = torch.load('data/OHE_encoded_samples.pt')
    datasets = [ESM_dataset, AAindex_dataset, OHE_dataset]

    # Models 
    model_fns = [ESM_seperateCNN, AAindex_seperateCNN, OHE_seperateCNN]
    model_names = ["ESM", "AAindex", "OHE"]

    for model_name, model_fn, dataset in zip(model_names, model_fns, datasets):
        dataset = ImmunoDataset(dataset)
        train_data, test_data = random_split(dataset, [0.8,0.2])
        
        study = optuna.create_study(study_name="hyperparam_optim",direction='minimize')
        study.optimize(lambda trial: objective(trial, model_fn, train_data), n_trials=2)  # number of trials: 20
        print(f"{model_name} Best Hyperparameters: {study.best_params}")



if __name__ == '__main__':
    main()
