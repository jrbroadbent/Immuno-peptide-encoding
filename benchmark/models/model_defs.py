# moved imports to inside function definition to avoid segmentation fault when loading pytorch/transformers and tensorflow at the same time

import torch
from torch import nn

# PyTorch implementation
class ESM_separateCNN(nn.Module):
    def __init__(self, dropout = 0.2):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(4,320)),  # [9,1,16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2,1)),   # [8,1,32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # [4,1,32]
            nn.Flatten()  # [128]
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(17,320)),  # [32,1,16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # [16,1,16]
            nn.Conv2d(16, 32, kernel_size=(9,1)),    # [8,1,32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)), # [4,1,32]
            nn.Flatten()  # [128]
        )
        self.combined = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x[0] # [1, 12, 320]
        x2 = x[1] # [1, 48, 320]
        output1 = self.cnn1(x1)
        output2 = self.cnn2(x2)
        combined = torch.cat((output1, output2), 1)
        logits = self.combined(combined)
        return logits 


class AAindex_separateCNN(nn.Module):
    def __init__(self, dropout = 0.2):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(2,553)),  # [9,1,16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2,1)),   # [8,1,32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # [4,1,32]
            nn.Flatten()  # [128]
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(15,553)),  # [32,1,16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # [16,1,16]
            nn.Conv2d(16, 32, kernel_size=(9,1)),    # [8,1,32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)), # [4,1,32]
            nn.Flatten()  # [128]
        )
        self.combined = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x[0] # --> does this need to be [1, 10, 12]  ?
        x2 = x[1] # --> does this need to be [1, 46, 12]  ?
        output1 = self.cnn1(x1)
        output2 = self.cnn2(x2)
        combined = torch.cat((output1, output2), 1)
        logits = self.combined(combined)
        return logits 


class AAindex_pca_separateCNN(nn.Module):
    def __init__(self, dropout = 0.2):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(2,12)),  # [9,1,16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2,1)),   # [8,1,32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # [4,1,32]
            nn.Flatten()  # [128]
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(15,12)),  # [32,1,16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # [16,1,16]
            nn.Conv2d(16, 32, kernel_size=(9,1)),    # [8,1,32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)), # [4,1,32]
            nn.Flatten()  # [128]
        )
        self.combined = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x[0] # --> does this need to be [1, 10, 12]  ?
        x2 = x[1] # --> does this need to be [1, 46, 12]  ?
        output1 = self.cnn1(x1)
        output2 = self.cnn2(x2)
        combined = torch.cat((output1, output2), 1)
        logits = self.combined(combined)
        return logits 
    

class OHE_separateCNN(nn.Module):
    def __init__(self, dropout = 0.2):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(2,21)),  # [9,1,16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2,1)),   # [8,1,32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # [4,1,32]
            nn.Flatten()  # [128]
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(15,21)),  # [32,1,16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # [16,1,16]
            nn.Conv2d(16, 32, kernel_size=(9,1)),    # [8,1,32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)), # [4,1,32]
            nn.Flatten()  # [128]
        )
        self.combined = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x[0]  # [1, 10, 21]
        x2 = x[1]  # [1, 46, 21]
        output1 = self.cnn1(x1)
        output2 = self.cnn2(x2)
        combined = torch.cat((output1, output2), 1)
        logits = self.combined(combined)
        return logits 


class Linear_Classifier(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(size,1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
            # x[0] has shape [batch_size,1,L,D] --> squeeze dim 1 to get [batch_size,L,D] --> mean dim 1 to get [batch_size,D]
            x1 = torch.squeeze(x[0], 1) 
            x2 = torch.squeeze(x[1], 1)

            # Take mean over sequence positions
            x1_mean = torch.mean(x1, dim=1) 
            x2_mean = torch.mean(x2, dim=1)
            # print("x1_mean size:",x1_mean.size()) # batch_size, 320
            # print("x2_mean size:", x2_mean.size()) # batch_size, 320
            
            combined = torch.cat((x1_mean, x2_mean), dim=1)
            # print("combined size:", combined.size()) # batch_size, 640

            prob = self.linear(combined)

            return prob

