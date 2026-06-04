# moved imports to inside function definition to avoid segmentation fault when loading pytorch/transformers and tensorflow at the same time

import torch
from torch import nn

# PyTorch implementation
class ESM_seperateCNN(nn.Module):
    def __init__(self):
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
            nn.Conv2d(1, 16, kernel_size=(15,320)),  # [32,1,16]
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
            nn.Dropout(p=0.2),
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


class AAindex_seperateCNN(nn.Module):
    def __init__(self):
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
            nn.Dropout(p=0.2),
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


class OHE_seperateCNN(nn.Module):
    def __init__(self):
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
            nn.Dropout(p=0.2),
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



## original model (tensorflow implementation)
# def seperateCNN():
#     import tensorflow.keras as keras
#     from tensorflow.keras import layers
    
#     input1 = keras.Input(shape=(12, 320, 1)) # squeeze(0).unsqueeze(-1)
#     input2 = keras.Input(shape=(48, 320, 1))


#     # x = layers.Dense()(input1) # projection layer
#     x = layers.Conv2D(filters=16, kernel_size=(4, 320))(input1)  # [9, 1, 16]
#     x = layers.BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)    # 8
#     x = layers.BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)  # 4
#     x = layers.Flatten()(x)
#     x = keras.Model(inputs=input1, outputs=x)

#     # y = layers.Dense()(input2) # projection layer
#     # y = layers.Conv2D(filters=16, kernel_size=(17, 960))(input2)     # [32, 1, 16]
#     y = layers.Conv2D(filters=16, kernel_size=(15, 320))(input2)     # [32, 1, 16]
#     y = layers.BatchNormalization()(y)
#     y = keras.activations.relu(y)
#     y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)  # 16
#     y = layers.Conv2D(filters=32,kernel_size=(9,1))(y)    # 8
#     y = layers.BatchNormalization()(y)
#     y = keras.activations.relu(y)
#     y = layers.MaxPool2D(pool_size=(2, 1),strides=(2,1))(y)  # 4
#     y = layers.Flatten()(y)
#     y = keras.Model(inputs=input2,outputs=y)

#     combined = layers.concatenate([x.output,y.output])
#     z = layers.Dense(128,activation='relu')(combined)
#     z = layers.Dropout(0.2)(z)
#     z = layers.Dense(1,activation='sigmoid')(z)

#     model = keras.Model(inputs=[input1,input2],outputs=z)
#     return model


# def seperateCNN():
#     input1 = keras.Input(shape=(12, 960, 1)) # squeeze(0).unsqueeze(-1)
#     input2 = keras.Input(shape=(48, 960, 1))


#     # x = layers.Dense()(input1) # projection layer
#     x = layers.Conv2D(filters=16, kernel_size=(4, 960))(input1)  # [9, 1, 16]
#     x = layers.BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)    # 8
#     x = layers.BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)  # 4
#     x = layers.Flatten()(x)
#     x = keras.Model(inputs=input1, outputs=x)

#     # y = layers.Dense()(input2) # projection layer
#     # y = layers.Conv2D(filters=16, kernel_size=(17, 960))(input2)     # [32, 1, 16]
#     y = layers.Conv2D(filters=16, kernel_size=(15, 12))(input2)     # [32, 1, 16]
#     y = layers.BatchNormalization()(y)
#     y = keras.activations.relu(y)
#     y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)  # 16
#     y = layers.Conv2D(filters=32,kernel_size=(9,1))(y)    # 8
#     y = layers.BatchNormalization()(y)
#     y = keras.activations.relu(y)
#     y = layers.MaxPool2D(pool_size=(2, 1),strides=(2,1))(y)  # 4
#     y = layers.Flatten()(y)
#     y = keras.Model(inputs=input2,outputs=y)

#     combined = layers.concatenate([x.output,y.output])
#     z = layers.Dense(128,activation='relu')(combined)
#     z = layers.Dropout(0.2)(z)
#     z = layers.Dense(1,activation='sigmoid')(z)

#     model = keras.Model(inputs=[input1,input2],outputs=z)
#     return model