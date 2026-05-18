# moved imports to inside function definition to avoid segmentation fault when loading pytorch/transformers and tensorflow at the same time
# import tensorflow.keras as keras
# from tensorflow.keras import layers


def seperateCNN():
    import tensorflow.keras as keras
    from tensorflow.keras import layers
    
    input1 = keras.Input(shape=(12, 320, 1)) # squeeze(0).unsqueeze(-1)
    input2 = keras.Input(shape=(48, 320, 1))


    # x = layers.Dense()(input1) # projection layer
    x = layers.Conv2D(filters=16, kernel_size=(4, 320))(input1)  # [9, 1, 16]
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)    # 8
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)  # 4
    x = layers.Flatten()(x)
    x = keras.Model(inputs=input1, outputs=x)

    # y = layers.Dense()(input2) # projection layer
    # y = layers.Conv2D(filters=16, kernel_size=(17, 960))(input2)     # [32, 1, 16]
    y = layers.Conv2D(filters=16, kernel_size=(15, 320))(input2)     # [32, 1, 16]
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)  # 16
    y = layers.Conv2D(filters=32,kernel_size=(9,1))(y)    # 8
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1),strides=(2,1))(y)  # 4
    y = layers.Flatten()(y)
    y = keras.Model(inputs=input2,outputs=y)

    combined = layers.concatenate([x.output,y.output])
    z = layers.Dense(128,activation='relu')(combined)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1,activation='sigmoid')(z)

    model = keras.Model(inputs=[input1,input2],outputs=z)
    return model

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