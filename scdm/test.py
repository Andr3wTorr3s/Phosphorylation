import numpy as np

import tensorflow as tf

from sklearn.model_selection import KFold

from tensorflow.keras import layers, models

import os

import sys






batch_size = 34000

scdm_train=np.load("s_scdm_training.npy")# list of 19 by 19
#add by the transpose
scdmt=np.zeros_like(scdm_train)

for i in range(len(scdm_train)):
    scdmt[i]=scdm_train[i]+np.transpose(scdm_train[i])

scdm_train=scdmt


y_train=np.load("s_y_training.npy")

#randomly permute data set
perm =np.random.permutation(scdm_train.shape[0])
scdm_train=scdm_train[perm]
y_train=y_train[perm]


#define splits
kf=KFold(n_splits=5,shuffle=True,random_state=0)


model =models.Sequential([
    layers.Conv2D(32,(3,3),activation="relu",input_shape=(19,19,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation="relu"),
    layers.Dense(1,activation="sigmoid")
])


model.compile(optimizer="adam",loss='binary_crossentropy',metrics=["accuracy"])

for fold, (train_idx,val_idx) in enumerate(kf.split(scdm_train)):

    print(f"training fold {fold+1}/{5}")
    x_train,y_train_fold=scdm_train[train_idx],y_train[train_idx]
    x_val,y_val_fold=scdm_train[val_idx],y_train[val_idx]

    x_train=x_train.reshape(x_train.shape[0],19,19,1)
    x_val= x_val.reshape(x_val.shape[0],19,19,1)

    model_file= f"model_fold_{fold+1}.h5"
    

    #check if saved for this fold, if not train new model

    if os.path.exists(model_file) and "--new" not in sys.argv:
        print(f"load model from file: {model_file}")
        model =tf.keras.models.load_model(model_file)
    else:
        print(f"training new mode")


    #train this fold
    print(np.shape(x_train),np.shape(y_train_fold))
    history = model.fit(x_train,y_train_fold,epochs=10,batch_size=batch_size,validation_data=(x_val,y_val_fold))
    
    #store validation accuracy and loss for rthisfold
    val_acc_history = history.history["val_accuracy"]
    val_loss_history =  history.history["val_loss"]

    model.save(model_file)
    print(f"final validation accuracy for fold {fold+1}: {val_acc_history[-1]:.4f}")
    print(f"final validation loss for fold {fold+1}: {val_loss_history[-1]:.4f}")



"""
input_shape = (19, 19, 1) # (height, width, channels)

model = Sequential()

# Add a 2D convolutional layer with 32 filters, a 3x3 kernel size, and a ReLU activation function
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))

# Add a flatten layer to convert the output of the convolutional layer to a 1D feature vector
model.add(Flatten())

# Add a dense layer with 128 neurons and a ReLU activation function
model.add(Dense(128, activation='relu'))

# Add the output layer with a sigmoid activation function (since we want a binary output)
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss and the Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on your data (X_train and y_train are your training data)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)

#print(trainpd)
"""




