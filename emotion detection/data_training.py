import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

is_init = False
size = -1
label = []
dictionary = {}
c = 0

# Load data
for i in os.listdir():
    if i.endswith(".npy") and not i.startswith("labels"):  
        data = np.load(i)
        if not is_init:
            is_init = True 
            X = data
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, data), axis=0)
            y = np.concatenate((y, np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)), axis=0)

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

# Encode labels
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")
y = to_categorical(y)

# Initialize new arrays for shuffled data
X_new = np.empty_like(X)
y_new = np.empty_like(y)
counter = 0

# Shuffle data
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter += 1

# Define model architecture
ip = Input(shape=(X.shape[1],))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 
model = Model(inputs=ip, outputs=op)

# Compile and train model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
model.fit(X_new, y_new, epochs=50)

# Save model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
