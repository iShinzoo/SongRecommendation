import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import load_model

# Initialize variables
is_initialized = False
data_size = -1
labels = []
label_to_num_dict = {}
counter = 0

# Iterate over files to load data
for file_name in os.listdir():
    if file_name.split(".")[-1] == "npy" and not (file_name.split(".")[0] == "labels"):
        if not is_initialized:
            is_initialized = True
            X = np.load(file_name)
            
            # Reshape data if necessary (e.g., ensure 2D shape for dense layers)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            data_size = X.shape[0]
            y = np.array([file_name.split('.')[0]] * data_size).reshape(-1, 1)
        else:
            new_data = np.load(file_name)
            
            if new_data.ndim == 1:
                new_data = new_data.reshape(-1, 1)
            
            X = np.concatenate((X, new_data))
            new_labels = np.array([file_name.split('.')[0]] * new_data.shape[0]).reshape(-1, 1)
            y = np.concatenate((y, new_labels))

        labels.append(file_name.split('.')[0])
        label_to_num_dict[file_name.split('.')[0]] = counter
        counter += 1

# Convert labels to numerical format
for i in range(y.shape[0]):
    y[i, 0] = label_to_num_dict[y[i, 0]]

y = np.array(y, dtype="int32")
y = to_categorical(y)

# Shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_shuffled = X[indices]
y_shuffled = y[indices]

# Print shape of X to ensure it's correct
print("Shape of X:", X.shape)

# Build the model
input_layer = Input(shape=(X.shape[1],))  # Ensure input shape is a tuple
hidden_layer_1 = Dense(512, activation="relu")(input_layer)
hidden_layer_2 = Dense(256, activation="relu")(hidden_layer_1)
output_layer = Dense(y.shape[1], activation="softmax")(hidden_layer_2)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X_shuffled, y_shuffled, epochs=50)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(labels))
