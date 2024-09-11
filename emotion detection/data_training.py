import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

# Importing layers and models separately
from keras.layers import Input, Dense
from keras.models import Model

# Initialize variables
is_initialized = False  # To track if the initial data has been loaded
data_size = -1  # To store the size of the loaded data

# To store labels and corresponding numerical values
labels = []
label_to_num_dict = {}
counter = 0  # Counter to assign numerical values to labels

# Iterate over files in the current directory
for file_name in os.listdir():
    # Check if the file is a numpy array file (ends with .npy) and not the labels file
    if file_name.split(".")[-1] == "npy" and not (file_name.split(".")[0] == "labels"):
        # If this is the first file, initialize the dataset
        if not is_initialized:
            is_initialized = True  # Mark as initialized
            X = np.load(file_name)  # Load the numpy array
            data_size = X.shape[0]  # Get the number of samples in the array
            y = np.array([file_name.split('.')[0]] * data_size).reshape(-1, 1)  # Initialize labels array
        else:
            # Concatenate new data to the existing data
            new_data = np.load(file_name)
            X = np.concatenate((X, new_data))  # Add new data to X
            new_labels = np.array([file_name.split('.')[0]] * data_size).reshape(-1, 1)
            y = np.concatenate((y, new_labels))  # Add new labels to y

        # Append the label to the labels list
        labels.append(file_name.split('.')[0])
        
        # Assign a numerical value to the label and store it in the dictionary
        label_to_num_dict[file_name.split('.')[0]] = counter
        counter += 1  # Increment the counter for the next label

# Convert string labels to numerical labels
for i in range(y.shape[0]):
    # Replace each label string with its corresponding numerical value
    y[i, 0] = label_to_num_dict[y[i, 0]]

# Convert the labels to integer type
y = np.array(y, dtype="int32")

# Convert integer labels to one-hot encoded vectors
y = to_categorical(y)

# Initialize new arrays for shuffled data
X_shuffled = X.copy()
y_shuffled = y.copy()
shuffle_counter = 0  # Counter to keep track of the shuffle index

# Create an array of indices and shuffle them
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

# Shuffle the data and labels
for idx in indices:
    X_shuffled[shuffle_counter] = X[idx]
    y_shuffled[shuffle_counter] = y[idx]
    shuffle_counter += 1  # Increment the shuffle counter

# Define the input layer for the model
input_layer = Input(shape=(X.shape[1]))

# Add dense layers to the model
hidden_layer_1 = Dense(512, activation="relu")(input_layer)
hidden_layer_2 = Dense(256, activation="relu")(hidden_layer_1)

# Define the output layer with softmax activation for classification
output_layer = Dense(y.shape[1], activation="softmax")(hidden_layer_2)

# Create the model by specifying the input and output layers
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model on the data for a fixed number of epochs
model.fit(X_shuffled, y_shuffled, epochs=50)

# Save the trained model to a file
model.save("model.h5")

# Save the labels to a numpy file for future use
np.save("labels.npy", np.array(labels))
