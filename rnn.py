import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Masking
import matplotlib.pyplot as plt


def read_data(path): 
    data_arr = []
    with open(path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_arr .append(data)
    return data_arr

def covert_res_to_csv(predictions, filename="result.csv"):
    # Convert probabilities to class labels (0 or 1). Adjust this based on your model's output.
    predicted_classes = np.where(predictions >= 0.5, 1, 0)

    # Create a DataFrame
    df = pd.DataFrame({
        'id': range(len(predicted_classes)),
        'class': predicted_classes.flatten()  # Flatten in case it's a 2D array
    })

    # Save to CSV
    df.to_csv(filename, index=False)

def create_model(EMBEDDING_DIM):
    # Define the RNN model
    VOCAB_SIZE = 5000
    
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True)) # mask_zero=True allows the network to ignore padded values
    model.add(Masking(mask_value=0))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def padding_dataset(data):
    MAX_LENGTH = max(len(item["text"]) for item in data)
    train_data_x = tf.keras.preprocessing.sequence.pad_sequences([item["text"] for item in data], maxlen=MAX_LENGTH)
    train_data_y = np.array([item["label"] for item in data])
    
    return train_data_x, train_data_y, MAX_LENGTH


def predict(model, test_data, MAX_LENGTH): 
    padded_test_data = tf.keras.preprocessing.sequence.pad_sequences([item["text"] for item in test_data], maxlen=MAX_LENGTH)
    predictions = model.predict(padded_test_data)
    return predictions    

def train_model(model, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return history


def plot_learning_curve(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation loss values
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Model loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation accuracy values
    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.set_title('Model accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.show()

