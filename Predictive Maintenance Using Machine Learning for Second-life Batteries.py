import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and preprocess the dataset
data = pd.read_csv("battery_data.csv")  # Replace with the actual dataset path

# Assume the dataset has columns ['voltage', 'current', 'temperature', 'SoH']
# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values)

# Define sequence length for LSTM
sequence_length = 30

# Function to create sequences for LSTM input
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length, -1]  # Assuming SoH is the target (last column)
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Create sequences
X, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)  # Regression output for SoH
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Plot actual vs predicted SoH
plt.scatter(y_test, y_pred)
plt.xlabel("Actual SoH")
plt.ylabel("Predicted SoH")
plt.title("Actual vs Predicted SoH")
plt.show()
