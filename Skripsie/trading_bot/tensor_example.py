import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
data = pd.read_csv('data/2023/2023_dayly_price.csv')
print(data)
# Let's assume the dataset has 'Date', 'Open', 'High', 'Low', 'Close', and other columns.
# Use the 'Close' price for prediction
data['date'] = pd.to_datetime(data['timestamp'])
data.set_index('date', inplace=True)

# Normalize the 'Close' prices using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data['Normalized_Close'] = scaler.fit_transform(data['close'].values.reshape(-1, 1))

# Plot the normalized 'Close' prices
data['Normalized_Close'].plot(title='Normalized Bitcoin Close Prices')
plt.show()

# Function to create sequences for LSTM
def create_sequences(data, lookback):
    sequences = []
    targets = []
    for i in range(len(data) - lookback):
        sequences.append(data[i:i + lookback])
        targets.append(data[i + lookback])
    return np.array(sequences), np.array(targets)

# Define a lookback period (number of previous timesteps to consider)
lookback = 10

# Create sequences and targets from the normalized data
sequences, targets = create_sequences(data['Normalized_Close'], lookback)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, shuffle=False)

# Create an LSTM model
model = keras.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.Dropout(0.2),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display the model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=16,
    verbose=2
)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict using the test set
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values to get back to original scale
predictions_inverse = scaler.inverse_transform(predictions)
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs predicted
plt.plot(y_test_inverse, label='Actual Prices')
plt.plot(predictions_inverse, label='Predicted Prices')
plt.title('Actual vs Predicted Bitcoin Prices')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price (USD)')
plt.legend()
plt.show()

