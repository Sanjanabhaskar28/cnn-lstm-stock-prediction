import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import os

# =========================
# SETTINGS
# =========================
DAY = 3

# =========================
# DATA LOADING FUNCTION (FIXED)
# =========================
def load_and_clean_data(file_path):
    x_data, y_data = [], []
    print(f"Reading {file_path}...")
    with open(file_path, 'rb') as f:
        raw_data = pickle.load(f)
        for seq in raw_data:
            try:
                # Force each sequence into (10 timesteps, 6 features)
                # This fixes the 'inhomogeneous shape' error
                reshaped_x = np.array(seq[0]).reshape(10, 6)
                x_data.append(reshaped_x)
                y_data.append(seq[1])
            except ValueError:
                # Skip sequences that don't fit the (10, 6) shape
                continue
    return np.array(x_data), np.array(y_data)

# =========================
# LOAD DATA
# =========================
if not os.path.exists("./data"):
    print("Error: 'data' folder not found!")
else:
    print("Files in data folder:", os.listdir("./data"))

train_x, train_y = load_and_clean_data(f'./data/train{DAY}.pkl')
test_x, test_y = load_and_clean_data(f'./data/test{DAY}.pkl')

print("Train shape:", train_x.shape)
print("Test shape:", test_x.shape)

# =========================
# AUTO DETECT SHAPE
# =========================
# Using the shapes from the successfully loaded data
TIMESTEPS = train_x.shape[1]
FEATURES = train_x.shape[2]

print(f"Detected Timesteps: {TIMESTEPS}, Features: {FEATURES}")

# =========================
# NORMALIZATION
# =========================
scaler = MinMaxScaler()

# Flatten to 2D for scaling, then back to 3D for the model
train_x_reshaped = train_x.reshape(-1, FEATURES)
test_x_reshaped = test_x.reshape(-1, FEATURES)

train_x = scaler.fit_transform(train_x_reshaped).reshape(train_x.shape)
test_x = scaler.transform(test_x_reshaped).reshape(test_x.shape)

# =========================
# MODEL (CNN-LSTM HYBRID)
# =========================
model = Sequential([
    # CNN Layer: Extracts local patterns/features from the 10-day window
    Conv1D(64, 3, activation='relu', input_shape=(TIMESTEPS, FEATURES), padding='same'),
    Dropout(0.3),

    # LSTM Layers: Learns long-term dependencies/trends
    LSTM(64, return_sequences=True),
    Dropout(0.3),

    LSTM(32),

    # Fully Connected Layers: For the final price regression
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['mae']
)

print(model.summary())

# =========================
# CALLBACKS
# =========================
# Ensure models folder exists
if not os.path.exists("./models/day3"):
    os.makedirs("./models/day3")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    f"./models/day3/NEXT{DAY}-BEST.h5",
    monitor="val_loss",
    save_best_only=True
)

# =========================
# TRAIN
# =========================
print("Starting training...")

history = model.fit(
    train_x, train_y,
    epochs=20,
    batch_size=64,
    validation_data=(test_x, test_y),
    callbacks=[early_stop, checkpoint]
)

# =========================
# EVALUATE & PREDICT
# =========================
print("Evaluating...")
loss, mae = model.evaluate(test_x, test_y)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

predictions = model.predict(test_x)
print("Sample predictions (first 5):")
print(predictions[:5])