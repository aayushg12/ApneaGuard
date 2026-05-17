import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import wfdb
from scipy import signal

print("MINIMAL TRAINING TEST")
print("="*60)

# Load data
print("Loading a01...")
record = wfdb.rdrecord('data/a01')
annotation = wfdb.rdann('data/a01', 'apn')
ecg = record.p_signal[:, 0]

# Preprocess
print("Preprocessing...")
sos = signal.butter(4, [0.5, 40], btype='band', fs=100, output='sos')
ecg_filtered = signal.sosfilt(sos, ecg)
ecg_normalized = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)

# Segment
print("Creating segments...")
segments = []
labels = []
for i in range(0, len(ecg_normalized) - 6000, 3000):
    segment = ecg_normalized[i:i+6000]
    minute = i // 6000
    if minute < len(annotation.symbol):
        label = 1 if annotation.symbol[minute] == 'A' else 0
        segments.append(segment)
        labels.append(label)

X = np.array(segments)
y = np.array(labels)

print(f"Created {len(X)} segments")
print(f"Apnea: {np.sum(y)} ({100*np.mean(y):.1f}%)")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = X_train.reshape(-1, 6000, 1)
X_test = X_test.reshape(-1, 6000, 1)

# Model
print("Building model...")
model = keras.Sequential([
    keras.layers.Conv1D(32, 11, activation='relu', input_shape=(6000, 1)),
    keras.layers.MaxPooling1D(4),
    keras.layers.Conv1D(64, 11, activation='relu'),
    keras.layers.MaxPooling1D(4),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
print("Training...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32, verbose=1)

print("\nRESULTS:")
print(f"Accuracy: {history.history['val_accuracy'][-1]:.3f}")
print(f"Loss: {history.history['val_loss'][-1]:.3f}")

if history.history['val_loss'][-1] < 1.0:
    print("\n✅ TEST PASSED - Pipeline works!")
else:
    print("\n❌ TEST FAILED - Fundamental issue")