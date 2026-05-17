"""
K-Fold Cross-Validation for ApneaGuard Model
Provides more robust performance estimates
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import wfdb
from scipy import signal
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import sys

# Force output flushing
sys.stdout.flush()

print("="*70, flush=True)
print("APNEAGUARD - K-FOLD CROSS-VALIDATION", flush=True)
print("="*70, flush=True)
print("Starting script...", flush=True)

# Configuration
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 32

print(f"\nConfiguration:", flush=True)
print(f"  K-Folds: {K_FOLDS}", flush=True)
print(f"  Epochs: {EPOCHS}", flush=True)
print(f"  Batch Size: {BATCH_SIZE}", flush=True)

# Load data
data_dir = 'data'
print(f"\nChecking data directory: {data_dir}", flush=True)

if not os.path.exists(data_dir):
    print(f"ERROR: Data directory '{data_dir}' not found!", flush=True)
    sys.exit(1)

all_files = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.dat')]
records = sorted([f for f in all_files if not f.endswith('r')])

print(f"Found {len(records)} record files", flush=True)
print(f"First few: {records[:5]}", flush=True)

all_segments = []
all_labels = []

print(f"\nLoading and preprocessing data...", flush=True)

for idx, record_name in enumerate(records):
    try:
        if idx % 5 == 0:  # Print progress every 5 records
            print(f"  Processing record {idx+1}/{len(records)}: {record_name}", flush=True)
        
        record = wfdb.rdrecord(f'{data_dir}/{record_name}')
        annotation = wfdb.rdann(f'{data_dir}/{record_name}', 'apn')
        
        ecg = record.p_signal[:, 0]
        
        # Preprocess
        sos = signal.butter(4, [0.5, 40], btype='band', fs=100, output='sos')
        ecg_filtered = signal.sosfilt(sos, ecg)
        ecg_normalized = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)
        
        # Segment
        for i in range(0, len(ecg_normalized) - 6000, 3000):
            segment = ecg_normalized[i:i+6000]
            minute = i // 6000
            
            if minute < len(annotation.symbol):
                label = 1 if annotation.symbol[minute] == 'A' else 0
                all_segments.append(segment)
                all_labels.append(label)
    except Exception as e:
        print(f"  Warning: Could not load {record_name}: {e}", flush=True)
        continue

X = np.array(all_segments)
y = np.array(all_labels)

print(f"\n✓ Data loaded successfully!", flush=True)
print(f"  Total segments: {len(X):,}", flush=True)
print(f"  Apnea segments: {np.sum(y):,} ({100*np.mean(y):.1f}%)", flush=True)
print(f"  Normal segments: {len(y)-np.sum(y):,} ({100*(1-np.mean(y)):.1f}%)", flush=True)

if len(X) == 0:
    print("\nERROR: No data loaded! Check your data directory.", flush=True)
    sys.exit(1)

# Prepare for K-Fold
print(f"\nInitializing {K_FOLDS}-Fold Cross-Validation...", flush=True)
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Store results
fold_results = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'auc': []
}

print(f"\n{'='*70}", flush=True)
print(f"RUNNING {K_FOLDS}-FOLD CROSS-VALIDATION", flush=True)
print(f"{'='*70}", flush=True)

# K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n{'='*70}", flush=True)
    print(f"FOLD {fold}/{K_FOLDS}", flush=True)
    print(f"{'='*70}", flush=True)
    
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Reshape
    X_train = X_train.reshape(-1, 6000, 1)
    X_val = X_val.reshape(-1, 6000, 1)
    
    print(f"Train: {len(X_train):,} samples | Val: {len(X_val):,} samples", flush=True)
    print(f"Building model...", flush=True)
    
    # Build model
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
    
    print(f"Training model (this may take a few minutes)...", flush=True)
    
    # Train with progress
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1  # Show training progress
    )
    
    print(f"\nEvaluating fold {fold}...", flush=True)
    
    # Predict
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_prob)
    
    # Store results
    fold_results['accuracy'].append(acc)
    fold_results['precision'].append(prec)
    fold_results['recall'].append(rec)
    fold_results['f1'].append(f1)
    fold_results['auc'].append(auc)
    
    # Print fold results
    print(f"\n✓ Fold {fold} Results:", flush=True)
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)", flush=True)
    print(f"  Precision: {prec:.4f}", flush=True)
    print(f"  Recall:    {rec:.4f}", flush=True)
    print(f"  F1-Score:  {f1:.4f}", flush=True)
    print(f"  AUC:       {auc:.4f}", flush=True)
    
    # Clean up
    del model
    keras.backend.clear_session()

# Calculate statistics
print(f"\n{'='*70}", flush=True)
print("CROSS-VALIDATION RESULTS SUMMARY", flush=True)
print(f"{'='*70}\n", flush=True)

for metric_name, values in fold_results.items():
    mean = np.mean(values)
    std = np.std(values)
    print(f"{metric_name.upper():12s}: {mean:.4f} ± {std:.4f} (mean ± std)", flush=True)
    print(f"              Range: [{np.min(values):.4f}, {np.max(values):.4f}]", flush=True)
    print(flush=True)

# Create output directory if needed
os.makedirs('plots', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Visualize results
print("Creating visualization...", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('K-Fold Cross-Validation Results', fontsize=16, fontweight='bold')

# Plot each metric
for idx, (metric_name, values) in enumerate(fold_results.items()):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Bar plot
    folds = list(range(1, K_FOLDS + 1))
    bars = ax.bar(folds, values, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Color bars
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add mean line
    mean = np.mean(values)
    ax.axhline(y=mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    
    # Formatting
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title(metric_name.upper())
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (fold, value) in enumerate(zip(folds, values)):
        ax.text(fold, value + 0.02, f'{value:.3f}', 
                ha='center', fontsize=9, fontweight='bold')

# Remove extra subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig('plots/kfold_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved plots/kfold_results.png", flush=True)

# Save results
with open('output/kfold_results.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("APNEAGUARD - K-FOLD CROSS-VALIDATION RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Configuration:\n")
    f.write(f"  Number of folds: {K_FOLDS}\n")
    f.write(f"  Total segments: {len(X):,}\n\n")
    
    f.write("RESULTS PER FOLD\n")
    f.write("="*70 + "\n")
    for fold in range(K_FOLDS):
        f.write(f"\nFold {fold + 1}:\n")
        for metric_name, values in fold_results.items():
            f.write(f"  {metric_name:12s}: {values[fold]:.4f}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("="*70 + "\n")
    for metric_name, values in fold_results.items():
        mean = np.mean(values)
        std = np.std(values)
        f.write(f"\n{metric_name.upper()}:\n")
        f.write(f"  Mean:   {mean:.4f}\n")
        f.write(f"  Std:    {std:.4f}\n")
        f.write(f"  Min:    {np.min(values):.4f}\n")
        f.write(f"  Max:    {np.max(values):.4f}\n")

print("✓ Saved output/kfold_results.txt", flush=True)

print(f"\n{'='*70}", flush=True)
print("K-FOLD CROSS-VALIDATION COMPLETE!", flush=True)
print(f"{'='*70}\n", flush=True)