"""
COMPLETE APNEAGUARD TRAINING - With Full Visualizations (FIXED)
Uses proven model architecture + all available data
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import wfdb
from scipy import signal
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("APNEAGUARD - COMPLETE TRAINING WITH VISUALIZATIONS")
print("="*70)

# Create output directories
os.makedirs('saved_models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Get ALL available records (a, b, c records, excluding *r files)
data_dir = 'data'
all_files = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.dat')]
records = sorted([f for f in all_files if not f.endswith('r')])

print(f"\nFound {len(records)} records")
print(f"Records: {', '.join(records[:10])}{'...' if len(records) > 10 else ''}")

# Load and preprocess all data
all_segments = []
all_labels = []
sample_ecg_raw = None
sample_ecg_processed = None
record_stats = []

print("\nLoading and preprocessing data...")
for idx, record_name in enumerate(tqdm(records, desc="Processing records")):
    try:
        # Load record
        record = wfdb.rdrecord(f'{data_dir}/{record_name}')
        annotation = wfdb.rdann(f'{data_dir}/{record_name}', 'apn')
        
        # Get ECG signal
        ecg = record.p_signal[:, 0]
        
        # Save sample for visualization (first record)
        if idx == 0:
            sample_ecg_raw = ecg[:30000]  # First 5 minutes
        
        # Bandpass filter
        sos = signal.butter(4, [0.5, 40], btype='band', fs=100, output='sos')
        ecg_filtered = signal.sosfilt(sos, ecg)
        
        # Normalize (per-record, as proven to work)
        ecg_normalized = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)
        
        # Save processed sample
        if idx == 0:
            sample_ecg_processed = ecg_normalized[:30000]
        
        # Segment
        record_segments = 0
        record_apnea = 0
        
        for i in range(0, len(ecg_normalized) - 6000, 3000):
            segment = ecg_normalized[i:i+6000]
            minute = i // 6000
            
            if minute < len(annotation.symbol):
                label = 1 if annotation.symbol[minute] == 'A' else 0
                all_segments.append(segment)
                all_labels.append(label)
                record_segments += 1
                if label == 1:
                    record_apnea += 1
        
        # Track stats
        apnea_pct = 100 * record_apnea / record_segments if record_segments > 0 else 0
        record_stats.append({
            'name': record_name,
            'segments': record_segments,
            'apnea_pct': apnea_pct
        })
    
    except Exception as e:
        print(f"\nWarning: Could not load {record_name}: {e}")
        continue

# Convert to arrays
X = np.array(all_segments)
y = np.array(all_labels)

print(f"\n{'='*70}")
print(f"DATASET SUMMARY")
print(f"{'='*70}")
print(f"Total records:    {len(records)}")
print(f"Total segments:   {len(X):,}")
print(f"Apnea segments:   {np.sum(y):,} ({100*np.mean(y):.1f}%)")
print(f"Normal segments:  {len(y)-np.sum(y):,} ({100*(1-np.mean(y)):.1f}%)")
print(f"{'='*70}\n")

# ============================================================
# VISUALIZATION 1: Raw vs Processed ECG
# ============================================================
print("Creating visualizations...")
print("  1/8 - Raw vs Processed ECG...")

plt.figure(figsize=(15, 8))

# Raw ECG
plt.subplot(2, 1, 1)
time_raw = np.arange(len(sample_ecg_raw)) / 100
plt.plot(time_raw, sample_ecg_raw, 'b-', linewidth=0.5)
plt.title('Raw ECG Signal (First 5 Minutes)', fontsize=14, fontweight='bold')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

# Processed ECG
plt.subplot(2, 1, 2)
time_proc = np.arange(len(sample_ecg_processed)) / 100
plt.plot(time_proc, sample_ecg_processed, 'r-', linewidth=0.5)
plt.title('Preprocessed ECG Signal (Filtered & Normalized)', fontsize=14, fontweight='bold')
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Amplitude')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/raw_ecg_sample.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# VISUALIZATION 2: Sample Segments
# ============================================================
print("  2/8 - Sample segments...")

plt.figure(figsize=(15, 10))

apnea_indices = np.where(y == 1)[0][:3]
normal_indices = np.where(y == 0)[0][:3]
time_seg = np.arange(6000) / 100

for i, idx in enumerate(apnea_indices):
    plt.subplot(3, 2, 2*i + 1)
    plt.plot(time_seg, X[idx], 'r-', linewidth=0.5)
    plt.title(f'Apnea Segment {i+1}', fontsize=12, fontweight='bold', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

for i, idx in enumerate(normal_indices):
    plt.subplot(3, 2, 2*i + 2)
    plt.plot(time_seg, X[idx], 'g-', linewidth=0.5)
    plt.title(f'Normal Segment {i+1}', fontsize=12, fontweight='bold', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/preprocessed_segments.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# VISUALIZATION 3: Class Distribution
# ============================================================
print("  3/8 - Class distribution...")

plt.figure(figsize=(10, 6))
labels_text = ['Normal', 'Apnea']
counts = [np.sum(y == 0), np.sum(y == 1)]
colors = ['#2ecc71', '#e74c3c']

bars = plt.bar(labels_text, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
plt.title('Class Distribution', fontsize=16, fontweight='bold')
plt.ylabel('Number of Segments', fontsize=12)
plt.xlabel('Class', fontsize=12)

for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 500,
             f'{count:,}\n({100*count/len(y):.1f}%)',
             ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Reshape for CNN
X_train = X_train.reshape(-1, 6000, 1)
X_test = X_test.reshape(-1, 6000, 1)

print(f"\nData split:")
print(f"  Training:  {len(X_train):,} samples ({100*len(X_train)/len(X):.1f}%)")
print(f"  Test:      {len(X_test):,} samples ({100*len(X_test)/len(X):.1f}%)")

# Build model (EXACT same as test that worked!)
print("\nBuilding CNN model...")
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
    metrics=['accuracy', keras.metrics.Precision(name='precision'), 
             keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
)

print(f"Model parameters: {model.count_params():,}")

# Callbacks
callbacks = [
    ModelCheckpoint(
        'saved_models/apnea_detector_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Train
print(f"\n{'='*70}")
print("TRAINING")
print(f"{'='*70}\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# VISUALIZATION 4: Training History
# ============================================================
print("\n  4/8 - Training history...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], 'b-', label='Training', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], 'b-', label='Training', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], 'b-', label='Training', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], 'r-', label='Validation', linewidth=2)
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], 'b-', label='Training', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], 'r-', label='Validation', linewidth=2)
axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# VISUALIZATION 5: AUC History
# ============================================================
print("  5/8 - AUC history...")

plt.figure(figsize=(10, 6))
plt.plot(history.history['auc'], 'b-', label='Training AUC', linewidth=2)
plt.plot(history.history['val_auc'], 'r-', label='Validation AUC', linewidth=2)
plt.title('Model AUC (Area Under Curve)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/auc_history.png', dpi=300, bbox_inches='tight')
plt.close()

# Evaluate
print("\n  6/8 - Evaluating model...")

y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

test_results = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'='*70}")
print("TEST SET PERFORMANCE")
print(f"{'='*70}")
print(f"Accuracy:   {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
print(f"Precision:  {test_results[2]:.4f}")
print(f"Recall:     {test_results[3]:.4f}")
print(f"AUC:        {test_results[4]:.4f}")
print(f"Loss:       {test_results[0]:.4f}")
print(f"{'='*70}")

# Calculate F1 Score
f1_score = 2 * (test_results[2] * test_results[3]) / (test_results[2] + test_results[3])
print(f"F1-Score:   {f1_score:.4f}")

# ============================================================
# VISUALIZATION 6: Confusion Matrix
# ============================================================
print("  7/8 - Confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Apnea'],
            yticklabels=['Normal', 'Apnea'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Add percentages
for i in range(2):
    for j in range(2):
        pct = 100 * cm[i, j] / cm.sum()
        plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                ha='center', va='center', fontsize=11, color='gray')

plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# VISUALIZATION 7: ROC Curve
# ============================================================
print("  8/8 - ROC curve...")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, 'b-', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# VISUALIZATION 8: Metrics Comparison Bar Chart
# ============================================================
plt.figure(figsize=(10, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
metrics_values = [test_results[1], test_results[2], test_results[3], f1_score, test_results[4]]

colors_metrics = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
bars = plt.bar(metrics_names, metrics_values, color=colors_metrics, alpha=0.7, edgecolor='black', linewidth=2)

plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.ylim([0, 1.1])
plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% Threshold')
plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% Threshold')

for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.3f}\n({value*100:.1f}%)',
             ha='center', fontsize=11, fontweight='bold')

plt.legend()
plt.tight_layout()
plt.savefig('plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Save detailed metrics
print("\nSaving detailed metrics...")

with open('output/evaluation_metrics.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("APNEAGUARD - MODEL EVALUATION RESULTS\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Dataset Information:\n")
    f.write(f"  Total records: {len(records)}\n")
    f.write(f"  Total segments: {len(X):,}\n")
    f.write(f"  Training samples: {len(X_train):,}\n")
    f.write(f"  Test samples: {len(X_test):,}\n\n")
    
    f.write(f"Test Set Performance:\n")
    f.write(f"  Accuracy:  {test_results[1]:.4f} ({test_results[1]*100:.2f}%)\n")
    f.write(f"  Precision: {test_results[2]:.4f}\n")
    f.write(f"  Recall:    {test_results[3]:.4f}\n")
    f.write(f"  F1-Score:  {f1_score:.4f}\n")
    f.write(f"  AUC:       {test_results[4]:.4f}\n")
    f.write(f"  Loss:      {test_results[0]:.4f}\n\n")
    
    f.write("Confusion Matrix:\n")
    f.write(f"  True Negatives:  {cm[0][0]:,}\n")
    f.write(f"  False Positives: {cm[0][1]:,}\n")
    f.write(f"  False Negatives: {cm[1][0]:,}\n")
    f.write(f"  True Positives:  {cm[1][1]:,}\n\n")
    
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Normal', 'Apnea']))
    
    f.write("\n" + "="*70 + "\n")
    f.write("Per-Record Statistics:\n")
    f.write("="*70 + "\n")
    for stat in record_stats:
        f.write(f"{stat['name']}: {stat['segments']:4d} segments, {stat['apnea_pct']:5.1f}% apnea\n")

# Save final model
model.save('saved_models/apnea_detector_final.h5')

print(f"\n{'='*70}")
print("TRAINING COMPLETE!")
print(f"{'='*70}")
print("\nGenerated Files:")
print("  Models:")
print("    ✓ saved_models/apnea_detector_best.h5 (best checkpoint)")
print("    ✓ saved_models/apnea_detector_final.h5 (final model)")
print("\n  Plots:")
print("    ✓ plots/raw_ecg_sample.png")
print("    ✓ plots/preprocessed_segments.png")
print("    ✓ plots/class_distribution.png")
print("    ✓ plots/training_history.png (4 subplots)")
print("    ✓ plots/auc_history.png")
print("    ✓ plots/confusion_matrix.png")
print("    ✓ plots/roc_curve.png")
print("    ✓ plots/metrics_comparison.png")
print("\n  Metrics:")
print("    ✓ output/evaluation_metrics.txt")
print(f"{'='*70}\n")