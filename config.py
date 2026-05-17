"""
Configuration file for ApneaGuard project.
Contains all hyperparameters and settings for the sleep apnea detection pipeline.
"""

# ==================== DATA SETTINGS ====================
# Sampling frequency of ECG signals in the PhysioNet database
SAMPLING_RATE = 100  # Hz

# Window size for segmentation (in seconds)
WINDOW_SIZE = 60  # seconds

# Overlap between consecutive windows (in seconds)
WINDOW_OVERLAP = 30  # seconds (50% overlap)

# Train/validation/test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42


# ==================== PREPROCESSING SETTINGS ====================
# Bandpass filter parameters (in Hz)
LOWCUT = 0.5   # Remove baseline wander
HIGHCUT = 40.0  # Remove high-frequency noise

# Filter order
FILTER_ORDER = 4

# Normalization method: 'z-score' or 'min-max'
NORMALIZATION_METHOD = 'z-score'


# ==================== FEATURE EXTRACTION SETTINGS ====================
# Whether to extract HRV features in addition to raw signal
EXTRACT_HRV_FEATURES = True

# HRV time-domain features to extract
HRV_FEATURES = [
    'mean_rr',      # Mean RR interval
    'sdnn',         # Standard deviation of RR intervals
    'rmssd',        # Root mean square of successive differences
    'pnn50',        # Percentage of successive RR differences > 50ms
    'mean_hr',      # Mean heart rate
    'std_hr'        # Standard deviation of heart rate
]


# ==================== MODEL SETTINGS ====================
# Model architecture: 'cnn' or 'cnn_lstm'
MODEL_TYPE = 'cnn'

# CNN architecture parameters
CNN_FILTERS = [64, 128, 256]  # Number of filters in each conv layer
CNN_KERNEL_SIZE = 7           # Kernel size for conv layers
CNN_POOL_SIZE = 2             # Pool size for max pooling

# LSTM parameters (for CNN-LSTM model)
LSTM_UNITS = [128, 64]        # Number of units in each LSTM layer

# Dense layer parameters
DENSE_UNITS = [128, 64]       # Number of units in dense layers

# Dropout rate for regularization
DROPOUT_RATE = 0.5


# ==================== TRAINING SETTINGS ====================
# Number of training epochs
EPOCHS = 50

# Batch size for training
BATCH_SIZE = 32

# Learning rate for optimizer
LEARNING_RATE = 0.0005  # Half the original

# Optimizer: 'adam', 'sgd', 'rmsprop'
OPTIMIZER = 'adam'

# Loss function: 'binary_crossentropy'
LOSS_FUNCTION = 'binary_crossentropy'

# Early stopping patience (epochs without improvement)
EARLY_STOPPING_PATIENCE = 10

# Model checkpoint settings
SAVE_BEST_ONLY = True
CHECKPOINT_DIR = 'saved_models'
MODEL_NAME = 'apnea_detector'


# ==================== EVALUATION SETTINGS ====================
# Metrics to compute during evaluation
METRICS = ['accuracy', 'precision', 'recall', 'auc']

# Threshold for binary classification
CLASSIFICATION_THRESHOLD = 0.5


# ==================== VISUALIZATION SETTINGS ====================
# Figure size for plots (width, height)
FIGURE_SIZE = (12, 6)

# DPI for saving figures
FIGURE_DPI = 100

# Number of sample ECG segments to visualize
NUM_SAMPLES_TO_PLOT = 5

# Whether to save plots to disk
SAVE_PLOTS = True
PLOTS_DIR = 'plots'


# ==================== PATHS ====================
# Default data directory (can be overridden via command line)
DATA_DIR = './data/apnea-ecg'

# Output directory for results
OUTPUT_DIR = './output'

# Logs directory
LOGS_DIR = './logs'


# ==================== CLASS LABELS ====================
# Mapping of annotation labels to class indices
LABEL_MAPPING = {
    'N': 0,  # Normal (no apnea)
    'A': 1,  # Apnea
}

# Class names for display
CLASS_NAMES = ['Normal', 'Apnea']


# ==================== INFERENCE SETTINGS ====================
# Minimum confidence threshold for apnea detection
INFERENCE_CONFIDENCE_THRESHOLD = 0.7

# Whether to smooth predictions over consecutive windows
SMOOTH_PREDICTIONS = True

# Number of windows to use for smoothing
SMOOTHING_WINDOW = 3
