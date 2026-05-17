"""
Visualization module for ApneaGuard.
Provides utilities for visualizing ECG signals, segments, and features.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
import os


def plot_ecg_signal(ecg_signal: np.ndarray, sampling_rate: int, 
                   title: str = "ECG Signal", duration: float = None,
                   annotations: np.ndarray = None, save_path: str = None):
    """
    Plot raw ECG signal with optional annotations.
    
    Args:
        ecg_signal: ECG signal array
        sampling_rate: Sampling frequency in Hz
        title: Plot title
        duration: Duration to plot in seconds (None = all)
        annotations: Minute-by-minute annotations (optional)
        save_path: Path to save the plot (None = display only)
    """
    # Calculate time axis
    total_duration = len(ecg_signal) / sampling_rate
    time = np.arange(len(ecg_signal)) / sampling_rate
    
    # Limit duration if specified
    if duration is not None:
        samples_to_plot = int(duration * sampling_rate)
        ecg_signal = ecg_signal[:samples_to_plot]
        time = time[:samples_to_plot]
    
    plt.figure(figsize=config.FIGURE_SIZE)
    
    # Plot ECG signal
    plt.plot(time, ecg_signal, linewidth=0.5, color='blue')
    
    # Add annotations if provided
    if annotations is not None:
        # Shade apnea regions
        for i, label in enumerate(annotations):
            if label == 1:  # Apnea
                start_time = i * 60  # Convert minute to seconds
                end_time = (i + 1) * 60
                
                # Only shade if within plotted range
                if start_time < time[-1]:
                    plt.axvspan(start_time, end_time, alpha=0.2, color='red')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.2, label='Apnea'),
            Patch(facecolor='white', label='Normal')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"ECG plot saved to: {save_path}")
    
    plt.show()


def plot_ecg_segments(segments: np.ndarray, labels: np.ndarray, 
                     num_samples: int, sampling_rate: int,
                     save_path: str = None):
    """
    Plot multiple ECG segments with their labels.
    
    Args:
        segments: Array of ECG segments
        labels: Corresponding labels
        num_samples: Number of segments to plot
        sampling_rate: Sampling frequency in Hz
        save_path: Path to save the plot (None = display only)
    """
    # Limit number of samples
    num_samples = min(num_samples, len(segments))
    
    # Select random samples
    indices = np.random.choice(len(segments), num_samples, replace=False)
    
    # Create subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        segment = segments[idx]
        label = labels[idx]
        label_name = config.CLASS_NAMES[label]
        
        # Create time axis
        time = np.arange(len(segment)) / sampling_rate
        
        # Plot segment
        axes[i].plot(time, segment, linewidth=0.8)
        axes[i].set_ylabel('Amplitude', fontsize=10)
        axes[i].set_title(f'Segment {idx} - {label_name}', 
                         fontsize=11, fontweight='bold',
                         color='red' if label == 1 else 'green')
        axes[i].grid(alpha=0.3)
        
        # Only show x-label on bottom plot
        if i == num_samples - 1:
            axes[i].set_xlabel('Time (seconds)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Segments plot saved to: {save_path}")
    
    plt.show()


def plot_preprocessing_comparison(original: np.ndarray, filtered: np.ndarray,
                                 normalized: np.ndarray, sampling_rate: int,
                                 duration: float = 10, save_path: str = None):
    """
    Compare original, filtered, and normalized ECG signals.
    
    Args:
        original: Original ECG signal
        filtered: Filtered ECG signal
        normalized: Normalized ECG signal
        sampling_rate: Sampling frequency in Hz
        duration: Duration to plot in seconds
        save_path: Path to save the plot (None = display only)
    """
    # Limit to specified duration
    samples = int(duration * sampling_rate)
    original = original[:samples]
    filtered = filtered[:samples]
    normalized = normalized[:samples]
    
    # Create time axis
    time = np.arange(len(original)) / sampling_rate
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    # Plot original
    axes[0].plot(time, original, linewidth=0.8, color='blue')
    axes[0].set_title('Original Signal', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Plot filtered
    axes[1].plot(time, filtered, linewidth=0.8, color='green')
    axes[1].set_title('After Bandpass Filtering', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=10)
    axes[1].grid(alpha=0.3)
    
    # Plot normalized
    axes[2].plot(time, normalized, linewidth=0.8, color='red')
    axes[2].set_title('After Normalization', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Amplitude', fontsize=10)
    axes[2].set_xlabel('Time (seconds)', fontsize=10)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Preprocessing comparison saved to: {save_path}")
    
    plt.show()


def plot_hrv_features(features: np.ndarray, labels: np.ndarray,
                     save_path: str = None):
    """
    Visualize HRV feature distributions for apnea vs normal.
    
    Args:
        features: Array of HRV features, shape (num_samples, num_features)
        labels: Corresponding labels
        save_path: Path to save the plot (None = display only)
    """
    feature_names = ['Mean RR', 'SDNN', 'RMSSD', 'pNN50', 'Mean HR', 'Std HR']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        # Separate by class
        normal_features = features[labels == 0, i]
        apnea_features = features[labels == 1, i]
        
        # Plot distributions
        ax.hist(normal_features, bins=30, alpha=0.5, label='Normal', color='green')
        ax.hist(apnea_features, bins=30, alpha=0.5, label='Apnea', color='red')
        
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('HRV Feature Distributions: Apnea vs Normal', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"HRV features plot saved to: {save_path}")
    
    plt.show()


def plot_class_distribution(labels: np.ndarray, title: str = "Class Distribution",
                           save_path: str = None):
    """
    Plot class distribution as pie and bar charts.
    
    Args:
        labels: Array of class labels
        title: Plot title
        save_path: Path to save the plot (None = display only)
    """
    # Count classes
    unique, counts = np.unique(labels, return_counts=True)
    percentages = 100 * counts / len(labels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    colors = ['lightgreen', 'lightcoral']
    ax1.pie(counts, labels=config.CLASS_NAMES, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax1.set_title('Class Distribution (Pie)', fontsize=12, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(config.CLASS_NAMES, counts, color=colors, alpha=0.8)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Class Distribution (Bar)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Class distribution saved to: {save_path}")
    
    plt.show()


def visualize_predictions(segments: np.ndarray, y_true: np.ndarray, 
                         y_pred: np.ndarray, y_pred_proba: np.ndarray,
                         num_samples: int, sampling_rate: int,
                         save_path: str = None):
    """
    Visualize predictions vs ground truth for sample segments.
    
    Args:
        segments: ECG segments
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        num_samples: Number of samples to plot
        sampling_rate: Sampling frequency in Hz
        save_path: Path to save the plot (None = display only)
    """
    # Select samples (prioritize misclassifications)
    misclassified = np.where(y_true != y_pred)[0]
    correct = np.where(y_true == y_pred)[0]
    
    # Mix of correct and incorrect
    if len(misclassified) >= num_samples // 2:
        indices = np.concatenate([
            np.random.choice(misclassified, num_samples // 2, replace=False),
            np.random.choice(correct, num_samples - num_samples // 2, replace=False)
        ])
    else:
        indices = np.random.choice(len(segments), num_samples, replace=False)
    
    # Create subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 2*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        segment = segments[idx]
        true_label = config.CLASS_NAMES[y_true[idx]]
        pred_label = config.CLASS_NAMES[y_pred[idx]]
        confidence = y_pred_proba[idx][0] if y_pred[idx] == 1 else 1 - y_pred_proba[idx][0]
        
        # Create time axis
        time = np.arange(len(segment)) / sampling_rate
        
        # Determine color (green = correct, red = incorrect)
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        
        # Plot segment
        axes[i].plot(time, segment, linewidth=0.8, color='blue')
        axes[i].set_ylabel('Amplitude', fontsize=9)
        
        # Title with prediction info
        title = f'True: {true_label} | Predicted: {pred_label} (confidence: {confidence:.2%})'
        axes[i].set_title(title, fontsize=10, fontweight='bold', color=color)
        axes[i].grid(alpha=0.3)
        
        # Only show x-label on bottom plot
        if i == num_samples - 1:
            axes[i].set_xlabel('Time (seconds)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Predictions visualization saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    """
    Test visualization functions.
    """
    print("Testing visualization module...")
    
    # Generate dummy data
    sampling_rate = config.SAMPLING_RATE
    duration = 10  # seconds
    num_samples = duration * sampling_rate
    
    # Create dummy ECG signal
    t = np.linspace(0, duration, num_samples)
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(num_samples)
    
    print("\nPlotting sample ECG signal...")
    plot_ecg_signal(ecg, sampling_rate, title="Sample ECG Signal", duration=duration)
    
    # Create dummy segments
    segments = np.random.randn(20, 600)
    labels = np.random.randint(0, 2, 20)
    
    print("\nPlotting sample segments...")
    plot_ecg_segments(segments, labels, num_samples=3, sampling_rate=sampling_rate)
    
    print("\nVisualization test complete!")
