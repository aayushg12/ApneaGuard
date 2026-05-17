"""
Evaluation module for ApneaGuard.
Computes comprehensive metrics and generates evaluation reports.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import config
import os


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model on test set and compute comprehensive metrics.
    
    Metrics computed:
    - Accuracy: (TP + TN) / Total
    - Precision: TP / (TP + FP) - How many predicted apneas are correct
    - Recall: TP / (TP + FN) - How many actual apneas are detected
    - F1-Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under ROC curve - Overall discriminative ability
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    print("\n" + "="*50)
    print("Evaluating Model on Test Set")
    print("="*50)
    
    # Reshape data for model
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Get predictions
    y_pred_proba = model.predict(X_test_reshaped, verbose=0)
    y_pred = (y_pred_proba > config.CLASSIFICATION_THRESHOLD).astype(int).flatten()
    
    # Calculate metrics
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    metrics['f1_score'] = f1_score(y_test, y_pred)
    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Extract TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    
    # Specificity (True Negative Rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Apnea")
    print(f"  Actual Normal  {tn:5d}  {fp:5d}")
    print(f"         Apnea   {fn:5d}  {tp:5d}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=config.CLASS_NAMES,
        digits=4
    ))
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot (None = display only)
    """
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(y_test: np.ndarray, y_pred_proba: np.ndarray, 
                   roc_auc: float, save_path: str = None):
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    The ROC curve shows the trade-off between:
    - True Positive Rate (Recall/Sensitivity)
    - False Positive Rate (1 - Specificity)
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        roc_auc: Area under ROC curve
        save_path: Path to save the plot (None = display only)
    """
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve - Sleep Apnea Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot (None = display only)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics: dict, save_path: str = None):
    """
    Plot bar chart comparing different metrics.
    
    Args:
        metrics: Dictionary of metric values
        save_path: Path to save the plot (None = display only)
    """
    # Select metrics to plot
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'specificity']
    metric_values = [metrics[name] for name in metric_names]
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity']
    
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(metric_labels, metric_values, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.ylim([0, 1.1])
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Metrics comparison saved to: {save_path}")
    
    plt.show()


def generate_evaluation_report(model: keras.Model, 
                               X_test: np.ndarray, 
                               y_test: np.ndarray,
                               history: dict = None,
                               output_dir: str = None) -> dict:
    """
    Generate complete evaluation report with all metrics and visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        history: Training history (optional)
        output_dir: Directory to save plots (None = config default)
        
    Returns:
        Dictionary of evaluation metrics
    """
    if output_dir is None:
        output_dir = config.PLOTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Get predictions for plotting
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_pred_proba = model.predict(X_test_reshaped, verbose=0)
    
    # Generate plots
    if config.SAVE_PLOTS:
        print("\nGenerating evaluation plots...")
        
        # Confusion matrix
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        # ROC curve
        plot_roc_curve(
            y_test, y_pred_proba, metrics['roc_auc'],
            save_path=os.path.join(output_dir, 'roc_curve.png')
        )
        
        # Metrics comparison
        plot_metrics_comparison(
            metrics,
            save_path=os.path.join(output_dir, 'metrics_comparison.png')
        )
        
        # Training history (if provided)
        if history is not None:
            plot_training_history(
                history,
                save_path=os.path.join(output_dir, 'training_history.png')
            )
    
    return metrics


if __name__ == "__main__":
    """
    Test evaluation with dummy data.
    """
    print("Testing evaluation module...")
    
    # Create dummy predictions
    num_samples = 500
    y_test = np.random.randint(0, 2, num_samples)
    y_pred_proba = np.random.rand(num_samples, 1)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics manually
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(cm)
    
    print("\nPlotting ROC curve...")
    auc = roc_auc_score(y_test, y_pred_proba)
    plot_roc_curve(y_test, y_pred_proba, auc)
    
    print("\nEvaluation test complete!")
