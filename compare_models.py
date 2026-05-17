"""
Model comparison script for ApneaGuard.
Compare performance of different model architectures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
import data_loader
import preprocessing
import model
import train
import evaluate


def train_and_evaluate_model(model_type: str, X_train, y_train, X_val, y_val, 
                             X_test, y_test, epochs: int = 10) -> dict:
    """
    Train and evaluate a single model configuration.
    
    Args:
        model_type: Type of model ('cnn' or 'cnn_lstm')
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        epochs: Number of training epochs
        
    Returns:
        Dictionary with model performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*60}")
    
    # Create model
    window_samples = config.WINDOW_SIZE * config.SAMPLING_RATE
    input_shape = (window_samples, 1)
    
    current_model = model.create_model(model_type, input_shape)
    
    # Train model
    history = train.train_model(
        current_model,
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=32
    )
    
    # Evaluate model
    metrics = evaluate.evaluate_model(current_model, X_test, y_test)
    
    # Add model type to metrics
    metrics['model_type'] = model_type
    metrics['history'] = history.history
    
    return metrics


def compare_models(X_train, y_train, X_val, y_val, X_test, y_test, 
                  epochs: int = 20):
    """
    Compare CNN and CNN-LSTM models.
    
    Args:
        Training, validation, and test data
        epochs: Number of epochs for each model
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    # Train CNN
    cnn_metrics = train_and_evaluate_model(
        'cnn', X_train, y_train, X_val, y_val, X_test, y_test, epochs
    )
    results.append(cnn_metrics)
    
    # Train CNN-LSTM
    cnn_lstm_metrics = train_and_evaluate_model(
        'cnn_lstm', X_train, y_train, X_val, y_val, X_test, y_test, epochs
    )
    results.append(cnn_lstm_metrics)
    
    # Create comparison table
    comparison_data = []
    for metrics in results:
        comparison_data.append({
            'Model': metrics['model_type'].upper(),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'Specificity': metrics['specificity']
        })
    
    df = pd.DataFrame(comparison_data)
    
    return df, results


def plot_model_comparison(df: pd.DataFrame, save_path: str = None):
    """
    Create visualization comparing model performance.
    
    Args:
        df: DataFrame with comparison results
        save_path: Path to save plot
    """
    # Prepare data for plotting
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Plot bars for each model
    for i, (_, row) in enumerate(df.iterrows()):
        offset = width * (i - 0.5)
        values = [row[m] for m in metrics]
        bars = ax.bar(x + offset, values, width, label=row['Model'], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"\nComparison plot saved to: {save_path}")
    
    plt.show()


def plot_training_comparison(results: list, save_path: str = None):
    """
    Compare training curves of different models.
    
    Args:
        results: List of result dictionaries with history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, metrics in enumerate(results):
        model_type = metrics['model_type'].upper()
        history = metrics['history']
        color = colors[i % len(colors)]
        
        # Plot loss
        axes[0].plot(history['loss'], label=f'{model_type} Train', 
                    color=color, linestyle='-', linewidth=2)
        axes[0].plot(history['val_loss'], label=f'{model_type} Val', 
                    color=color, linestyle='--', linewidth=2)
        
        # Plot accuracy
        axes[1].plot(history['accuracy'], label=f'{model_type} Train', 
                    color=color, linestyle='-', linewidth=2)
        axes[1].plot(history['val_accuracy'], label=f'{model_type} Val', 
                    color=color, linestyle='--', linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"Training comparison plot saved to: {save_path}")
    
    plt.show()


def main():
    """
    Run model comparison.
    """
    print("="*60)
    print("APNEAGUARD - MODEL COMPARISON")
    print("="*60)
    
    # Option 1: Use synthetic data for quick testing
    print("\nWould you like to use:")
    print("1. Synthetic data (fast demo)")
    print("2. Real PhysioNet data (requires download)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        print("\nGenerating synthetic data...")
        from demo import generate_synthetic_dataset
        signals, annotations = generate_synthetic_dataset(
            num_records=10,
            record_duration=480
        )
    else:
        print("\nLoading real data...")
        data_dir = input("Enter path to data directory: ").strip()
        if not data_dir:
            data_dir = config.DATA_DIR
        
        signals, annotations = data_loader.load_dataset(
            data_dir,
            max_records=10  # Limit for faster comparison
        )
    
    # Preprocess
    print("\nPreprocessing data...")
    X, y = preprocessing.preprocess_dataset(signals, annotations)
    
    # Split data
    print("\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = train.split_data(X, y)
    
    # Compare models
    print("\nComparing models...")
    print("Note: Using 10 epochs for quick comparison")
    print("For final results, use 30-50 epochs")
    
    df, results = compare_models(
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=10
    )
    
    # Print results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print("\n" + df.to_string(index=False))
    print("\n" + "="*60)
    
    # Determine best model
    best_idx = df['F1-Score'].idxmax()
    best_model = df.loc[best_idx, 'Model']
    best_f1 = df.loc[best_idx, 'F1-Score']
    
    print(f"\nBest Model: {best_model} (F1-Score: {best_f1:.4f})")
    
    # Save results
    import os
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    df.to_csv(os.path.join(config.OUTPUT_DIR, 'model_comparison.csv'), index=False)
    print(f"\nResults saved to: {os.path.join(config.OUTPUT_DIR, 'model_comparison.csv')}")
    
    # Visualize
    print("\nGenerating comparison plots...")
    plot_model_comparison(
        df,
        save_path=os.path.join(config.PLOTS_DIR, 'model_comparison.png')
    )
    
    plot_training_comparison(
        results,
        save_path=os.path.join(config.PLOTS_DIR, 'training_comparison.png')
    )
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
