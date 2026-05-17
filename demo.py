"""
Demo script for ApneaGuard.
Demonstrates the complete pipeline using synthetic ECG data.
Useful for testing the system before downloading the PhysioNet database.
"""

import numpy as np
import os
import config
import model
import train
import evaluate
import visualize


def generate_synthetic_ecg(duration_seconds: int, sampling_rate: int, 
                          has_apnea: bool = False) -> np.ndarray:
    """
    Generate synthetic ECG signal for demonstration purposes.
    
    This creates a simplified ECG-like signal with:
    - Regular heartbeat pattern (QRS complexes)
    - Baseline wander
    - Random noise
    - Heart rate variations (especially during "apnea")
    
    Args:
        duration_seconds: Duration of signal in seconds
        sampling_rate: Sampling frequency in Hz
        has_apnea: Whether to simulate apnea-like patterns
        
    Returns:
        Synthetic ECG signal
    """
    num_samples = duration_seconds * sampling_rate
    t = np.arange(num_samples) / sampling_rate
    
    # Base heart rate (beats per minute)
    if has_apnea:
        # During apnea: irregular heart rate with variations
        hr = 65 + 15 * np.sin(2 * np.pi * 0.05 * t)  # More variation
    else:
        # Normal: steady heart rate
        hr = 70 + 5 * np.sin(2 * np.pi * 0.02 * t)  # Less variation
    
    # Convert to instantaneous frequency
    freq = hr / 60.0
    
    # Generate QRS complexes (simplified)
    ecg = np.zeros(num_samples)
    phase = np.cumsum(2 * np.pi * freq / sampling_rate)
    
    # R-peaks (main spike)
    ecg += 1.0 * np.sin(phase)
    
    # Add harmonics for more realistic shape
    ecg += 0.3 * np.sin(2 * phase)
    ecg += 0.1 * np.sin(3 * phase)
    
    # Add baseline wander
    ecg += 0.2 * np.sin(2 * np.pi * 0.1 * t)
    
    # Add noise
    ecg += 0.05 * np.random.randn(num_samples)
    
    # For apnea, add characteristic patterns
    if has_apnea:
        # Add cyclic variations (apnea cycles)
        ecg += 0.3 * np.sin(2 * np.pi * 0.03 * t)
    
    return ecg


def generate_synthetic_dataset(num_records: int = 10, 
                               record_duration: int = 480) -> tuple:
    """
    Generate a synthetic dataset for demonstration.
    
    Args:
        num_records: Number of records to generate
        record_duration: Duration of each record in seconds
        
    Returns:
        Tuple of (signals, annotations) in the same format as data_loader
    """
    print(f"\nGenerating synthetic dataset...")
    print(f"  Records: {num_records}")
    print(f"  Duration per record: {record_duration} seconds")
    
    signals = []
    annotations = []
    
    for i in range(num_records):
        # Randomly decide if this record has apnea
        has_apnea = i % 2 == 0  # Alternate between apnea and normal
        
        # Generate ECG signal
        ecg_signal = generate_synthetic_ecg(
            record_duration,
            config.SAMPLING_RATE,
            has_apnea=has_apnea
        )
        
        # Generate minute-by-minute annotations
        num_minutes = record_duration // 60
        if has_apnea:
            # Random pattern of apnea and normal minutes
            labels = np.random.choice([0, 1], num_minutes, p=[0.4, 0.6])
        else:
            # Mostly normal with occasional apnea
            labels = np.random.choice([0, 1], num_minutes, p=[0.9, 0.1])
        
        # Create metadata
        metadata = {
            'sampling_rate': config.SAMPLING_RATE,
            'signal_name': f'synthetic_{i}',
            'units': 'mV',
            'num_samples': len(ecg_signal),
            'duration_seconds': record_duration
        }
        
        signals.append((ecg_signal, metadata))
        annotations.append(labels)
        
        print(f"  Generated record {i+1}/{num_records}: "
              f"{np.sum(labels==1)} apnea minutes, {np.sum(labels==0)} normal minutes")
    
    print(f"\nSynthetic dataset generated successfully!")
    
    return signals, annotations


def run_demo():
    """
    Run complete demonstration of ApneaGuard pipeline.
    """
    print("="*60)
    print("APNEAGUARD DEMONSTRATION")
    print("Using Synthetic ECG Data")
    print("="*60)
    
    # Create output directories
    demo_dir = './demo_output'
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # ==================== GENERATE SYNTHETIC DATA ====================
    print("\n" + "="*60)
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print("="*60)
    
    signals, annotations = generate_synthetic_dataset(
        num_records=10,
        record_duration=480  # 8 minutes per record
    )
    
    # ==================== VISUALIZE RAW DATA ====================
    print("\n" + "="*60)
    print("STEP 2: VISUALIZING RAW DATA")
    print("="*60)
    
    # Plot sample raw ECG
    ecg_sample, metadata = signals[0]
    labels_sample = annotations[0]
    
    visualize.plot_ecg_signal(
        ecg_sample,
        config.SAMPLING_RATE,
        title="Synthetic ECG Signal (with simulated apnea)",
        duration=240,  # 4 minutes
        annotations=labels_sample,
        save_path=os.path.join(config.PLOTS_DIR, 'demo_raw_ecg.png')
    )
    
    # ==================== PREPROCESSING ====================
    print("\n" + "="*60)
    print("STEP 3: PREPROCESSING")
    print("="*60)
    
    import preprocessing
    X, y = preprocessing.preprocess_dataset(signals, annotations)
    
    # Visualize segments
    visualize.plot_ecg_segments(
        X, y,
        num_samples=5,
        sampling_rate=config.SAMPLING_RATE,
        save_path=os.path.join(config.PLOTS_DIR, 'demo_segments.png')
    )
    
    visualize.plot_class_distribution(
        y,
        title="Demo Segment Class Distribution",
        save_path=os.path.join(config.PLOTS_DIR, 'demo_class_dist.png')
    )
    
    # ==================== SPLIT DATA ====================
    print("\n" + "="*60)
    print("STEP 4: SPLITTING DATA")
    print("="*60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = train.split_data(X, y)
    
    # ==================== CREATE MODEL ====================
    print("\n" + "="*60)
    print("STEP 5: CREATING MODEL")
    print("="*60)
    
    window_samples = config.WINDOW_SIZE * config.SAMPLING_RATE
    input_shape = (window_samples, 1)
    
    demo_model = model.create_model('cnn', input_shape)
    
    # ==================== TRAIN MODEL ====================
    print("\n" + "="*60)
    print("STEP 6: TRAINING MODEL (SHORT DEMO)")
    print("="*60)
    print("Note: Using only 5 epochs for demonstration")
    print("Real training would use 30-50 epochs")
    
    history = train.train_model(
        demo_model,
        X_train, y_train,
        X_val, y_val,
        epochs=5,  # Short demo
        batch_size=16
    )
    
    # Plot training history
    evaluate.plot_training_history(
        history.history,
        save_path=os.path.join(config.PLOTS_DIR, 'demo_training.png')
    )
    
    # ==================== EVALUATE MODEL ====================
    print("\n" + "="*60)
    print("STEP 7: EVALUATING MODEL")
    print("="*60)
    
    metrics = evaluate.generate_evaluation_report(
        demo_model,
        X_test, y_test,
        history=history.history,
        output_dir=config.PLOTS_DIR
    )
    
    # ==================== VISUALIZE PREDICTIONS ====================
    print("\n" + "="*60)
    print("STEP 8: VISUALIZING PREDICTIONS")
    print("="*60)
    
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_pred_proba = demo_model.predict(X_test_reshaped, verbose=0)
    y_pred = (y_pred_proba > config.CLASSIFICATION_THRESHOLD).astype(int).flatten()
    
    visualize.visualize_predictions(
        X_test, y_test, y_pred, y_pred_proba,
        num_samples=5,
        sampling_rate=config.SAMPLING_RATE,
        save_path=os.path.join(config.PLOTS_DIR, 'demo_predictions.png')
    )
    
    # ==================== SAVE MODEL ====================
    print("\n" + "="*60)
    print("STEP 9: SAVING MODEL")
    print("="*60)
    
    model_path = os.path.join(demo_dir, 'demo_model.h5')
    demo_model.save(model_path)
    print(f"\nDemo model saved to: {model_path}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\nNote: These results are from SYNTHETIC data.")
    print("Real performance will depend on actual ECG data quality.")
    print(f"\nAll plots saved to: {config.PLOTS_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_demo()
