"""
Main execution script for ApneaGuard.
Orchestrates the complete pipeline from data loading to model evaluation.
"""

import argparse
import os
import pickle
import numpy as np
from tensorflow import keras
import config
import data_loader
import preprocessing
import feature_extraction
import model
import train
import evaluate
import visualize
import inference


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='ApneaGuard: Sleep Apnea Detection from ECG Signals',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default=config.DATA_DIR,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--max_records',
        type=int,
        default=None,
        help='Maximum number of records to load (None = all)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['cnn', 'cnn_lstm'],
        default=config.MODEL_TYPE,
        help='Type of model architecture'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.BATCH_SIZE,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=config.LEARNING_RATE,
        help='Learning rate for optimizer'
    )
    
    # Pipeline control
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='Skip training and load existing model'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to existing model (for inference only)'
    )
    parser.add_argument(
        '--extract_features',
        action='store_true',
        help='Extract HRV features (currently for visualization only)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default=config.OUTPUT_DIR,
        help='Directory for output files'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Disable plot generation'
    )
    
    return parser.parse_args()


def main():
    """
    Main execution function.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Update config with command line arguments
    if args.no_plots:
        config.SAVE_PLOTS = False
    
    print("="*60)
    print("APNEAGUARD - SLEEP APNEA DETECTION SYSTEM")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Model type: {args.model_type}")
    print(f"  Window size: {config.WINDOW_SIZE} seconds")
    print(f"  Sampling rate: {config.SAMPLING_RATE} Hz")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # ==================== STEP 1: LOAD DATA ====================
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    signals, annotations = data_loader.load_dataset(
        args.data_dir,
        max_records=args.max_records
    )
    
    # Print class distribution
    dist = data_loader.get_class_distribution(annotations)
    print(f"\nOverall Class Distribution:")
    print(f"  Normal: {dist['normal_count']} minutes ({dist['normal_percentage']:.1f}%)")
    print(f"  Apnea:  {dist['apnea_count']} minutes ({dist['apnea_percentage']:.1f}%)")
    
    # ==================== STEP 2: PREPROCESSING ====================
    print("\n" + "="*60)
    print("STEP 2: PREPROCESSING")
    print("="*60)
    
    # Visualize one raw signal before preprocessing
    if config.SAVE_PLOTS and len(signals) > 0:
        print("\nVisualizing sample raw ECG signal...")
        ecg_sample, metadata = signals[0]
        labels_sample = annotations[0]
        visualize.plot_ecg_signal(
            ecg_sample,
            int(metadata['sampling_rate']),
            title="Sample Raw ECG Signal",
            duration=300,  # 5 minutes
            annotations=labels_sample,
            save_path=os.path.join(config.PLOTS_DIR, 'raw_ecg_sample.png')
        )
    
    # Preprocess all data
    X, y = preprocessing.preprocess_dataset(signals, annotations)
    
    print(f"\nPreprocessed data shape:")
    print(f"  X (segments): {X.shape}")
    print(f"  y (labels): {y.shape}")
    
    # Visualize preprocessed segments
    if config.SAVE_PLOTS:
        print("\nVisualizing preprocessed segments...")
        visualize.plot_ecg_segments(
            X, y,
            num_samples=config.NUM_SAMPLES_TO_PLOT,
            sampling_rate=config.SAMPLING_RATE,
            save_path=os.path.join(config.PLOTS_DIR, 'preprocessed_segments.png')
        )
        
        visualize.plot_class_distribution(
            y,
            title="Segment Class Distribution",
            save_path=os.path.join(config.PLOTS_DIR, 'class_distribution.png')
        )
    
    # ==================== STEP 3: FEATURE EXTRACTION (OPTIONAL) ====================
    if args.extract_features:
        print("\n" + "="*60)
        print("STEP 3: FEATURE EXTRACTION")
        print("="*60)
        
        hrv_features = feature_extraction.extract_features_from_segments(
            X, config.SAMPLING_RATE
        )
        
        # Visualize HRV features
        if config.SAVE_PLOTS:
            print("\nVisualizing HRV features...")
            visualize.plot_hrv_features(
                hrv_features, y,
                save_path=os.path.join(config.PLOTS_DIR, 'hrv_features.png')
            )
        
        # Save features
        features_path = os.path.join(args.output_dir, 'hrv_features.npy')
        np.save(features_path, hrv_features)
        print(f"\nHRV features saved to: {features_path}")
    
    # ==================== STEP 4: TRAIN/VAL/TEST SPLIT ====================
    print("\n" + "="*60)
    print("STEP 4: SPLITTING DATA")
    print("="*60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = train.split_data(X, y)
    
    # ==================== STEP 5: MODEL TRAINING ====================
    if not args.skip_training:
        print("\n" + "="*60)
        print("STEP 5: MODEL TRAINING")
        print("="*60)
        
        # Calculate input shape
        window_samples = config.WINDOW_SIZE * config.SAMPLING_RATE
        input_shape = (window_samples, 1)
        
        # Create model
        sleep_apnea_model = model.create_model(args.model_type, input_shape)
        
        # Train model
        history = train.train_model(
            sleep_apnea_model,
            X_train, y_train,
            X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save training history
        history_path = os.path.join(args.output_dir, 'training_history.pkl')
        train.save_training_history(history, history_path)
        
        # Plot training history
        if config.SAVE_PLOTS:
            print("\nVisualizing training history...")
            evaluate.plot_training_history(
                history.history,
                save_path=os.path.join(config.PLOTS_DIR, 'training_history.png')
            )
        
        # Load best model
        best_model_path = os.path.join(
            config.CHECKPOINT_DIR,
            f'{config.MODEL_NAME}_best.h5'
        )
        print(f"\nLoading best model from: {best_model_path}")
        sleep_apnea_model = keras.models.load_model(best_model_path)
        
    else:
        # Load existing model
        print("\n" + "="*60)
        print("STEP 5: LOADING EXISTING MODEL")
        print("="*60)
        
        if args.model_path:
            sleep_apnea_model = inference.load_trained_model(args.model_path)
        else:
            model_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'{config.MODEL_NAME}_best.h5'
            )
            sleep_apnea_model = inference.load_trained_model(model_path)
        
        history = None
    
    # ==================== STEP 6: EVALUATION ====================
    print("\n" + "="*60)
    print("STEP 6: MODEL EVALUATION")
    print("="*60)
    
    # Load training history if available
    if history is None and not args.skip_training:
        history_path = os.path.join(args.output_dir, 'training_history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
    
    # Generate comprehensive evaluation report
    metrics = evaluate.generate_evaluation_report(
        sleep_apnea_model,
        X_test, y_test,
        history=history if isinstance(history, dict) else (history.history if history else None),
        output_dir=config.PLOTS_DIR
    )
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("APNEAGUARD - EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                f.write(f"{key}: {value}\n")
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Visualize predictions
    if config.SAVE_PLOTS:
        print("\nVisualizing predictions on test set...")
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_pred_proba = sleep_apnea_model.predict(X_test_reshaped, verbose=0)
        y_pred = (y_pred_proba > config.CLASSIFICATION_THRESHOLD).astype(int).flatten()
        
        visualize.visualize_predictions(
            X_test, y_test, y_pred, y_pred_proba,
            num_samples=5,
            sampling_rate=config.SAMPLING_RATE,
            save_path=os.path.join(config.PLOTS_DIR, 'prediction_samples.png')
        )
    
    # ==================== STEP 7: SAVE FINAL MODEL ====================
    print("\n" + "="*60)
    print("STEP 7: SAVING FINAL MODEL")
    print("="*60)
    
    final_model_path = os.path.join(args.output_dir, 'final_model.h5')
    sleep_apnea_model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # ==================== COMPLETION ====================
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Plots saved to: {config.PLOTS_DIR}")
    print(f"\nModel Performance Summary:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("\n" + "="*60)
    print("Thank you for using ApneaGuard!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
