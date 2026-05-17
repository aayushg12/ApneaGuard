"""
Inference module for ApneaGuard.
Load trained models and make predictions on new ECG data.
"""

import numpy as np
from tensorflow import keras
import os
import config
import data_loader
import preprocessing


def load_trained_model(model_path: str) -> keras.Model:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model file (.h5)
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    print("Model loaded successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    return model


def predict_single_segment(model: keras.Model, segment: np.ndarray) -> tuple:
    """
    Make prediction on a single ECG segment.
    
    Args:
        model: Trained Keras model
        segment: ECG segment array
        
    Returns:
        Tuple of (prediction, probability)
            - prediction: 0 (Normal) or 1 (Apnea)
            - probability: Confidence score [0, 1]
    """
    # Reshape for model input
    segment_reshaped = segment.reshape(1, len(segment), 1)
    
    # Get prediction
    probability = model.predict(segment_reshaped, verbose=0)[0][0]
    
    # Apply threshold
    prediction = 1 if probability > config.CLASSIFICATION_THRESHOLD else 0
    
    return prediction, probability


def predict_segments(model: keras.Model, segments: np.ndarray, 
                    smooth: bool = False) -> tuple:
    """
    Make predictions on multiple ECG segments.
    
    Args:
        model: Trained Keras model
        segments: Array of ECG segments
        smooth: Whether to apply smoothing to predictions
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    # Reshape for model input
    segments_reshaped = segments.reshape(segments.shape[0], segments.shape[1], 1)
    
    # Get predictions
    probabilities = model.predict(segments_reshaped, verbose=0)
    
    # Apply smoothing if requested
    if smooth and config.SMOOTH_PREDICTIONS:
        probabilities = smooth_predictions(probabilities, config.SMOOTHING_WINDOW)
    
    # Convert to binary predictions
    predictions = (probabilities > config.CLASSIFICATION_THRESHOLD).astype(int).flatten()
    
    return predictions, probabilities


def smooth_predictions(probabilities: np.ndarray, window_size: int) -> np.ndarray:
    """
    Smooth predictions using moving average.
    
    This reduces jitter in consecutive predictions by averaging
    over a sliding window of predictions.
    
    Args:
        probabilities: Array of prediction probabilities
        window_size: Size of smoothing window
        
    Returns:
        Smoothed probabilities
    """
    if window_size <= 1:
        return probabilities
    
    # Apply moving average
    smoothed = np.copy(probabilities)
    
    for i in range(len(probabilities)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(probabilities), i + window_size // 2 + 1)
        smoothed[i] = np.mean(probabilities[start_idx:end_idx])
    
    return smoothed


def predict_from_file(model: keras.Model, data_dir: str, record_name: str) -> dict:
    """
    Load ECG from file, preprocess, and make predictions.
    
    Args:
        model: Trained Keras model
        data_dir: Directory containing ECG files
        record_name: Name of record to process
        
    Returns:
        Dictionary with predictions and metadata
    """
    print(f"\n{'='*50}")
    print(f"Processing record: {record_name}")
    print(f"{'='*50}")
    
    # Load ECG signal
    ecg_signal, metadata = data_loader.load_ecg_signal(data_dir, record_name)
    
    # Load annotations if available (for comparison)
    annotations = data_loader.load_annotations(data_dir, record_name)
    
    # Preprocess signal
    print("\nPreprocessing signal...")
    segments, segment_labels = preprocessing.preprocess_record(
        ecg_signal, annotations, int(metadata['sampling_rate'])
    )
    
    # Make predictions
    print("\nMaking predictions...")
    predictions, probabilities = predict_segments(
        model, segments, smooth=config.SMOOTH_PREDICTIONS
    )
    
    # Calculate statistics
    num_apnea = np.sum(predictions == 1)
    num_normal = np.sum(predictions == 0)
    apnea_percentage = 100 * num_apnea / len(predictions)
    
    print(f"\nPrediction Results:")
    print(f"  Total segments: {len(predictions)}")
    print(f"  Apnea segments: {num_apnea} ({apnea_percentage:.1f}%)")
    print(f"  Normal segments: {num_normal} ({100-apnea_percentage:.1f}%)")
    
    # If annotations available, compare
    if segment_labels is not None:
        accuracy = np.mean(predictions == segment_labels)
        print(f"\n  Accuracy (vs annotations): {accuracy:.4f}")
    
    results = {
        'record_name': record_name,
        'segments': segments,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': segment_labels,
        'metadata': metadata,
        'num_apnea': num_apnea,
        'num_normal': num_normal,
        'apnea_percentage': apnea_percentage
    }
    
    return results


def generate_prediction_report(results: dict, output_path: str = None):
    """
    Generate a text report of prediction results.
    
    Args:
        results: Results dictionary from predict_from_file
        output_path: Path to save report (None = print only)
    """
    report = []
    report.append("="*60)
    report.append("APNEAGUARD - SLEEP APNEA DETECTION REPORT")
    report.append("="*60)
    report.append("")
    report.append(f"Record: {results['record_name']}")
    report.append(f"Duration: {results['metadata']['duration_seconds']:.1f} seconds")
    report.append(f"Sampling Rate: {results['metadata']['sampling_rate']} Hz")
    report.append("")
    report.append("-"*60)
    report.append("PREDICTION SUMMARY")
    report.append("-"*60)
    report.append(f"Total segments analyzed: {len(results['predictions'])}")
    report.append(f"Apnea segments detected: {results['num_apnea']} ({results['apnea_percentage']:.1f}%)")
    report.append(f"Normal segments: {results['num_normal']} ({100-results['apnea_percentage']:.1f}%)")
    report.append("")
    
    # If ground truth available
    if results['true_labels'] is not None:
        accuracy = np.mean(results['predictions'] == results['true_labels'])
        report.append("-"*60)
        report.append("VALIDATION (vs Ground Truth)")
        report.append("-"*60)
        report.append(f"Accuracy: {accuracy:.4f}")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        tn, fp, fn, tp = cm.ravel()
        
        report.append(f"True Positives (Apnea detected correctly): {tp}")
        report.append(f"False Positives (Normal detected as Apnea): {fp}")
        report.append(f"False Negatives (Apnea detected as Normal): {fn}")
        report.append(f"True Negatives (Normal detected correctly): {tn}")
        report.append("")
    
    # Confidence statistics
    report.append("-"*60)
    report.append("CONFIDENCE STATISTICS")
    report.append("-"*60)
    avg_confidence_apnea = np.mean(results['probabilities'][results['predictions'] == 1])
    avg_confidence_normal = np.mean(1 - results['probabilities'][results['predictions'] == 0])
    
    report.append(f"Average confidence (Apnea): {avg_confidence_apnea:.4f}")
    report.append(f"Average confidence (Normal): {avg_confidence_normal:.4f}")
    report.append("")
    report.append("="*60)
    
    # Join and print
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_path}")
    
    return report_text


def batch_inference(model: keras.Model, data_dir: str, 
                   record_names: list = None, output_dir: str = None) -> list:
    """
    Run inference on multiple records.
    
    Args:
        model: Trained Keras model
        data_dir: Directory containing ECG files
        record_names: List of record names (None = all available)
        output_dir: Directory to save reports
        
    Returns:
        List of results dictionaries
    """
    if record_names is None:
        record_names = data_loader.get_record_list(data_dir)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for i, record_name in enumerate(record_names, 1):
        print(f"\n[{i}/{len(record_names)}] Processing {record_name}...")
        
        try:
            results = predict_from_file(model, data_dir, record_name)
            all_results.append(results)
            
            # Generate report
            if output_dir:
                report_path = os.path.join(output_dir, f"{record_name}_report.txt")
                generate_prediction_report(results, report_path)
                
        except Exception as e:
            print(f"Error processing {record_name}: {str(e)}")
            continue
    
    return all_results


if __name__ == "__main__":
    """
    Test inference pipeline.
    """
    import model
    
    print("Testing inference module...")
    
    # Create a dummy model for testing
    window_samples = config.WINDOW_SIZE * config.SAMPLING_RATE
    input_shape = (window_samples, 1)
    test_model = model.create_model('cnn', input_shape)
    
    # Create dummy segment
    dummy_segment = np.random.randn(window_samples)
    
    print("\nTesting single segment prediction...")
    prediction, probability = predict_single_segment(test_model, dummy_segment)
    print(f"Prediction: {config.CLASS_NAMES[prediction]}")
    print(f"Probability: {probability:.4f}")
    
    # Create dummy segments
    dummy_segments = np.random.randn(10, window_samples)
    
    print("\nTesting batch prediction...")
    predictions, probabilities = predict_segments(test_model, dummy_segments)
    print(f"Predictions: {predictions}")
    
    print("\nInference test complete!")
