"""
Voice Print Verification Prediction Function

This module provides a function to predict which person (Erneste, Thierry, Idara, or Rodas)
is speaking in a given audio file using the trained voice print verification model.

Usage:
    # As a script
    python predict_voice.py <audio_path>
    
    # As a function
    from scripts.predict_voice import predict_voice
    
    person, confidence = predict_voice("path/to/audio.wav")
    if person:
        print(f"Identified as: {person} (confidence: {confidence:.2%})")
    else:
        print(f"Person not recognized (confidence: {confidence:.2%})")
"""

import pandas as pd
import numpy as np
import librosa
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def extract_features_from_audio(y, sr):
    """
    Extract features from an audio signal (same as training).
    
    Parameters:
    -----------
    y : numpy.ndarray
        Audio time series
    sr : int
        Sample rate
    
    Returns:
    --------
    dict : Dictionary of extracted features
    """
    features = {}
    
    # MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    for i in range(13):
        features[f'mfcc_{i}'] = mfccs_mean[i]
    
    # MFCCs standard deviation
    mfccs_std = np.std(mfccs, axis=1)
    for i in range(13):
        features[f'mfcc_std_{i}'] = mfccs_std[i]
    
    # Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(rolloff)
    features['spectral_rolloff_std'] = np.std(rolloff)
    
    # Energy/RMS
    rms = librosa.feature.rms(y=y)
    features['energy_mean'] = np.mean(rms)
    features['energy_std'] = np.std(rms)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(centroid)
    features['spectral_centroid_std'] = np.std(centroid)
    
    # Chroma Features (12 chroma bins)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    for i in range(12):
        features[f'chroma_{i}'] = chroma_mean[i]
    
    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(bandwidth)
    features['spectral_bandwidth_std'] = np.std(bandwidth)
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast_mean'] = np.mean(contrast)
    features['spectral_contrast_std'] = np.std(contrast)
    
    return features


def predict_voice(audio_path, confidence_threshold=0.8, results_dir=None):
    """
    Predict which person is speaking in the audio file.
    
    Parameters:
    -----------
    audio_path : str or Path
        Path to the audio file
    confidence_threshold : float, default=0.8
        Minimum confidence score to return a prediction. If confidence is below this,
        returns None. Known voices typically score 85-100%, so 0.8 helps filter unknown voices.
    results_dir : str or Path, optional
        Path to directory containing saved model files. If None, uses default location
    
    Returns:
    --------
    tuple : (person_name, confidence) or (None, confidence)
        - person_name: str or None - Name of the person (Erneste, Thierry, Idara, Rodas) or None
        - confidence: float - Confidence score of the prediction (0.0 to 1.0)
    """
    try:
        # Set up paths
        if results_dir is None:
            script_dir = Path(__file__).parent
            root_dir = script_dir.parent
            results_dir = root_dir / "model_notebook" / "voice_print_verification_results"
        else:
            results_dir = Path(results_dir)
        
        audio_path = Path(audio_path)
        
        # Check if audio file exists
        if not audio_path.exists():
            print(f"Error: Audio file not found at {audio_path}")
            return None, 0.0
        
        # Load model components
        model_path = results_dir / "voice_print_verification_model.pkl"
        scaler_path = results_dir / "voice_print_verification_feature_scaler.pkl"
        encoder_path = results_dir / "voice_print_verification_label_encoder.pkl"
        
        if not all([model_path.exists(), scaler_path.exists(), encoder_path.exists()]):
            print("Error: Model files not found. Please ensure the model has been trained.")
            return None, 0.0
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        
        # Load and process audio
        try:
            y, sr = librosa.load(str(audio_path), sr=None)
        except Exception as e:
            print(f"Error: Could not load audio from {audio_path}: {str(e)}")
            return None, 0.0
        
        # Extract features
        features = extract_features_from_audio(y, sr)
        
        # Create feature dictionary with all required fields
        feature_dict = {
            'sample_rate': sr,
            'duration': len(y) / sr,
        }
        
        # Add all extracted features
        for key, value in features.items():
            feature_dict[key] = value
        
        # Add categorical columns with default values (will be used for column alignment)
        feature_dict['phrase'] = 'yes_approve'  # Default value
        feature_dict['augmentation'] = 'Original'  # Default value
        feature_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create DataFrame
        df = pd.DataFrame([feature_dict])
        
        # Load training data structure to ensure column alignment
        script_dir = Path(__file__).parent
        root_dir = script_dir.parent
        training_data_path = root_dir / "Audio_Processing" / "audio_features.csv"
        
        if not training_data_path.exists():
            print("Error: Training data file not found. Cannot align feature columns.")
            return None, 0.0
        
        # Load ALL training data to get the complete column structure
        training_df = pd.read_csv(training_data_path)
        
        # Prepare features (same as model training)
        # Select only numerical features for modeling
        numerical_cols = training_df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove non-feature columns
        non_feature_cols = ['sample_rate', 'duration']
        numerical_cols = [col for col in numerical_cols if col not in non_feature_cols]
        
        training_X = training_df[numerical_cols]
        
        # Prepare prediction features
        X = df[numerical_cols]
        
        # Handle missing values (same as training)
        X = X.fillna(X.median())
        
        # Align columns with training data structure
        # Add missing columns (fill with 0) and remove extra columns
        X = X.reindex(columns=training_X.columns, fill_value=0)
        
        # Fill any remaining missing values
        X = X.fillna(X.median())
        
        # Scale features (same scaler as training)
        X_scaled = scaler.transform(X)
        
        # Make prediction
        y_pred = model.predict(X_scaled)[0]
        y_pred_proba = model.predict_proba(X_scaled)[0]
        
        # Get confidence (probability of predicted class)
        confidence = float(np.max(y_pred_proba))
        
        # Get predicted person name
        predicted_person = label_encoder.inverse_transform([y_pred])[0]
        
        # Verify the predicted person exists in training data
        # Get list of known persons from training data
        known_persons = set(training_df['member'].unique())
        
        # Return None if:
        # 1. Confidence is below threshold, OR
        # 2. Predicted person is not in training data (shouldn't happen, but safety check)
        if confidence < confidence_threshold:
            return None, confidence
        elif predicted_person not in known_persons:
            # This shouldn't happen if model is working correctly, but safety check
            print(f"Warning: Predicted person '{predicted_person}' not found in training data.")
            print(f"Known persons: {sorted(known_persons)}")
            return None, confidence
        else:
            return predicted_person, confidence
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0.0


def main():
    """Example usage of the predict_voice function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_voice.py <audio_path>")
        print("\nExample:")
        print("  python predict_voice.py ../Audio_Processing/Audios/Erneste_yes_approve.wav")
        return
    
    audio_path = sys.argv[1]
    confidence_threshold = 0.8  # Default threshold for unknown voice detection
    
    print(f"Predicting voice in: {audio_path}")
    print(f"Confidence threshold: {confidence_threshold}\n")
    
    person, confidence = predict_voice(audio_path, confidence_threshold=confidence_threshold)
    
    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print(f"Person: {person}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("=" * 70)
    
    if person is None:
        print("The person in the audio may not be one of the trained members.")
        print("Trained members: Erneste, Thierry, Idara, Rodas")
        print(f"If this person should be recognized, add their audio samples to the training data.")
    else:
        print(f"\n Identified as: {person}")


if __name__ == "__main__":
    main()

