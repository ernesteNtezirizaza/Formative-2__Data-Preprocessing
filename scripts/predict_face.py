"""
Facial Recognition Prediction Function

This module provides a function to predict which person (Erneste, Thierry, Idara, or Rodas)
is in a given image using the trained facial recognition model.

Usage:
    # As a script
    python predict_face.py <image_path> [confidence_threshold]
    
    # As a function
    from scripts.predict_face import predict_face
    
    person, confidence = predict_face("path/to/image.jpg")
    if person:
        print(f"Identified as: {person} (confidence: {confidence:.2%})")
    else:
        print(f"Person not recognized (confidence: {confidence:.2%})")
"""

import pandas as pd
import numpy as np
import cv2
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def extract_features_from_image(image):
    """
    Extract features from an image (same as training).
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image array loaded with cv2
    
    Returns:
    --------
    dict : Dictionary of extracted features
    """
    features = {}
    
    # Image dimensions
    features['height'] = image.shape[0]
    features['width'] = image.shape[1]
    features['channels'] = image.shape[2] if len(image.shape) == 3 else 1
    
    # Convert to RGB for consistent processing
    if len(image.shape) == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image
    
    # Color histogram features (8 bins per channel)
    if len(image.shape) == 3:
        for i, color in enumerate(['R', 'G', 'B']):
            hist = cv2.calcHist([img_rgb], [i], None, [8], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            for j, val in enumerate(hist):
                features[f'hist_{color}_{j}'] = val
    
    # Grayscale histogram features
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    hist_gray = cv2.calcHist([gray], [0], None, [8], [0, 256])
    hist_gray = hist_gray.flatten() / hist_gray.sum()
    for j, val in enumerate(hist_gray):
        features[f'hist_gray_{j}'] = val
    
    # Statistical features
    features['mean_intensity'] = np.mean(gray)
    features['std_intensity'] = np.std(gray)
    features['min_intensity'] = np.min(gray)
    features['max_intensity'] = np.max(gray)
    features['median_intensity'] = np.median(gray)
    
    # Texture features (edge density)
    edges = cv2.Canny(gray, 100, 200)
    features['edge_density'] = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    
    return features


def predict_face(image_path, confidence_threshold=0.8, results_dir=None):
    """
    Predict which person is in the image.
    
    Parameters:
    -----------
    image_path : str or Path
        Path to the image file
    confidence_threshold : float, default=0.8
        Minimum confidence score to return a prediction. If confidence is below this,
        returns None. Known faces typically score 85-100%, so 0.8 helps filter unknown faces.
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
            results_dir = root_dir / "model_notebook" / "facial_recognition_results"
        else:
            results_dir = Path(results_dir)
        
        image_path = Path(image_path)
        
        # Check if image exists
        if not image_path.exists():
            print(f"Error: Image file not found at {image_path}")
            return None, 0.0
        
        # Load model components
        model_path = results_dir / "facial_recognition_model.pkl"
        scaler_path = results_dir / "facial_recognition_scaler.pkl"
        encoder_path = results_dir / "facial_recognition_label_encoder.pkl"
        
        if not all([model_path.exists(), scaler_path.exists(), encoder_path.exists()]):
            print("Error: Model files not found. Please ensure the model has been trained.")
            return None, 0.0
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        
        # Load and process image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None, 0.0
        
        # Extract features
        features = extract_features_from_image(image)
        
        # Convert to DataFrame (single row)
        # We need to include the same categorical columns that were used during training
        # Create a DataFrame with all expected columns
        feature_dict = {
            'height': features['height'],
            'width': features['width'],
            'channels': features['channels'],
        }
        
        # Add histogram features
        for color in ['R', 'G', 'B']:
            for j in range(8):
                feature_dict[f'hist_{color}_{j}'] = features.get(f'hist_{color}_{j}', 0.0)
        
        for j in range(8):
            feature_dict[f'hist_gray_{j}'] = features.get(f'hist_gray_{j}', 0.0)
        
        # Add statistical features
        feature_dict['mean_intensity'] = features['mean_intensity']
        feature_dict['std_intensity'] = features['std_intensity']
        feature_dict['min_intensity'] = features['min_intensity']
        feature_dict['max_intensity'] = features['max_intensity']
        feature_dict['median_intensity'] = features['median_intensity']
        feature_dict['edge_density'] = features['edge_density']
        
        # Add categorical columns with default values (will be one-hot encoded)
        # These need to match what was used during training
        feature_dict['expression'] = 'Neutral'  # Default value
        feature_dict['augmentation'] = 'Original'  # Default value
        feature_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create DataFrame
        df = pd.DataFrame([feature_dict])
        
        # Load training data structure to ensure column alignment
        script_dir = Path(__file__).parent
        root_dir = script_dir.parent
        training_data_path = root_dir / "Image_Processing" / "image_features.csv"
        
        if not training_data_path.exists():
            print("Error: Training data file not found. Cannot align feature columns.")
            return None, 0.0
        
        # Load ALL training data to get the complete column structure after one-hot encoding
        training_df = pd.read_csv(training_data_path)
        training_X = training_df.drop(columns=['member'])
        
        # Prepare training features (same as model training) - use full dataset to get all columns
        training_categorical = training_X.select_dtypes(include=['object']).columns
        if len(training_categorical) > 0:
            training_X_encoded = pd.get_dummies(training_X, columns=training_categorical, drop_first=True)
        else:
            training_X_encoded = training_X
        
        # Prepare prediction features (same preprocessing)
        X = df.drop(columns=['timestamp'])  # Drop timestamp as it's not used in training
        
        # Handle categorical features (one-hot encoding) - must match training structure
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Align columns with training data structure
        # Add missing columns (fill with 0) and remove extra columns
        X = X.reindex(columns=training_X_encoded.columns, fill_value=0)
        
        # Fill missing values
        X = X.fillna(X.mean())
        
        # Scale features
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
    """Example usage of the predict_face function"""
    import sys
    
    if len(sys.argv) < 2:
        print("\nExample:")
        print("  python predict_face.py ../Image_Processing/Images/Erneste_Neutral.jpg")
        return
    
    image_path = sys.argv[1]
    confidence_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    
    print(f"Predicting face in: {image_path}")
    print(f"Confidence threshold: {confidence_threshold}\n")
    
    person, confidence = predict_face(image_path, confidence_threshold=confidence_threshold)
    
    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print(f"Person: {person}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("=" * 70)
    
    if person is None:
        print("The person in the image may not be one of the trained members.")
        print("Trained members: Erneste, Thierry, Idara, Rodas")
        print(f"If this person should be recognized, add their images to the training data.")
    else:
        print(f"\n✓ Identified as: {person}")


if __name__ == "__main__":
    main()

