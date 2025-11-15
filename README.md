# Formative-2: Data-Preprocessing

## User Identity and Product Recommendation System

This project implements a multimodal authentication and recommendation system that combines facial recognition, voice verification, and product recommendation models.

## System Flow

The system follows this sequential flow:
1. **Facial Recognition** → Authenticates user identity
2. **Product Recommendation** → Generates product suggestion (if face recognized)
3. **Voice Validation** → Final approval step
4. **Display Product** → Shows recommended product (if all checks pass)

If any authentication step fails, access is denied.

## Project Structure

```
Formative-2__Data-Preprocessing/
├── data/
│   ├── raw/
│   │   ├── customer_social_profiles.csv
│   │   └── customer_transactions.csv
│   └── processed/
│       └── merged_customer_data.csv
├── Image_Processing/
│   ├── Images/                    # Original images (3 per member)
│   ├── Augmented_Images/          # Augmented images
│   ├── Visualizations/            # Image visualizations
│   ├── image_features.csv         # Extracted image features
│   └── image_processing.ipynb     # Image processing notebook
├── Audio_Processing/
│   ├── Audios/                    # Original audio files (2 per member)
│   ├── Augmented_Audios/          # Augmented audio files
│   ├── Visualizations/            # Audio visualizations
│   ├── audio_features.csv         # Extracted audio features
│   └── audio_processing.ipynb     # Audio processing notebook
├── model_notebook/
│   ├── facial_recognition.ipynb
│   ├── facial_recognition_results/
│   ├── voice_print_verification.ipynb
│   ├── voice_print_verification_results/
│   ├── product_recommendation.ipynb
│   └── product_recommendation_results/
├── scripts/
│   ├── data_merging.py            # Data merge pipeline
│   ├── predict_face.py            # Facial recognition prediction
│   ├── predict_voice.py           # Voice validation prediction
│   ├── predict_product.py         # Product recommendation prediction
│   └── system_simulation.py       # Complete system simulation ⭐
└── README.md
```

## Quick Start

### 1. System Simulation (Main Demo)

Run the complete system flow:

```bash
# Full transaction with default test data
python scripts/system_simulation.py

# With custom image and audio
python scripts/system_simulation.py --image "Image_Processing/Images/Erneste_Neutral.jpg" --audio "Audio_Processing/Audios/Erneste_yes_approve.wav"

# Simulate unauthorized attempt
python scripts/system_simulation.py --unauthorized

# With custom customer data for product recommendation
python scripts/system_simulation.py --image "Image_Processing/Images/Thierry_Smile.jpg" --audio "Audio_Processing/Audios/Thierry_confirm_transaction.wav" --amount 250 --rating 4.0
```

### 2. Individual Model Predictions

```bash
# Facial recognition
python scripts/predict_face.py "Image_Processing/Images/Erneste_Neutral.jpg"

# Voice validation
python scripts/predict_voice.py "Audio_Processing/Audios/Erneste_yes_approve.wav"

# Product recommendation
python scripts/predict_product.py
```

## Models

### 1. Facial Recognition Model
- **Type**: RandomForestClassifier
- **Purpose**: Identifies person from facial image
- **Output**: Person name (Erneste, Thierry, Idara, Rodas) or None
- **Notebook**: `model_notebook/facial_recognition.ipynb`
- **Script**: `scripts/predict_face.py`

### 2. Voice Print Verification Model
- **Type**: LogisticRegression
- **Purpose**: Verifies person identity from voice sample
- **Output**: Person name or None
- **Notebook**: `model_notebook/voice_print_verification.ipynb`
- **Script**: `scripts/predict_voice.py`

### 3. Product Recommendation Model
- **Type**: XGBoost (best performing)
- **Purpose**: Recommends product category based on customer data
- **Output**: Product category (Books, Clothing, Electronics, Groceries, Sports)
- **Notebook**: `model_notebook/product_recommendation.ipynb`
- **Script**: `scripts/predict_product.py`

## Data Processing

### Image Processing
- **Input**: 3 images per member (Neutral, Smile, Surprised)
- **Augmentations**: Rotation, flipping, grayscale, brightness, etc.
- **Features**: Color histograms, statistical features, texture features
- **Output**: `Image_Processing/image_features.csv`

### Audio Processing
- **Input**: 2 audio phrases per member ("Yes, approve", "Confirm transaction")
- **Augmentations**: Pitch shift, time stretch, background noise, etc.
- **Features**: MFCCs, spectral roll-off, energy, zero-crossing rate, chroma, etc.
- **Output**: `Audio_Processing/audio_features.csv`

### Data Merging
- **Sources**: Customer social profiles + Customer transactions
- **Output**: `data/processed/merged_customer_data.csv`
- **Script**: `scripts/data_merging.py`

## Evaluation Metrics

All models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Log Loss**: Logarithmic loss for probability predictions

Performance metrics are saved in JSON format in each model's results folder.

## Requirements

Install required packages:

```bash
pip install pandas numpy scikit-learn xgboost opencv-python librosa matplotlib seaborn joblib
```

## Assignment Checklist

See `ASSIGNMENT_CHECKLIST.md` for a complete checklist of all assignment requirements.

## Team Members

- [List team members and their contributions]

## License

[Add license information if applicable]
