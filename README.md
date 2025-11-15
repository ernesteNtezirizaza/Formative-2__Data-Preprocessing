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
│   └── system_simulation.py       # Complete system simulation
└── README.md
```

## Running the System from Scratch

### Prerequisites

1. **Python 3.7+** installed on your system
2. **Required Python packages** (see Requirements section below)

### Step 1: Install Dependencies

```bash
# Install all required packages
pip install pandas numpy scikit-learn xgboost opencv-python librosa matplotlib seaborn joblib
```

### Step 2: Prepare Data

The system requires processed data files. If starting from scratch:

#### 2.1 Merge Customer Data

```bash
# Merge customer social profiles and transactions
python scripts/data_merging.py
```

This creates `data/processed/merged_customer_data.csv` which is used for product recommendation.

#### 2.2 Process Images

1. Place your images in `Image_Processing/Images/` (3 images per member: Neutral, Smile, Surprised)
2. Run the image processing notebook:
   ```bash
   jupyter notebook Image_Processing/image_processing.ipynb
   ```
   This will:
   - Apply augmentations to images
   - Extract features (color histograms, statistical features, texture features)
   - Generate `Image_Processing/image_features.csv`

#### 2.3 Process Audio

1. Place your audio files in `Audio_Processing/Audios/` (2 phrases per member)
2. Run the audio processing notebook:
   ```bash
   jupyter notebook Audio_Processing/audio_processing.ipynb
   ```
   This will:
   - Apply augmentations to audio files
   - Extract features (MFCCs, spectral roll-off, energy, etc.)
   - Generate `Audio_Processing/audio_features.csv`

### Step 3: Train Models

Train all three models using the Jupyter notebooks:

#### 3.1 Facial Recognition Model

```bash
jupyter notebook model_notebook/facial_recognition.ipynb
```

This trains a RandomForestClassifier and saves:
- `model_notebook/facial_recognition_results/facial_recognition_model.pkl`
- `model_notebook/facial_recognition_results/facial_recognition_scaler.pkl`
- `model_notebook/facial_recognition_results/facial_recognition_label_encoder.pkl`
- Performance metrics in JSON format

#### 3.2 Voice Print Verification Model

```bash
jupyter notebook model_notebook/voice_print_verification.ipynb
```

This trains a LogisticRegression model and saves:
- `model_notebook/voice_print_verification_results/voice_print_verification_model.pkl`
- `model_notebook/voice_print_verification_results/voice_print_verification_scaler.pkl`
- `model_notebook/voice_print_verification_results/voice_print_verification_label_encoder.pkl`
- Performance metrics in JSON format

#### 3.3 Product Recommendation Model

```bash
jupyter notebook model_notebook/product_recommendation.ipynb
```

This trains an XGBoost model (best performing) and saves:
- `model_notebook/product_recommendation_results/product_recommendation_model.pkl`
- `model_notebook/product_recommendation_results/product_recommendation_scaler.pkl`
- `model_notebook/product_recommendation_results/product_recommendation_label_encoders.pkl`
- Performance metrics in JSON format

### Step 4: Run the System

Once all models are trained, you can run the complete system:

#### 4.1 Interactive Mode (Recommended)

```bash
# Run the system simulation in interactive mode
python scripts/system_simulation.py
```

The system will:
1. Prompt you to enter an image name (searches in `Images/` and `Augmented_Images/`)
2. **Run facial recognition immediately** - if face is not recognized or not authorized, stops here
3. If face recognized, prompt for customer data (purchase amount, rating, etc.)
4. Generate product recommendation (name hidden until voice validation)
5. Prompt for audio file name (searches in `Audios/` and `Augmented_Audios/`)
6. Run voice validation
7. **Display product only if voice matches face**

#### 4.2 Command-Line Mode

```bash
# With image and audio file names (searches in Images/Audios folders)
python scripts/system_simulation.py
```

### Step 5: Test Individual Models

You can also test individual models separately:

```bash
# Facial recognition
python scripts/predict_face.py "Image_Processing/Images/Erneste_Neutral.jpg"

# Voice validation
python scripts/predict_voice.py "Audio_Processing/Audios/Erneste_yes_approve.wav"

# Product recommendation (uses default customer data)
python scripts/predict_product.py
```

## Quick Start (If Models Already Trained)

If the models are already trained and saved, you can skip Steps 2-3 and go directly to Step 4.

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

### Python Packages

Install required packages:

```bash
pip install pandas numpy scikit-learn xgboost opencv-python librosa matplotlib seaborn joblib
```

### Required Files Structure

For the system to run, ensure you have:

1. **Trained Models** (in `model_notebook/*_results/` folders):
   - Facial recognition model files
   - Voice verification model files
   - Product recommendation model files

2. **Feature Files**:
   - `Image_Processing/image_features.csv`
   - `Audio_Processing/audio_features.csv`
   - `data/processed/merged_customer_data.csv`

3. **Test Files**:
   - Images in `Image_Processing/Images/` or `Image_Processing/Augmented_Images/`
   - Audio files in `Audio_Processing/Audios/` or `Audio_Processing/Augmented_Audios/`

## Team Members

- Erneste Ntezirizaza
- Thierry Shyaka
- Idara Patrick
- Rodas Goniche
