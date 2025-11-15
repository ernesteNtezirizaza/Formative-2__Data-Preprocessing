"""
Product Recommendation Prediction Function

This module provides a function to predict which product category a customer is most likely
to purchase based on their transaction data and profile.

Usage:
    # As a script
    python predict_product.py
    
    # As a function
    from scripts.predict_product import predict_product
    
    customer_data = {
        'customer_id': 151,
        'purchase_amount': 300,
        'customer_rating': 4.5,
        'engagement_score': 75,
        'purchase_interest_score': 3.5,
        'social_media_platform': 'Facebook',
        'review_sentiment': 'Positive',
        'purchase_date': '2024-01-15'
    }

"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


def predict_product(customer_data, results_dir=None, customer_history_path=None):
    """
    Predict which product category a customer is most likely to purchase.
    
    Parameters:
    -----------
    customer_data : dict
        Dictionary containing customer transaction data. Required keys:
        - purchase_amount: float
        - customer_rating: float (1-5)
        - engagement_score: float (0-100)
        - purchase_interest_score: float (1-5)
        - social_media_platform: str (Facebook, Twitter, Instagram, LinkedIn, TikTok)
        - review_sentiment: str (Positive, Neutral, Negative)
        - purchase_date: str (YYYY-MM-DD format)
        - customer_id: int (optional, for aggregation features)
    
    results_dir : str or Path, optional
        Path to directory containing saved model files. If None, uses default location
    
    customer_history_path : str or Path, optional
        Path to CSV file with customer transaction history. If provided and customer_id
        is in customer_data, will calculate aggregation features from history.
        If None, uses default location (data/processed/merged_customer_data.csv)
    
    Returns:
    --------
    tuple : (product_category, confidence)
        - product_category: str - Recommended product category (Books, Clothing, Electronics, Groceries, Sports)
        - confidence: float - Confidence score of the prediction (0.0 to 1.0)
    """
    try:
        # Set up paths
        if results_dir is None:
            script_dir = Path(__file__).parent
            root_dir = script_dir.parent
            results_dir = root_dir / "model_notebook" / "product_recommendation_results"
        else:
            results_dir = Path(results_dir)
        
        # Load model components
        model_path = results_dir / "product_recommendation_model.pkl"
        scaler_path = results_dir / "product_recommendation_scaler.pkl"
        encoders_path = results_dir / "product_recommendation_label_encoders.pkl"
        
        if not all([model_path.exists(), scaler_path.exists(), encoders_path.exists()]):
            print("Error: Model files not found. Please ensure the model has been trained.")
            return None, 0.0
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)
        
        # Convert customer_data to DataFrame for processing
        df = pd.DataFrame([customer_data])
        
        # Feature Engineering - Time Features
        df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
        df['purchase_year'] = df['purchase_date'].dt.year
        df['purchase_month'] = df['purchase_date'].dt.month
        df['purchase_day'] = df['purchase_date'].dt.day
        df['purchase_dayofweek'] = df['purchase_date'].dt.dayofweek
        df['is_weekend'] = df['purchase_dayofweek'].isin([5, 6]).astype(int)
        df['purchase_quarter'] = df['purchase_date'].dt.quarter
        
        # Feature Engineering - Customer Aggregations
        # If customer_id is provided and history is available, calculate aggregations
        customer_id = customer_data.get('customer_id', None)
        if customer_id is not None:
            if customer_history_path is None:
                script_dir = Path(__file__).parent
                root_dir = script_dir.parent
                customer_history_path = root_dir / "data" / "processed" / "merged_customer_data.csv"
            
            customer_history_path = Path(customer_history_path)
            
            if customer_history_path.exists():
                try:
                    history_df = pd.read_csv(customer_history_path)
                    customer_id = customer_data['customer_id']
                    
                    # Filter history for this customer
                    customer_history = history_df[history_df['customer_id'] == customer_id].copy()
                    
                    if len(customer_history) > 0:
                        # Calculate aggregations
                        customer_agg = customer_history.groupby('customer_id').agg({
                            'transaction_id': 'count',
                            'purchase_amount': ['mean', 'sum', 'std', 'min', 'max'],
                            'customer_rating': ['mean', 'std'],
                            'engagement_score': ['mean', 'max'],
                            'purchase_interest_score': ['mean', 'max']
                        }).reset_index()
                        
                        customer_agg.columns = ['customer_id', 'transaction_count',
                                              'avg_purchase_amount', 'total_spent', 'std_purchase_amount',
                                              'min_purchase_amount', 'max_purchase_amount',
                                              'avg_customer_rating', 'std_customer_rating',
                                              'avg_engagement_score', 'max_engagement_score',
                                              'avg_purchase_interest', 'max_purchase_interest']
                        
                        # Merge with current transaction
                        df = df.merge(customer_agg, on='customer_id', how='left')
                    else:
                        # No history, use defaults
                        df['transaction_count'] = 1
                        df['avg_purchase_amount'] = df['purchase_amount']
                        df['total_spent'] = df['purchase_amount']
                        df['std_purchase_amount'] = 0.0
                        df['min_purchase_amount'] = df['purchase_amount']
                        df['max_purchase_amount'] = df['purchase_amount']
                        df['avg_customer_rating'] = df['customer_rating']
                        df['std_customer_rating'] = 0.0
                        df['avg_engagement_score'] = df['engagement_score']
                        df['max_engagement_score'] = df['engagement_score']
                        df['avg_purchase_interest'] = df['purchase_interest_score']
                        df['max_purchase_interest'] = df['purchase_interest_score']
                except Exception as e:
                    print(f"Warning: Could not load customer history: {e}")
                    # Use defaults
                    df['transaction_count'] = 1
                    df['avg_purchase_amount'] = df['purchase_amount']
                    df['total_spent'] = df['purchase_amount']
                    df['std_purchase_amount'] = 0.0
                    df['min_purchase_amount'] = df['purchase_amount']
                    df['max_purchase_amount'] = df['purchase_amount']
                    df['avg_customer_rating'] = df['customer_rating']
                    df['std_customer_rating'] = 0.0
                    df['avg_engagement_score'] = df['engagement_score']
                    df['max_engagement_score'] = df['engagement_score']
                    df['avg_purchase_interest'] = df['purchase_interest_score']
                    df['max_purchase_interest'] = df['purchase_interest_score']
            else:
                # No history file, use defaults
                df['transaction_count'] = 1
                df['avg_purchase_amount'] = df['purchase_amount']
                df['total_spent'] = df['purchase_amount']
                df['std_purchase_amount'] = 0.0
                df['min_purchase_amount'] = df['purchase_amount']
                df['max_purchase_amount'] = df['purchase_amount']
                df['avg_customer_rating'] = df['customer_rating']
                df['std_customer_rating'] = 0.0
                df['avg_engagement_score'] = df['engagement_score']
                df['max_engagement_score'] = df['engagement_score']
                df['avg_purchase_interest'] = df['purchase_interest_score']
                df['max_purchase_interest'] = df['purchase_interest_score']
        else:
            # No customer_id provided, use defaults
            df['transaction_count'] = 1
            df['avg_purchase_amount'] = df['purchase_amount']
            df['total_spent'] = df['purchase_amount']
            df['std_purchase_amount'] = 0.0
            df['min_purchase_amount'] = df['purchase_amount']
            df['max_purchase_amount'] = df['purchase_amount']
            df['avg_customer_rating'] = df['customer_rating']
            df['std_customer_rating'] = 0.0
            df['avg_engagement_score'] = df['engagement_score']
            df['max_engagement_score'] = df['engagement_score']
            df['avg_purchase_interest'] = df['purchase_interest_score']
            df['max_purchase_interest'] = df['purchase_interest_score']
        
        # Feature Engineering - Interaction Features
        df['amount_rating_interaction'] = df['purchase_amount'] * df['customer_rating']
        df['engagement_interest_interaction'] = df['engagement_score'] * df['purchase_interest_score']
        df['amount_per_transaction'] = df['total_spent'] / df['transaction_count']
        
        # Feature Engineering - Encoding Categorical Variables
        le_platform = label_encoders.get('platform')
        le_sentiment = label_encoders.get('sentiment')
        le_amount_cat = label_encoders.get('amount_category')
        
        if le_platform:
            # Handle unknown platforms
            try:
                df['platform_encoded'] = le_platform.transform(df['social_media_platform'])
            except ValueError:
                # Unknown platform, use most common or default
                df['platform_encoded'] = 0  # Default to first encoded value
        else:
            df['platform_encoded'] = 0
        
        if le_sentiment:
            # Handle unknown sentiments
            try:
                df['sentiment_encoded'] = le_sentiment.transform(df['review_sentiment'])
            except ValueError:
                df['sentiment_encoded'] = 1  # Default to Neutral
        else:
            df['sentiment_encoded'] = 1
        
        # Feature Engineering - Binned Features
        df['amount_category'] = pd.cut(df['purchase_amount'],
                                      bins=[0, 100, 300, 500],
                                      labels=['Low', 'Medium', 'High'])
        if le_amount_cat:
            try:
                df['amount_category_encoded'] = le_amount_cat.transform(df['amount_category'].astype(str))
            except ValueError:
                df['amount_category_encoded'] = 1  # Default to Medium
        else:
            df['amount_category_encoded'] = 1
        
        # Select features in the same order as training
        feature_cols = [
            'purchase_amount', 'customer_rating', 'engagement_score', 'purchase_interest_score',
            'platform_encoded', 'sentiment_encoded',
            'purchase_month', 'purchase_dayofweek', 'is_weekend', 'purchase_quarter',
            'transaction_count', 'avg_purchase_amount', 'total_spent', 'std_purchase_amount',
            'avg_customer_rating', 'avg_engagement_score', 'max_engagement_score',
            'avg_purchase_interest', 'max_purchase_interest',
            'amount_rating_interaction', 'engagement_interest_interaction',
            'amount_per_transaction', 'amount_category_encoded'
        ]
        
        X = df[feature_cols].copy()
        X = X.fillna(0)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        y_pred = model.predict(X_scaled)[0]
        y_pred_proba = model.predict_proba(X_scaled)[0]
        
        # Get confidence (probability of predicted class)
        confidence = float(np.max(y_pred_proba))
        
        # Get predicted product category
        le_target = label_encoders.get('target')
        if le_target:
            predicted_product = le_target.inverse_transform([y_pred])[0]
        else:
            # Fallback if target encoder not found
            product_categories = ['Books', 'Clothing', 'Electronics', 'Groceries', 'Sports']
            predicted_product = product_categories[y_pred] if y_pred < len(product_categories) else 'Unknown'
        
        return predicted_product, confidence
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0.0


def main():
    """Example usage of the predict_product function"""
    import sys
    
    # Example customer data
    customer_data = {
        'customer_id': 151,
        'purchase_amount': 300.0,
        'customer_rating': 4.5,
        'engagement_score': 75.0,
        'purchase_interest_score': 3.5,
        'social_media_platform': 'Facebook',
        'review_sentiment': 'Positive',
        'purchase_date': '2024-01-15'
    }
    
    print("=" * 70)
    print("PRODUCT RECOMMENDATION PREDICTION")
    print("=" * 70)
    print("\nCustomer Data:")
    for key, value in customer_data.items():
        print(f"  {key}: {value}")
    print()
    
    product, confidence = predict_product(customer_data)
    
    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print(f"Recommended Product: {product}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("=" * 70)
    
    if product:
        print(f"\nRecommended product category: {product}")
    else:
        print("\nCould not make a prediction.")


if __name__ == "__main__":
    main()

