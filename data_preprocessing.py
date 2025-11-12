"""
Data Merging and Feature Engineering Script
Prepares dataset for product recommendation modeling
Saves the final dataset to a CSV file
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_data_folders():
    """ Create necessary folders if they don't exist"""
    folders = ['data/raw', 'data/processed']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Created folder: {folder}")

def load_data():
    """ Load the datasets given from their folder paths"""
    try:
        social_df = pd.read_csv('data/raw/social_profiles.csv')
        transactions_df = pd.read_csv('data/raw/transactions.csv')
        logger.info(f"Social Profiles successfully loaded: {social_df.shape}")
        logger.info(f"Transaction records successfully loaded: {transactions_df.shape}")
        return social_df, transactions_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_social_data(social_df):
    """ Clean social profiles dataset"""
    logger.info("Cleaning social profiles data...")
    df_clean = social_df.copy()

    # handle missing values
    for col in ['social_media_platform', 'engagement_score',
                'purchase_interest_score', 'review_sentiment']:
        
        if col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col].fillna('Unknown', inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)

    # remove duplicates
    df_clean.drop_duplicates(inplace=True)

    # standardize sentiment text
    if 'review_sentiment' in df_clean.columns:
        df_clean['review_sentiment'] = df_clean['review_sentiment'].str.strip().str.capitalize()

    logger.info("Social dataset cleaned Successfully.")
    return df_clean

def clean_transaction_data(transactions_df):
    """ clean transaction dataset """
    logger.info("Cleaning transaction data...")
    df_clean = transactions_df.copy()

    # handle missing values according to column type
    df_clean.dropna(subset=['customer_id_legacy', 'transaction_id'], inplace=True)

    for col in ['purchase_amount', 'customer_rating']:
        if col in df_clean.columns:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

    # Convert date columns to datetime
    date_columns = ['transaction_date', 'purchase_date']
    for date_col in date_columns:
        if date_col in df_clean.columns:
            try:
                df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
                # Fill any remaining NaT values with a default date
                df_clean[date_col].fillna(pd.Timestamp('2025-01-01'), inplace=True)
            except Exception as e:
                logger.warning(f"Could not convert {date_col} to datetime: {e}")

    logger.info("Transaction dataset cleaned successfully.")
    return df_clean

def merge_datasets(social_df, transactions_df):
    """Merge social profiles and transaction datasets safely."""

    logger.info("Merging datasets...")

    # Standardize customer ID columns to strings
    social_df['customer_id_new'] = social_df['customer_id_new'].astype(str).str.strip()
    transactions_df['customer_id_legacy'] = transactions_df['customer_id_legacy'].astype(str).str.strip()

    # Handle duplicates in social profiles
    # If a customer has multiple social rows, keep the first (or aggregate if needed)
    social_unique = social_df.drop_duplicates(subset=['customer_id_new'], keep='first')

    # Merge datasets
    merged_df = transactions_df.merge(
        social_unique,
        left_on='customer_id_legacy',
        right_on='customer_id_new',
        how='left',  # keep all transactions even if social data is missing
        validate='many_to_one'  # transactions many -> social one
    )

    # Cleanup: unify customer_id column
    merged_df = merged_df.rename(columns={'customer_id_legacy': 'customer_id'})
    merged_df = merged_df.drop(columns=['customer_id_new'])

    logger.info(f"Merged dataset shape: {merged_df.shape}")
    return merged_df

def engineer_social_features(df):
    """Create features from social media engagement data"""
    logger.info("Engineering social features...")
    logger.info(f"Input social data shape: {df.shape}")
    
    if df.empty:
        logger.error("Input DataFrame is empty!")
        return df
        
    df = df.copy()

    # Encode the platform (categorical → numeric)
    if 'social_media_platform' in df.columns:
        platform_dummies = pd.get_dummies(df['social_media_platform'], prefix='platform')
        df = pd.concat([df, platform_dummies], axis=1)
        df.drop(columns=['social_media_platform'], inplace=True)

    # Normalize engagement_score (optional)
    if 'engagement_score' in df.columns:
        # Normalize to 0-1 range
        df['engagement_score_normalized'] = (
            df['engagement_score'] - df['engagement_score'].min()
        ) / (df['engagement_score'].max() - df['engagement_score'].min())

    # Handle review sentiment (Positive / Neutral / Negative → numeric)
    if 'review_sentiment' in df.columns:
        sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        df['sentiment_score'] = df['review_sentiment'].map(sentiment_map).fillna(0)
        df.drop(columns=['review_sentiment'], inplace=True)

    # Derive engagement strength score
    # Combine engagement intensity and sentiment positivity
    if all(col in df.columns for col in ['engagement_score_normalized', 'sentiment_score']):
        df['engagement_strength'] = (
            df['engagement_score_normalized'] * 0.7 + df['sentiment_score'] * 0.3
        )

    # Flag users with high purchase interest
    if 'purchase_interest_score' in df.columns:
        df['high_interest_flag'] = (df['purchase_interest_score'] > 3).astype(int)

    # Ensure customer_id column exists
    if 'customer_id_new' in df.columns:
        df = df.rename(columns={'customer_id_new': 'customer_id'})
    elif 'customer_id' not in df.columns:
        logger.warning("No customer_id column found in social features")

    logger.info(f"Output social features shape: {df.shape}")
    return df

def engineer_transaction_features(df):
    """Create customer-level transaction features"""
    logger.info("Engineering transaction features...")
    logger.info(f"Input transaction data shape: {df.shape}")
    
    if df.empty:
        logger.error("Input transaction DataFrame is empty!")
        return pd.DataFrame()
    
    df = df.copy()
    
    # Ensure there is a unified customer_id column
    if 'customer_id_legacy' in df.columns:
        df['customer_id'] = df['customer_id_legacy'].astype(str).str.strip()
    elif 'customer_id' not in df.columns:
        raise KeyError("No customer ID column found in transactions dataset")
    
    # Convert date columns to datetime before aggregation
    date_columns = ['transaction_date', 'purchase_date']
    for date_col in date_columns:
        if date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not convert {date_col} to datetime: {e}")
    
    # Aggregation functions
    agg_functions = {
        'transaction_amount': ['sum', 'mean', 'count'],
        'purchase_amount': ['sum', 'mean', 'count'],  # Added purchase_amount as alternative
        'transaction_date': ['max', 'min'],
        'purchase_date': ['max', 'min'],  # Added purchase_date as alternative
        'product_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
        'customer_rating': 'mean'
    }
    
    # Only include columns that exist
    available_columns = {}
    for col in agg_functions:
        if col in df.columns:
            available_columns[col] = agg_functions[col]
    
    if not available_columns:
        logger.warning("No aggregatable columns found in transaction data")
        return pd.DataFrame()
    
    # Group by customer_id
    customer_features = df.groupby('customer_id').agg(available_columns).reset_index()
    
    # Flatten column names - FIXED: Handle single-level columns properly
    new_columns = []
    for col in customer_features.columns:
        if isinstance(col, tuple):
            # Multi-level column: join with underscore
            new_col = '_'.join(col).strip('_')
        else:
            # Single-level column: keep as is
            new_col = col
        new_columns.append(new_col)
    
    customer_features.columns = new_columns
    
    # Rename for clarity
    column_rename = {
        'transaction_amount_sum': 'total_spent',
        'transaction_amount_mean': 'avg_purchase_value',
        'transaction_amount_count': 'purchase_frequency',
        'purchase_amount_sum': 'total_spent',
        'purchase_amount_mean': 'avg_purchase_value', 
        'purchase_amount_count': 'purchase_frequency',
        'transaction_date_max': 'last_purchase_date',
        'transaction_date_min': 'first_purchase_date',
        'purchase_date_max': 'last_purchase_date',
        'purchase_date_min': 'first_purchase_date',
        'product_category_<lambda>': 'most_frequent_category',
        'customer_rating_mean': 'avg_customer_rating'
    }
    
    # Only rename columns that exist and need renaming
    customer_features = customer_features.rename(columns={
        k: v for k, v in column_rename.items() if k in customer_features.columns
    })
    
    # Time-based features - FIXED: Ensure dates are datetime objects
    current_date = pd.Timestamp.now()
    
    # Convert date columns to datetime if they exist
    if 'last_purchase_date' in customer_features.columns:
        if not pd.api.types.is_datetime64_any_dtype(customer_features['last_purchase_date']):
            customer_features['last_purchase_date'] = pd.to_datetime(customer_features['last_purchase_date'], errors='coerce')
        
        customer_features['last_purchase_days_ago'] = (
            current_date - customer_features['last_purchase_date']
        ).dt.days
    
    if all(col in customer_features.columns for col in ['last_purchase_date', 'first_purchase_date']):
        if not pd.api.types.is_datetime64_any_dtype(customer_features['first_purchase_date']):
            customer_features['first_purchase_date'] = pd.to_datetime(customer_features['first_purchase_date'], errors='coerce')
        
        customer_features['customer_tenure_days'] = (
            customer_features['last_purchase_date'] - customer_features['first_purchase_date']
        ).dt.days
    
    logger.info(f"Output transaction features shape: {customer_features.shape}")
    return customer_features

def create_final_dataset(social_with_features, transaction_features):
    """Combine all features into final dataset"""
    logger.info("Creating final dataset...")
    
    # Debug: Check input shapes
    logger.info(f"Social features shape: {social_with_features.shape}")
    logger.info(f"Transaction features shape: {transaction_features.shape}")

    # Ensure both have the same ID column name
    if 'customer_id_new' in social_with_features.columns:
        social_with_features = social_with_features.rename(columns={'customer_id_new': 'customer_id'})
    if 'customer_id_legacy' in transaction_features.columns:
        transaction_features = transaction_features.rename(columns={'customer_id_legacy': 'customer_id'})

    # Check if customer_id exists in both datasets
    if 'customer_id' not in social_with_features.columns:
        logger.error("customer_id not found in social features")
        return pd.DataFrame()
    if 'customer_id' not in transaction_features.columns:
        logger.error("customer_id not found in transaction features")
        return pd.DataFrame()

    # Merge on customer_id
    final_df = social_with_features.merge(transaction_features, on='customer_id', how='inner')
    
    if final_df.empty:
        logger.warning("Inner merge resulted in empty dataset - no common customer_ids")
        return final_df

    # Engagement-to-Spend ratio (if applicable)
    if all(col in final_df.columns for col in ['engagement_score', 'total_spent']):
        final_df['engagement_to_purchase_ratio'] = (
            final_df['engagement_score'] / (final_df['total_spent'] + 1)
        )

    # Customer Value Score (simplified for your data)
    value_components = []
    if 'total_spent' in final_df.columns:
        value_components.append(final_df['total_spent'] * 0.5)
    if 'engagement_score' in final_df.columns:
        value_components.append(final_df['engagement_score'] * 0.3)
    if 'purchase_interest_score' in final_df.columns:
        value_components.append(final_df['purchase_interest_score'] * 0.2)

    if value_components:
        final_df['customer_value_score'] = sum(value_components)

    logger.info(f"Final merged dataset shape: {final_df.shape}")
    return final_df

def make_ml_ready(final_df):
    """Prepare final dataset for ML: fill missing, encode categoricals, drop unused cols"""
    df = final_df.copy()

    # Fill missing numeric columns with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Encode categorical features
    if 'most_frequent_category' in df.columns:
        df = pd.get_dummies(df, columns=['most_frequent_category'], prefix='category')

    # Optional: drop text columns if any remain
    text_cols = df.select_dtypes(include=['object']).columns
    text_cols = [col for col in text_cols if col != 'customer_id']
    df.drop(columns=text_cols, inplace=True)

    # Ensure customer_id is first column
    cols = ['customer_id'] + [c for c in df.columns if c != 'customer_id']
    df = df[cols]

    return df

def save_processed_data(df, filename='final_customer_dataset.csv', save_sample=True):
    """Save processed dataset safely to data/processed/ folder."""
    
    # Ensure the processed data folder exists
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Define full file paths
    filepath = os.path.join(processed_dir, filename)
    
    try:
        # Save full dataset
        df.to_csv(filepath, index=False)
        logger.info(f"Final dataset saved to: {filepath}")
        logger.info(f"Dataset shape: {df.shape}")

        # Optionally save a smaller sample (first 100 rows)
        if save_sample and not df.empty:
            sample_filename = filename.replace('.csv', '_sample.csv')
            sample_filepath = os.path.join(processed_dir, sample_filename)
            df.head(100).to_csv(sample_filepath, index=False)
            logger.info(f"Sample dataset saved to: {sample_filepath}")

    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise e

def generate_summary_report(df, id_column='customer_id'):
    """Generate a summary report of the final dataset"""
    logger.info("\n" + "="*60)
    logger.info("FINAL DATASET SUMMARY REPORT")
    logger.info("="*60)
    
    if df.empty:
        logger.warning("Dataset is empty - cannot generate summary")
        return
        
    logger.info(f"Dataset shape: {df.shape}")
    
    if id_column in df.columns:
        logger.info(f"Unique customers: {df[id_column].nunique()}")
    else:
        logger.warning(f"ID column '{id_column}' not found in dataset.")
    
    logger.info(f"Number of features: {len(df.columns)}")
    
    logger.info("\nColumn Overview:")
    for col in df.columns:
        logger.info(f"  - {col}: {df[col].dtype}")
    
    logger.info("\nMissing Values Summary:")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        for col, count in missing_cols.items():
            percent = (count / len(df)) * 100
            logger.info(f"  - {col}: {count} missing ({percent:.2f}%)")
    else:
        logger.info("  - No missing values detected")
    
    logger.info("\nDataset preparation completed successfully!")

def extract_features_from_merged(merged_df):
    """Extract social and transaction features from merged dataset"""
    logger.info("Extracting features from merged dataset...")
    
    # Extract social features (one row per customer)
    social_cols = [col for col in merged_df.columns if col in [
        'customer_id', 'social_media_platform', 'engagement_score', 
        'purchase_interest_score', 'review_sentiment'
    ]]
    social_data = merged_df[social_cols].drop_duplicates(subset=['customer_id'], keep='first')
    social_features = engineer_social_features(social_data)
    
    # Extract transaction features (aggregate all transactions per customer)
    transaction_features = engineer_transaction_features(merged_df)
    
    return social_features, transaction_features

def main():
    """Main function to run the entire data preprocessing pipeline"""
    try:
        logger.info("Starting full data preprocessing pipeline...")

        # Step 1: Create folder structure
        create_data_folders()

        # Step 2: Load raw data
        social_df, transactions_df = load_data()

        # Step 3: Clean data
        social_clean = clean_social_data(social_df)
        transactions_clean = clean_transaction_data(transactions_df)

        # Step 4: Merge datasets
        merged_data = merge_datasets(social_clean, transactions_clean)

        # Debug: Check if we have data after merge
        logger.info(f"Merged data shape: {merged_data.shape}")
        if merged_data.empty:
            logger.error("Merged dataset is empty! Check if customer IDs match between datasets.")
            return

        # Step 5: Extract and engineer features from merged data
        social_features, transaction_features = extract_features_from_merged(merged_data)

        # Step 6: Create final dataset
        final_dataset = create_final_dataset(social_features, transaction_features)

        if final_dataset.empty:
            logger.error("Final dataset is empty! No common customers with complete data.")
            return

        # Step 6b: Make dataset ML-ready
        ml_ready_df = make_ml_ready(final_dataset)
        logger.info(f"ML-ready dataset shape: {ml_ready_df.shape}")

        # Step 7: Save processed data
        save_processed_data(ml_ready_df, filename='ml_ready_customer_dataset.csv')

        # Step 8: Generate summary report
        generate_summary_report(ml_ready_df, id_column='customer_id')

        logger.info("Data preprocessing pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed due to: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
