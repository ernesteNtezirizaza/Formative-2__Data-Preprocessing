import pandas as pd
import numpy as np
import os

def load_data():
    """Load the raw datasets from the specified paths"""
    try:
        # Load customer transactions data
        transactions_path = '../data/raw/customer_transactions.csv'
        social_profiles_path = '../data/raw/customer_social_profiles.csv'
        
        transactions_df = pd.read_csv(transactions_path)
        social_profiles_df = pd.read_csv(social_profiles_path)
        
        print("Data loaded successfully:")
        print(f"Transactions data shape: {transactions_df.shape}")
        print(f"Social profiles data shape: {social_profiles_df.shape}")
        
        return transactions_df, social_profiles_df
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def clean_transactions_data(df):
    """Clean the transactions dataset"""
    print("\nCleaning transactions data...")
    
    # Display initial info
    print(f"Initial shape: {df.shape}")
    print(f"Initial null values:\n{df.isnull().sum()}")
    
    # Handle duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"Removed {duplicates_removed} duplicate rows")
    
    # Data type conversion
    df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
    df['customer_rating'] = pd.to_numeric(df['customer_rating'], errors='coerce')
    df['purchase_amount'] = pd.to_numeric(df['purchase_amount'], errors='coerce')
    df['transaction_id'] = df['transaction_id'].astype(str)
    
    # Handle null values
    numeric_cols = ['purchase_amount', 'customer_rating']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    categorical_cols = ['product_category']
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    # Validate data ranges
    df = df[df['purchase_amount'] >= 0]
    df = df[(df['customer_rating'] >= 1) & (df['customer_rating'] <= 5)]
    
    # Rename customer_id_legacy to customer_id
    df = df.rename(columns={'customer_id_legacy': 'customer_id'})
    
    print(f"Final shape after cleaning: {df.shape}")
    print(f"Final null values:\n{df.isnull().sum()}")
    
    return df

def clean_social_profiles_data(df):
    """Clean the social profiles dataset"""
    print("\nCleaning social profiles data...")
    
    # Display initial info
    print(f"Initial shape: {df.shape}")
    print(f"Initial null values:\n{df.isnull().sum()}")
    
    # Handle duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"Removed {duplicates_removed} duplicate rows")
    
    # Extract numeric customer ID from customer_id_new and rename to customer_id
    df['customer_id'] = df['customer_id_new'].str.extract('(\d+)').astype(float)
    
    # Drop the original customer_id_new column
    df = df.drop('customer_id_new', axis=1)
    
    # Data type conversion
    df['engagement_score'] = pd.to_numeric(df['engagement_score'], errors='coerce')
    df['purchase_interest_score'] = pd.to_numeric(df['purchase_interest_score'], errors='coerce')
    df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce')
    
    # Handle null values
    numeric_cols = ['engagement_score', 'purchase_interest_score', 'customer_id']
    for col in numeric_cols:
        if col == 'customer_id':
            df = df.dropna(subset=['customer_id'])
        else:
            df[col] = df[col].fillna(df[col].mean())
    
    categorical_cols = ['social_media_platform', 'review_sentiment']
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    # Validate data ranges
    df = df[(df['engagement_score'] >= 0) & (df['engagement_score'] <= 100)]
    df = df[(df['purchase_interest_score'] >= 1) & (df['purchase_interest_score'] <= 5)]
    df = df[df['customer_id'].notna()]
    
    # Convert customer_id to integer (remove decimal places)
    df['customer_id'] = df['customer_id'].astype(int)
    
    print(f"Final shape after cleaning: {df.shape}")
    print(f"Final null values:\n{df.isnull().sum()}")
    
    return df

def create_mapping_key(transactions_df, social_profiles_df):
    """
    Create a mapping between customer IDs from both datasets
    """
    print("\nCreating customer ID mapping...")
    
    # Find common numeric IDs
    transactions_numeric_ids = set(transactions_df['customer_id'].unique())
    social_numeric_ids = set(social_profiles_df['customer_id'].unique())
    
    common_ids = transactions_numeric_ids.intersection(social_numeric_ids)
    
    print(f"Unique numeric IDs in transactions: {len(transactions_numeric_ids)}")
    print(f"Unique numeric IDs in social profiles: {len(social_numeric_ids)}")
    print(f"Common numeric IDs: {len(common_ids)}")
    
    return common_ids

def merge_datasets(transactions_df, social_profiles_df, common_ids):
    """Merge the datasets using INNER JOIN logic"""
    print("\nMerging datasets...")
    
    # Filter both datasets to only include common IDs
    transactions_filtered = transactions_df[transactions_df['customer_id'].isin(common_ids)].copy()
    social_profiles_filtered = social_profiles_df[social_profiles_df['customer_id'].isin(common_ids)].copy()
    
    print(f"Transactions rows for merging: {len(transactions_filtered)}")
    print(f"Social profiles rows for merging: {len(social_profiles_filtered)}")
    
    # Perform the merge
    merged_df = pd.merge(
        transactions_filtered,
        social_profiles_filtered,
        on='customer_id',
        how='inner',
        validate='many_to_many'
    )
    
    print(f"Merged dataset shape: {merged_df.shape}")
    
    return merged_df

def validate_merge(merged_df, transactions_df, social_profiles_df):
    """Validate the merge results"""
    print("\nValidating merge results...")
    
    # Check for null values in merged data
    print("Null values in merged data:")
    print(merged_df.isnull().sum())
    
    # Check data types
    print("\nData types in merged data:")
    print(merged_df.dtypes)
    
    # Check merge statistics
    original_transaction_count = len(transactions_df)
    original_social_count = len(social_profiles_df)
    merged_count = len(merged_df)
    
    print(f"\nMerge Statistics:")
    print(f"Original transactions records: {original_transaction_count}")
    print(f"Original social profiles records: {original_social_count}")
    print(f"Merged records: {merged_count}")
    print(f"Merge coverage (transactions): {merged_count/original_transaction_count*100:.2f}%")
    
    # Check for duplicate combinations
    key_columns = ['transaction_id', 'customer_id', 'social_media_platform']
    duplicate_count = merged_df.duplicated(subset=key_columns).sum()
    print(f"Duplicate key combinations: {duplicate_count}")
    
    # Verify customer_id is numeric and consistent
    print(f"\nCustomer ID verification:")
    print(f"Customer ID data type: {merged_df['customer_id'].dtype}")
    print(f"Customer ID range: {merged_df['customer_id'].min()} to {merged_df['customer_id'].max()}")
    print(f"Unique customers in merged data: {merged_df['customer_id'].nunique()}")
    
    return merged_df

def save_merged_data(merged_df):
    """Save the merged data to the specified location"""
    # Create directory if it doesn't exist
    os.makedirs('../data/processed', exist_ok=True)
    
    # Save to CSV
    output_path = '../data/processed/merged_customer_data.csv'
    merged_df.to_csv(output_path, index=False)
    
    print(f"\nMerged data saved to: {output_path}")
    print(f"Final merged data shape: {merged_df.shape}")
    
    return output_path

def main():
    """Main function to execute the data merging pipeline"""
    print("Starting data merging process...")
    
    # Load data
    transactions_df, social_profiles_df = load_data()
    
    if transactions_df is None or social_profiles_df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Clean datasets
    transactions_clean = clean_transactions_data(transactions_df)
    social_profiles_clean = clean_social_profiles_data(social_profiles_df)
    
    # Create mapping and merge
    common_ids = create_mapping_key(transactions_clean, social_profiles_clean)
    
    if not common_ids:
        print("No common customer IDs found for merging. Exiting.")
        return
    
    # Merge datasets
    merged_df = merge_datasets(transactions_clean, social_profiles_clean, common_ids)
    
    if merged_df.empty:
        print("Merge resulted in empty dataset. Exiting.")
        return
    
    # Validate merge
    validated_df = validate_merge(merged_df, transactions_clean, social_profiles_clean)
    
    # Save results
    output_path = save_merged_data(validated_df)
    
    # Final summary
    print("\n" + "="*50)
    print("DATA MERGING COMPLETE")
    print("="*50)
    print(f"Final merged dataset columns: {list(validated_df.columns)}")
    print(f"Total records in merged data: {len(validated_df)}")
    print(f"File saved at: {output_path}")
    
    # Display sample of merged data
    print("\nSample of merged data (first 5 rows):")
    print(validated_df.head().to_string(index=False))

if __name__ == "__main__":
    main()