import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Removes duplicate rows from the DataFrame.
    Returns the DataFrame with duplicate rows removed and a report message.
    """
    original_rows = df.shape[0]
    df_cleaned = df.drop_duplicates(subset=subset)
    removed_rows = original_rows - df_cleaned.shape[0]
    
    message = ""
    if removed_rows > 0:
        message = f"Removed {removed_rows} duplicate rows."
    else:
        message = "No duplicate rows found or removed."
    return df_cleaned, message

def fill_missing_values(df, column, method='mean', value=None):
    """
    Fills missing values in a specified column using various methods.
    Returns the DataFrame with missing values filled and a report message.
    """
    if column not in df.columns:
        return df, f"Column '{column}' not found in DataFrame. No missing values filled."

    missing_count = df[column].isnull().sum()
    if missing_count == 0:
        return df, f"No missing values found in column '{column}'."

    message = ""
    df_processed = df.copy() # Work on a copy to avoid modifying original df directly before returning

    if method == 'mean':
        if pd.api.types.is_numeric_dtype(df_processed[column]):
            df_processed[column] = df_processed[column].fillna(df_processed[column].mean())
            message = f"Filled {missing_count} missing values in column '{column}' using method 'mean'."
        else:
            # Fallback to mode for non-numeric columns if mean is requested
            mode_val_series = df_processed[column].mode()
            if not mode_val_series.empty:
                df_processed[column] = df_processed[column].fillna(mode_val_series[0])
                message = f"Cannot apply 'mean' method to non-numeric column '{column}'. Filled {missing_count} missing values using 'mode' instead."
            else:
                message = f"Cannot apply 'mean' method to non-numeric column '{column}' and no mode found. No action taken."
                return df, message # Return original df if no action taken
    elif method == 'median':
        if pd.api.types.is_numeric_dtype(df_processed[column]):
            df_processed[column] = df_processed[column].fillna(df_processed[column].median())
            message = f"Filled {missing_count} missing values in column '{column}' using method 'median'."
        else:
            # Fallback to mode for non-numeric columns if median is requested
            mode_val_series = df_processed[column].mode()
            if not mode_val_series.empty:
                df_processed[column] = df_processed[column].fillna(mode_val_series[0])
                message = f"Cannot apply 'median' method to non-numeric column '{column}'. Filled {missing_count} missing values using 'mode' instead."
            else:
                message = f"Cannot apply 'median' method to non-numeric column '{column}' and no mode found. No action taken."
                return df, message
    elif method == 'mode':
        mode_val_series = df_processed[column].mode()
        if not mode_val_series.empty:
            df_processed[column] = df_processed[column].fillna(mode_val_series[0])
            message = f"Filled {missing_count} missing values in column '{column}' using method 'mode'."
        else:
            message = f"No mode found for column '{column}'. No missing values filled."
            return df, message
    elif method == 'ffill':
        df_processed[column] = df_processed[column].fillna(method='ffill')
        message = f"Filled {missing_count} missing values in column '{column}' using method 'ffill'."
    elif method == 'bfill':
        df_processed[column] = df_processed[column].fillna(method='bfill')
        message = f"Filled {missing_count} missing values in column '{column}' using method 'bfill'."
    elif method == 'value':
        if value is not None:
            df_processed[column] = df_processed[column].fillna(value)
            message = f"Filled {missing_count} missing values in column '{column}' with specified value."
        else:
            message = "Value for filling missing values is required when method is 'value'. No action taken."
            return df, message
    elif method == 'drop':
        rows_before_drop = df_processed.shape[0]
        df_processed = df_processed.dropna(subset=[column])
        rows_after_drop = df_processed.shape[0]
        message = f"Dropped {rows_before_drop - rows_after_drop} rows with missing values in column '{column}'."
    else:
        message = f"Invalid method '{method}' for filling missing values. No action taken."
        return df, message
        
    return df_processed, message

def convert_data_type(df, column, new_type):
    """
    Converts the data type of a specified column.
    Returns the DataFrame with the column converted and a report message.
    """
    if column not in df.columns:
        return df, f"Column '{column}' not found in DataFrame. No type conversion performed."

    df_processed = df.copy()
    message = ""
    try:
        if new_type == 'int':
            df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce').astype(pd.Int64Dtype())
        elif new_type == 'float':
            df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce')
        elif new_type == 'str':
            df_processed[column] = df_processed[column].astype(str)
        elif new_type == 'datetime':
            df_processed[column] = pd.to_datetime(df_processed[column], errors='coerce')
        elif new_type == 'bool':
            # Convert to boolean, handle common string representations
            df_processed[column] = df_processed[column].replace({'True': True, 'False': False, '1': True, '0': False, 1: True, 0: False})
            df_processed[column] = df_processed[column].astype(bool)
        else:
            return df, f"Unsupported new type '{new_type}'. Supported types: 'int', 'float', 'str', 'datetime', 'bool'. No type conversion performed."
            
        message = f"Column '{column}' converted to type '{new_type}'. NaNs may be introduced if conversion failed."
    except Exception as e:
        message = f"Error converting column '{column}' to '{new_type}': {e}. Original type retained."
        return df, message
    return df_processed, message

import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers(df, column=None, columns=None, method='iqr', threshold=1.5, z_thresh=3):
    """
    Remove outliers from a DataFrame using various methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    column : str, optional
        Single column name to process (for backward compatibility)
    columns : list, optional
        List of column names to process
    method : str, default 'iqr'
        Method to use for outlier detection ('iqr', 'z_score', 'modified_z_score')
    threshold : float, default 1.5
        Threshold multiplier for IQR method
    z_thresh : float, default 3
        Threshold for z-score methods
    
    Returns:
    --------
    tuple: (cleaned_df, message)
        cleaned_df : pandas.DataFrame with outliers removed
        message : str describing the outlier removal process
    """
    
    # Handle parameter compatibility
    if columns is None and column is not None:
        columns = [column]
    elif columns is None and column is None:
        raise ValueError("Either 'column' or 'columns' parameter must be provided")
    
    # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]
    
    # Validate columns exist in DataFrame
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Ensure columns are numeric
    numeric_columns = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        else:
            print(f"Warning: Column '{col}' is not numeric and will be skipped")
    
    if not numeric_columns:
        return df.copy(), "No numeric columns found for outlier removal"
    
    df_cleaned = df.copy()
    total_outliers = 0
    outlier_details = []
    
    for col in numeric_columns:
        initial_count = len(df_cleaned)
        
        if method.lower() == 'iqr':
            # Interquartile Range method
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Remove outliers
            mask = (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
            df_cleaned = df_cleaned[mask]
            
        elif method.lower() == 'z_score':
            # Z-score method
            z_scores = np.abs(stats.zscore(df_cleaned[col].dropna()))
            mask = z_scores < z_thresh
            
            # Apply mask to non-null values
            valid_indices = df_cleaned[col].dropna().index
            outlier_indices = valid_indices[~mask]
            df_cleaned = df_cleaned.drop(outlier_indices)
            
        elif method.lower() == 'modified_z_score':
            # Modified Z-score method using median
            median = df_cleaned[col].median()
            mad = np.median(np.abs(df_cleaned[col] - median))
            
            if mad == 0:
                # If MAD is 0, use standard deviation
                modified_z_scores = np.abs(df_cleaned[col] - median) / df_cleaned[col].std()
            else:
                modified_z_scores = 0.6745 * (df_cleaned[col] - median) / mad
            
            mask = np.abs(modified_z_scores) < z_thresh
            df_cleaned = df_cleaned[mask]
            
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'iqr', 'z_score', 'modified_z_score'")
        
        # Calculate outliers removed for this column
        outliers_removed = initial_count - len(df_cleaned)
        total_outliers += outliers_removed
        
        if outliers_removed > 0:
            outlier_details.append(f"{col}: {outliers_removed} outliers removed")
    
    # Create summary message
    if total_outliers > 0:
        message = f"Outlier removal using {method.upper()} method:\n"
        message += f"Total outliers removed: {total_outliers}\n"
        message += f"Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}\n"
        message += "Details:\n" + "\n".join(outlier_details)
    else:
        message = f"No outliers detected using {method.upper()} method"
    
    return df_cleaned, message

def standardize_text(df, column):
    """
    Standardizes text in a column (e.g., converts to lowercase, removes leading/trailing spaces).
    Returns the DataFrame with text standardized and a report message.
    """
    if column not in df.columns:
        return df, f"Column '{column}' not found in DataFrame. No text standardization applied."
        
    df_processed = df.copy()
    message = ""
    if pd.api.types.is_string_dtype(df_processed[column]):
        df_processed[column] = df_processed[column].astype(str).str.lower().str.strip() # Ensure it's string first
        message = f"Text in column '{column}' standardized (lowercase, stripped spaces)."
    else:
        message = f"Column '{column}' is not of string type. No text standardization applied."
    return df_processed, message


def clean_data(df):
    """Simple, working data cleaning."""
    df_cleaned = df.copy()
    report_messages = []

    # 1. Remove Duplicates
    df_cleaned, msg = remove_duplicates(df_cleaned)
    if msg:
        report_messages.append(msg)

    # 2. Fill Missing Values
    for col in df_cleaned.columns:
        missing_count = int(df_cleaned[col].isnull().sum())  # Convert to int
        if missing_count > 0:
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                df_cleaned, msg = fill_missing_values(df_cleaned, col, method='mean')
            else:
                df_cleaned, msg = fill_missing_values(df_cleaned, col, method='mode')
            if msg:
                report_messages.append(msg)

    final_report = "Data Cleaning Summary:\n" + "\n".join(report_messages)
    return df_cleaned, final_report