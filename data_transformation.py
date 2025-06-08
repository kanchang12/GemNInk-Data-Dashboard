# data_transformation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import json

def apply_log_transformation(df, column):
    """Applies log1p transformation to a specified numerical column."""
    df_transformed = df.copy()
    if column in df_transformed.columns and pd.api.types.is_numeric_dtype(df_transformed[column]):
        # Add 1 to avoid issues with log(0)
        df_transformed[column] = np.log1p(df_transformed[column])
        return df_transformed, f"Log transformation applied to '{column}'."
    else:
        return df_transformed, f"Column '{column}' not found or not numeric for log transformation."

def apply_standard_scaling(df, columns=None):
    """Applies standard scaling (Z-score normalization) to specified numerical columns."""
    df_transformed = df.copy()
    if columns is None:
        columns = df_transformed.select_dtypes(include=np.number).columns.tolist()

    if not columns:
        return df_transformed, "No numerical columns found for standard scaling."

    scaler = StandardScaler()
    try:
        # Handle cases where all values in a column are constant (std=0)
        # scaler.fit_transform will handle this gracefully, resulting in 0 for those columns.
        df_transformed[columns] = scaler.fit_transform(df_transformed[columns].fillna(df_transformed[columns].mean())) # Fill NaNs before scaling
        return df_transformed, f"Standard scaling applied to columns: {', '.join(columns)}."
    except Exception as e:
        return df_transformed, f"Error during standard scaling: {str(e)}"

def apply_minmax_scaling(df, columns=None):
    """Applies Min-Max scaling to specified numerical columns."""
    df_transformed = df.copy()
    if columns is None:
        columns = df_transformed.select_dtypes(include=np.number).columns.tolist()

    if not columns:
        return df_transformed, "No numerical columns found for Min-Max scaling."

    scaler = MinMaxScaler()
    try:
        df_transformed[columns] = scaler.fit_transform(df_transformed[columns].fillna(df_transformed[columns].mean())) # Fill NaNs before scaling
        return df_transformed, f"Min-Max scaling applied to columns: {', '.join(columns)}."
    except Exception as e:
        return df_transformed, f"Error during Min-Max scaling: {str(e)}"


def apply_one_hot_encoding(df, column):
    """Applies one-hot encoding to a specified categorical column."""
    df_transformed = df.copy()
    if column in df_transformed.columns and (df_transformed[column].dtype == 'object' or df_transformed[column].dtype == 'category'):
        # drop_first=True to avoid multicollinearity
        try:
            dummies = pd.get_dummies(df_transformed[column], prefix=column, drop_first=True)
            df_transformed = pd.concat([df_transformed.drop(columns=[column]), dummies], axis=1)
            return df_transformed, f"One-Hot Encoding applied to '{column}'."
        except Exception as e:
            return df_transformed, f"Error during One-Hot Encoding for '{column}': {str(e)}"
    else:
        return df_transformed, f"Column '{column}' not found or not categorical for One-Hot Encoding."

def apply_label_encoding(df, column):
    """Applies Label Encoding to a specified categorical column."""
    df_transformed = df.copy()
    if column in df_transformed.columns and (df_transformed[column].dtype == 'object' or df_transformed[column].dtype == 'category'):
        try:
            le = LabelEncoder()
            # Handle potential NaNs before encoding
            if df_transformed[column].isnull().any():
                df_transformed[column] = df_transformed[column].astype(str) # Convert to string to allow encoding of NaN if desired
            df_transformed[column] = le.fit_transform(df_transformed[column])
            return df_transformed, f"Label Encoding applied to '{column}'."
        except Exception as e:
            return df_transformed, f"Error during Label Encoding for '{column}': {str(e)}"
    else:
        return df_transformed, f"Column '{column}' not found or not categorical for Label Encoding."

def convert_column_type(df, column, target_type):
    """Converts a column to a specified data type."""
    df_transformed = df.copy()
    if column not in df_transformed.columns:
        return df_transformed, f"Column '{column}' not found for type conversion."

    try:
        if target_type == 'numeric':
            df_transformed[column] = pd.to_numeric(df_transformed[column], errors='coerce')
        elif target_type == 'datetime':
            df_transformed[column] = pd.to_datetime(df_transformed[column], errors='coerce')
        elif target_type == 'category':
            df_transformed[column] = df_transformed[column].astype('category')
        elif target_type == 'string':
            df_transformed[column] = df_transformed[column].astype(str)
        else:
            return df_transformed, f"Unsupported target type '{target_type}' for column conversion."
        
        return df_transformed, f"Column '{column}' converted to '{target_type}'. NaNs may be introduced if conversion failed."
    except Exception as e:
        return df_transformed, f"Error converting column '{column}' to '{target_type}': {str(e)}"

# You can add a main transformation function to route requests
def apply_transformation(df, transformation_type, params=None):
    """
    Applies a specified transformation to the DataFrame based on type and parameters.
    Params example: {'column': 'Age', 'target_type': 'numeric'}
    """
    if params is None:
        params = {}
    
    if transformation_type == 'log_transform':
        return apply_log_transformation(df, params.get('column'))
    elif transformation_type == 'standard_scale':
        return apply_standard_scaling(df, params.get('columns'))
    elif transformation_type == 'minmax_scale':
        return apply_minmax_scaling(df, params.get('columns'))
    elif transformation_type == 'one_hot_encode':
        return apply_one_hot_encoding(df, params.get('column'))
    elif transformation_type == 'label_encode':
        return apply_label_encoding(df, params.get('column'))
    elif transformation_type == 'convert_type':
        return convert_column_type(df, params.get('column'), params.get('target_type'))
    else:
        return df, f"Unsupported transformation type: {transformation_type}"