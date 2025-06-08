import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder
import json

def safe_serialize(obj):
    """
    Convert pandas and numpy types to JSON serializable types.
    This can be used when returning results from analysis functions.
    """
    import pandas as pd
    import numpy as np
    import json
    
    class NumpyPandasEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Series):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if hasattr(obj, 'dtype'):
                return str(obj.dtype)
            if pd.isna(obj):
                return None
            return super().default(obj)
    
    return json.loads(json.dumps(obj, cls=NumpyPandasEncoder))


def convert_to_serializable(obj):
    pass


def calculate_summary_statistics(df):
    pass

    
def get_overall_summary(df):
    pass

def analyze_numerical_data(df):
    pass
    

def analyze_categorical_data(df):
    pass

def analyze_correlations(df):
    pass


def analyze_feature_importance(df, target_column):
    pass

def _describe_data(df):
    """Provides a descriptive summary of the DataFrame."""
    try:
        description = df.describe(include='all').to_dict()
        # Convert non-serializable types if any (e.g., NaT, inf/-inf)
        for col, stats_dict in description.items():
            for stat, value in stats_dict.items():
                if pd.isna(value):
                    stats_dict[stat] = None
                elif isinstance(value, (np.integer, np.int64)):
                    stats_dict[stat] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    stats_dict[stat] = round(float(value), 4) # Round floats for readability
        return description
    except Exception as e:
        return {"error": f"Could not describe data: {str(e)}"}

def _perform_aggregation(df, column_to_aggregate, aggregation_type, group_by_column=None):
    """Performs various aggregation operations on a specified numeric column."""
    if column_to_aggregate not in df.columns:
        return {"error": f"Column '{column_to_aggregate}' not found for aggregation."}
    
    df_temp = df.copy()
    df_temp[column_to_aggregate] = pd.to_numeric(df_temp[column_to_aggregate], errors='coerce')
    df_temp.dropna(subset=[column_to_aggregate], inplace=True)

    if df_temp[column_to_aggregate].empty:
        return {"message": f"No valid numeric data in '{column_to_aggregate}' for aggregation after cleaning."}

    valid_aggregations = ['mean', 'sum', 'min', 'max', 'median', 'count', 'std']
    if aggregation_type not in valid_aggregations:
        return {"error": f"Invalid aggregation type: '{aggregation_type}'. Choose from {', '.join(valid_aggregations)}."}

    if group_by_column:
        if group_by_column not in df_temp.columns:
            return {"error": f"Group-by column '{group_by_column}' not found for aggregation."}
        
        # Attempt to handle time-based grouping robustly
        if pd.api.types.is_string_dtype(df_temp[group_by_column]) or pd.api.types.is_object_dtype(df_temp[group_by_column]):
            try:
                temp_dt_col = pd.to_datetime(df_temp[group_by_column], errors='coerce')
                if not temp_dt_col.isnull().all():
                    df_temp[group_by_column + '_parsed'] = temp_dt_col.dt.hour # Extract hour for grouping
                    group_col = group_by_column + '_parsed'
                else:
                    group_col = group_by_column
            except Exception:
                group_col = group_by_column
        else:
            group_col = group_by_column

        grouped_result = df_temp.groupby(group_col)[column_to_aggregate].agg(aggregation_type)
        serializable_result = {str(k): round(v, 2) if isinstance(v, (float, np.float64)) else v for k, v in grouped_result.to_dict().items()}
        return {"aggregation_type": aggregation_type, "column_aggregated": column_to_aggregate, "grouped_by": group_by_column, "result": serializable_result}
    else:
        overall_result = getattr(df_temp[column_to_aggregate], aggregation_type)()
        return {"aggregation_type": aggregation_type, "column_aggregated": column_to_aggregate, "result": round(overall_result, 2) if isinstance(overall_result, (float, np.float64)) else overall_result}

def _filter_data(df, column, operator, value):
    """Filters the DataFrame based on a condition."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found for filtering."}
    
    df_filtered = df.copy()
    
    # Try to convert value to appropriate type for comparison
    try:
        if pd.api.types.is_numeric_dtype(df_filtered[column]):
            value = float(value)
        elif pd.api.types.is_datetime64_any_dtype(df_filtered[column]) or (pd.api.types.is_string_dtype(df_filtered[column]) and any(df_filtered[column].apply(lambda x: pd.to_datetime(x, errors='coerce') is not pd.NaT))):
            value = pd.to_datetime(value) # Attempt to convert filter value to datetime
        # Add more type conversions as needed (e.g., boolean)
    except ValueError:
        pass # Keep as original type if conversion fails

    try:
        if operator == '>':
            filtered_df = df_filtered[df_filtered[column] > value]
        elif operator == '<':
            filtered_df = df_filtered[df_filtered[column] < value]
        elif operator == '==':
            filtered_df = df_filtered[df_filtered[column] == value]
        elif operator == '!=':
            filtered_df = df_filtered[df_filtered[column] != value]
        elif operator == '>=':
            filtered_df = df_filtered[df_filtered[column] >= value]
        elif operator == '<=':
            filtered_df = df_filtered[df_filtered[column] <= value]
        elif operator == 'contains': # For string columns
            filtered_df = df_filtered[df_filtered[column].astype(str).str.contains(str(value), case=False, na=False)]
        else:
            return {"error": f"Unsupported filter operator: '{operator}'. Supported: >, <, ==, !=, >=, <=, contains."}
        
        return {"filtered_rows_count": len(filtered_df), "sample_data": filtered_df.head().to_dict(orient='records')}
    except Exception as e:
        import traceback
        print(f"Error in _filter_data: {traceback.format_exc()}")
        return {"error": f"An error occurred during filtering: {str(e)}. Check column type, operator, and value."}

def _sort_data(df, column, ascending=True):
    """Sorts the DataFrame by a specified column."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found for sorting."}
    
    try:
        sorted_df = df.sort_values(by=column, ascending=ascending)
        return {"message": f"Data sorted by '{column}' in {'ascending' if ascending else 'descending'} order.", "sample_data": sorted_df.head().to_dict(orient='records')}
    except Exception as e:
        return {"error": f"An error occurred during sorting: {str(e)}. Check column type."}

def _perform_t_test(df, column1, column2=None, group_column=None, group1_value=None, group2_value=None):
    """Performs an independent samples t-test between two groups or two columns."""
    if column1 not in df.columns:
        return {"error": f"Column '{column1}' not found for t-test."}
    
    df_temp = df.copy()
    df_temp[column1] = pd.to_numeric(df_temp[column1], errors='coerce').dropna()

    if df_temp[column1].empty:
        return {"error": f"No valid numeric data in '{column1}' for t-test."}

    try:
        if group_column and group1_value is not None and group2_value is not None:
            # Independent samples t-test between two groups
            if group_column not in df_temp.columns:
                return {"error": f"Group column '{group_column}' not found for t-test."}
            
            group1_data = df_temp[df_temp[group_column] == group1_value][column1].dropna()
            group2_data = df_temp[df_temp[group_column] == group2_value][column1].dropna()

            if group1_data.empty or group2_data.empty:
                return {"error": "Insufficient data in one or both groups for t-test."}
            
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False) # Welch's t-test
            return {"test_type": "independent_t_test", "column1": column1, "group_column": group_column,
                    "group1": group1_value, "group2": group2_value,
                    "t_statistic": round(t_stat, 4), "p_value": round(p_value, 4)}
        elif column2:
            # Two-sample t-test on two separate columns
            if column2 not in df_temp.columns:
                return {"error": f"Column '{column2}' not found for t-test."}
            
            df_temp[column2] = pd.to_numeric(df_temp[column2], errors='coerce').dropna()
            
            if df_temp[[column1, column2]].dropna().empty:
                return {"error": "Insufficient data in both columns for t-test."}

            t_stat, p_value = stats.ttest_ind(df_temp[column1].dropna(), df_temp[column2].dropna(), equal_var=False) # Welch's t-test
            return {"test_type": "two_sample_t_test", "column1": column1, "column2": column2,
                    "t_statistic": round(t_stat, 4), "p_value": round(p_value, 4)}
        else:
            return {"error": "Invalid parameters for t-test. Specify two columns or a group column with two group values."}
    except Exception as e:
        import traceback
        print(f"Error in _perform_t_test: {traceback.format_exc()}")
        return {"error": f"An error occurred during t-test: {str(e)}. Ensure data is numeric."}

def _perform_anova(df, value_column, group_column):
    """Performs a one-way ANOVA test."""
    if value_column not in df.columns or group_column not in df.columns:
        return {"error": f"Required columns '{value_column}' or '{group_column}' not found for ANOVA."}
    
    df_temp = df.copy()
    df_temp[value_column] = pd.to_numeric(df_temp[value_column], errors='coerce')
    df_temp.dropna(subset=[value_column, group_column], inplace=True)

    if df_temp.empty:
        return {"error": "No valid data available for ANOVA after cleaning."}

    try:
        groups = [df_temp[value_column][df_temp[group_column] == g].dropna() for g in df_temp[group_column].unique()]
        groups = [g for g in groups if not g.empty] # Filter out empty groups

        if len(groups) < 2:
            return {"error": "ANOVA requires at least two groups with data."}
        
        f_stat, p_value = stats.f_oneway(*groups)
        return {"test_type": "anova", "value_column": value_column, "group_column": group_column,
                "f_statistic": round(f_stat, 4), "p_value": round(p_value, 4)}
    except Exception as e:
        import traceback
        print(f"Error in _perform_anova: {traceback.format_exc()}")
        return {"error": f"An error occurred during ANOVA: {str(e)}. Ensure data is numeric and groups are valid."}

# --- Main Comprehensive Analysis Function ---

def perform_comprehensive_analysis(df, analysis_type, **params):
    """
    Acts as a central dispatcher for various data analysis tasks.
    The AI model should call this single function for most data analysis requests,
    passing the analysis_type and specific parameters.

    Args:
        df (pd.DataFrame): The input DataFrame from the user's session.
        analysis_type (str): The type of analysis to perform.
            Valid types: 'describe', 'aggregate', 'filter', 'sort', 't_test', 'anova', 'unique_values', 'feature_correlation'.
        **params: Keyword arguments specific to the chosen analysis_type.

    Returns:
        dict: The result of the analysis, or an error message if inputs are invalid or an error occurs.
    """
    if df is None or df.empty:
        return {"error": "No data available in the session to perform analysis. Please upload a dataset first."}

    # Clean column names by stripping whitespace for robustness
    df.columns = df.columns.str.strip()

    try:
        if analysis_type == 'describe':
            return _describe_data(df)
        elif analysis_type == 'aggregate':
            required_params = ['column_to_aggregate', 'aggregation_type']
            if not all(p in params for p in required_params):
                return {"error": f"Missing parameters for 'aggregate' analysis. Required: {', '.join(required_params)}."}
            return _perform_aggregation(df, params['column_to_aggregate'], params['aggregation_type'], params.get('group_by_column'))
        elif analysis_type == 'filter':
            required_params = ['column', 'operator', 'value']
            if not all(p in params for p in required_params):
                return {"error": f"Missing parameters for 'filter' analysis. Required: {', '.join(required_params)}."}
            return _filter_data(df, params['column'], params['operator'], params['value'])
        elif analysis_type == 'sort':
            required_params = ['column']
            if not all(p in params for p in required_params):
                return {"error": f"Missing parameters for 'sort' analysis. Required: {', '.join(required_params)}."}
            return _sort_data(df, params['column'], params.get('ascending', True))
        elif analysis_type == 't_test':
            if 'column1' in params and 'group_column' in params and 'group1_value' in params and 'group2_value' in params:
                return _perform_t_test(df, params['column1'], group_column=params['group_column'], group1_value=params['group1_value'], group2_value=params['group2_value'])
            elif 'column1' in params and 'column2' in params:
                 return _perform_t_test(df, params['column1'], params['column2'])
            else:
                 return {"error": "Missing parameters for 't_test' analysis. Required: (column1, column2) OR (column1, group_column, group1_value, group2_value)."}
        elif analysis_type == 'anova':
            required_params = ['value_column', 'group_column']
            if not all(p in params for p in required_params):
                return {"error": f"Missing parameters for 'anova' analysis. Required: {', '.join(required_params)}."}
            return _perform_anova(df, params['value_column'], params['group_column'])
        elif analysis_type == 'unique_values': # Re-incorporate existing general analysis
            if 'column' not in params:
                return {"error": "Missing 'column' parameter for 'unique_values' analysis."}
            return {"unique_values": df[params['column']].dropna().unique().tolist(), "count": df[params['column']].nunique()}
        elif analysis_type == 'feature_correlation': # Re-incorporate existing general analysis
            if 'column1' not in params or 'column2' not in params:
                return {"error": "Missing 'column1' or 'column2' for 'feature_correlation' analysis."}
            
            df_numeric = df[[params['column1'], params['column2']]].apply(pd.to_numeric, errors='coerce').dropna()
            if df_numeric.empty:
                return {"error": "No valid numeric data in specified columns for correlation."}
            
            correlation = df_numeric[params['column1']].corr(df_numeric[params['column2']])
            return {"correlation": round(correlation, 4), "column1": params['column1'], "column2": params['column2']}
        else:
            return {"error": f"Unsupported analysis type: '{analysis_type}'. Choose from 'describe', 'aggregate', 'filter', 'sort', 't_test', 'anova', 'unique_values', 'feature_correlation'."}
    except Exception as e:
        import traceback
        print(f"Error in perform_comprehensive_analysis: {traceback.format_exc()}")
        return {"error": f"An unexpected error occurred during analysis: {str(e)}."}
