from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash, make_response
import pandas as pd
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename
import traceback
import json
from functools import wraps
from flask import session, redirect, url_for, flash
import copy
import io
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

from dotenv import load_dotenv
load_dotenv()

# --- Vertex AI Configuration ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content, FunctionDeclaration, Tool, GenerationConfig

import pymongo
from pymongo.errors import PyMongoError
from flask_session import Session
from bson import json_util

# Import custom modules
# IMPORTANT: Ensure these modules are present and contain the expected functions as provided in previous turns
import data_analysis
import data_visualization
import data_cleaning
import data_transformation

# Import plotly express for chart creation
import plotly.express as px # Add this import
import plotly.graph_objects as go # Add this import


app = Flask(__name__)
app.config['APPLICATION_NAME'] = 'GemNInk'

# --- Configuration ---
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_USE_SIGNER'] = True

# --- MongoDB Configuration ---
MONGO_URI = "mongodb+srv://dbuser:4fRiCewZkPuywKVM@hackathoncluster.cqwqixu.mongodb.net/?retryWrites=true&w=majority&appName=HackathonCluster"
DB_NAME = os.getenv('MONGO_DB_NAME', 'gemink_intelligence')

mongo_client = None
if MONGO_URI:
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI)
        # The ping command is cheap and does not require auth.
        mongo_client.admin.command('ping')
        print("MongoDB connection successful!")
    except PyMongoError as e:
        print(f"Could not connect to MongoDB: {e}")
        mongo_client = None
else:
    print("MONGO_URI not set. MongoDB features will be unavailable.")

def get_db():
    if mongo_client:
        return mongo_client[DB_NAME]
    return None

# --- Session Configuration ---
Session(app)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'info')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Vertex AI Initialization ---
PROJECT_ID = "gen-lang-client-0035881252"
REGION = 'us-central1'

vertex_ai_model = None

# Setup credentials
def get_credentials():
    credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
    if credentials_json:
        try:
            credentials_info = json.loads(credentials_json)
            from google.oauth2 import service_account
            return service_account.Credentials.from_service_account_info(credentials_info)
        except Exception as e:
            print(f"Error loading credentials: {e}")
            return None
    return None

if PROJECT_ID and REGION:
    try:
        credentials = get_credentials()
        if credentials:
            vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)
        else:
            vertexai.init(project=PROJECT_ID, location=REGION)
        vertex_ai_model = GenerativeModel(
            "gemini-2.0-flash-001",
            tools=[
                Tool(
                    function_declarations=[
                        FunctionDeclaration(
                            name="clean_data",
                            description="Applies a series of data cleaning operations to the DataFrame, including handling missing values, removing duplicates, and standardizing text columns. Returns a summary of changes.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "remove_outliers": {"type": "boolean", "description": "Whether to remove outliers from numerical columns (default: false)."},
                                    "z_score_threshold": {"type": "number", "description": "Z-score threshold for outlier removal (default: 5.0, only if remove_outliers is true)."}
                                }
                            }
                        ),
                        FunctionDeclaration(
                            name="analyze_numerical_data",
                            description="Performs comprehensive statistical analysis on numerical columns in the DataFrame, including descriptive statistics, distribution analysis, and outlier detection.",
                            parameters={"type": "object", "properties": {}}
                        ),
                        FunctionDeclaration(
                            name="analyze_categorical_data",
                            description="Analyzes categorical data patterns, including frequency counts, unique value counts, and identifies potential issues like high cardinality or inconsistent entries.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "columns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Optional: List of specific categorical columns to analyze. If not provided, all categorical columns will be analyzed."
                                    }
                                }
                            }
                        ),
                        FunctionDeclaration(
                            name="perform_correlation_analysis",
                            description="Calculates correlation matrices for numerical columns and identifies highly correlated features.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "method": {"type": "string", "description": "Correlation method (e.g., 'pearson', 'spearman', 'kendall'). Defaults to 'pearson'."}
                                }
                            }
                        ),
                        FunctionDeclaration(
                            name="create_visualization",
                            description="Generates a specified type of data visualization (e.g., histogram, scatter plot, bar chart) for given columns.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "chart_type": {"type": "string", "enum": ["histogram", "scatter", "bar", "line", "box", "heatmap", "pie"], "description": "Type of chart to create."},
                                    "x_column": {"type": "string", "description": "The column for the X-axis."},
                                    "y_column": {"type": "string", "description": "The column for the Y-axis (if applicable, e.g., for scatter/line charts)."},
                                    "title": {"type": "string", "description": "Optional: Title for the chart."},
                                    "color_by": {"type": "string", "description": "Optional: Column to use for coloring data points/bars."},
                                    "facet_by": {"type": "string", "description": "Optional: Column to use for faceting/sub-plotting."},
                                    "aggregate_by": {"type": "string", "description": "Optional: Aggregation type for bar/line charts (e.g., 'sum', 'mean', 'count')."}
                                },
                                "required": ["chart_type", "x_column"]
                            }
                        ),
                         FunctionDeclaration(
                            name="remove_outliers",
                            description="Removes outliers from numerical columns based on the Z-score method. Can be applied to all or specific columns.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "columns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Optional: List of specific numerical columns from which to remove outliers. If not provided, all numerical columns will be considered."
                                    },
                                    "z_score_threshold": {"type": "number", "description": "Z-score threshold for outlier removal (default: 3.0)."}
                                }
                            }
                        ),
                        FunctionDeclaration(
                            name="apply_transformation",
                            description="Applies a specified data transformation (e.g., log, one-hot encoding, scaling) to one or more columns.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "transformation_type": {"type": "string", "enum": ["log_transform", "standard_scale", "minmax_scale", "one_hot_encode"], "description": "Type of transformation to apply."},
                                    "columns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of columns to apply the transformation to."
                                    },
                                    "output_prefix": {"type": "string", "description": "Optional: Prefix for new columns created by transformations like one-hot encoding."}
                                },
                                "required": ["transformation_type", "columns"]
                            }
                        ),
                        FunctionDeclaration(
                            name="get_dataframe_info",
                            description="Retrieves a summary of the current DataFrame, including column names, data types, and non-null counts. Useful for understanding the dataset structure.",
                            parameters={"type": "object", "properties": {}}
                        ),
                        FunctionDeclaration(
                            name="get_sample_data",
                            description="Retrieves a small sample (e.g., first 5 rows) of the current DataFrame. Useful for quick inspection.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "num_rows": {"type": "integer", "description": "Number of rows to sample (default: 5). Max 100 rows."}
                                }
                            }
                        ),
                        FunctionDeclaration(
                            name="analyze_data",
                            description="Analyzes a specific column or the entire dataset, providing summary statistics or missing value information.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string", "description": "Optional: The column to analyze. If not provided, a general overview is given."},
                                    "type": {"type": "string", "enum": ["summary", "missing", "types"], "description": "Type of analysis to perform (summary, missing, types). Defaults to 'summary'."}
                                }
                            }
                        ),
                        FunctionDeclaration(
                            name="filter_data",
                            description="Filters the dataset based on a specified column, condition, and value.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string", "description": "The column to filter by."},
                                    "condition": {"type": "string", "enum": ["equals", "greater_than", "less_than"], "description": "The filtering condition."},
                                    "value": {"description": "The value to filter against."}
                                },
                                "required": ["column", "condition", "value"]
                            }
                        ),
                        FunctionDeclaration(
                            name="get_statistics",
                            description="Retrieves general statistics about the dataset, such as row count, column count, or missing values.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["overview", "missing", "types"], "description": "Type of statistics to retrieve (overview, missing, types). Defaults to 'overview'."}
                                }
                            }
                        )
                    ]
                )
            ]
        )
        print("Vertex AI model initialized successfully!")
    except Exception as e:
        print(f"Could not initialize Vertex AI model: {e}")
        vertex_ai_model = None
else:
    print("Google Cloud Project or Location not set. Vertex AI features will be unavailable.")

# --- Helper functions ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls', 'json', 'tsv'}

def get_df_info(df, filename='Unknown'):
    num_rows = int(df.shape[0])  # Convert to regular Python int
    num_columns = int(df.shape[1])  # Convert to regular Python int
    
    missing_values_count = int(df.isnull().sum().sum())  # Convert to regular Python int
    duplicate_rows_count = int(df.duplicated().sum())  # Convert to regular Python int
    
    column_info_list = []
    for col in df.columns:
        column_info_list.append({
            'name': str(col),
            'dtype': str(df[col].dtype),
            'unique_values': int(df[col].nunique()),  # Convert to regular Python int
            'missing_count': int(df[col].isnull().sum())  # Convert to regular Python int
        })

    return {
        'filename': str(filename),
        'rows': num_rows,
        'columns': num_columns,
        'missing_values': missing_values_count,
        'duplicate_rows': duplicate_rows_count,
        'data_quality_score': 'Good',
        'column_info': column_info_list
    }

def get_sample_for_ai(df, num_rows=5):
    MAX_SAMPLE_ROWS = 100
    if num_rows > MAX_SAMPLE_ROWS:
        num_rows = MAX_SAMPLE_ROWS
    
    sample_df = df.head(num_rows)
    return sample_df.to_dict('records')

def execute_tool_call(tool_name, tool_args, df):
    """Executes a tool call based on the AI model's function call."""
    try:
        if tool_name == "clean_data":
            cleaned_df, cleaning_report_message = data_cleaning.clean_data(df.copy())
            if tool_args.get('remove_outliers'):
                z_thresh = tool_args.get('z_score_threshold', 5)
                numerical_cols = cleaned_df.select_dtypes(include=np.number).columns
                original_rows_before_outlier_removal = cleaned_df.shape[0]
                cleaned_df, outlier_message = data_cleaning.remove_outliers(cleaned_df, columns=list(numerical_cols), z_thresh=z_thresh)
                rows_removed = original_rows_before_outlier_removal - cleaned_df.shape[0]
                return {"status": "success", "message": f"Data cleaned. Removed {rows_removed} outliers.", "df": cleaned_df.to_dict('records')} # Return dict for session storage
            return {"status": "success", "message": "Data cleaning applied.", "df": cleaned_df.to_dict('records')} # Return dict for session storage
        
        elif tool_name == "analyze_numerical_data":
            # Corrected: Removed the 'columns' argument
            result = data_analysis.analyze_numerical_data(df)
            return {"status": "success", "result": result}
        
        elif tool_name == "analyze_categorical_data":
            result = data_analysis.analyze_categorical_data(df, columns=tool_args.get('columns'))
            return {"status": "success", "result": result}
        
        elif tool_name == "perform_correlation_analysis":
            result = data_analysis.analyze_correlations(df, method=tool_args.get('method', 'pearson'))
            return {"status": "success", "result": result}
            
        elif tool_name == "create_visualization":
            chart_spec = {
                "chart_type": tool_args.get('chart_type'),
                "x_column": tool_args.get('x_column'),
                "y_column": tool_args.get('y_column'),
                "title": tool_args.get('title'),
                "color_by": tool_args.get('color_by'),
                "facet_by": tool_args.get('facet_by'),
                "aggregate_by": tool_args.get('aggregate_by')
            }
            # Directly call the chart creation function
            chart_html = create_chart_from_function_call(df, chart_spec)
            return {"status": "success", "message": "Visualization created.", "chart_html": chart_html}

        elif tool_name == "remove_outliers":
            z_thresh = tool_args.get('z_score_threshold', 3.0)
            numerical_cols = df.select_dtypes(include=np.number).columns
            columns_to_process = tool_args.get('columns', list(numerical_cols))
            
            original_rows_before_outlier_removal = df.shape[0]
            cleaned_df, outlier_message = data_cleaning.remove_outliers(df.copy(), columns=columns_to_process, z_thresh=z_thresh)
            rows_removed = original_rows_before_outlier_removal - cleaned_df.shape[0]
            return {"status": "success", "message": f"Removed {rows_removed} outliers. {outlier_message}", "df": cleaned_df.to_dict('records')}

        elif tool_name == "apply_transformation":
            transformed_df, message = data_transformation.apply_transformation(
                df.copy(),
                tool_args.get('transformation_type'),
                tool_args.get('columns'),
                output_prefix=tool_args.get('output_prefix')
            )
            return {"status": "success", "message": message, "df": transformed_df.to_dict('records')}

        elif tool_name == "get_dataframe_info":
            info = get_df_info(df, "current_dataframe")
            return {"status": "success", "info": info}

        elif tool_name == "get_sample_data":
            sample = get_sample_for_ai(df, num_rows=tool_args.get('num_rows', 5))
            return {"status": "success", "sample": sample}
        
        elif tool_name == "analyze_data":
            return {"status": "success", "result": analyze_data(df, tool_args)}

        elif tool_name == "filter_data":
            return {"status": "success", "result": filter_data(df, tool_args)}

        elif tool_name == "get_statistics":
            return {"status": "success", "result": get_statistics(df, tool_args)}

        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

    except Exception as e:
        app.logger.error(f"Error executing tool '{tool_name}': {traceback.format_exc()}")
        return {"status": "error", "message": f"Error executing tool '{tool_name}': {str(e)}"}

def read_complex_file(file_stream, filename):
    file_ext = filename.rsplit('.', 1)[1].lower()
    df = None
    try:
        if file_ext == 'csv':
            # Store the original stream position
            original_pos = file_stream.tell()
            
            # List of delimiters to try, in order of preference
            possible_delimiters = [',', ';', '\t'] # comma, semicolon, tab
            for delim in possible_delimiters:
                try:
                    file_stream.seek(original_pos) # Reset stream for each try
                    df = pd.read_csv(file_stream, encoding='utf-8', sep=delim)
                    # If read_csv succeeds and DataFrame is not empty, break the loop
                    if not df.empty:
                        break
                except Exception as e_delim:
                    # Log the error for this delimiter but continue to the next
                    app.logger.debug(f"Failed to read CSV with delimiter '{delim}': {e_delim}")
            
            # If df is still None or empty after trying all delimiters, raise an error
            if df is None or df.empty:
                raise Exception("Could not read CSV file with common delimiters. It might be malformed or use an unsupported delimiter.")
                
        elif file_ext == 'xlsx' or file_ext == 'xls':
            df = pd.read_excel(file_stream)
        else:
            raise Exception(f"Unsupported file type: {file_ext}")
            
        # Basic validation after reading
        if df is None or df.empty:
            raise Exception("File is empty or could not be read into a DataFrame.")
            
        # Replace non-standard NaN representations
        df.replace(['', 'NA', 'N/A', 'NaN', 'null'], np.nan, inplace=True)
        
        # Handle problematic columns for JSON serialization
        df_safe = df.copy()
        for col in df_safe.columns:
            try:
                # Test if column can be JSON serialized
                df_safe[col].to_json()
            except (TypeError, ValueError) as json_error:
                # If not, convert to string and skip the problematic column
                print(f"Warning: Converting problematic column '{col}' to string due to: {json_error}")
                try:
                    df_safe[col] = df_safe[col].astype(str)
                except Exception as convert_error:
                    print(f"Warning: Could not convert column '{col}', filling with 'DATA_ERROR': {convert_error}")
                    df_safe[col] = 'DATA_ERROR'
        
        return df_safe
        
    except Exception as e:
        app.logger.error(f"Error reading file: {traceback.format_exc()}")
        raise Exception(f"Error reading {file_ext.upper()} file: {str(e)}")

# Locate this function in your app3.py and replace it.
def serialize_content(content):
    if content is None:
        return None
    
    serialized_parts = []
    for part in content.parts:
        if hasattr(part, 'text') and part.text is not None:
            serialized_parts.append({"text": part.text})
        elif hasattr(part, 'function_call') and part.function_call is not None:
            serialized_parts.append({
                "function_call": {
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args)
                }
            })
        elif hasattr(part, 'function_response') and part.function_response is not None:
            # CRITICAL FIX: Ensure the response payload is fully JSON-serializable
            # by converting it to a JSON string and then back to a Python object.
            try:
                # Use json_util.dumps (from bson) to convert potentially complex Python/upb objects
                # into a JSON string, handling various types that pickle might struggle with.
                serialized_response_str = json_util.dumps(part.function_response.response)
                # Then, use json.loads to parse that JSON string back into a pure Python dict/list structure.
                deserialized_response_for_pickle = json.loads(serialized_response_str)
            except Exception as e:
                print(f"Error during deep serialization of function_response payload: {e}")
                # As a fallback, convert to string if deep serialization fails (less ideal but prevents crash)
                deserialized_response_for_pickle = str(part.function_response.response)

            serialized_parts.append({
                "function_response": {
                    "name": part.function_response.name,
                    "response": deserialized_response_for_pickle # Store the fully serialized payload
                }
            })
        else:
            print(f"Warning: serialize_content encountered an unhandled or empty part type: {type(part)}")
            pass
            
    return {
        "role": content.role,
        "parts": serialized_parts
    }


def deserialize_content(serialized_content):
    """Converts a serializable dictionary back to a vertexai.generative_models.Content object."""
    if not isinstance(serialized_content, dict) or 'role' not in serialized_content or 'parts' not in serialized_content:
        app.logger.warning(f"Attempted to deserialize malformed content: {serialized_content}")
        return None

    deserialized_parts = []
    for part_data in serialized_content['parts']:
        if 'text' in part_data:
            deserialized_parts.append(Part.from_text(part_data['text']))
        elif 'function_call' in part_data:
            fc_data = part_data['function_call']
            # Reconstruct FunctionCall object directly
            function_call = vertexai.generative_models.FunctionCall(name=fc_data['name'], **fc_data.get('args', {}))
            deserialized_parts.append(Part.from_function_call(function_call))
        elif 'function_response' in part_data:
            fr_data = part_data['function_response']
            # Reconstruct FunctionResponse object directly
            function_response = vertexai.generative_models.FunctionResponse(name=fr_data['name'], response=fr_data.get('response', {}))
            deserialized_parts.append(Part.from_function_response(function_response))
        else:
            app.logger.warning(f"Skipping unknown part type during deserialization: {part_data}")
            continue

    return Content(role=serialized_content['role'], parts=deserialized_parts)


# --- Routes ---

@app.route('/')
@login_required
def index():
    if 'username' in session:  # User is logged in
        return render_template('index.html', app_name=app.config['APPLICATION_NAME'])
    else:  # User is not logged in
        return render_template('new-page.html', app_name=app.config['APPLICATION_NAME'])

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            
            # Read the file
            df = read_complex_file(file.stream, filename)
            
            # Validate DataFrame
            if df.empty:
                return jsonify({'error': 'The uploaded file is empty or could not be processed.'}), 400
            
            # --- AUTOMATED PROCESSING STARTS HERE ---
            initial_rows = df.shape[0]
            initial_missing = df.isnull().sum().sum()

            # 1. Automated Data Cleaning
            cleaned_df, cleaning_report_message = data_cleaning.clean_data(df.copy())
            
            # Fixed outlier removal section
            numerical_cols = cleaned_df.select_dtypes(include=np.number).columns
            total_outliers_removed = 0
            
            if len(numerical_cols) > 0:  # Check if there are numerical columns
                original_rows_before_outlier_removal = cleaned_df.shape[0]
                
                # Apply outlier removal to all numerical columns at once
                cleaned_df, outlier_message = data_cleaning.remove_outliers(cleaned_df, columns=list(numerical_cols), method='iqr', z_thresh=5)
                total_outliers_removed = original_rows_before_outlier_removal - cleaned_df.shape[0]
                print(f"Automated outlier removal: Removed {total_outliers_removed} outliers.")

            rows_after_cleaning = cleaned_df.shape[0]
            missing_after_cleaning = cleaned_df.isnull().sum().sum()
            cleaning_message = (f"Data cleaning applied automatically! "
                                f"Original rows: {initial_rows:,}, Cleaned rows: {rows_after_cleaning:,}. "
                                f"Missing values reduced from {initial_missing:,} to {missing_after_cleaning:,}.")
            
            # 2. Automated Insight Generation
            overall_summary = data_analysis.get_overall_summary(cleaned_df)
            numerical_analysis = data_analysis.analyze_numerical_data(cleaned_df)
            categorical_analysis = data_analysis.analyze_categorical_data(cleaned_df)
            correlation_analysis = data_analysis.analyze_correlations(cleaned_df)
            
            all_insights = {
                "summary": overall_summary,
                "numerical_stats": numerical_analysis,
                "categorical_stats": categorical_analysis,
                "correlations": correlation_analysis
            }
            
            # 3. Generate AI Summary
            ai_insights_summary = "AI summary generation skipped for automated upload. Can be generated via chat."
            if vertex_ai_model:
                try:
                    ai_analysis_prompt = f"""
                    You are an expert data analyst. Based on the initial automated data cleaning and analysis results for the dataset '{filename}',
                    provide a concise overview (around 100-150 words). Highlight key changes from cleaning, initial data quality, and any immediate interesting observations.
                    
                    Automated Data Analysis Results (JSON format):
                    {json.dumps(all_insights, indent=2)}
                    """
                    chat_session_auto = vertex_ai_model.start_chat() # Use a separate chat session for automated
                    response = chat_session_auto.send_message(ai_analysis_prompt)
                    ai_insights_summary = response.text
                except Exception as e:
                    ai_insights_summary = f"Error generating automated AI summary: {str(e)}"
                    app.logger.error(f"Automated AI Insight summary generation error: {traceback.format_exc()}")

            # 4. Generate Dashboard Explanation (NEW)
            dashboard_explanation = "Data insights and patterns visualization"
            if vertex_ai_model:
                try:
                    explanation_prompt = f"""
                    Based on this data analysis, provide ONE sentence (maximum 15 words) explaining what this dataset is about.
                    
                    Dataset: {filename}
                    Rows: {cleaned_df.shape[0]}
                    Columns: {list(cleaned_df.columns)}
                    Sample Data: {cleaned_df.head(3).to_dict('records')}
                    
                    Respond with just one clear sentence about what this data represents.
                    """
                    
                    explanation_chat = vertex_ai_model.start_chat()
                    explanation_response = explanation_chat.send_message(explanation_prompt)
                    dashboard_explanation = explanation_response.text.strip()
                    
                    # Clean up explanation - remove quotes and extra text
                    dashboard_explanation = dashboard_explanation.replace('"', '').replace("'", "")
                    if len(dashboard_explanation) > 100:
                        dashboard_explanation = dashboard_explanation[:97] + "..."
                        
                except Exception as e:
                    dashboard_explanation = "Data insights and patterns visualization"
                    app.logger.error(f"Dashboard explanation generation error: {traceback.format_exc()}")

            # Add explanation to insights
            all_insights['dashboard_explanation'] = dashboard_explanation

            # 5. Automated Dashboard Generation
            df_for_dashboard = cleaned_df.copy()
            dashboard_html = data_visualization.generate_dashboard_html(df_for_dashboard, all_insights)
            
            # Save dashboard HTML to a file and return its URL
            dashboard_filename = f"dashboard_{uuid.uuid4().hex}.html"
            dashboard_path = os.path.join(app.root_path, 'static', 'dashboards', dashboard_filename)
            os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            dashboard_url = url_for('static', filename=f'dashboards/{dashboard_filename}')

            # --- STORE PROCESSED DATA IN SESSION ---
            dataset_id = str(uuid.uuid4())
            session[dataset_id] = {
                'df': cleaned_df.to_dict('records'),
                'info': get_df_info(cleaned_df, filename),
                'chat_history': [],
                'upload_timestamp': datetime.now().isoformat(),
                'dashboard_url': dashboard_url,
                'initial_insights': all_insights
            }
            session['active_dataset_id'] = dataset_id
            
            # Log to MongoDB if available
            if mongo_client:
                try:
                    db = get_db()
                    db.upload_logs.insert_one({
                        'dataset_id': dataset_id,
                        'filename': filename,
                        'upload_time': datetime.now(),
                        'rows': cleaned_df.shape[0],
                        'columns': cleaned_df.shape[1],
                        'user_session': session.get('username', 'anonymous'),
                        'automated_cleaning_summary': cleaning_message,
                        'initial_ai_summary': ai_insights_summary,
                        'dashboard_url': dashboard_url,
                        'dashboard_explanation': dashboard_explanation
                    })
                except Exception as e:
                    app.logger.warning(f"Could not log to MongoDB: {e}")
            
            return jsonify({
                'message': f'File processed automatically by GemNInk!',
                'dataset_id': dataset_id,
                'df_info': session[dataset_id]['info'],
                'cleaning_summary': cleaning_message,
                'initial_ai_summary': ai_insights_summary,
                'dashboard_url': dashboard_url,
                'dashboard_explanation': dashboard_explanation,
                'initial_insights': all_insights
            }), 200
            
        except Exception as e:
            app.logger.error(f"Error processing file upload: {traceback.format_exc()}")
            return jsonify({'error': f'GemNInk failed to process file automatically: {str(e)}'}), 500
    return jsonify({'error': 'File type not supported by GemNInk. Supported formats: CSV, Excel, JSON, TSV'}), 400

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        db = get_db()
        if db is not None: # Corrected: check for None explicitly
            users_collection = db.users
            user = users_collection.find_one({'username': username})

            # Corrected: Check for 'password' key instead of 'password_hash'
            if user and user.get('password') is not None:
                if check_password_hash(user['password'], password): # Corrected: use 'password' key
                    session['logged_in'] = True
                    session['username'] = username
                    flash('Logged in successfully!', 'success')
                    return redirect(url_for('index'))
                else:
                    flash('Invalid username or password', 'danger')
            else:
                flash('Invalid username or password', 'danger')
        else:
            flash('Database not available.', 'danger')
            return render_template('login.html')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        
        db = get_db()
        if db is not None: # Corrected: check for None explicitly
            users_collection = db.users

            if users_collection.find_one({'username': username}):
                flash('Username already exists. Please choose a different one.', 'danger')
                return render_template('register.html')

            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            # Corrected: Store hash under 'password' key
            users_collection.insert_one({'username': username, 'password': hashed_password})
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Database not available.', 'danger')
            return render_template('register.html')
    return render_template('register.html')



@app.route('/api/process_step/<dataset_id>', methods=['POST'])
@login_required
def process_step(dataset_id):
    if dataset_id not in session or 'df' not in session[dataset_id]:
        return jsonify({'error': 'Dataset not found in session for processing. Please upload a dataset first.'}), 404

    df_data = session[dataset_id]['df']
    if not isinstance(df_data, list):
        return jsonify({'error': 'Internal server error: DataFrame data not in expected format in session.'}), 500
    df = pd.DataFrame(df_data)

    original_filename = session[dataset_id]['info'].get('filename', 'Unknown')
    
    data = request.json
    step_type = data.get('step_type') 
    
    if not step_type:
        return jsonify({'error': 'Missing step_type in request payload.'}), 400

    try:
        if step_type == 'data_cleaning':
            initial_rows = df.shape[0]
            initial_missing = df.isnull().sum().sum()

            cleaned_df, cleaning_message = data_cleaning.clean_data(df.copy())
            
            if data.get('remove_outliers'):
                z_thresh = data.get('z_score_threshold', 5)
                numerical_cols = cleaned_df.select_dtypes(include=np.number).columns
                original_rows_before_outlier_removal = cleaned_df.shape[0]
                cleaned_df, outlier_message = data_cleaning.remove_outliers(cleaned_df, columns=list(numerical_cols), z_thresh=z_thresh)
                if cleaned_df.shape[0] < original_rows_before_outlier_removal:
                    print(f"Removed {original_rows_before_outlier_removal - cleaned_df.shape[0]} outliers.")
            
            session[dataset_id]['df'] = cleaned_df.to_dict('records')
            session[dataset_id]['info'] = get_df_info(cleaned_df, original_filename)

            rows_after_cleaning = cleaned_df.shape[0]
            missing_after_cleaning = cleaned_df.isnull().sum().sum()
            
            message = (f"Data cleaning applied successfully! "
                       f"Original rows: {initial_rows:,}, Cleaned rows: {rows_after_cleaning:,}. "
                       f"Missing values reduced from {initial_missing:,} to {missing_after_cleaning:,}.")
            
            return jsonify({
                "message": message,
                "dataset_id": dataset_id,
                "df_info": session[dataset_id]['info'] 
            }), 200

        elif step_type == 'data_transformation':
            transformation_type = data.get('transformation_type')
            transformation_params = data.get('params', {})

            if not transformation_type:
                return jsonify({'error': 'Missing transformation_type for data_transformation step.'}), 400
            
            # Pass transformation_params correctly as a dictionary to apply_transformation
            # The apply_transformation function in data_transformation.py needs specific args.
            # You might need to adjust this depending on how apply_transformation expects params.
            transformed_df, message = data_transformation.apply_transformation(
                df.copy(),
                transformation_type,
                columns=transformation_params.get('columns', []), # Example: expecting 'columns' key
                output_prefix=transformation_params.get('output_prefix') # Example: expecting 'output_prefix' key
            )

            session[dataset_id]['df'] = transformed_df.to_dict('records')
            session[dataset_id]['info'] = get_df_info(transformed_df, original_filename)
            
            return jsonify({
                "message": f"Data transformation '{transformation_type}' applied. {message}",
                "dataset_id": dataset_id,
                "df_info": session[dataset_id]['info'] 
            }), 200
            
        elif step_type == 'insight_generation':
            overall_summary = data_analysis.get_overall_summary(df)
            numerical_analysis = data_analysis.analyze_numerical_data(df)
            categorical_analysis = data_analysis.analyze_categorical_data(df)
            correlation_analysis = data_analysis.analyze_correlations(df)
            
            feature_importance_analysis = {"message": "No target column specified for feature importance or no suitable numeric target found."}
            potential_targets = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1]
            
            requested_target = data.get('target_column') 

            if requested_target and requested_target in potential_targets:
                feature_importance_analysis = data_analysis.analyze_feature_importance(df, requested_target)
            elif potential_targets: 
                default_target = potential_targets[0]
                feature_importance_analysis = data_analysis.analyze_feature_importance(df, default_target)
                feature_importance_analysis['note'] = f"Feature importance calculated using default target: '{default_target}'. To specify a different target, pass 'target_column' in the request."
            
            all_insights = {
                "summary": overall_summary,
                "numerical_stats": numerical_analysis,
                "categorical_stats": categorical_analysis,
                "correlations": correlation_analysis,
                "feature_importance": feature_importance_analysis
            }
            
            ai_insights_summary = "No AI summary generated (AI model not configured or prompt failed)."
            if vertex_ai_model:
                try:
                    ai_analysis_prompt = f"""
                    You are an expert data analyst. Based on the following structured data analysis results for the dataset '{original_filename}',
                    provide a concise, actionable summary (around 200-300 words).
                    Highlight:
                    - Key findings (e.g., significant correlations, important features).
                    - Potential data quality issues (e.g., high missing values in critical columns, high duplicate rows).
                    - Interesting trends or observations from the statistics.
                    - Suggestions for next steps, deeper investigations, or relevant visualizations.

                    Data Analysis Results (JSON format):
                    {json.dumps(all_insights, indent=2)}
                    """
                    
                    chat_session_insight = vertex_ai_model.start_chat() # Use a separate chat session
                    response = chat_session_insight.send_message(ai_analysis_prompt)
                    ai_insights_summary = response.text
                except Exception as e:
                    ai_insights_summary = f"Error generating AI summary: {str(e)}"
                    app.logger.error(f"AI Insight summary generation error: {traceback.format_exc()}")

            return jsonify({
                "message": "Insight generation complete.",
                "dataset_id": dataset_id,
                "insights": all_insights,
                "ai_summary": ai_insights_summary
            }), 200

        elif step_type == 'report_dashboard':
            current_df = pd.DataFrame(session[dataset_id]['df'])
            
            # Generate dashboard HTML
            dashboard_title = data.get('title', f"Dashboard for {original_filename}")
            dashboard_html = data_visualization.create_dashboard_html(current_df, "Generic Data Overview") # Always use generic for now
            
            # Save dashboard HTML to a file
            dashboard_filename = f"dashboard_{uuid.uuid4().hex}.html"
            dashboard_path = os.path.join(app.root_path, 'static', 'dashboards', dashboard_filename)
            os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
            
            with open(dashboard_path, 'w') as f:
                f.write(dashboard_html)
            
            dashboard_url = url_for('static', filename=f'dashboards/{dashboard_filename}')
            
            # Update session with dashboard URL
            session[dataset_id]['dashboard_url'] = dashboard_url
            
            return jsonify({
                "message": "Dashboard generated successfully.",
                "dataset_id": dataset_id,
                "dashboard_url": dashboard_url
            }), 200

        else:
            return jsonify({'error': f'Unknown step_type: {step_type}'}), 400

    except Exception as e:
        app.logger.error(f"Error processing step '{step_type}': {traceback.format_exc()}")
        return jsonify({'error': f'Error processing step: {str(e)}'}), 500


@app.route('/api/chat/<dataset_id>', methods=['POST'])
@login_required
def chat(dataset_id):
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Retrieve dataset from session
        if dataset_id not in session or 'df' not in session[dataset_id]:
            return jsonify({"error": "No data available for this dataset ID. Please upload a file first or select an active dataset."}), 400
        
        df = pd.DataFrame(session[dataset_id]['df'])
        chat_history = session[dataset_id].get('chat_history', [])

        # Deserialize chat history for Vertex AI
        vertex_ai_chat_history = [deserialize_content(item) for item in chat_history if deserialize_content(item) is not None]
        
        # Start or continue chat session with the model
        chat_session = vertex_ai_model.start_chat(history=vertex_ai_chat_history)
        
        # Create context message with data info
        data_info = f"""
        You are an expert data analyst assistant. The current dataset has {len(df)} rows and {len(df.columns)} columns.
        Here's a summary of its structure:
        - Columns: {', '.join(df.columns.tolist())}
        - Column types: {df.dtypes.to_dict()}
        - Sample data (first 5 rows): {get_sample_for_ai(df, 5)}
        
        Based on this data, please respond to the user's request.Make sure none of your answers are more than 20 words at any time. Keep simple and short If a function call is needed, propose it.
        """
        
        # Send data context and user message to AI
        response = chat_session.send_message([data_info, user_message]) # Send context and user message
        
        # Initialize response components
        ai_response_text = ""
        function_results = []
        updated_df_data = df.to_dict('records') # Start with current df data
        
        # Handle the response
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            if hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    # Handle text parts
                    if hasattr(part, 'text') and part.text:
                        ai_response_text += part.text
                    
                    # Handle function calls
                    elif hasattr(part, 'function_call') and part.function_call:
                        func_name = part.function_call.name
                        func_args = dict(part.function_call.args) if part.function_call.args else {}
                        
                        # Execute the function call with the current DataFrame
                        tool_execution_result = execute_tool_call(func_name, func_args, pd.DataFrame(updated_df_data))
                        
                        # Update df_data if the tool returned a modified DataFrame
                        if tool_execution_result.get('df') is not None:
                            updated_df_data = tool_execution_result['df']
                            # Update session DataFrame and info immediately if DF changed
                            session[dataset_id]['df'] = updated_df_data
                            session[dataset_id]['info'] = get_df_info(pd.DataFrame(updated_df_data), session[dataset_id]['info'].get('filename', 'Unknown'))
                            tool_execution_result['message'] = tool_execution_result.get('message', '') + " (DataFrame updated in session)."
                            
                        function_results.append({
                            'function': func_name,
                            'args': func_args,
                            'result': tool_execution_result
                        })

                        # Respond to the model with tool output
                        response_from_tool = chat_session.send_message(
                            Content(
                                role="function",
                                parts=[Part.from_function_response(name=func_name, response=tool_execution_result)]
                            )
                        )
                        # Append any new text response from the model after tool execution
                        if hasattr(response_from_tool, 'candidates') and response_from_tool.candidates:
                            for res_part in response_from_tool.candidates[0].content.parts:
                                if hasattr(res_part, 'text') and res_part.text:
                                    ai_response_text += "\n" + res_part.text
                                elif hasattr(res_part, 'function_call') and res_part.function_call:
                                    # This indicates the model is making another function call after tool output.
                                    # For this example, we're not set up for nested tool calls within one turn
                                    # but in a more advanced multi-turn system, you might handle it here.
                                    print(f"Debug: Model suggested another function call after tool execution: {res_part.function_call.name}")
                                    pass # Ignore for now, as it's not text output
                                elif hasattr(res_part, 'function_response') and res_part.function_response:
                                    # This is the problematic case where the model echoes the function_response back.
                                    # We explicitly ignore it for text aggregation.
                                    print(f"Debug: Model incorrectly returned a function_response part: {res_part.function_response.name}")
                                    pass
                                else:
                                    # Log any other unexpected part types for debugging
                                    print(f"Debug: Unhandled part type in model's follow-up response: {type(res_part)}")
        
        # If no text but we have function results, create a response
        if not ai_response_text and function_results:
            ai_response_text = "I've processed your request. Here are the results:"
        
        # Build the complete response
        complete_response = ai_response_text
        
        # Add function results to response, especially if they contain HTML for charts
        for func_result in function_results:
            if func_result['function'] == 'create_visualization' and func_result['result'].get('chart_html'):
                complete_response += f"\n\n{func_result['result']['chart_html']}"
            elif func_result['function'] == 'perform_comprehensive_analysis': # Handle our single comprehensive tool
                analysis_output = func_result['result']
                analysis_type = analysis_output.get('analysis_type')
                error_message = analysis_output.get('error')
                
                if error_message:
                    complete_response += f"\n\nAnalysis encountered an error: {error_message}"
                else:
                    if analysis_type == 'describe':
                        complete_response += "\n\n**Dataset Description:**\n"
                        description_result = analysis_output.get('result', {})
                        for col, stats_dict in description_result.items():
                            complete_response += f"- **{col}**:\n"
                            for stat, val in stats_dict.items():
                                if isinstance(val, dict): # Handle nested dicts for some stats (e.g., 'top', 'freq')
                                    complete_response += f"  - {stat}: {val.get('top', str(val))}\n" # Simplified, adjust as needed
                                else:
                                    complete_response += f"  - {stat}: {val}\n"
                    elif analysis_type == 'aggregate':
                        col_agg = analysis_output.get('column_aggregated')
                        agg_type = analysis_output.get('aggregation_type')
                        group_by = analysis_output.get('grouped_by')
                        result = analysis_output.get('result')

                        if group_by:
                            complete_response += f"\n\n**{agg_type.capitalize()} of {col_agg} grouped by {group_by}:**\n"
                            # Sort grouped results for consistent display, especially for numerical groups like hours
                            sorted_results = sorted(result.items(), key=lambda item: item[0]) if all(isinstance(k, (int, float)) for k in result.keys()) else result.items()
                            for group, val in sorted_results:
                                complete_response += f"- {group}: {val}\n"
                        else:
                            complete_response += f"\n\n**Overall {agg_type.capitalize()} of {col_agg}:** {result}\n"
                    elif analysis_type == 'filter':
                        count = analysis_output.get('filtered_rows_count')
                        sample = analysis_output.get('sample_data')
                        complete_response += f"\n\n**Filtered Data:** {count} rows match your criteria.\n"
                        if sample:
                            complete_response += "Here's a sample of the filtered data:\n```json\n" + json.dumps(sample, indent=2, default=json_util.default) + "\n```\n"
                    elif analysis_type == 'sort':
                        message = analysis_output.get('message')
                        sample = analysis_output.get('sample_data')
                        complete_response += f"\n\n**{message}**\n"
                        if sample:
                            complete_response += "Here's a sample of the sorted data:\n```json\n" + json.dumps(sample, indent=2, default=json_util.default) + "\n```\n"
                    elif analysis_type == 't_test':
                        test_type = analysis_output.get('test_type')
                        t_stat = analysis_output.get('t_statistic')
                        p_val = analysis_output.get('p_value')
                        
                        complete_response += f"\n\n**T-Test Result ({test_type.replace('_', ' ').title()}):**\n"
                        complete_response += f"  - T-Statistic: {t_stat}\n"
                        complete_response += f"  - P-Value: {p_val}\n"
                        if p_val is not None:
                            if p_val < 0.05:
                                complete_response += "  (Result is statistically significant at alpha=0.05)\n"
                            else:
                                complete_response += "  (Result is not statistically significant at alpha=0.05)\n"
                    elif analysis_type == 'anova':
                        value_col = analysis_output.get('value_column')
                        group_col = analysis_output.get('group_column')
                        f_stat = analysis_output.get('f_statistic')
                        p_val = analysis_output.get('p_value')
                        
                        complete_response += f"\n\n**ANOVA Result (for {value_col} by {group_col}):**\n"
                        complete_response += f"  - F-Statistic: {f_stat}\n"
                        complete_response += f"  - P-Value: {p_val}\n"
                        if p_val is not None:
                            if p_val < 0.05:
                                complete_response += "  (Result is statistically significant at alpha=0.05)\n"
                            else:
                                complete_response += "  (Result is not statistically significant at alpha=0.05)\n"
                    elif analysis_type == 'unique_values':
                        column = analysis_output.get('column')
                        values = analysis_output.get('unique_values')
                        count = analysis_output.get('count')
                        complete_response += f"\n\n**Unique values for '{column}' ({count} unique values):**\n"
                        complete_response += ", ".join(map(str, values[:10])) # Show first 10
                        if count > 10:
                            complete_response += f"...\n(Showing top 10 of {count} unique values)"
                        complete_response += "\n"
                    elif analysis_type == 'feature_correlation':
                        col1 = analysis_output.get('column1')
                        col2 = analysis_output.get('column2')
                        correlation = analysis_output.get('correlation')
                        complete_response += f"\n\n**Correlation between '{col1}' and '{col2}':** {correlation}\n"
                    else:
                        # Fallback for unknown analysis types or general results
                        try:
                            result_str = json.dumps(analysis_output, indent=2, default=json_util.default)
                        except TypeError:
                            result_str = str(analysis_output)
                        complete_response += f"\n\nTool execution result:\n```json\n{result_str}\n```"
            else:
                # Existing fallback for other non-perform_comprehensive_analysis tool results (e.g., visualizations)
                try:
                    result_str = json.dumps(func_result['result'], indent=2, default=json_util.default)
                except TypeError:
                    result_str = str(func_result['result']) # Fallback to string representation
                complete_response += f"\n\nFunction '{func_result['function']}' executed with result:\n```json\n{result_str}\n```"

        # Update conversation history in session (serialize content)
        # Store user message
        chat_history.append(serialize_content(Content(role="user", parts=[Part.from_text(user_message)])))
        # Store AI's initial response (if any text) and function call
        if ai_response_text and not function_results: # If only text, store it
            chat_history.append(serialize_content(Content(role="model", parts=[Part.from_text(ai_response_text)])))
        elif function_results: # If function calls, store the model's function call part
            # Reconstruct the model's original content including function calls for history
            model_parts = []
            if ai_response_text:
                model_parts.append(Part.from_text(ai_response_text))
            for func_res in function_results:
                model_parts.append(Part.from_function_response(name=func_res['function'], response=func_res['result'])) # Store tool output as well
            chat_history.append(serialize_content(Content(role="model", parts=model_parts)))

        session[dataset_id]['chat_history'] = chat_history
        
        return jsonify({
            "response": complete_response,
            "function_calls": function_results,
            "df_info": session[dataset_id]['info'] # Return updated DF info
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Chat error: {str(e)}"}), 500


def create_chart_from_function_call(df, args): # Now accepts df as argument
    """Create chart based on AI function call arguments"""
    try:
        if df is None or df.empty:
            return "<p style='color: red;'>No data available to create chart.</p>"
        
        x_col = args.get('x_column')
        y_col = args.get('y_column')
        chart_type = args.get('chart_type', 'bar').lower()
        title = args.get('title')
        color_by = args.get('color_by')
        
        print(f"Creating chart: {chart_type}, x={x_col}, y={y_col}, title={title}, color={color_by}")
        
        # Validate columns exist
        if x_col and x_col not in df.columns:
            available_cols = list(df.columns)
            return f"<p style='color: red;'>Column '{x_col}' not found. Available columns: {available_cols}</p>"
        
        if y_col and y_col not in df.columns and y_col is not None:
            available_cols = list(df.columns)
            return f"<p style='color: red;'>Column '{y_col}' not found. Available columns: {available_cols}</p>"

        if color_by and color_by not in df.columns and color_by is not None:
            available_cols = list(df.columns)
            return f"<p style='color: red;'>Color-by column '{color_by}' not found. Available columns: {available_cols}</p>"
        
        fig = None
        
        if chart_type == 'bar':
            if x_col and y_col:
                # Bar chart with x and y columns
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    # Aggregate y values by x
                    grouped = df.groupby(x_col)[y_col].mean().reset_index()
                    fig = px.bar(grouped, x=x_col, y=y_col, 
                               title=f"Average {y_col} by {x_col}" if not title else title,
                               color=color_by)
                else:
                    # Cross-tabulation for categorical data
                    # For bar chart, if y_col is not numeric, we can show counts per x_col category
                    df_plot = df.groupby([x_col, y_col]).size().reset_index(name='count')
                    fig = px.bar(df_plot, x=x_col, y='count', color=y_col,
                               title=f"Counts of {y_col} by {x_col}" if not title else title)
            elif x_col:
                # Single column bar chart (frequency)
                value_counts = df[x_col].value_counts().head(20) # Limit to top 20 for readability
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Distribution of {x_col}" if not title else title,
                           labels={'x': x_col, 'y': 'Count'},
                           color=color_by if color_by and color_by in value_counts.index else None) # If color_by is one of the top values
            else:
                return "<p style='color: red;'>Please specify at least an x_column for bar chart</p>"
        
        elif chart_type == 'histogram':
            if x_col and pd.api.types.is_numeric_dtype(df[x_col]):
                fig = px.histogram(df, x=x_col, title=f"Histogram of {x_col}" if not title else title, color=color_by)
            else:
                return f"<p style='color: red;'>Column '{x_col}' must be numeric for histogram or x_column is missing</p>"
        
        elif chart_type == 'scatter':
            if x_col and y_col:
                if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    fig = px.scatter(df, x=x_col, y=y_col, 
                                   title=f"{y_col} vs {x_col}" if not title else title, color=color_by)
                else:
                    return f"<p style='color: red;'>Both columns must be numeric for scatter plot</p>"
            else:
                return "<p style='color: red;'>Please specify both x_column and y_column for scatter plot</p>"
        
        elif chart_type == 'line':
            if x_col and y_col:
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    # Sort by x column for proper line connection
                    df_sorted = df.sort_values(x_col)
                    fig = px.line(df_sorted, x=x_col, y=y_col, 
                                title=f"{y_col} over {x_col}" if not title else title, color=color_by)
                else:
                    return f"<p style='color: red;'>Y column '{y_col}' must be numeric for line chart</p>"
            else:
                return "<p style='color: red;'>Please specify both x_column and y_column for line chart</p>"
        
        elif chart_type == 'pie':
            if x_col:
                value_counts = df[x_col].value_counts().head(10) # Limit to top 10 categories
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Pie Chart of {x_col}" if not title else title)
            else:
                return "<p style='color: red;'>Please specify x_column for pie chart</p>"
        
        elif chart_type == 'box':
            if x_col and pd.api.types.is_numeric_dtype(df[x_col]):
                fig = px.box(df, y=x_col, title=f"Box Plot of {x_col}" if not title else title, color=color_by)
            elif x_col and y_col and pd.api.types.is_numeric_dtype(df[y_col]):
                 fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col} by {x_col}" if not title else title, color=color_by)
            else:
                return "<p style='color: red;'>Please specify a numeric column for box plot (y_column for grouping is optional).</p>"
        
        elif chart_type == 'heatmap':
            # For heatmap, typically used for correlation matrix or pivot table
            if x_col and y_col and pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                # Create a 2D histogram/heatmap for two numerical columns
                fig = go.Figure(go.Histogram2dContour(x=df[x_col], y=df[y_col], colorscale='Blues'))
                fig.update_layout(title=f"Heatmap of {y_col} vs {x_col}" if not title else title,
                                  xaxis_title=x_col, yaxis_title=y_col)
            elif len(df.select_dtypes(include=np.number).columns) > 1:
                # If no specific columns, show correlation heatmap for all numerical columns
                corr_matrix = df.select_dtypes(include=np.number).corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                title="Correlation Heatmap" if not title else title)
            else:
                return "<p style='color: red;'>Heatmap requires at least two numeric columns or specific x_column and y_column.</p>"

        else:
            return f"<p style='color: red;'>Unsupported chart type: {chart_type}. Supported types: bar, histogram, scatter, line, pie, box, heatmap</p>"
        
        if fig:
            # Style the chart
            fig.update_layout(
                height=500,
                margin=dict(l=50, r=50, t=80, b=50),
                font=dict(size=12),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Convert to HTML
            chart_html = fig.to_html(
                include_plotlyjs='cdn',
                div_id=f"chart_{abs(hash(str(args)))}",
                config={'displayModeBar': True, 'responsive': True}
            )
            
            return chart_html
        else:
            return "<p style='color: red;'>Failed to create chart with the specified parameters</p>"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<p style='color: red;'>Error creating chart: {str(e)}</p>"


def analyze_data(df, args):
    """Analyze data based on arguments"""
    try:
        column = args.get('column')
        analysis_type = args.get('type', 'summary')
        
        if column and column in df.columns:
            if analysis_type == 'summary':
                if pd.api.types.is_numeric_dtype(df[column]):
                    stats = df[column].describe()
                    return f"Summary statistics for {column}:\n{stats.to_string()}"
                else:
                    counts = df[column].value_counts().head(10)
                    return f"Top values for {column}:\n{counts.to_string()}"
            elif analysis_type == 'missing':
                missing = df[column].isnull().sum()
                total = len(df)
                pct = (missing / total) * 100
                return f"Missing values in {column}: {missing} out of {total} ({pct:.1f}%)"
            elif analysis_type == 'types': # Added for consistency with get_statistics
                return f"Data type for {column}: {df[column].dtype}"
        elif not column and analysis_type == 'types': # Handle overall types
            return f"Column data types:\n{df.dtypes.to_string()}"
        elif not column and analysis_type == 'summary': # Handle overall summary
            return data_analysis.get_overall_summary(df)
        
        return f"Could not analyze column: {column}. Column might not exist or analysis type is invalid."
        
    except Exception as e:
        return f"Error in data analysis: {str(e)}"


def filter_data(df, args):
    """Filter data based on arguments"""
    try:
        column = args.get('column')
        condition = args.get('condition')
        value = args.get('value')
        
        if column in df.columns:
            filtered_df = pd.DataFrame() # Initialize an empty DataFrame
            if condition == 'equals':
                filtered_df = df[df[column] == value]
            elif condition == 'greater_than':
                if pd.api.types.is_numeric_dtype(df[column]):
                    filtered_df = df[df[column] > value]
                else:
                    return f"Column '{column}' is not numeric for 'greater_than' comparison."
            elif condition == 'less_than':
                if pd.api.types.is_numeric_dtype(df[column]):
                    filtered_df = df[df[column] < value]
                else:
                    return f"Column '{column}' is not numeric for 'less_than' comparison."
            else:
                return f"Unknown condition: {condition}"
            
            return {"filtered_rows_count": len(filtered_df), "sample_data": get_sample_for_ai(filtered_df, 5)}
        
        return f"Column {column} not found"
        
    except Exception as e:
        return f"Error filtering data: {str(e)}"


def get_statistics(df, args):
    """Get statistics for the dataset"""
    try:
        stat_type = args.get('type', 'overview')
        
        if stat_type == 'overview':
            return {"rows": len(df), "columns": len(df.columns), "memory_usage_mb": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"}
        elif stat_type == 'missing':
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            result = {}
            for col in df.columns:
                result[col] = {"missing_count": int(missing[col]), "missing_percentage": round(float(missing_pct[col]), 2)}
            return result
        elif stat_type == 'types':
            return df.dtypes.astype(str).to_dict()
        
        return f"Statistics for type '{stat_type}' could not be generated."
        
    except Exception as e:
        return f"Error getting statistics: {str(e)}"


# Removed global current_dataframe and associated functions as data is now in session

@app.route('/api/datasets')
@login_required
def list_datasets():
    """List all datasets in the current session."""
    datasets = []
    for key, value in session.items():
        if isinstance(value, dict) and 'df' in value and 'info' in value:
            datasets.append({
                'dataset_id': key,
                'info': value['info'],
                'upload_timestamp': value.get('upload_timestamp'),
                'has_dashboard': 'dashboard_url' in value
            })
    
    active_dataset = session.get('active_dataset_id')
    return jsonify({
        'datasets': datasets,
        'active_dataset_id': active_dataset
    })

@app.route('/api/dataset/<dataset_id>')
@login_required
def get_dataset_info(dataset_id):
    """Get detailed information about a specific dataset."""
    if dataset_id not in session or 'df' not in session[dataset_id]:
        return jsonify({'error': 'Dataset not found.'}), 404
    
    dataset_info = session[dataset_id]
    df = pd.DataFrame(dataset_info['df'])
    
    # Deserialize chat history for display if needed, but not for direct use by AI
    serializable_chat_history = [
        serialize_content(deserialize_content(item)) # Re-serialize after deserializing to ensure consistent format
        for item in dataset_info.get('chat_history', []) 
        if deserialize_content(item) is not None
    ]

    return jsonify({
        'dataset_id': dataset_id,
        'info': dataset_info['info'],
        'sample_data': get_sample_for_ai(df, 10),
        'upload_timestamp': dataset_info.get('upload_timestamp'),
        'dashboard_url': dataset_info.get('dashboard_url'),
        'initial_insights': dataset_info.get('initial_insights'),
        'chat_history_length': len(dataset_info.get('chat_history', [])),
        'chat_history_preview': serializable_chat_history[-5:] # Last 5 messages for preview
    })

@app.route('/api/dataset/<dataset_id>/sample')
@login_required
def get_dataset_sample(dataset_id):
    """Get a sample of the dataset for preview."""
    if dataset_id not in session or 'df' not in session[dataset_id]:
        return jsonify({'error': 'Dataset not found.'}), 404
    
    df = pd.DataFrame(session[dataset_id]['df'])
    num_rows = min(request.args.get('rows', 20, type=int), 100)
    
    sample_data = df.head(num_rows).to_dict('records')
    
    return jsonify({
        'sample_data': sample_data,
        'total_rows': len(df),
        'columns': df.columns.tolist()
    })

@app.route('/api/dataset/<dataset_id>/export')
@login_required
def export_dataset(dataset_id):
    """Export the current state of a dataset as CSV."""
    if dataset_id not in session or 'df' not in session[dataset_id]:
        return jsonify({'error': 'Dataset not found.'}), 404
    
    df = pd.DataFrame(session[dataset_id]['df'])
    filename = session[dataset_id]['info'].get('filename', 'exported_data')
    
    # Remove file extension and add _processed
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    export_filename = f"{base_name}_processed.csv"
    
    # Create CSV in memory
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename={export_filename}'
    
    return response

@app.route('/api/visualize/<dataset_id>', methods=['POST'])
@login_required
def create_visualization_api(dataset_id): # Renamed to avoid clash with function name
    """Create a visualization for the dataset."""
    if dataset_id not in session or 'df' not in session[dataset_id]:
        return jsonify({'error': 'Dataset not found.'}), 404
    
    df = pd.DataFrame(session[dataset_id]['df'])
    data = request.json
    
    try:
        chart_type = data.get('chart_type')
        x_column = data.get('x_column') 
        y_column = data.get('y_column')
        title = data.get('title')
        color_by = data.get('color_by')
        
        if not chart_type or not x_column:
            return jsonify({'error': 'chart_type and x_column are required.'}), 400
        
        # Call the helper function that creates charts
        chart_html = create_chart_from_function_call(df, {
            'chart_type': chart_type,
            'x_column': x_column,
            'y_column': y_column,
            'title': title,
            'color_by': color_by
        })
        
        if "<p style='color: red;'>" in chart_html: # Check for error message from chart creation
            return jsonify({'error': chart_html.replace("<p style='color: red;'>", "").replace("</p>", "")}), 400

        # Save chart HTML
        chart_filename = f"chart_{uuid.uuid4().hex}.html"
        chart_path = os.path.join(app.root_path, 'static', 'charts', chart_filename)
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        
        with open(chart_path, 'w') as f:
            f.write(chart_html)
        
        chart_url = url_for('static', filename=f'charts/{chart_filename}')
        
        return jsonify({
            'message': 'Visualization created successfully.',
            'chart_url': chart_url,
            'chart_type': chart_type
        })
        
    except Exception as e:
        app.logger.error(f"Error creating visualization: {traceback.format_exc()}")
        return jsonify({'error': f'Error creating visualization: {str(e)}'}), 500

@app.route('/health')
@login_required
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'vertex_ai': vertex_ai_model is not None,
            'mongodb': mongo_client is not None
        }
    }
    return jsonify(status)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Please upload a smaller file.'}), 413

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"Internal server error: {traceback.format_exc()}")
    return jsonify({'error': 'Internal server error occurred.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port)
