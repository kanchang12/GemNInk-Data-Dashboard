from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash, make_response
import pandas as pd
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename
import traceback
import json
from functools import wraps
import copy
import io
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import SimpleConnectionPool
import openai

load_dotenv()

# Import custom modules
import data_analysis
import data_visualization
import data_cleaning
import data_transformation
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)
app.config['APPLICATION_NAME'] = 'GemNInk'
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_USE_SIGNER'] = True

from flask_session import Session
Session(app)

# --- PostgreSQL Configuration ---
DATABASE_URL = "postgres://koyeb-adm:npg_6fcBpeKIWtq5@ep-raspy-morning-a2q7q3op.eu-central-1.pg.koyeb.app/koyebdb"

# Create connection pool
db_pool = None
try:
    db_pool = SimpleConnectionPool(1, 20, DATABASE_URL)
    print("PostgreSQL connection pool created successfully!")
except Exception as e:
    print(f"Could not create PostgreSQL connection pool: {e}")
    db_pool = None

def get_db_connection():
    """Get a connection from the pool"""
    if db_pool:
        return db_pool.getconn()
    return None

def return_db_connection(conn):
    """Return connection to the pool"""
    if db_pool and conn:
        db_pool.putconn(conn)

def init_database():
    """Initialize database tables if they don't exist"""
    conn = get_db_connection()
    if not conn:
        print("Cannot initialize database - no connection available")
        return
    
    try:
        cur = conn.cursor()
        
        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create upload_logs table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS upload_logs (
                id SERIAL PRIMARY KEY,
                dataset_id VARCHAR(255) UNIQUE NOT NULL,
                filename VARCHAR(255) NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                rows INTEGER,
                columns INTEGER,
                user_session VARCHAR(255),
                automated_cleaning_summary TEXT,
                initial_ai_summary TEXT,
                dashboard_url TEXT,
                dashboard_explanation TEXT
            )
        """)
        
        # Create datasets table (optional - for persistence)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id SERIAL PRIMARY KEY,
                dataset_id VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(255),
                data JSONB,
                info JSONB,
                chat_history JSONB,
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dashboard_url TEXT,
                initial_insights JSONB
            )
        """)
        
        conn.commit()
        print("Database tables initialized successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"Error initializing database: {e}")
        traceback.print_exc()
    finally:
        cur.close()
        return_db_connection(conn)
def create_default_user():
    """Create default admin user if no users exist"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        
        if count == 0:
            hashed = generate_password_hash('admin123', method='pbkdf2:sha256')
            cur.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                ('admin', hashed)
            )
            conn.commit()
            print("Default user created - Username: admin, Password: admin123")
        
        cur.close()
    except Exception as e:
        conn.rollback()
        print(f"Error creating default user: {e}")
    finally:
        return_db_connection(conn)
# Initialize database on startup
init_database()
create_default_user()
# --- Perplexity AI Configuration ---
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')

# Initialize OpenAI client for Perplexity
perplexity_client = None
if PERPLEXITY_API_KEY:
    perplexity_client = openai.OpenAI(
        api_key=PERPLEXITY_API_KEY,
        base_url="https://api.perplexity.ai"
    )
    print("Perplexity API client initialized successfully!")
else:
    print("PERPLEXITY_API_KEY not set. AI features will be unavailable.")

# Define tools for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "clean_data",
            "description": "Applies data cleaning operations including handling missing values, removing duplicates, and standardizing text columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "remove_outliers": {
                        "type": "boolean",
                        "description": "Whether to remove outliers from numerical columns"
                    },
                    "z_score_threshold": {
                        "type": "number",
                        "description": "Z-score threshold for outlier removal (default: 5.0)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_numerical_data",
            "description": "Performs statistical analysis on numerical columns including descriptive statistics and outlier detection.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_categorical_data",
            "description": "Analyzes categorical data patterns including frequency counts and unique values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific categorical columns to analyze"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_correlation_analysis",
            "description": "Calculates correlation matrices for numerical columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Correlation method: pearson, spearman, or kendall"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_visualization",
            "description": "Generates data visualizations like histogram, scatter plot, bar chart, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["histogram", "scatter", "bar", "line", "box", "heatmap", "pie"],
                        "description": "Type of chart to create"
                    },
                    "x_column": {"type": "string", "description": "Column for X-axis"},
                    "y_column": {"type": "string", "description": "Column for Y-axis"},
                    "title": {"type": "string", "description": "Chart title"},
                    "color_by": {"type": "string", "description": "Column to color by"},
                    "facet_by": {"type": "string", "description": "Column to facet by"},
                    "aggregate_by": {"type": "string", "description": "Aggregation type"}
                },
                "required": ["chart_type", "x_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_outliers",
            "description": "Removes outliers from numerical columns using Z-score method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Numerical columns to process"
                    },
                    "z_score_threshold": {
                        "type": "number",
                        "description": "Z-score threshold (default: 3.0)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_transformation",
            "description": "Applies transformations like log, scaling, or one-hot encoding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transformation_type": {
                        "type": "string",
                        "enum": ["log_transform", "standard_scale", "minmax_scale", "one_hot_encode"]
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to transform"
                    },
                    "output_prefix": {"type": "string", "description": "Prefix for new columns"}
                },
                "required": ["transformation_type", "columns"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataframe_info",
            "description": "Retrieves DataFrame summary including columns, types, and counts.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sample_data",
            "description": "Retrieves sample rows from the DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_rows": {"type": "integer", "description": "Number of rows (max 100)"}
                }
            }
        }
    }
]

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'info')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls', 'json', 'tsv'}

def get_df_info(df, filename='Unknown'):
    return {
        'filename': str(filename),
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'data_quality_score': 'Good',
        'column_info': [
            {
                'name': str(col),
                'dtype': str(df[col].dtype),
                'unique_values': int(df[col].nunique()),
                'missing_count': int(df[col].isnull().sum())
            }
            for col in df.columns
        ]
    }

def get_sample_for_ai(df, num_rows=5):
    num_rows = min(num_rows, 100)
    return df.head(num_rows).to_dict('records')

def execute_tool_call(tool_name, tool_args, df):
    """Execute a tool call based on AI function call"""
    try:
        if tool_name == "clean_data":
            cleaned_df, cleaning_report = data_cleaning.clean_data(df.copy())
            if tool_args.get('remove_outliers'):
                z_thresh = tool_args.get('z_score_threshold', 5)
                numerical_cols = cleaned_df.select_dtypes(include=np.number).columns
                original_rows = cleaned_df.shape[0]
                cleaned_df, _ = data_cleaning.remove_outliers(cleaned_df, columns=list(numerical_cols), z_thresh=z_thresh)
                rows_removed = original_rows - cleaned_df.shape[0]
                return {"status": "success", "message": f"Data cleaned. Removed {rows_removed} outliers.", "df": cleaned_df.to_dict('records')}
            return {"status": "success", "message": "Data cleaning applied.", "df": cleaned_df.to_dict('records')}
        
        elif tool_name == "analyze_numerical_data":
            result = data_analysis.analyze_numerical_data(df)
            return {"status": "success", "result": result}
        
        elif tool_name == "analyze_categorical_data":
            result = data_analysis.analyze_categorical_data(df, columns=tool_args.get('columns'))
            return {"status": "success", "result": result}
        
        elif tool_name == "perform_correlation_analysis":
            result = data_analysis.analyze_correlations(df, method=tool_args.get('method', 'pearson'))
            return {"status": "success", "result": result}
        
        elif tool_name == "create_visualization":
            chart_html = create_chart_from_function_call(df, tool_args)
            return {"status": "success", "message": "Visualization created.", "chart_html": chart_html}
        
        elif tool_name == "remove_outliers":
            z_thresh = tool_args.get('z_score_threshold', 3.0)
            columns = tool_args.get('columns', list(df.select_dtypes(include=np.number).columns))
            original_rows = df.shape[0]
            cleaned_df, msg = data_cleaning.remove_outliers(df.copy(), columns=columns, z_thresh=z_thresh)
            rows_removed = original_rows - cleaned_df.shape[0]
            return {"status": "success", "message": f"Removed {rows_removed} outliers.", "df": cleaned_df.to_dict('records')}
        
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
        
        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    
    except Exception as e:
        app.logger.error(f"Error executing tool '{tool_name}': {traceback.format_exc()}")
        return {"status": "error", "message": f"Error: {str(e)}"}

def read_complex_file(file_stream, filename):
    file_ext = filename.rsplit('.', 1)[1].lower()
    try:
        if file_ext == 'csv':
            original_pos = file_stream.tell()
            for delim in [',', ';', '\t']:
                try:
                    file_stream.seek(original_pos)
                    df = pd.read_csv(file_stream, encoding='utf-8', sep=delim)
                    if not df.empty:
                        break
                except:
                    continue
            if df is None or df.empty:
                raise Exception("Could not read CSV with common delimiters")
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_stream)
        else:
            raise Exception(f"Unsupported file type: {file_ext}")
        
        if df.empty:
            raise Exception("File is empty")
        
        df.replace(['', 'NA', 'N/A', 'NaN', 'null'], np.nan, inplace=True)
        
        # Handle problematic columns
        for col in df.columns:
            try:
                df[col].to_json()
            except:
                df[col] = df[col].astype(str)
        
        return df
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def create_chart_from_function_call(df, args):
    """Create chart based on function call arguments"""
    try:
        if df is None or df.empty:
            return "<p style='color: red;'>No data available</p>"
        
        x_col = args.get('x_column')
        y_col = args.get('y_column')
        chart_type = args.get('chart_type', 'bar').lower()
        title = args.get('title')
        color_by = args.get('color_by')
        
        if x_col and x_col not in df.columns:
            return f"<p style='color: red;'>Column '{x_col}' not found</p>"
        
        fig = None
        
        if chart_type == 'bar':
            if x_col and y_col and pd.api.types.is_numeric_dtype(df[y_col]):
                grouped = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(grouped, x=x_col, y=y_col, title=title or f"{y_col} by {x_col}", color=color_by)
            elif x_col:
                counts = df[x_col].value_counts().head(20)
                fig = px.bar(x=counts.index, y=counts.values, title=title or f"Distribution of {x_col}")
        
        elif chart_type == 'histogram' and x_col:
            fig = px.histogram(df, x=x_col, title=title or f"Histogram of {x_col}", color=color_by)
        
        elif chart_type == 'scatter' and x_col and y_col:
            fig = px.scatter(df, x=x_col, y=y_col, title=title or f"{y_col} vs {x_col}", color=color_by)
        
        elif chart_type == 'line' and x_col and y_col:
            fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, title=title or f"{y_col} over {x_col}", color=color_by)
        
        elif chart_type == 'pie' and x_col:
            counts = df[x_col].value_counts().head(10)
            fig = px.pie(values=counts.values, names=counts.index, title=title or f"Pie Chart of {x_col}")
        
        elif chart_type == 'box' and x_col:
            fig = px.box(df, y=x_col, title=title or f"Box Plot of {x_col}", color=color_by)
        
        elif chart_type == 'heatmap':
            corr = df.select_dtypes(include=np.number).corr()
            fig = px.imshow(corr, text_auto=True, title=title or "Correlation Heatmap")
        
        if fig:
            fig.update_layout(height=500, margin=dict(l=50, r=50, t=80, b=50))
            return fig.to_html(include_plotlyjs='cdn', config={'responsive': True})
        
        return "<p style='color: red;'>Could not create chart</p>"
    
    except Exception as e:
        return f"<p style='color: red;'>Error: {str(e)}</p>"

# --- Routes ---

@app.route('/')
@login_required
def index():
    if 'username' in session:
        return render_template('index.html', app_name=app.config['APPLICATION_NAME'])
    return render_template('new-page.html', app_name=app.config['APPLICATION_NAME'])

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        filename = secure_filename(file.filename)
        df = read_complex_file(file.stream, filename)
        
        if df.empty:
            return jsonify({'error': 'File is empty'}), 400
        
        # Automated cleaning
        initial_rows = df.shape[0]
        initial_missing = df.isnull().sum().sum()
        
        cleaned_df, _ = data_cleaning.clean_data(df.copy())
        
        numerical_cols = cleaned_df.select_dtypes(include=np.number).columns
        total_outliers_removed = 0
        if len(numerical_cols) > 0:
            original_rows = cleaned_df.shape[0]
            cleaned_df, _ = data_cleaning.remove_outliers(cleaned_df, columns=list(numerical_cols), method='iqr', z_thresh=5)
            total_outliers_removed = original_rows - cleaned_df.shape[0]
        
        rows_after = cleaned_df.shape[0]
        missing_after = cleaned_df.isnull().sum().sum()
        cleaning_message = f"Cleaned: {initial_rows:,}→{rows_after:,} rows, {initial_missing:,}→{missing_after:,} missing values"
        
        # Generate insights
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
        
        # AI summary
        ai_summary = "AI summary unavailable"
        dashboard_explanation = "Data insights visualization"
        
        if perplexity_client:
            try:
                response = perplexity_client.chat.completions.create(
                    model="llama-3.1-sonar-small-128k-online",
                    messages=[{
                        "role": "user",
                        "content": f"Provide a 100-word summary of this dataset analysis:\n{json.dumps(all_insights, indent=2)}"
                    }]
                )
                ai_summary = response.choices[0].message.content
                
                # Get explanation
                expl_response = perplexity_client.chat.completions.create(
                    model="llama-3.1-sonar-small-128k-online",
                    messages=[{
                        "role": "user",
                        "content": f"In ONE sentence (max 15 words), what is this dataset about?\nFile: {filename}\nColumns: {list(cleaned_df.columns)}"
                    }]
                )
                dashboard_explanation = expl_response.choices[0].message.content.strip()[:100]
            except Exception as e:
                app.logger.error(f"AI summary error: {e}")
        
        all_insights['dashboard_explanation'] = dashboard_explanation
        
        # Generate dashboard
        dashboard_html = data_visualization.generate_dashboard_html(cleaned_df, all_insights)
        dashboard_filename = f"dashboard_{uuid.uuid4().hex}.html"
        dashboard_path = os.path.join(app.root_path, 'static', 'dashboards', dashboard_filename)
        os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        dashboard_url = url_for('static', filename=f'dashboards/{dashboard_filename}')
        
        # Store in session
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
        
        # Log to PostgreSQL
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO upload_logs 
                    (dataset_id, filename, rows, columns, user_session, automated_cleaning_summary, 
                     initial_ai_summary, dashboard_url, dashboard_explanation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (dataset_id, filename, cleaned_df.shape[0], cleaned_df.shape[1],
                      session.get('username', 'anonymous'), cleaning_message, ai_summary,
                      dashboard_url, dashboard_explanation))
                conn.commit()
                cur.close()
            except Exception as e:
                conn.rollback()
                app.logger.warning(f"Could not log to database: {e}")
            finally:
                return_db_connection(conn)
        
        return jsonify({
            'message': 'File processed successfully!',
            'dataset_id': dataset_id,
            'df_info': session[dataset_id]['info'],
            'cleaning_summary': cleaning_message,
            'initial_ai_summary': ai_summary,
            'dashboard_url': dashboard_url,
            'dashboard_explanation': dashboard_explanation,
            'initial_insights': all_insights
        }), 200
    
    except Exception as e:
        app.logger.error(f"Upload error: {traceback.format_exc()}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/chat/<dataset_id>', methods=['POST'])
@login_required
def chat(dataset_id):
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        if dataset_id not in session or 'df' not in session[dataset_id]:
            return jsonify({"error": "Dataset not found"}), 400
        
        df = pd.DataFrame(session[dataset_id]['df'])
        chat_history = session[dataset_id].get('chat_history', [])
        
        # Build context
        data_info = f"""Dataset: {len(df)} rows, {len(df.columns)} columns
Columns: {', '.join(df.columns.tolist())}
Types: {df.dtypes.to_dict()}
Sample: {get_sample_for_ai(df, 5)}

Respond in max 20 words. Use function calls when needed."""
        
        # Build messages
        messages = [{"role": "system", "content": "You are a data analyst assistant. Keep responses under 20 words."}]
        
        # Add history
        for msg in chat_history[-10:]:  # Last 10 messages
            messages.append(msg)
        
        messages.append({"role": "user", "content": f"{data_info}\n\n{user_message}"})
        
        # Call Perplexity
        response = perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        ai_response = response.choices[0].message
        ai_response_text = ai_response.content or ""
        function_results = []
        updated_df_data = df.to_dict('records')
        
        # Handle tool calls
        if ai_response.tool_calls:
            for tool_call in ai_response.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                result = execute_tool_call(func_name, func_args, pd.DataFrame(updated_df_data))
                
                if result.get('df'):
                    updated_df_data = result['df']
                    session[dataset_id]['df'] = updated_df_data
                    session[dataset_id]['info'] = get_df_info(pd.DataFrame(updated_df_data), 
                                                               session[dataset_id]['info'].get('filename', 'Unknown'))
                
                function_results.append({
                    'function': func_name,
                    'args': func_args,
                    'result': result
                })
                
                # Get follow-up response
                messages.append({"role": "assistant", "content": ai_response_text, "tool_calls": [tool_call.dict()]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
                
                follow_up = perplexity_client.chat.completions.create(
                    model="llama-3.1-sonar-small-128k-online",
                    messages=messages
                )
                ai_response_text += "\n" + (follow_up.choices[0].message.content or "")
        
        complete_response = ai_response_text
        
        # Add visualization HTML
        for func_result in function_results:
            if func_result['function'] == 'create_visualization' and func_result['result'].get('chart_html'):
                complete_response += f"\n\n{func_result['result']['chart_html']}"
        
        # Update history
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": complete_response})
        session[dataset_id]['chat_history'] = chat_history
        
        return jsonify({
            "response": complete_response,
            "function_calls": function_results,
            "df_info": session[dataset_id]['info']
        })
    
    except Exception as e:
        app.logger.error(f"Chat error: {traceback.format_exc()}")
        return jsonify({"error": f"Chat error: {str(e)}"}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute("SELECT * FROM users WHERE username = %s", (username,))
                user = cur.fetchone()
                cur.close()
                
                if user and check_password_hash(user['password'], password):
                    session['logged_in'] = True
                    session['username'] = username
                    flash('Logged in successfully!', 'success')
                    return redirect(url_for('index'))
                else:
                    flash('Invalid credentials', 'danger')
            except Exception as e:
                app.logger.error(f"Login error: {e}")
                flash('Login error', 'danger')
            finally:
                return_db_connection(conn)
        else:
            flash('Database unavailable', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT id FROM users WHERE username = %s", (username,))
                if cur.fetchone():
                    flash('Username exists', 'danger')
                    cur.close()
                    return render_template('register.html')
                
                hashed = generate_password_hash(password, method='pbkdf2:sha256')
                cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed))
                conn.commit()
                cur.close()
                flash('Registration successful!', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                conn.rollback()
                flash(f'Registration error: {e}', 'danger')
            finally:
                return_db_connection(conn)
        else:
            flash('Database unavailable', 'danger')
    
    return render_template('register.html')

@app.route('/api/datasets')
@login_required
def list_datasets():
    datasets = []
    for key, value in session.items():
        if isinstance(value, dict) and 'df' in value and 'info' in value:
            datasets.append({
                'dataset_id': key,
                'info': value['info'],
                'upload_timestamp': value.get('upload_timestamp'),
                'has_dashboard': 'dashboard_url' in value
            })
    return jsonify({'datasets': datasets, 'active_dataset_id': session.get('active_dataset_id')})

@app.route('/api/dataset/<dataset_id>/export')
@login_required
def export_dataset(dataset_id):
    if dataset_id not in session or 'df' not in session[dataset_id]:
        return jsonify({'error': 'Dataset not found'}), 404
    
    df = pd.DataFrame(session[dataset_id]['df'])
    filename = session[dataset_id]['info'].get('filename', 'data')
    base_name = filename.rsplit('.', 1)[0]
    
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename={base_name}_processed.csv'
    return response

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'perplexity_api': perplexity_client is not None,
            'postgresql': db_pool is not None
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


