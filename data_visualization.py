import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import uuid

# 40 BRIGHT COLOR PALETTES
BRIGHT_PALETTES = {
    'palette_1': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'],
    'palette_2': ['#FF5722', '#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#00BCD4', '#CDDC39'],
    'palette_3': ['#E91E63', '#009688', '#3F51B5', '#FF5722', '#795548', '#607D8B', '#8BC34A'],
    'palette_4': ['#F44336', '#00E676', '#2979FF', '#FF6D00', '#AA00FF', '#00E5FF', '#C6FF00'],
    'palette_5': ['#FF1744', '#00C853', '#2962FF', '#FF3D00', '#D500F9', '#00B8D4', '#AEEA00'],
    'palette_6': ['#FF073A', '#39FF14', '#0080FF', '#FF8C00', '#8A2BE2', '#00FFFF', '#FFFF00'],
    'palette_7': ['#DC143C', '#32CD32', '#1E90FF', '#FFA500', '#9400D3', '#00CED1', '#ADFF2F'],
    'palette_8': ['#B22222', '#228B22', '#4169E1', '#FF7F50', '#BA55D3', '#48D1CC', '#9AFF9A'],
    'palette_9': ['#CD5C5C', '#20B2AA', '#6495ED', '#F4A460', '#DDA0DD', '#5F9EA0', '#98FB98'],
    'palette_10': ['#FA8072', '#3CB371', '#87CEEB', '#DAA520', '#EE82EE', '#66CDAA', '#F0E68C'],
    'palette_11': ['#FF4500', '#00FF7F', '#00BFFF', '#FFD700', '#FF69B4', '#40E0D0', '#ADFF2F'],
    'palette_12': ['#FF6347', '#00FA9A', '#87CEFA', '#FFFF00', '#FF1493', '#7FFFD4', '#GREENYELLOW'],
    'palette_13': ['#FF7F50', '#90EE90', '#ADD8E6', '#F0E68C', '#DA70D6', '#AFEEEE', '#F5DEB3'],
    'palette_14': ['#FFA07A', '#98FB98', '#B0E0E6', '#FFFFE0', '#PLUM', '#E0FFFF', '#WHEAT'],
    'palette_15': ['#FFB6C1', '#LIGHTGREEN', '#LIGHTBLUE', '#LIGHTYELLOW', '#THISTLE', '#LIGHTCYAN', '#MOCCASIN'],
    'palette_16': ['#FF69B4', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500'],
    'palette_17': ['#FF1493', '#32CD32', '#1E90FF', '#FFD700', '#8A2BE2', '#00CED1', '#FF4500'],
    'palette_18': ['#DC143C', '#228B22', '#4169E1', '#FF8C00', '#9400D3', '#48D1CC', '#ADFF2F'],
    'palette_19': ['#B22222', '#2E8B57', '#6495ED', '#FFA500', '#BA55D3', '#5F9EA0', '#9AFF9A'],
    'palette_20': ['#CD5C5C', '#3CB371', '#87CEEB', '#DAA520', '#DDA0DD', '#66CDAA', '#F0E68C'],
    'palette_21': ['#FA8072', '#20B2AA', '#ADD8E6', '#F0E68C', '#EE82EE', '#AFEEEE', '#F5DEB3'],
    'palette_22': ['#FFA07A', '#90EE90', '#B0E0E6', '#FFFFE0', '#D8BFD8', '#E0FFFF', '#DEB887'],
    'palette_23': ['#FFB6C1', '#98FB98', '#87CEFA', '#FFFACD', '#THISTLE', '#LIGHTCYAN', '#WHEAT'],
    'palette_24': ['#FF7F50', '#00FF7F', '#00BFFF', '#FFD700', '#FF69B4', '#40E0D0', '#GREENYELLOW'],
    'palette_25': ['#FF6347', '#00FA9A', '#87CEEB', '#FFFF00', '#FF1493', '#7FFFD4', '#ADFF2F'],
    'palette_26': ['#FF4500', '#32CD32', '#1E90FF', '#FFA500', '#8A2BE2', '#00CED1', '#9AFF9A'],
    'palette_27': ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500'],
    'palette_28': ['#FF69B4', '#32CD32', '#4169E1', '#FFD700', '#9400D3', '#48D1CC', '#FF4500'],
    'palette_29': ['#DC143C', '#228B22', '#6495ED', '#FF8C00', '#BA55D3', '#5F9EA0', '#ADFF2F'],
    'palette_30': ['#B22222', '#2E8B57', '#87CEEB', '#DAA520', '#DDA0DD', '#66CDAA', '#F0E68C'],
    'palette_31': ['#CD5C5C', '#3CB371', '#ADD8E6', '#F0E68C', '#EE82EE', '#AFEEEE', '#F5DEB3'],
    'palette_32': ['#FA8072', '#20B2AA', '#B0E0E6', '#FFFFE0', '#D8BFD8', '#E0FFFF', '#DEB887'],
    'palette_33': ['#FFA07A', '#90EE90', '#87CEFA', '#FFFACD', '#THISTLE', '#LIGHTCYAN', '#WHEAT'],
    'palette_34': ['#FFB6C1', '#98FB98', '#00BFFF', '#FFD700', '#FF69B4', '#40E0D0', '#GREENYELLOW'],
    'palette_35': ['#FF7F50', '#00FF7F', '#87CEEB', '#FFFF00', '#FF1493', '#7FFFD4', '#ADFF2F'],
    'palette_36': ['#FF6347', '#00FA9A', '#1E90FF', '#FFA500', '#8A2BE2', '#00CED1', '#9AFF9A'],
    'palette_37': ['#FF4500', '#32CD32', '#4169E1', '#FFD700', '#9400D3', '#48D1CC', '#FF4500'],
    'palette_38': ['#FF0000', '#00FF00', '#6495ED', '#FF8C00', '#BA55D3', '#5F9EA0', '#ADFF2F'],
    'palette_39': ['#FF69B4', '#228B22', '#87CEEB', '#DAA520', '#DDA0DD', '#66CDAA', '#F0E68C'],
    'palette_40': ['#DC143C', '#2E8B57', '#ADD8E6', '#F0E68C', '#EE82EE', '#AFEEEE', '#F5DEB3']
}

def safe_json_serialize(df):
    """Make DataFrame safe for JSON serialization."""
    df_safe = df.copy()
    for col in df_safe.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df_safe[col]):
                df_safe[col] = df_safe[col].astype(str)
            elif pd.api.types.is_integer_dtype(df_safe[col]):
                df_safe[col] = df_safe[col].astype(int)
            elif pd.api.types.is_float_dtype(df_safe[col]):
                df_safe[col] = df_safe[col].astype(float)
        except Exception as e:
            df_safe[col] = df_safe[col].astype(str)
    return df_safe

def analyze_data_structure(df):
    """Analyze data and return structure info"""
    
    # Get actual data characteristics
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    
    for col in df.columns:
        # Skip ID columns
        if df[col].nunique() == len(df):
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 0:
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif df[col].dtype == 'object' and df[col].notna().sum() > 0:
            # Check if it's reasonable for visualization (not too many unique values)
            if df[col].nunique() <= len(df) * 0.8:
                categorical_cols.append(col)
    
    return {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'datetime_cols': datetime_cols,
        'total_rows': len(df),
        'total_cols': len(df.columns)
    }

def select_template_and_palette(data_structure):
    """Select best template and palette based on data structure"""
    
    num_numeric = len(data_structure['numeric_cols'])
    num_categorical = len(data_structure['categorical_cols'])
    total_rows = data_structure['total_rows']
    
    # Simple logic for template selection
    if total_rows > 50000:
        template_id = 1  # Simple layout for large data
    elif num_numeric >= 5:
        template_id = 2  # Multiple numeric - good for correlations
    elif num_categorical >= 4:
        template_id = 3  # Category heavy
    elif num_numeric >= 2 and num_categorical >= 2:
        template_id = 1  # Balanced
    else:
        template_id = 1  # Default
    
    # Select palette based on data size
    palette_id = f'palette_{(template_id + total_rows) % 40 + 1}'
    
    return template_id, palette_id

def create_chart_safe(chart_type, df, data_structure, palette_colors, chart_index, title):
    """Create chart with proper error handling"""
    
    try:
        numeric_cols = data_structure['numeric_cols']
        categorical_cols = data_structure['categorical_cols']
        
        if chart_type == 'bar':
            if not categorical_cols:
                return f"<div style='padding:20px;text-align:center;color:#666;'>No categorical data for bar chart</div>"
            
            col = categorical_cols[chart_index % len(categorical_cols)]
            value_counts = df[col].value_counts().head(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                marker_color=palette_colors[0],
                text=value_counts.values,
                textposition='outside'
            ))
            
        elif chart_type == 'pie':
            if not categorical_cols:
                return f"<div style='padding:20px;text-align:center;color:#666;'>No categorical data for pie chart</div>"
            
            col = categorical_cols[chart_index % len(categorical_cols)]
            value_counts = df[col].value_counts().head(6)
            
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=value_counts.index,
                values=value_counts.values,
                marker_colors=palette_colors[:len(value_counts)],
                hole=0.3
            ))
            
        elif chart_type == 'line':
            if not numeric_cols:
                return f"<div style='padding:20px;text-align:center;color:#666;'>No numeric data for line chart</div>"
            
            col = numeric_cols[chart_index % len(numeric_cols)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=df[col],
                mode='lines+markers',
                line=dict(color=palette_colors[0], width=2),
                marker=dict(size=4, color=palette_colors[1])
            ))
            
        elif chart_type == 'scatter':
            if len(numeric_cols) < 2:
                return f"<div style='padding:20px;text-align:center;color:#666;'>Need 2+ numeric columns for scatter</div>"
            
            x_col = numeric_cols[0]
            y_col = numeric_cols[1 % len(numeric_cols)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='markers',
                marker=dict(color=palette_colors[0], size=6)
            ))
            
        elif chart_type == 'histogram':
            if not numeric_cols:
                return f"<div style='padding:20px;text-align:center;color:#666;'>No numeric data for histogram</div>"
            
            col = numeric_cols[chart_index % len(numeric_cols)]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[col],
                marker_color=palette_colors[0],
                opacity=0.7,
                nbinsx=20
            ))
            
        elif chart_type == 'box':
            if not numeric_cols:
                return f"<div style='padding:20px;text-align:center;color:#666;'>No numeric data for box plot</div>"
            
            col = numeric_cols[chart_index % len(numeric_cols)]
            
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=df[col],
                marker_color=palette_colors[0],
                boxpoints='outliers'
            ))
            
        elif chart_type == 'heatmap':
            if len(numeric_cols) < 2:
                return f"<div style='padding:20px;text-align:center;color:#666;'>Need 2+ numeric columns for heatmap</div>"
            
            # Use up to 6 numeric columns for correlation
            cols_for_corr = numeric_cols[:6]
            corr_matrix = df[cols_for_corr].corr()
            
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='Viridis',
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 8}
            ))
            
        else:
            return f"<div style='padding:20px;text-align:center;color:#666;'>Unknown chart type: {chart_type}</div>"
        
        # Apply layout with EXACT sizing for grid alignment
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Segoe UI, Arial, sans-serif', size=10),
            margin=dict(l=40, r=40, t=40, b=40),
            height=200,
            showlegend=False,
            autosize=True
        )
        
        # Add title separately - NO CONFLICTS
        if title:
            fig.update_layout(title=dict(text=title, x=0.5, y=0.95, font=dict(size=11)))
        
        return fig.to_html(include_plotlyjs='cdn')
        
    except Exception as e:
        return f"<div style='padding:20px;text-align:center;color:#666;'>Chart Error: {str(e)}</div>"

def generate_dashboard_html(df, app3_analysis_results="Data Analysis Dashboard"):
    """
    FIXED VERSION - Generate dashboard with App3 AI results
    """
    
    try:
        # Ensure DataFrame is safe
        df_safe = safe_json_serialize(df)
        
        # Analyze data structure properly
        data_structure = analyze_data_structure(df_safe)
        
        # Select template and palette
        template_id, palette_id = select_template_and_palette(data_structure)
        palette_colors = BRIGHT_PALETTES[palette_id]
        
        # Extract title and explanation from App3 results
        if isinstance(app3_analysis_results, dict):
            dashboard_title = app3_analysis_results.get('title', 'Data Analysis Dashboard')
            dashboard_subtitle = app3_analysis_results.get('subtitle', 'AI-powered data visualization')
            ai_explanation = app3_analysis_results.get('dashboard_explanation', 'Smart visualization of your data patterns and insights')
        elif isinstance(app3_analysis_results, str):
            lines = app3_analysis_results.split('\n')
            dashboard_title = lines[0][:60] if lines else "Data Analysis Dashboard"
            dashboard_subtitle = lines[1][:80] if len(lines) > 1 else "AI-powered data visualization"
            ai_explanation = lines[2][:120] if len(lines) > 2 else "Smart visualization of your data patterns and insights"
        else:
            dashboard_title = "Data Analysis Dashboard"
            dashboard_subtitle = "AI-powered data visualization"
            ai_explanation = "Smart visualization of your data patterns and insights"
        
        # Define chart types for layout
        chart_types = ['bar', 'pie', 'line', 'scatter', 'histogram', 'box', 'heatmap']
        
        # Generate charts
        charts_html = []
        for i, chart_type in enumerate(chart_types):
            chart_title = f"{chart_type.title()} Analysis {i+1}"
            chart_html = create_chart_safe(chart_type, df_safe, data_structure, palette_colors, i, chart_title)
            charts_html.append(f'<div class="chart-item">{chart_html}</div>')
        
        # Generate final HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dashboard_title}</title>
    <style>
        body {{
            margin: 0;
            padding: 8px;
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, {palette_colors[0]} 0%, {palette_colors[1]} 100%);
            min-height: 100vh;
            zoom: 0.85;
        }}
        
        .dashboard-header {{
            text-align: center;
            color: white;
            margin-bottom: 12px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            padding: 8px;
        }}
        
        .dashboard-title {{
            font-size: 1.6rem;
            margin: 0 0 6px 0;
            font-weight: 700;
        }}
        
        .dashboard-subtitle {{
            font-size: 0.85rem;
            margin: 0 0 4px 0;
            opacity: 0.9;
        }}
        
        .ai-info {{
            font-size: 0.75rem;
            opacity: 0.8;
            font-style: italic;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            grid-template-rows: repeat(2, 1fr);
            gap: 6px;
            height: calc(100vh - 90px);
            max-width: 100vw;
            margin: 0 auto;
            padding: 0 6px;
        }}
        
        .chart-item {{
            background: white;
            border-radius: 6px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            position: relative;
            cursor: zoom-in;
            min-height: 160px;
        }}
        
        .chart-item:hover {{
            transform: scale(1.03);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            border-color: {palette_colors[2]};
            z-index: 10;
        }}
        
        .chart-item:nth-child(1) {{
            grid-column: span 2;
            border-left: 4px solid {palette_colors[0]};
        }}
        
        .chart-item:nth-child(2) {{
            grid-column: span 2;
            border-left: 4px solid {palette_colors[1]};
        }}
        
        .chart-item:nth-child(3) {{
            border-left: 4px solid {palette_colors[2]};
        }}
        
        .chart-item:nth-child(4) {{
            border-left: 4px solid {palette_colors[3]};
        }}
        
        .chart-item:nth-child(5) {{
            border-left: 4px solid {palette_colors[4]};
        }}
        
        .chart-item:nth-child(6) {{
            border-left: 4px solid {palette_colors[5]};
        }}
        
        .chart-item:nth-child(7) {{
            border-left: 4px solid {palette_colors[6]};
        }}
        
        .chart-item .plotly-graph-div {{
            height: 100% !important;
            width: 100% !important;
        }}
        
        .chart-item.zoomed {{
            position: fixed;
            top: 3%;
            left: 3%;
            width: 94vw;
            height: 94vh;
            z-index: 1000;
            transform: none;
            cursor: zoom-out;
            border-radius: 12px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.3);
        }}
        
        .chart-backdrop {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0,0,0,0.7);
            z-index: 999;
            display: none;
        }}
        
        .chart-backdrop.active {{
            display: block;
        }}
        
        .ai-badge {{
            position: absolute;
            top: 12px;
            right: 12px;
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.65rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
        }}
        
        @media (max-width: 1400px) {{
            body {{ zoom: 0.75; }}
            .charts-grid {{
                grid-template-columns: repeat(3, 1fr);
                grid-template-rows: repeat(3, auto);
                height: calc(100vh - 80px);
                gap: 4px;
            }}
        }}
        
        @media (max-width: 1000px) {{
            body {{ zoom: 0.7; }}
            .charts-grid {{
                grid-template-columns: repeat(2, 1fr);
                gap: 4px;
            }}
        }}
        
        @media (max-width: 700px) {{
            body {{ zoom: 1.0; }}
            .charts-grid {{
                grid-template-columns: 1fr;
                height: auto;
                gap: 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="ai-badge">
        Visualization AI: Template {template_id}
    </div>
    
    <div class="dashboard-header">
        <h1 class="dashboard-title">{dashboard_title}</h1>
        <p class="dashboard-subtitle">{dashboard_subtitle}</p>
        <div class="ai-explanation">{ai_explanation}</div>
        <p class="ai-info">
            AI Analysis: {data_structure['total_rows']} Records • {data_structure['total_cols']} Columns • 
            {len(data_structure['numeric_cols'])} Numeric • {len(data_structure['categorical_cols'])} Categorical
        </p>
    </div>
    
    <div class="charts-grid">
        {''.join(charts_html)}
    </div>
    
    <div class="chart-backdrop"></div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const backdrop = document.querySelector('.chart-backdrop');
            const charts = document.querySelectorAll('.chart-item');
            
            charts.forEach((chart, index) => {{
                chart.style.animationDelay = `${{index * 0.1}}s`;
                chart.style.animation = 'slideInUp 0.6s ease forwards';
                
                chart.addEventListener('click', function() {{
                    if (chart.classList.contains('zoomed')) {{
                        exitZoom();
                    }} else {{
                        enterZoom(chart);
                    }}
                }});
            }});
            
            backdrop.addEventListener('click', exitZoom);
            
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape') exitZoom();
            }});
            
            function enterZoom(chart) {{
                exitZoom();
                backdrop.classList.add('active');
                chart.classList.add('zoomed');
                
                setTimeout(() => {{
                    const plotlyDiv = chart.querySelector('.plotly-graph-div');
                    if (plotlyDiv && window.Plotly) {{
                        window.Plotly.Plots.resize(plotlyDiv);
                    }}
                }}, 300);
                
                document.body.style.overflow = 'hidden';
            }}
            
            function exitZoom() {{
                backdrop.classList.remove('active');
                charts.forEach(chart => {{
                    chart.classList.remove('zoomed');
                    setTimeout(() => {{
                        const plotlyDiv = chart.querySelector('.plotly-graph-div');
                        if (plotlyDiv && window.Plotly) {{
                            window.Plotly.Plots.resize(plotlyDiv);
                        }}
                    }}, 300);
                }});
                document.body.style.overflow = 'auto';
            }}
        }});
        
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInUp {{
                from {{ opacity: 0; transform: translateY(30px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .chart-item {{ opacity: 0; }}
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
        """
        
        return html_content
        
    except Exception as e:
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Dashboard Error</title></head>
        <body style="padding: 50px; text-align: center; font-family: Arial;">
            <h1 style="color: #dc3545;">Dashboard Generation Error</h1>
            <p style="color: #6c757d;">Error: {str(e)}</p>
            <p>Data Structure Debug:</p>
            <pre>{str(analyze_data_structure(df)) if 'df' in locals() else 'No data'}</pre>
        </body>
        </html>
        """