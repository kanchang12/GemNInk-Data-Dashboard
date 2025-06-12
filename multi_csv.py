import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import io
import traceback
from werkzeug.utils import secure_filename

class MultiCSVProcessor:
    def __init__(self):
        self.tables = {}
        self.join_suggestions = []
    
    def read_csv_file(self, file_stream, filename):
        """Read a single CSV file - same logic as your main app"""
        try:
            # Store the original stream position
            original_pos = file_stream.tell()
            
            # List of delimiters to try
            possible_delimiters = [',', ';', '\t']
            df = None
            
            for delim in possible_delimiters:
                try:
                    file_stream.seek(original_pos)
                    df = pd.read_csv(file_stream, encoding='utf-8', sep=delim)
                    if not df.empty:
                        break
                except Exception:
                    continue
            
            if df is None or df.empty:
                raise Exception("Could not read CSV file")
                
            # Clean the data
            df.replace(['', 'NA', 'N/A', 'NaN', 'null'], np.nan, inplace=True)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error reading CSV file {filename}: {str(e)}")
    
    def add_table(self, file_stream, filename):
        """Add a CSV file as a table"""
        try:
            df = self.read_csv_file(file_stream, filename)
            table_name = filename.rsplit('.', 1)[0]  # Remove .csv extension
            self.tables[table_name] = df
            return table_name, df.shape
        except Exception as e:
            raise Exception(f"Failed to add table {filename}: {str(e)}")
    
    def similarity(self, a, b):
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def find_join_suggestions(self):
        """Auto-detect potential join keys between tables"""
        self.join_suggestions = []
        table_names = list(self.tables.keys())
        
        for i, table1 in enumerate(table_names):
            for j, table2 in enumerate(table_names[i+1:], i+1):
                df1 = self.tables[table1]
                df2 = self.tables[table2]
                
                # Look for similar column names
                for col1 in df1.columns:
                    for col2 in df2.columns:
                        similarity_score = self.similarity(col1, col2)
                        
                        # If columns are very similar or identical
                        if similarity_score > 0.8:
                            # Check if they have overlapping values
                            common_values = len(set(df1[col1].dropna()) & set(df2[col2].dropna()))
                            
                            if common_values > 0:
                                self.join_suggestions.append({
                                    'table1': table1,
                                    'column1': col1,
                                    'table2': table2,
                                    'column2': col2,
                                    'similarity': similarity_score,
                                    'common_values': common_values,
                                    'confidence': min(similarity_score + (common_values/100), 1.0)
                                })
        
        # Sort by confidence
        self.join_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        return self.join_suggestions
    
    def auto_join_tables(self, join_type='inner'):
        """Automatically join tables based on best suggestions"""
        if len(self.tables) < 2:
            raise Exception("Need at least 2 tables to join")
        
        suggestions = self.find_join_suggestions()
        
        if not suggestions:
            # If no good suggestions, just concatenate tables side by side
            return self.concatenate_tables()
        
        # Start with the first suggested join
        best_join = suggestions[0]
        
        table1_name = best_join['table1']
        table2_name = best_join['table2']
        col1 = best_join['column1']
        col2 = best_join['column2']
        
        df1 = self.tables[table1_name].copy()
        df2 = self.tables[table2_name].copy()
        
        # Rename columns to avoid conflicts (except join key)
        df1_cols = {col: f"{table1_name}_{col}" for col in df1.columns if col != col1}
        df2_cols = {col: f"{table2_name}_{col}" for col in df2.columns if col != col2}
        
        df1.rename(columns=df1_cols, inplace=True)
        df2.rename(columns=df2_cols, inplace=True)
        
        # Rename join columns to be the same
        join_key = f"join_key_{col1}"
        df1.rename(columns={col1: join_key}, inplace=True)
        df2.rename(columns={col2: join_key}, inplace=True)
        
        # Perform the join
        merged_df = pd.merge(df1, df2, on=join_key, how=join_type)
        
        # Add any remaining tables
        for table_name, df in self.tables.items():
            if table_name not in [table1_name, table2_name]:
                # Try to find a way to join this table
                for suggestion in suggestions:
                    if suggestion['table1'] == table_name and suggestion['table2'] in [table1_name, table2_name]:
                        # Found a way to join this table
                        df_to_join = df.copy()
                        
                        # Rename columns
                        join_col = suggestion['column1']
                        target_col = suggestion['column2']
                        
                        # Rename to match existing join key if possible
                        if target_col in [col1, col2]:
                            df_to_join.rename(columns={join_col: join_key}, inplace=True)
                        else:
                            df_to_join.rename(columns={join_col: f"join_key_{join_col}"}, inplace=True)
                        
                        # Rename other columns
                        other_cols = {col: f"{table_name}_{col}" for col in df_to_join.columns if not col.startswith('join_key')}
                        df_to_join.rename(columns=other_cols, inplace=True)
                        
                        # Try to merge
                        try:
                            common_keys = set(merged_df.columns) & set(df_to_join.columns)
                            if common_keys:
                                merge_key = list(common_keys)[0]
                                merged_df = pd.merge(merged_df, df_to_join, on=merge_key, how='left')
                        except:
                            pass  # Skip if join fails
                        break
        
        return merged_df, {
            'join_type': join_type,
            'primary_join': best_join,
            'total_tables': len(self.tables),
            'final_shape': merged_df.shape
        }
    
    def concatenate_tables(self):
        """Fallback: concatenate tables horizontally with prefixes"""
        all_dfs = []
        
        for table_name, df in self.tables.items():
            df_copy = df.copy()
            # Add table prefix to all columns
            df_copy.columns = [f"{table_name}_{col}" for col in df_copy.columns]
            all_dfs.append(df_copy)
        
        # Concatenate horizontally
        result_df = pd.concat(all_dfs, axis=1)
        
        return result_df, {
            'join_type': 'concatenation',
            'total_tables': len(self.tables),
            'final_shape': result_df.shape
        }
    
    def manual_join(self, table1_name, table2_name, col1, col2, join_type='inner'):
        """Manually specify join between two tables"""
        if table1_name not in self.tables or table2_name not in self.tables:
            raise Exception("One or both tables not found")
        
        df1 = self.tables[table1_name].copy()
        df2 = self.tables[table2_name].copy()
        
        if col1 not in df1.columns or col2 not in df2.columns:
            raise Exception("Join columns not found in tables")
        
        # Rename columns to avoid conflicts
        df1_cols = {col: f"{table1_name}_{col}" for col in df1.columns if col != col1}
        df2_cols = {col: f"{table2_name}_{col}" for col in df2.columns if col != col2}
        
        df1.rename(columns=df1_cols, inplace=True)
        df2.rename(columns=df2_cols, inplace=True)
        
        # Perform join
        merged_df = pd.merge(df1, df2, left_on=col1, right_on=col2, how=join_type)
        
        return merged_df, {
            'join_type': join_type,
            'table1': table1_name,
            'table2': table2_name,
            'join_columns': f"{col1} = {col2}",
            'final_shape': merged_df.shape
        }
    
    def get_tables_info(self):
        """Get information about all loaded tables"""
        info = {}
        for table_name, df in self.tables.items():
            info[table_name] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'sample': df.head(3).to_dict('records')
            }
        return info


# Helper function to integrate with your main Flask app
def process_multiple_csvs(files_list):
    """
    Process multiple CSV files and return a merged DataFrame
    
    Args:
        files_list: List of file objects from Flask request.files
    
    Returns:
        tuple: (merged_dataframe, processing_info, error_message)
    """
    try:
        processor = MultiCSVProcessor()
        
        # Add all files
        for file in files_list:
            if file.filename and file.filename.endswith('.csv'):
                secure_name = secure_filename(file.filename)
                table_name, shape = processor.add_table(file.stream, secure_name)
                print(f"Added table '{table_name}' with shape {shape}")
        
        if len(processor.tables) == 0:
            return None, None, "No valid CSV files found"
        
        if len(processor.tables) == 1:
            # Only one table, return it as-is
            table_name = list(processor.tables.keys())[0]
            df = processor.tables[table_name]
            info = {
                'single_table': True,
                'table_name': table_name,
                'shape': df.shape
            }
            return df, info, None
        
        # Multiple tables - try to join them
        merged_df, join_info = processor.auto_join_tables()
        
        # Add suggestions to info
        suggestions = processor.find_join_suggestions()
        join_info['suggestions'] = suggestions[:3]  # Top 3 suggestions
        join_info['tables_info'] = processor.get_tables_info()
        
        return merged_df, join_info, None
        
    except Exception as e:
        print(f"Error in process_multiple_csvs: {traceback.format_exc()}")
        return None, None, str(e)


# Example usage for testing
if __name__ == "__main__":
    # This is just for testing the module
    processor = MultiCSVProcessor()
    
    # You would normally add files like this:
    # processor.add_table(file_stream, "customers.csv")
    # processor.add_table(file_stream, "orders.csv")
    # merged_df, info = processor.auto_join_tables()
    
    print("Multi-CSV processor ready!")
