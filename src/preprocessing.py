import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """Loads dataset and performs basic cleaning (missing values)."""
    df = pd.read_csv(filepath)
    
    # Handle missing values for numerical columns
    num_cols = ['Visibility(mi)', 'Temperature(F)', 'Humidity(%)', 'Wind_Speed(mph)']
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
            
    # Handle missing values for categorical columns
    cat_cols = ['Weather_Condition', 'City', 'County']
    for c in cat_cols:
        if c in df.columns:
            mode_val = df[c].mode()
            df[c] = df[c].fillna(mode_val[0] if not mode_val.empty else 'Unknown')
            
    # Convert time columns to datetime
    if 'Start_Time' in df.columns:
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    if 'End_Time' in df.columns:
        df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
        
    df.dropna(subset=['Start_Lat', 'Start_Lng', 'Severity'], inplace=True)
    return df
