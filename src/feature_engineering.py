import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_temporal_features(df):
    if 'Start_Time' in df.columns:
        df['Hour'] = df['Start_Time'].dt.hour
        df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
        df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def generate_risk_score(df):
    # Depending on severity (1-4), scale to risk_score (0-1)
    if 'Severity' in df.columns:
        df['risk_score'] = (df['Severity'] - 1) / 3.0
    return df

def encode_categorical(df):
    # Using Label Encoding for trees logic compatibility
    le = LabelEncoder()
    cat_cols = ['Weather_Condition', 'City', 'County', 'State']
    for c in cat_cols:
        if c in df.columns:
            df[c] = le.fit_transform(df[c].astype(str))
            
    # boolean columns mapping
    bool_cols = ['Amenity', 'Junction', 'Crossing']
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(int)
            
    return df

def feature_engineering_pipeline(df):
    df = create_temporal_features(df)
    df = generate_risk_score(df)
    df = encode_categorical(df)
    
    # drop original time columns, they are inherently not numeric
    cols_to_drop = ['Start_Time', 'End_Time']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    return df
