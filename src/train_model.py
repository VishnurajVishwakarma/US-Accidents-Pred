import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

from preprocessing import load_and_clean_data
from feature_engineering import feature_engineering_pipeline

def train_models(data_path, models_dir="models"):
    print(f"Loading data from {data_path}...")
    df = load_and_clean_data(data_path)
    
    print("Applying feature engineering...")
    df = feature_engineering_pipeline(df)
    
    # Feature matrix X and target y
    target_col = 'risk_score'
    exclude_cols = ['Severity', 'risk_score', 'Start_Lat', 'Start_Lng']
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Let's include Lat/Lng in features so the model can learn spatial risk patterns
    features_with_spatial = features + ['Start_Lat', 'Start_Lng']
    
    X = df[features_with_spatial]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("Training RandomForest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    print(f"RandomForest RMSE: {rf_rmse:.4f}")
    
    print("Training XGBoost...")
    xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    print(f"XGBoost RMSE: {xgb_rmse:.4f}")
    
    print("Training LightGBM...")
    lgb = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    lgb.fit(X_train, y_train)
    lgb_preds = lgb.predict(X_test)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
    print(f"LightGBM RMSE: {lgb_rmse:.4f}")
    
    # Pick the best model
    models = {'RandomForest': (rf, rf_rmse), 'XGBoost': (xgb, xgb_rmse), 'LightGBM': (lgb, lgb_rmse)}
    best_name = min(models.items(), key=lambda x: x[1][1])[0]
    best_model = models[best_name][0]
    
    print(f"Best model is {best_name} with RMSE {models[best_name][1]:.4f}")
    
    model_path = os.path.join(models_dir, 'best_risk_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

    # Save features list
    features_path = os.path.join(models_dir, 'model_features.pkl')
    joblib.dump(features_with_spatial, features_path)

if __name__ == "__main__":
    train_models('data/US_Accidents.csv')
