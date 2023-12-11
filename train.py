import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import os

file_path = './spotify-2023.csv'
songs = pd.read_csv(file_path, delimiter=',', encoding='ISO-8859-1')

# Filtering data
songs['in_charts_under_10'] = songs['in_spotify_charts'] <= 10
songs['in_charts_under_50'] = songs['in_spotify_charts'] <= 50

# Features
X = songs[['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']]

# Labels for under 10 and under 50
y_10 = songs['in_charts_under_10']
y_50 = songs['in_charts_under_50']

# Split data into train and test sets for both scenarios
X_train_10, X_test_10, y_10_train, y_10_test = train_test_split(X, y_10, test_size=0.2, random_state=42)
X_train_50, X_test_50, y_50_train, y_50_test = train_test_split(X, y_50, test_size=0.2, random_state=42)

# Create and train XGBoost models for both scenarios
xgb_model_10 = xgb.XGBClassifier()  # You can set parameters here
xgb_model_50 = xgb.XGBClassifier()  # You can set parameters here

xgb_model_10.fit(X_train_10, y_10_train)
xgb_model_50.fit(X_train_50, y_50_train)

# Save the trained models
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(xgb_model_10, 'models/xgb_model_under_10.pkl')
joblib.dump(xgb_model_50, 'models/xgb_model_under_50.pkl')
