import pandas as pd
import joblib

# Load the saved models
xgb_model_10 = joblib.load('models/xgb_model_under_10.pkl')
xgb_model_50 = joblib.load('models/xgb_model_under_50.pkl')

# Load your test file or create a sample test dataframe with similar columns
# For example, let's create a sample test dataframe with similar columns as your training data
test_data = pd.DataFrame({
    'danceability_%': [0.6],
    'valence_%': [0.7],
    'energy_%': [0.8],
    'acousticness_%': [0.2],
    'instrumentalness_%': [0.1],
    'liveness_%': [0.5],
    'speechiness_%': [0.3]
})

# Perform inference using the loaded models
features_10 = test_data[['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']]
features_50 = test_data[['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']]

# Make predictions for under 10 and under 50 using the test features
probability_under_10 = xgb_model_10.predict_proba(features_10)[:, 1] * 100
probability_under_50 = xgb_model_50.predict_proba(features_50)[:, 1] * 100

print("Probability for under 10:", probability_under_10)
print("Probability for under 50:", probability_under_50)
