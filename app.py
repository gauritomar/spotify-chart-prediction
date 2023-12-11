from flask import Flask, render_template, request
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pandas as pd
import joblib
import random

app = Flask(__name__)

xgb_model_10 = joblib.load('models/xgb_model_under_10.pkl')
xgb_model_50 = joblib.load('models/xgb_model_under_50.pkl')

# Load Spotify credentials from auth.json
with open('auth.json') as f:
    auth_data = json.load(f)
    SPOTIPY_CLIENT_ID = auth_data['SPOTIPY_CLIENT_ID']
    SPOTIPY_CLIENT_SECRET = auth_data['SPOTIPY_CLIENT_SECRET']

# Initialize Spotipy client with your credentials
auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

def predict_probabilities(xgb_model_10, xgb_model_50, features_10, features_50):
    # try:
        # Make prediction for under 10 and under 50
        probability_under_10 = xgb_model_10.predict_proba(features_10)[:, 1] * 100
        probability_under_50 = xgb_model_50.predict_proba(features_50)[:, 1] * 100
    # except Exception as e:
    #     probability_under_10 = random.uniform(0, 50) 
    #     probability_under_50 = probability_under_10 + 10 

    # Ensure the difference between the probabilities
    # if probability_under_50 <= probability_under_10 + 10:
    #     probability_under_50 = probability_under_10 + 10

        return probability_under_10, probability_under_50

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_song():
    query = request.form['song_query']

    if not query:
        query = "adele hello"

    result = sp.search(q=query, limit=1, type='track')

    if not result['tracks']['items']:
        # If the search result is empty, modify query to "adele hello"
        query = "adele hello"
        result = sp.search(q=query, limit=1, type='track')
    
    if result['tracks']['items']:
        track_uri = result['tracks']['items'][0]['uri']
        # Fetching audio features
        audio_features = sp.audio_features(track_uri)[0]

        # Creating DataFrame for inference
        test_data = pd.DataFrame({
            'danceability_%': [audio_features['danceability'] * 100],
            'valence_%': [audio_features['valence'] * 100],
            'energy_%': [audio_features['energy'] * 100],
            'acousticness_%': [audio_features['acousticness'] * 100],
            'instrumentalness_%': [audio_features['instrumentalness'] * 100],
            'liveness_%': [audio_features['liveness'] * 100],
            'speechiness_%': [audio_features['speechiness'] * 100]
        })

        # Extracting features for under 10 and under 50
        features_10 = test_data[['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']]
        features_50 = test_data[['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']]

        # Make prediction for under 10 and under 50
        probability_under_10, probability_under_50 = predict_probabilities(xgb_model_10, xgb_model_50, features_10, features_50)


        return render_template('result.html', audio_features=audio_features,
                               probability_under_10=probability_under_10[0],
                               probability_under_50=probability_under_50[0])
    else:
        return render_template('result.html', audio_features=None)

if __name__ == '__main__':
    app.run(debug=True)