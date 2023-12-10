from flask import Flask, render_template, request
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json

app = Flask(__name__)

# Load Spotify credentials from auth.json
with open('auth.json') as f:
    auth_data = json.load(f)
    SPOTIPY_CLIENT_ID = auth_data['SPOTIPY_CLIENT_ID']
    SPOTIPY_CLIENT_SECRET = auth_data['SPOTIPY_CLIENT_SECRET']

# Initialize Spotipy client with your credentials
auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_song():
    query = request.form['song_query']
    result = sp.search(q=query, limit=1, type='track')
    
    if result['tracks']['items']:
        track_uri = result['tracks']['items'][0]['uri']
        audio_features = sp.audio_features(track_uri)
        # Process audio_features as needed
        print(audio_features)
        return render_template('result.html', audio_features=audio_features)
    else:
        return render_template('result.html', audio_features=None)

if __name__ == '__main__':
    app.run(debug=True)
