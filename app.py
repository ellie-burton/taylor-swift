from flask import Flask, render_template, request
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import os


# Set up your credentials

client_id=os.getenv("SPOTIPY_CLIENT_ID")
client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")

# Authorize access to Spotify API
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Initialize Flask app
app = Flask(__name__)

# Function to fetch audio features for a list of song URIs
def get_audio_features(song_uris):
    audio_features = []
    for uri in song_uris:
        features = sp.audio_features(uri)
        if features[0]:
            audio_features.append(features[0])
    return audio_features

# Search for Taylor Swift's songs and get their URIs
def get_artist_songs(artist_name):
    results = sp.search(q='artist:' + artist_name, type='track', limit=50)
    songs = results['tracks']['items']
    
    # Get song URIs and names
    song_uris = [song['uri'] for song in songs]
    song_names = [song['name'] for song in songs]
    
    return song_uris, song_names

# Function to extract relevant features for similarity calculation
def extract_features(df):
    return df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the form submission
@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        friend_songs_input = request.form['friend_songs']
        friend_songs = [uri.strip() for uri in friend_songs_input.split(',') if uri.strip()]

        # Fetch Taylor Swift's songs
        taylor_song_uris, taylor_song_names = get_artist_songs("Taylor Swift")
        taylor_audio_features = get_audio_features(taylor_song_uris)

        # Convert audio features to a DataFrame
        taylor_audio_features_df = pd.DataFrame(taylor_audio_features)

        # Extract specific audio features
        taylor_features = extract_features(taylor_audio_features_df)

        # Get friend's song features
        friend_audio_features = get_audio_features(friend_songs)
        friend_audio_features_df = pd.DataFrame(friend_audio_features)

        # Extract friend's song features
        friend_features = extract_features(friend_audio_features_df)

        # Calculate cosine similarity
        similarities = cosine_similarity(taylor_features, friend_features)
        average_similarity = np.mean(similarities, axis=1)

        # Add similarity scores to the DataFrame
        taylor_songs_df = pd.DataFrame({
            'Song': taylor_song_names,
            'Similarity': average_similarity
        })
        taylor_songs_df = taylor_songs_df.sort_values(by='Similarity', ascending=False)

        # Output top Taylor Swift songs
        top_songs = taylor_songs_df[['Song', 'Similarity']].head(5).to_html()

        return render_template('results.html', top_songs=top_songs)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
