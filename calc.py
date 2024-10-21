import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up your credentials

client_id=os.getenv("SPOTIPY_CLIENT_ID")
client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
# Authorize access to Spotify API
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

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

# Fetch Taylor Swift's songs
taylor_song_uris, taylor_song_names = get_artist_songs("Taylor Swift")
taylor_audio_features = get_audio_features(taylor_song_uris)

# Check if we got valid features and names
if len(taylor_audio_features) != len(taylor_song_names):
    print("The number of songs and audio features do not match!")
else:
    # Convert audio features to a DataFrame
    taylor_audio_features_df = pd.DataFrame(taylor_audio_features)
    print("taylor_audio_features_df")
    print(taylor_audio_features_df.head())
    
    # Extract specific audio features (e.g., danceability, energy, tempo)
    taylor_songs_df = pd.DataFrame({
        'Song': taylor_song_names,
        'Danceability': taylor_audio_features_df['danceability'],
        'Energy': taylor_audio_features_df['energy'],
        'Tempo': taylor_audio_features_df['tempo'],
        'Key': taylor_audio_features_df['key'],
        'Mode': taylor_audio_features_df['mode'],
        'Acousticness': taylor_audio_features_df['acousticness'],
        'Instrumentalness': taylor_audio_features_df['instrumentalness'],
        'Liveness': taylor_audio_features_df['liveness'],
        'Valence': taylor_audio_features_df['valence'],
        'Speechiness': taylor_audio_features_df['speechiness'],
        'Loudness': taylor_audio_features_df['loudness'],

        # Add more features as needed
    })
    
    # Display the DataFrame with the song names and audio features
    print(taylor_songs_df.head())

    # Now, let's extract the features for similarity calculation
    taylor_features = extract_features(taylor_audio_features_df)

    # Example: Let's say these are your friend's favorite songs (URIs)
    friend_song_uris = [
        'https://open.spotify.com/track/0uxSUdBrJy9Un0EYoBowng?si=6587965e8aa0493a',  # Replace with actual URIs of songs
    ]

    # Get your friend's song features
    friend_audio_features = get_audio_features(friend_song_uris)
    friend_audio_features_df = pd.DataFrame(friend_audio_features)
    
    # Extract friend's song features
    friend_features = extract_features(friend_audio_features_df)
    
    # Calculate cosine similarity between Taylor Swift's songs and friend's songs
    similarities = cosine_similarity(taylor_features, friend_features)
    
    # Take the average similarity for each of Taylor Swift's songs
    average_similarity = np.mean(similarities, axis=1)
    
    # Add similarity scores to the DataFrame and sort by similarity
    taylor_songs_df['Similarity'] = average_similarity
    taylor_songs_df = taylor_songs_df.sort_values(by='Similarity', ascending=False)

    # Output top Taylor Swift songs that are most similar to your friend's favorites
    print(taylor_songs_df[['Song', 'Similarity']].head(5))
