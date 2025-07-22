import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from tqdm import tqdm

# Spotify authorization scope
SCOPE = 'user-library-read'

# Authenticate
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope=SCOPE
))

# Get all liked songs
def get_all_liked_tracks():
    tracks = []
    limit = 50
    offset = 0

    print("Fetching liked songs...")
    while True:
        results = sp.current_user_saved_tracks(limit=limit, offset=offset)
        if not results['items']:
            break
        tracks.extend(results['items'])
        offset += limit
    return tracks

# Extract metadata and audio features
def extract_track_data(tracks):
    data = []
    print("Extracting metadata and audio features...")

    for chunk_start in tqdm(range(0, len(tracks), 50)):
        chunk = tracks[chunk_start:chunk_start + 50]
        track_ids = [item['track']['id'] for item in chunk]
        features = sp.audio_features(tracks=track_ids)

        for item, feat in zip(chunk, features):
            track = item['track']
            artist_id = track['artists'][0]['id']
            artist_info = sp.artist(artist_id)
            genres = artist_info.get('genres', [])

            data.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'release_date': track['album']['release_date'],
                'id': track['id'],
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms'],
                'explicit': track['explicit'],
                'genres': genres,
                'danceability': feat['danceability'] if feat else None,
                'energy': feat['energy'] if feat else None,
                'key': feat['key'] if feat else None,
                'loudness': feat['loudness'] if feat else None,
                'mode': feat['mode'] if feat else None,
                'speechiness': feat['speechiness'] if feat else None,
                'acousticness': feat['acousticness'] if feat else None,
                'instrumentalness': feat['instrumentalness'] if feat else None,
                'liveness': feat['liveness'] if feat else None,
                'valence': feat['valence'] if feat else None,
                'tempo': feat['tempo'] if feat else None,
                'time_signature': feat['time_signature'] if feat else None,
            })

    return pd.DataFrame(data)

# Main execution
if __name__ == "__main__":
    liked_tracks = get_all_liked_tracks()
    df = extract_track_data(liked_tracks)
    df.to_csv('liked_songs_with_metadata.csv', index=False)
    print("Saved to liked_songs_with_metadata.csv")