import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope="user-library-read"
))

"""
	•	user-library-read — access your liked songs
	•	playlist-read-private — access your private playlists
	•	user-top-read — get top tracks and artists
	•	user-read-playback-state — access playback info
	•	user-modify-playback-state — control playback
"""

results = sp.current_user_saved_tracks(limit=10)

for idx, item in enumerate(results['items']):
    track = item['track']
    print(f"{idx + 1}. {track['artists'][0]['name']} - {track['name']}")