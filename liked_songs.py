import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from tqdm import tqdm
from typing import List, Optional

# Spotify authorization scope
SCOPE = "user-library-read"

# Authenticate
sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope=SCOPE,
    )
)


# Get most recent liked songs (up to max_tracks)
def get_recent_liked_tracks(max_tracks: int = 100):
    tracks = []
    page_limit = 50  # Spotify API maximum for this endpoint
    offset = 0

    print(f"Fetching up to {max_tracks} most recent liked songs...")
    while len(tracks) < max_tracks:
        to_fetch = min(page_limit, max_tracks - len(tracks))
        results = sp.current_user_saved_tracks(limit=to_fetch, offset=offset)
        if not results or not results.get("items"):
            break
        items = results.get("items") or []
        tracks.extend(items)
        if len(items) < to_fetch:
            break
        offset += page_limit
    return tracks[:max_tracks]


# Extract metadata and audio features
def extract_track_data(tracks):
    data = []
    print("Extracting metadata and audio features...")

    # Filter out items with missing track or id to avoid None errors
    valid_items = []
    for item in tracks:
        track = (item or {}).get("track")
        if not track or not track.get("id"):
            continue
        valid_items.append(item)

    def fetch_audio_features_with_fallback(
        track_ids: List[str],
    ) -> List[Optional[dict]]:
        try:
            features_batch = sp.audio_features(tracks=track_ids)
            return features_batch or [None] * len(track_ids)
        except Exception:
            # Fallback: fetch per-track to skip problematic IDs
            fallback_features: List[Optional[dict]] = []
            for track_id in track_ids:
                try:
                    single = sp.audio_features(tracks=[track_id])
                    fallback_features.append(single[0] if single else None)
                except Exception:
                    fallback_features.append(None)
            return fallback_features

    def fetch_audio_analysis_fields(track_id: str) -> dict:
        try:
            analysis = sp.audio_analysis(track_id)
            track_info = analysis.get("track", {}) if isinstance(analysis, dict) else {}
            beats = analysis.get("beats", []) if isinstance(analysis, dict) else []

            beats_count = len(beats) if isinstance(beats, list) else None
            first_beat_start = (
                beats[0].get("start")
                if beats_count and isinstance(beats[0], dict)
                else None
            )
            average_beat_duration = (
                sum(b.get("duration", 0) for b in beats if isinstance(b, dict))
                / beats_count
                if beats_count and beats_count > 0
                else None
            )

            return {
                "key_confidence": track_info.get("key_confidence"),
                "mode_confidence": track_info.get("mode_confidence"),
                "tempo_confidence": track_info.get("tempo_confidence"),
                "time_signature_confidence": track_info.get(
                    "time_signature_confidence"
                ),
                "beats_count": beats_count,
                "first_beat_start": first_beat_start,
                "average_beat_duration": average_beat_duration,
            }
        except Exception:
            return {
                "key_confidence": None,
                "mode_confidence": None,
                "tempo_confidence": None,
                "time_signature_confidence": None,
                "beats_count": None,
                "first_beat_start": None,
                "average_beat_duration": None,
            }

    for chunk_start in tqdm(range(0, len(valid_items), 50)):
        chunk = valid_items[chunk_start : chunk_start + 50]
        track_ids = [item["track"]["id"] for item in chunk]
        features = fetch_audio_features_with_fallback(track_ids)

        # Ensure features list aligns with track_ids length
        if len(features) < len(track_ids):
            features = features + [None] * (len(track_ids) - len(features))
        elif len(features) > len(track_ids):
            features = features[: len(track_ids)]

        for item, feat in zip(chunk, features):
            track = item.get("track", {})
            artist_id = None
            try:
                artist_id = (track.get("artists") or [{}])[0].get("id")
            except Exception:
                artist_id = None

            artist_info = None
            if artist_id:
                try:
                    artist_info = sp.artist(artist_id)
                except Exception:
                    artist_info = None
            genres = (
                artist_info.get("genres", []) if isinstance(artist_info, dict) else []
            )

            analysis_fields = (
                fetch_audio_analysis_fields(track.get("id")) if track.get("id") else {}
            )

            data.append(
                {
                    "name": track.get("name"),
                    "artist": ((track.get("artists") or [{}])[0]).get("name"),
                    "album": (track.get("album") or {}).get("name"),
                    "release_date": (track.get("album") or {}).get("release_date"),
                    "id": track.get("id"),
                    "popularity": track.get("popularity"),
                    "duration_ms": track.get("duration_ms"),
                    "explicit": track.get("explicit"),
                    "genres": genres,
                    "danceability": feat.get("danceability")
                    if isinstance(feat, dict)
                    else None,
                    "energy": feat.get("energy") if isinstance(feat, dict) else None,
                    "key": feat.get("key") if isinstance(feat, dict) else None,
                    "loudness": feat.get("loudness")
                    if isinstance(feat, dict)
                    else None,
                    "mode": feat.get("mode") if isinstance(feat, dict) else None,
                    "speechiness": feat.get("speechiness")
                    if isinstance(feat, dict)
                    else None,
                    "acousticness": feat.get("acousticness")
                    if isinstance(feat, dict)
                    else None,
                    "instrumentalness": feat.get("instrumentalness")
                    if isinstance(feat, dict)
                    else None,
                    "liveness": feat.get("liveness")
                    if isinstance(feat, dict)
                    else None,
                    "valence": feat.get("valence") if isinstance(feat, dict) else None,
                    "tempo": feat.get("tempo") if isinstance(feat, dict) else None,
                    "time_signature": feat.get("time_signature")
                    if isinstance(feat, dict)
                    else None,
                    # Analysis-derived fields
                    "key_confidence": analysis_fields.get("key_confidence"),
                    "mode_confidence": analysis_fields.get("mode_confidence"),
                    "tempo_confidence": analysis_fields.get("tempo_confidence"),
                    "time_signature_confidence": analysis_fields.get(
                        "time_signature_confidence"
                    ),
                    "beats_count": analysis_fields.get("beats_count"),
                    "first_beat_start": analysis_fields.get("first_beat_start"),
                    "average_beat_duration": analysis_fields.get(
                        "average_beat_duration"
                    ),
                }
            )

    return pd.DataFrame(data)


# Main execution
if __name__ == "__main__":
    liked_tracks = get_recent_liked_tracks(max_tracks=10)
    df = extract_track_data(liked_tracks)
    df.to_csv("liked_songs.csv", index=False)
    print("Saved to liked_songs.csv")
