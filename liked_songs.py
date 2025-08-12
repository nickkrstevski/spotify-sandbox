import os
import logging
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
import time

# Spotify authorization scope
SCOPE = "user-library-read"

# Authenticate
# Reduce noisy HTTP error logs
logging.getLogger("spotipy").setLevel(logging.ERROR)
logging.getLogger("spotipy.client").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Two separate clients: user-auth for library access; app-auth for features/analysis
sp_user = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope=SCOPE,
    )
)

sp_app = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
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
        results = sp_user.current_user_saved_tracks(limit=to_fetch, offset=offset)
        if not results or not results.get("items"):
            break
        items = results.get("items") or []
        tracks.extend(items)
        if len(items) < to_fetch:
            break
        offset += page_limit
    return tracks[:max_tracks]


def get_user_playlist_tracks_by_name(playlist_name: str) -> List[dict]:
    print(f"Searching for playlist named '{playlist_name}'...")
    playlist_id: Optional[str] = None
    limit = 50
    offset = 0
    while True:
        playlists = sp_user.current_user_playlists(limit=limit, offset=offset) or {}
        items = playlists.get("items") or []
        for pl in items:
            if (pl.get("name") or "").strip().lower() == playlist_name.strip().lower():
                playlist_id = pl.get("id")
                break
        if playlist_id or not playlists.get("next") or len(items) == 0:
            break
        offset += limit

    if not playlist_id:
        raise RuntimeError(f"Playlist '{playlist_name}' not found")

    print(f"Fetching tracks from playlist '{playlist_name}' ({playlist_id})...")
    tracks: List[dict] = []
    limit = 100
    offset = 0
    while True:
        page = sp_user.playlist_items(playlist_id, limit=limit, offset=offset) or {}
        items = page.get("items") or []
        tracks.extend(items)
        if not page.get("next") or len(items) == 0:
            break
        offset += limit

    return tracks


# Extract metadata and audio features
def extract_track_data(tracks, include_analysis: bool = False):
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
            # Prefer user-auth client; fallback handled below
            features_batch = sp_user.audio_features(tracks=track_ids)
            return features_batch or [None] * len(track_ids)
        except Exception:
            # Fallback: fetch per-track to skip problematic IDs
            fallback_features: List[Optional[dict]] = []
            for track_id in track_ids:
                single_feat: Optional[dict] = None
                for attempt in range(4):
                    try:
                        # Try user client first, then app client
                        try:
                            single = sp_user.audio_features(tracks=[track_id])
                        except Exception:
                            single = sp_app.audio_features(tracks=[track_id])
                        single_feat = single[0] if single else None
                        break
                    except Exception:
                        time.sleep(2**attempt * 0.5)
                        continue
                fallback_features.append(single_feat)
            return fallback_features

    def fetch_audio_analysis_fields(track_id: str) -> dict:
        try:
            # Small retry for analysis as well
            analysis = None
            last_exc = None
            for attempt in range(3):
                try:
                    analysis = sp_app.audio_analysis(track_id)
                    break
                except Exception as exc:
                    last_exc = exc
                    time.sleep(2**attempt * 0.5)
            if analysis is None:
                raise last_exc or Exception("analysis failed")
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
                    # Prefer user client; fallback to app client
                    try:
                        artist_info = sp_user.artist(artist_id)
                    except Exception:
                        artist_info = sp_app.artist(artist_id)
                except Exception:
                    artist_info = None
            genres = (
                artist_info.get("genres", []) if isinstance(artist_info, dict) else []
            )

            analysis_fields = (
                fetch_audio_analysis_fields(track.get("id"))
                if include_analysis and track.get("id")
                else {}
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
    # Build CSV from playlist 'Movement'
    playlist_tracks = get_user_playlist_tracks_by_name("Movement")
    df = extract_track_data(playlist_tracks, include_analysis=False)

    # Derive BPM and human-readable key
    def key_index_to_name(key_index: Optional[int]) -> Optional[str]:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        if key_index is None:
            return None
        try:
            i = int(key_index)
        except Exception:
            return None
            if 0 <= i <= 11:
                return names[i]
        return None

    def key_with_mode(row) -> Optional[str]:
        key_name = key_index_to_name(row.get("key"))
        if key_name is None:
            return None
        mode = row.get("mode")
        suffix = "major" if mode == 1 else "minor" if mode == 0 else None
        return f"{key_name} {suffix}" if suffix else key_name

    if not df.empty:
        if "tempo" in df.columns:
            df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
            df["bpm"] = df["tempo"].round(1)
        else:
            df["bpm"] = None
        df["key_name"] = df.apply(key_with_mode, axis=1)

        preferred_cols = [
            "name",
            "artist",
            "album",
            "release_date",
            "id",
            "bpm",
            "key_name",
        ]
        other_cols = [c for c in df.columns if c not in preferred_cols]
        df = df[preferred_cols + other_cols]

    df.to_csv("liked_songs.csv", index=False)
    print("Saved to liked_songs.csv")
