import os
import re
import shlex
import subprocess
import time
from typing import List, Optional

import spotipy
from spotipy.oauth2 import SpotifyOAuth


# -------- Configuration --------
PLAYLIST_NAME = "Movement"
OUTPUT_DIR = os.path.join(os.getcwd(), "recordings")
BLACKHOLE_DEVICE_NAME = "BlackHole 2ch"
PRE_ROLL_SECONDS = 0.5  # Start recording just before starting playback
POST_ROLL_SECONDS = 0.8  # Record a touch longer than track duration
SKIP_IF_EXISTS = True
MAX_TRACKS: Optional[int] = None  # Set to an int to limit tracks; None for all
OUTPUT_FORMAT = "wav"  # fixed to WAV only per request
INTER_TRACK_GAP_SECONDS = 5  # Safer gap between tracks
BUFFER_PRELOAD_SECONDS = 0.5  # Time to let Spotify buffer after load, before recording

# Requires SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI in env
SCOPE = "user-library-read playlist-read-private playlist-read-collaborative"


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def run(cmd: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return proc.returncode, proc.stdout, proc.stderr


def start_sox_recording(duration_seconds: float, output_path: str) -> subprocess.Popen:
    # Build SoX command to capture from CoreAudio device by name
    record_ceiling = max(1.0, duration_seconds)
    # Explicit WAV format and show progress; inherit stdio to avoid pipe deadlocks and show logs live
    base_cmd = (
        f"sox -V1 -t coreaudio {shlex.quote(BLACKHOLE_DEVICE_NAME)} "
        f"-r 44100 -b 16 -c 2 -e signed-integer {shlex.quote(output_path)} "
        f"trim 0 {record_ceiling:.2f}"
    )
    print(f"Running: {base_cmd}")
    return subprocess.Popen(base_cmd, shell=True, stdout=None, stderr=None, text=True)


def get_spotify_client() -> spotipy.Spotify:
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
            scope=SCOPE,
        )
    )


def get_user_playlist_tracks_by_name(
    sp: spotipy.Spotify, playlist_name: str
) -> List[dict]:
    limit = 50
    offset = 0
    playlist_id: Optional[str] = None
    while True:
        playlists = sp.current_user_playlists(limit=limit, offset=offset) or {}
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

    tracks: List[dict] = []
    limit = 100
    offset = 0
    while True:
        page = sp.playlist_items(playlist_id, limit=limit, offset=offset) or {}
        items = page.get("items") or []
        tracks.extend(items)
        if not page.get("next") or len(items) == 0:
            break
        offset += limit
    return tracks


def ensure_spotify_open() -> None:
    run("open -a Spotify")
    time.sleep(1.0)


def play_spotify_track(track_uri: str) -> None:
    # Disable shuffle to keep order
    run("osascript -e 'tell application \"Spotify\" to set shuffling to false'")
    run(f'osascript -e \'tell application "Spotify" to play track "{track_uri}"\'')


def pause_spotify() -> None:
    run("osascript -e 'tell application \"Spotify\" to pause'")


def preload_spotify_track(
    track_uri: str, buffer_seconds: float = BUFFER_PRELOAD_SECONDS
) -> None:
    # Load and buffer the track without proceeding immediately
    run("osascript -e 'tell application \"Spotify\" to set shuffling to false'")
    run(f'osascript -e \'tell application "Spotify" to play track "{track_uri}"\'')
    time.sleep(max(0.0, buffer_seconds))
    run("osascript -e 'tell application \"Spotify\" to pause'")
    run("osascript -e 'tell application \"Spotify\" to set player position to 0'")


def play_from_start() -> None:
    run("osascript -e 'tell application \"Spotify\" to play'")


## MP3 and metadata tagging removed per request


def main() -> None:
    print("Preparing environment...")
    ensure_output_dir(OUTPUT_DIR)
    print(f"Using CoreAudio device: {BLACKHOLE_DEVICE_NAME}")

    sp = get_spotify_client()
    ensure_spotify_open()

    print(f"Fetching playlist: {PLAYLIST_NAME}")
    items = get_user_playlist_tracks_by_name(sp, PLAYLIST_NAME)
    if MAX_TRACKS is not None:
        items = items[:MAX_TRACKS]
    print(f"Found {len(items)} tracks")

    for idx, item in enumerate(items, start=1):
        pause_spotify()
        track = (item or {}).get("track") or {}
        track_id = track.get("id")
        track_uri = track.get("uri")
        if not track_id or not track_uri:
            print(f"[{idx}] Skipping item with no track")
            continue
        name = track.get("name") or "Unknown"
        artist = ((track.get("artists") or [{}])[0]).get("name") or "Unknown"
        duration_ms = int(track.get("duration_ms") or 0)
        duration_seconds = max(
            1.0, duration_ms / 1000.0 + PRE_ROLL_SECONDS + POST_ROLL_SECONDS
        )
        # duration_seconds = 5

        # Remove numeric prefix; use "Artist - Title"
        base = f"{sanitize_filename(artist)} - {sanitize_filename(name)}"
        wav_path = os.path.join(OUTPUT_DIR, base + ".wav")
        flac_path = os.path.join(OUTPUT_DIR, base + ".flac")
        if SKIP_IF_EXISTS and os.path.exists(wav_path):
            print(f"[{idx}] Exists, skipping: {os.path.basename(wav_path)}")
            continue

        print(
            f"[{idx}] Recording {duration_seconds:.1f}s → {os.path.basename(wav_path)}"
        )

        preload_spotify_track(track_uri)
        proc = start_sox_recording(duration_seconds, wav_path)
        time.sleep(PRE_ROLL_SECONDS)
        play_from_start()
        # Record for the intended ceiling
        time.sleep(duration_seconds)
        pause_spotify()
        time.sleep(INTER_TRACK_GAP_SECONDS)
        try:
            proc.wait()
        except subprocess.TimeoutExpired:
            print(f"[{idx}] Recorder did not stop on silence; forcing stop...")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        pause_spotify()
        rc = proc.returncode
        if rc != 0:
            print(f"[{idx}] recorder exited with code {rc}")
        else:
            print(f"[{idx}] Saved {wav_path}")

        # Ensure a brief settle gap before starting the next recording

    print("Done.")


if __name__ == "__main__":
    main()
