import os
import time
import tempfile
import math
from typing import Dict, List, Optional, Tuple

import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth


def create_spotify_client() -> spotipy.Spotify:
    scopes = [
        "user-library-read",
        "user-read-playback-state",
        "user-modify-playback-state",
    ]
    scope_str = " ".join(scopes)
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
            scope=scope_str,
        )
    )


def get_most_recent_liked_track(sp: spotipy.Spotify) -> Optional[Dict]:
    results = sp.current_user_saved_tracks(limit=1)
    items = results.get("items") or []
    if not items:
        return None
    return items[0].get("track")


def get_active_or_available_device(sp: spotipy.Spotify) -> Optional[str]:
    devices_resp = sp.devices() or {}
    devices = devices_resp.get("devices") or []
    active_devices = [d for d in devices if d.get("is_active")]
    if active_devices:
        return active_devices[0].get("id")
    # Fallback: take the first available device if any
    if devices:
        return devices[0].get("id")
    return None


def ensure_playback_on_device(
    sp: spotipy.Spotify, device_id: Optional[str]
) -> Optional[str]:
    if device_id:
        return device_id
    print(
        "No Spotify Connect devices found. Open Spotify on your Mac/phone, start playing any track to activate a device, then re-run."
    )
    return None


def start_playback(sp: spotipy.Spotify, device_id: str, track_uri: str) -> None:
    try:
        sp.start_playback(device_id=device_id, uris=[track_uri])
    except spotipy.SpotifyException as exc:
        print(f"Failed to start playback: {exc}")


def seconds_from_ms(ms: Optional[int]) -> float:
    if not ms:
        return 30.0
    return max(1.0, ms / 1000.0)


def find_loopback_input_device_name(
    preferred_keywords: Optional[List[str]] = None,
) -> Optional[str]:
    try:
        import sounddevice as sd  # type: ignore
    except Exception:
        return None

    if preferred_keywords is None:
        preferred_keywords = ["blackhole", "loopback", "soundflower"]

    preferred_keywords = [k.lower() for k in preferred_keywords]

    try:
        devices = sd.query_devices()
    except Exception:
        return None

    best_match_name: Optional[str] = None
    for dev in devices:
        if dev.get("max_input_channels", 0) <= 0:
            continue
        name = str(dev.get("name", ""))
        lower = name.lower()
        if any(k in lower for k in preferred_keywords):
            best_match_name = name
            break
    return best_match_name


def record_system_audio(
    device_name: str,
    duration_seconds: float,
    samplerate: int = 44100,
    channels: int = 2,
    output_wav_path: Optional[str] = None,
) -> Optional[str]:
    try:
        import sounddevice as sd  # type: ignore
        import soundfile as sf  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        print(f"Recording deps missing (sounddevice, soundfile, numpy): {exc}")
        return None

    # Resolve device index by name
    try:
        device_index = None
        for idx, dev in enumerate(sd.query_devices()):
            if (
                str(dev.get("name", "")) == device_name
                and dev.get("max_input_channels", 0) > 0
            ):
                device_index = idx
                break
        if device_index is None:
            print(f"Could not find input device named '{device_name}'.")
            return None
    except Exception as exc:
        print(f"Failed to enumerate audio devices: {exc}")
        return None

    output_path = output_wav_path or os.path.join(
        tempfile.gettempdir(), "captured_audio.wav"
    )
    total_frames = int(math.ceil(duration_seconds * samplerate))

    print(
        f"Recording system audio from '{device_name}' for {duration_seconds:.1f}s → {output_path}"
    )
    try:
        sd.default.samplerate = samplerate
        sd.default.channels = channels
        with sd.InputStream(
            device=device_index, channels=channels, samplerate=samplerate
        ):
            audio = sd.rec(
                frames=total_frames,
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
            )
            sd.wait()
        sf.write(output_path, audio, samplerate)
        return output_path
    except Exception as exc:
        print(f"Recording failed: {exc}")
        return None


def download_preview_audio(
    preview_url: str, output_path: Optional[str] = None
) -> Optional[str]:
    if not preview_url:
        return None
    output = output_path or os.path.join(tempfile.gettempdir(), "preview.mp3")
    try:
        with requests.get(preview_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(output, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return output
    except Exception as exc:
        print(f"Failed to download preview audio: {exc}")
        return None


def estimate_key_from_chroma(chroma: "np.ndarray") -> Tuple[str, float]:
    # Lazy import to avoid heavy deps if not analyzing
    import numpy as np  # type: ignore

    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )
    key_names = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]

    chroma_mean = chroma.mean(axis=1)
    chroma_norm = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)

    best_corr = -1.0
    best_key = "Unknown"

    for i in range(12):
        rot_major = np.roll(major_profile, i)
        rot_minor = np.roll(minor_profile, i)
        rot_major = rot_major / (np.linalg.norm(rot_major) + 1e-9)
        rot_minor = rot_minor / (np.linalg.norm(rot_minor) + 1e-9)

        corr_major = float(np.dot(chroma_norm, rot_major))
        corr_minor = float(np.dot(chroma_norm, rot_minor))

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = f"{key_names[i]} major"
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = f"{key_names[i]} minor"

    confidence = max(0.0, min(1.0, (best_corr + 1.0) / 2.0))
    return best_key, confidence


def analyze_audio(audio_path: str) -> Dict[str, object]:
    import numpy as np  # type: ignore
    import librosa  # type: ignore

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Key estimation via chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_name, key_conf = estimate_key_from_chroma(chroma)

    return {
        "sample_rate": sr,
        "duration_sec": float(len(y)) / float(sr),
        "tempo_bpm": float(tempo),
        "num_beats": int(len(beat_times)),
        "first_8_beat_times_sec": [float(t) for t in beat_times[:8]],
        "key": key_name,
        "key_confidence": float(key_conf),
    }


def main() -> None:
    print("Authenticating with Spotify…")
    sp = create_spotify_client()

    track = get_most_recent_liked_track(sp)
    if not track:
        print("No liked tracks found on your account.")
        return

    track_name = track.get("name") or "Unknown"
    artists = ", ".join([a.get("name", "?") for a in (track.get("artists") or [])])
    track_uri = track.get("uri")
    preview_url = track.get("preview_url")
    duration_ms = track.get("duration_ms") or 0

    print(f"Most recent liked track: {artists} — {track_name}")

    device_id = get_active_or_available_device(sp)
    device_id = ensure_playback_on_device(sp, device_id)
    if device_id and track_uri:
        print("Starting Spotify playback on your device…")
        start_playback(sp, device_id, track_uri)
    else:
        print(
            "Skipping playback (no device available). You'll still get analysis if audio can be captured or preview is available."
        )

    # Try to capture system audio first (requires loopback device)
    audio_path: Optional[str] = None
    loopback_name = find_loopback_input_device_name()
    track_duration_sec = seconds_from_ms(duration_ms)

    if loopback_name and device_id and track_uri:
        # Small delay to allow playback to ramp up before recording
        time.sleep(1.0)
        audio_path = record_system_audio(
            loopback_name, duration_seconds=track_duration_sec
        )
        if not audio_path:
            print(
                "System-audio capture failed; will try preview_url fallback if available."
            )
    else:
        if not loopback_name:
            print(
                "No loopback input device found (e.g., BlackHole). Install and set your system/output to route Spotify audio if you want full-track capture."
            )

    # Fallback to 30s preview download if capture was not possible
    if not audio_path and preview_url:
        print("Downloading 30s preview for analysis…")
        audio_path = download_preview_audio(preview_url)

    if not audio_path:
        print("No audio to analyze (neither loopback capture nor preview available).")
        return

    print(f"Analyzing audio: {audio_path}")
    try:
        results = analyze_audio(audio_path)
    except Exception as exc:
        print(f"Audio analysis failed: {exc}")
        return

    print("Analysis results:")
    for k, v in results.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    """
    Requirements (install via pip):
      - spotipy
      - requests
      - librosa
      - sounddevice, soundfile, numpy  (only needed if capturing system audio)

    Notes for macOS system-audio capture:
      - Install a loopback driver like BlackHole (two-channel is fine).
      - In macOS Audio MIDI Setup, create a Multi-Output device if you want to hear audio while capturing.
      - Set Spotify output to route through the loopback device so this script can record it.
    """
    main()
