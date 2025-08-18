import csv
import os
import re
from typing import Dict, List, Optional

import spotipy
from spotipy.oauth2 import SpotifyOAuth


PLAYLIST_NAME = "Movement"
OUTPUT_DIR = os.path.join(os.getcwd(), "flac")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

# Requires SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI in env
SCOPE = "playlist-read-private playlist-read-collaborative"


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Sanitize a string for safe cross-platform filenames."""
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_spotify_client() -> spotipy.Spotify:
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
            scope=SCOPE,
        )
    )


def find_playlist_id_by_name(sp: spotipy.Spotify, playlist_name: str) -> str:
    limit = 50
    offset = 0
    while True:
        playlists = sp.current_user_playlists(limit=limit, offset=offset) or {}
        items = playlists.get("items") or []
        for pl in items:
            if (pl.get("name") or "").strip().lower() == playlist_name.strip().lower():
                pid = pl.get("id")
                if pid:
                    return pid
        if not playlists.get("next") or len(items) == 0:
            break
        offset += limit
    raise RuntimeError(f"Playlist '{playlist_name}' not found")


def get_playlist_items(sp: spotipy.Spotify, playlist_id: str) -> List[dict]:
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


def fetch_artist_genres_map(
    sp: spotipy.Spotify, artist_ids: List[str]
) -> Dict[str, List[str]]:
    """Batch-fetch artist genres; returns {artist_id: [genres...]}"""
    unique_ids = [i for i in dict.fromkeys([aid for aid in artist_ids if aid])]
    result: Dict[str, List[str]] = {}
    for i in range(0, len(unique_ids), 50):
        batch = unique_ids[i : i + 50]
        if not batch:
            continue
        resp = sp.artists(batch) or {}
        artists = resp.get("artists") or []
        for a in artists:
            aid = a.get("id")
            if not aid:
                continue
            result[aid] = a.get("genres") or []
    return result


def first_or_empty(values: List[str]) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def reduce_release_date(date_str: Optional[str]) -> str:
    if not date_str:
        return ""
    # Spotify dates can be YYYY, YYYY-MM-DD, or YYYY-MM
    return (date_str or "").split("-")[0]


def build_rows_for_csv(sp: spotipy.Spotify, items: List[dict]) -> List[Dict[str, str]]:
    # Collect primary artist IDs for genre lookup
    artist_ids: List[str] = []
    for item in items:
        track = (item or {}).get("track") or {}
        primary_artist = ((track.get("artists") or [{}])[0]) or {}
        aid = primary_artist.get("id")
        if aid:
            artist_ids.append(aid)

    artist_id_to_genres = fetch_artist_genres_map(sp, artist_ids)

    rows: List[Dict[str, str]] = []
    for item in items:
        track = (item or {}).get("track") or {}
        if not track:
            continue

        title = track.get("name") or ""
        artists = track.get("artists") or []
        primary_artist = (artists[0] if artists else {}) or {}
        artist_name = primary_artist.get("name") or ""
        artist_id = primary_artist.get("id") or ""

        album = track.get("album") or {}
        album_name = album.get("name") or ""
        album_artists = album.get("artists") or []
        album_artist_name = (
            first_or_empty(
                [(album_artists[0] or {}).get("name") if album_artists else ""]
            )
            or artist_name
        )

        # Genres from primary artist
        genres_list = artist_id_to_genres.get(artist_id, [])
        genre = ", ".join(genres_list)

        # Additional helpful tags
        date_year = reduce_release_date(album.get("release_date"))
        track_number = track.get("track_number") or ""
        disc_number = track.get("disc_number") or ""
        isrc = ((track.get("external_ids") or {}).get("isrc")) or ""
        spotify_id = track.get("id") or ""
        spotify_uri = track.get("uri") or ""

        # Album art: pick the largest image URL if present
        images = album.get("images") or []
        # Spotify returns images largest->smallest usually, but sort just in case
        try:
            images = sorted(
                images,
                key=lambda im: int((im or {}).get("width") or 0),
                reverse=True,
            )
        except Exception:
            images = images or []
        album_art_url = ((images[0] or {}).get("url") if images else "") or ""

        base_filename = (
            f"{sanitize_filename(artist_name)} - {sanitize_filename(title)}.flac"
        )

        rows.append(
            {
                # Core FLAC/Vorbis comment fields
                "title": title,
                "artist": artist_name,
                "album": album_name,
                "albumartist": album_artist_name,
                "genre": genre,
                "date": date_year,
                "tracknumber": str(track_number) if track_number != "" else "",
                "discnumber": str(disc_number) if disc_number != "" else "",
                # Identifiers and convenience
                "isrc": isrc,
                "spotify_id": spotify_id,
                "spotify_uri": spotify_uri,
                "suggested_filename": base_filename,
                "album_art_url": album_art_url,
            }
        )

    return rows


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    ensure_output_dir(os.path.dirname(path))
    fieldnames = [
        "title",
        "artist",
        "album",
        "albumartist",
        "genre",
        "date",
        "tracknumber",
        "discnumber",
        "isrc",
        "spotify_id",
        "spotify_uri",
        "suggested_filename",
        "album_art_url",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    print(f"Preparing to export FLAC metadata for playlist: {PLAYLIST_NAME}")
    sp = get_spotify_client()
    playlist_id = find_playlist_id_by_name(sp, PLAYLIST_NAME)
    items = get_playlist_items(sp, playlist_id)
    print(f"Found {len(items)} playlist items")

    rows = build_rows_for_csv(sp, items)
    write_csv(OUTPUT_CSV, rows)
    print(f"Wrote {len(rows)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
