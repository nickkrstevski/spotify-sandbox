import csv
import os
from typing import Dict, List, Optional

import requests
from mutagen.flac import FLAC, Picture


FLAC_DIR = os.path.join(os.getcwd(), "flac")
METADATA_CSV = os.path.join(FLAC_DIR, "metadata.csv")


TAG_FIELDS: List[str] = [
    "title",
    "artist",
    "album",
    "albumartist",
    "genre",
    "date",
    "tracknumber",
    "discnumber",
    "isrc",
]


def download_image(url: str) -> Optional[bytes]:
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        return None
    return None


def tag_file(
    path: str, tags: Dict[str, str], cover_bytes: Optional[bytes], mime: Optional[str]
) -> None:
    audio = FLAC(path)
    # Clear existing fields we will set to avoid duplicates
    for key in TAG_FIELDS:
        if key in audio:
            del audio[key]
    for key in TAG_FIELDS:
        value = (tags.get(key) or "").strip()
        if value:
            audio[key] = [value]
    # Replace front cover if provided
    if cover_bytes:
        picture = Picture()
        picture.type = 3  # Front cover
        picture.mime = mime or "image/jpeg"
        picture.desc = "Front Cover"
        picture.data = cover_bytes
        # Remove existing pictures
        audio.clear_pictures()
        audio.add_picture(picture)
    audio.save()


def main() -> None:
    if not os.path.isfile(METADATA_CSV):
        raise SystemExit(f"Missing CSV: {METADATA_CSV}")
    if not os.path.isdir(FLAC_DIR):
        raise SystemExit(f"Missing FLAC directory: {FLAC_DIR}")

    updated = 0
    missing = 0
    with open(METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            suggested = (row.get("suggested_filename") or "").strip()
            if not suggested:
                continue
            flac_path = os.path.join(FLAC_DIR, suggested)
            if not os.path.isfile(flac_path):
                missing += 1
                print(f"Skip (file not found): {suggested}")
                continue

            tags = {k: (row.get(k) or "").strip() for k in TAG_FIELDS}
            cover_url = (row.get("album_art_url") or "").strip()
            cover_bytes: Optional[bytes] = None
            cover_mime: Optional[str] = None
            if cover_url:
                cover_bytes = download_image(cover_url)
                if cover_bytes:
                    # Guess mime from URL extension
                    lower = cover_url.lower()
                    if lower.endswith(".png"):
                        cover_mime = "image/png"
                    else:
                        cover_mime = "image/jpeg"
            try:
                tag_file(flac_path, tags, cover_bytes, cover_mime)
                updated += 1
                print(f"Tagged: {os.path.basename(flac_path)}")
            except Exception as exc:
                print(f"Failed to tag {suggested}: {exc}")

    print(f"Done. Updated {updated} file(s). Missing {missing} file(s).")


if __name__ == "__main__":
    main()
