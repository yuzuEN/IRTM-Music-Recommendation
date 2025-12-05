import json
import re
from collections import defaultdict
from difflib import SequenceMatcher


# ===============================================================
# Helpers
# ===============================================================
def normalize_title(s):
    """Normalize titles for dedupe: lowercase, remove (), trim."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()

    # remove (feat...), (with...), (remix...)
    s = re.sub(r"\(.*?\)", "", s)

    # remove non-alphanum
    s = re.sub(r"[^a-z0-9 ]+", "", s)

    # collapse spaces
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def normalize_artist(s):
    """Normalize artist string."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()

    # remove featuring parts
    s = re.sub(r"feat.*", "", s)
    s = re.sub(r"with.*", "", s)

    # remove symbols
    s = re.sub(r"[^a-z0-9 ]+", "", s)

    s = re.sub(r"\s+", " ", s)
    return s.strip()


def similarity(a, b):
    """Return fuzzy similarity score between 0 and 1."""
    return SequenceMatcher(None, a, b).ratio()


# ===============================================================
# Duplicate Checker
# ===============================================================
def check_duplicates(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} songs.")

    exact_dupes = defaultdict(list)
    fuzzy_dupes = []

    # =====================================================
    # Step 1: exact duplicate check
    # =====================================================
    for song in data:
        artist = normalize_artist(song.get("artist", ""))
        title = normalize_title(song.get("title", ""))

        key = (artist, title)
        exact_dupes[key].append(song)

    # Only keep groups with duplicates
    exact_duplicates = {
        f"{k[0]}::{k[1]}": v     # convert tuple → string
        for k, v in exact_dupes.items()
        if len(v) > 1
    }

    print(f"\n=== Exact duplicates found: {len(exact_duplicates)} groups ===")

    # =====================================================
    # Step 2: fuzzy duplicate check
    # =====================================================
    songs_norm = []
    for song in data:
        norm_title = normalize_title(song.get("title", ""))
        norm_artist = normalize_artist(song.get("artist", ""))
        songs_norm.append((norm_artist, norm_title, song))

    # compare each pair
    for i in range(len(songs_norm)):
        artist1, title1, song1 = songs_norm[i]

        for j in range(i + 1, len(songs_norm)):
            artist2, title2, song2 = songs_norm[j]

            # fuzzy match only within same artist
            if artist1 != artist2:
                continue

            score = similarity(title1, title2)

            if score > 0.85 and title1 != title2:
                fuzzy_dupes.append({
                    "artist": artist1,
                    "title_1": title1,
                    "title_2": title2,
                    "similarity": score,
                    "song_1_id": song1["song_id"],
                    "song_2_id": song2["song_id"]
                })

    print(f"=== Fuzzy duplicates found (same artist, similar title): {len(fuzzy_dupes)} ===")

    # =====================================================
    # Save report
    # =====================================================
    report = {
        "total_songs": len(data),
        "exact_duplicates": exact_duplicates,
        "fuzzy_duplicates": fuzzy_dupes
    }

    with open("duplicates_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n[Saved] duplicates_report.json")
    print("Done.")


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    check_duplicates("data/raw/lyrics/all_lyrics_raw.json")
