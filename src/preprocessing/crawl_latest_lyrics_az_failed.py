import billboard
import requests
import time
import json
import os
from bs4 import BeautifulSoup


# ===========================================================
# Billboard utilities
# ===========================================================
def fetch_chart(chart, date=None, retries=3):
    for _ in range(retries):
        try:
            return billboard.ChartData(chart, date=date)
        except:
            time.sleep(1)
    return None


def fetch_billboard_songs(target_count=500):
    print("\n=== Fetching Billboard Songs (Goal: 500 unique) ===")

    songs = []
    seen = set()

    # --- Step 1: Year-end 2024 ---
    print("Fetching Year-End 2024 Hot 100 ...")
    chart2024 = fetch_chart("hot-100", date="2024-12-31")
    if chart2024:
        for e in chart2024:
            key = (e.artist.lower(), e.title.lower())
            if key not in seen:
                seen.add(key)
                songs.append({"artist": e.artist, "title": e.title})
    print(f"Added {len(songs)} so far.")

    # --- Step 2: Try Year-end 2025, fallback weekly ---
    print("Fetching Year-End 2025 (fallback to weekly if unavailable) ...")
    chart2025 = fetch_chart("hot-100", date="2025-12-31")
    if chart2025 and len(chart2025) > 0:
        weekly = chart2025
    else:
        weekly = fetch_chart("hot-100")

    for e in weekly:
        key = (e.artist.lower(), e.title.lower())
        if key not in seen:
            seen.add(key)
            songs.append({"artist": e.artist, "title": e.title})
    print(f"Added {len(songs)} so far.")

    # --- Step 3: Keep going weekly backwards until 500 unique ---
    chart = weekly
    while len(songs) < target_count:
        print(f"Fetching previous week (current count: {len(songs)}) ...")
        if not chart.previousDate:
            break
        chart = fetch_chart("hot-100", date=chart.previousDate)
        if not chart:
            break

        for e in chart:
            key = (e.artist.lower(), e.title.lower())
            if key not in seen:
                seen.add(key)
                songs.append({"artist": e.artist, "title": e.title})

    print(f"\n=== Final Billboard unique songs: {len(songs)} ===")
    return songs[:target_count]


# ===========================================================
# AZLyrics utilities
# ===========================================================
def normalize_az(name):
    """Convert artist/title to AZLyrics URL format"""
    name = name.lower()
    name = "".join(c for c in name if c.isalnum())
    return name


def get_azlyrics(artist, title):
    base_artist = normalize_az(artist)
    base_title = normalize_az(title)

    url = f"https://www.azlyrics.com/lyrics/{base_artist}/{base_title}.html"
    print(f"→ Fetching: {artist} - {title}")

    try:
        r = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0"
        })
        if r.status_code != 200:
            print("  [×] Not found")
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        # Lyrics block is the first <div> after the first 15 divs
        divs = soup.find_all("div")
        lyrics_block = None
        for d in divs:
            if d.get("class") is None and d.text.strip():
                lyrics_block = d.get_text("\n").strip()
                break

        if not lyrics_block:
            print("  [×] Lyrics not found in page")
            return None

        print("  [+] OK")
        return lyrics_block

    except Exception as e:
        print("  [×] Error:", e)
        return None


# ===========================================================
# MAIN: Billboard + AZLyrics full pipeline
# ===========================================================
def main():
    os.makedirs("data/raw/lyrics", exist_ok=True)

    # Step 1: Fetch Billboard songs
    billboard_songs = fetch_billboard_songs(target_count=500)

    # Step 2: AZLyrics crawler with dedup
    results = []
    seen = set()

    for song in billboard_songs:
        a = song["artist"]
        t = song["title"]
        key = (a.lower(), t.lower())

        if key in seen:
            continue
        seen.add(key)

        lyrics = get_azlyrics(a, t)
        time.sleep(1.2)  # avoid ban

        if lyrics:
            results.append({
                "artist": a,
                "title": t,
                "lyrics": lyrics
            })

    # Step 3: Save
    outpath = "data/raw/lyrics/billboard_azlyrics_500.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {outpath}")
    print(f"Total songs with lyrics: {len(results)}")


if __name__ == "__main__":
    main()
