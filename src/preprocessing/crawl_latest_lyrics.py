"""
Fetch Billboard Weekly HOT 100 for years 2024 and 2025,
then retrieve lyrics using Genius API.

Output:
data/raw/lyrics/latest_lyrics_raw.json
"""

import requests
from bs4 import BeautifulSoup
import lyricsgenius
import json
import os
import time
from tqdm import tqdm


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


def clean_text(s):
    """Clean song and artist names."""
    s = s.replace("–", "-").replace("’", "'").strip()
    tokens = ["feat", "Feat", "FEAT", "featuring", "ft.", "(with", "Ft."]
    for t in tokens:
        if t in s:
            s = s.split(t)[0].strip()
    return s.strip()


def fetch_billboard_week_chart(date_str):
    """
    Fetch Billboard Hot 100 chart for a specific date (YYYY-MM-DD).

    Example date_str: "2024-02-03"
    """
    url = f"https://www.billboard.com/charts/hot-100/{date_str}/"
    r = requests.get(url, headers=HEADERS)

    if r.status_code != 200:
        # Page may not exist for early 2024/2025 days
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    items = soup.find_all("li", class_="o-chart-results-list__item")

    songs = []

    for item in items:
        title_tag = item.find("h3")
        artist_tag = item.find("span")

        if not title_tag or not artist_tag:
            continue

        title = clean_text(title_tag.text.strip())
        artist = clean_text(artist_tag.text.strip())

        songs.append((title, artist))

    return songs


def fetch_year_hot100(year):
    """
    Fetch ALL Billboard weekly Hot 100 charts for a given year.
    Example years: 2024, 2025
    """
    print(f"\nFetching Billboard Hot 100 for year {year}...")

    songs = set()

    # Billboard 更新週數：每週六
    # 抓年份所有週
    from datetime import datetime, timedelta

    # 從年初開始
    date = datetime(year, 1, 6)

    while date.year == year:
        datestr = date.strftime("%Y-%m-%d")
        weekly_songs = fetch_billboard_week_chart(datestr)

        for t, a in weekly_songs:
            songs.add((t, a))

        date += timedelta(days=7)

    print(f"Collected {len(songs)} unique songs for {year}.")
    return list(songs)


def crawl_lyrics(song_list, token, output_path):
    genius = lyricsgenius.Genius(
        token,
        timeout=10,
        retries=2,
        remove_section_headers=True,
        skip_non_songs=True
    )

    results = {}

    print("\nFetching lyrics from Genius...")

    for title, artist in tqdm(song_list):
        try:
            # 1) 搜尋歌曲
            song = genius.search_song(title, artist)

            if not song:
                continue

            # 2) 只接受英文原版頁面
            url = song.url.lower()
            invalid_keywords = [
                "translation", "traduccion", "traducción",
                "übersetzung", "tradução",
                "перевод", "översättning"
            ]

            # 如果 URL 包含翻譯關鍵字 → 跳過
            if any(k in url for k in invalid_keywords):
                print(f"[SKIP] {title} (translation detected)")
                continue

            # 3) 存入資料
            key = f"{artist}_{title}"
            results[key] = {
                "artist": artist,
                "song": title,
                "lyrics": song.lyrics
            }

        except Exception as e:
            print(f"Error fetching {title}: {e}")

        time.sleep(1)

    # 4) Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} English-only songs to {output_path}")


# -------------------
# Main
# -------------------
if __name__ == "__main__":
    token = input("Enter your Genius API token: ").strip()
    if not token:
        raise ValueError("A Genius API token is required.")

    songs_2024 = fetch_year_hot100(2024)
    songs_2025 = fetch_year_hot100(2025)

    all_songs = songs_2024 + songs_2025
    print(f"\nTotal unique songs collected: {len(all_songs)}")

    crawl_lyrics(all_songs, token, "data/raw/lyrics/latest_lyrics_raw.json")
