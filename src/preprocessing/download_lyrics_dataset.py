"""
Download lyrics dataset (2011–2023, ~30k songs) from HuggingFace
and save to data/raw/lyrics/lyrics_raw.json

Dataset: https://huggingface.co/datasets/mrYou/lyrics-dataset
"""

from datasets import load_dataset
import json
import os


def download_lyrics_dataset(output_path="data/raw/lyrics/lyrics_raw.json"):
    print("Downloading dataset from HuggingFace...")

    # Load dataset
    ds = load_dataset("mrYou/lyrics-dataset")

    data = []
    for idx, item in enumerate(ds["train"]):

        title = item.get("title", None)
        artist = item.get("artist", None)
        year = item.get("year", None)
        lyrics = item.get("lyrics", "")

        # Auto-generate song_id (stable ID)
        song_id = f"hf_{idx+1:05d}"

        data.append({
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "year": year,
            "lyrics": lyrics
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Total songs: {len(data)}")


if __name__ == "__main__":
    download_lyrics_dataset()
