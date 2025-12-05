"""
Merge:
1. Billboard/Genius scraped lyrics (dict format)
2. HuggingFace 30k lyrics dataset (list format)

Output:
data/raw/lyrics/all_lyrics_raw.json
"""

import json
import os
import re
from langdetect import detect, LangDetectException


# ===============================================================
# Utility
# ===============================================================
def safe_lower(s):
    return s.lower().strip() if isinstance(s, str) else ""


def normalize_id(s):
    """Convert arbitrary key to a safe ID."""
    if not s:
        return "unknown"
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# ===============================================================
# LINE-LEVEL CLEANING
# ===============================================================
def is_english_line(line):
    """粗略判斷一行是不是英文（字母比例 >= 0.6）"""
    english_chars = re.findall(r"[A-Za-z]", line)
    ratio = len(english_chars) / max(1, len(line))
    return ratio >= 0.6


# ===============================================================
# LYRICS CLEANER（中等強度）
# ===============================================================
def clean_lyrics(text):
    """
    中等強度 lyrics cleaner：
    - 移除 Read More / Contributors / Translations 等資訊行
    - 移除開頭的多餘非英文說明（遇到英語才開始）
    - 移除過長敘述句 (>120 chars)
    - 移除括號式的非歌詞 (e.g., "(Intro)", "(Tony Coles)")
    """

    if not text:
        return ""

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""

    cleaned = []

    BAD_PATTERNS = [
        r"read more", r"contributors", r"translation",
        r"transliteration", r"lyrics? provided",
        r"produced by", r"written by", r"copyright",
        r"all rights reserved"
    ]
    METADATA_RE = re.compile("|".join(BAD_PATTERNS), re.IGNORECASE)

    def is_mostly_non_english(line):
        try:
            return detect(line) != "en"
        except LangDetectException:
            return True

    # ---------- Step 1: remove metadata ----------
    filtered = []
    for line in lines:
        low = line.lower()

        if METADATA_RE.search(low):
            continue

        # (Intro), (Tony Coles), (2x)
        if re.match(r"^\([^)]{2,30}\)$", line):
            continue

        filtered.append(line)

    if not filtered:
        return ""

    # ---------- Step 2: drop early non-English block ----------
    started = False
    for line in filtered:
        if not started:
            if is_mostly_non_english(line):
                continue
            started = True
        cleaned.append(line)

    # fallback
    if not cleaned:
        cleaned = filtered

    # ---------- Step 3: remove overly descriptive first line ----------
    if cleaned and len(cleaned[0]) > 120:
        cleaned.pop(0)

    return "\n".join(cleaned).strip()


# ===============================================================
# SONG-LEVEL check: 是否整首是英文？
# ===============================================================
def is_english_lyrics(text):
    try:
        return detect(text) == "en"
    except:
        return False


# ===============================================================
# NEW: 判斷歌曲是否要丟掉（太短 / 非英文比例高）
# ===============================================================
def should_discard_song(cleaned_text):
    """
    移除以下歌曲：
    - 全文 < 50 字
    - 行數 < 3 行
    - detect(text) 非英文
    """

    if not cleaned_text:
        return True

    lines = cleaned_text.split("\n")
    words = cleaned_text.split()

    # 太少行 → 移除
    if len(lines) < 3:
        return True

    # 單字數太少 → 移除
    if len(words) < 50:
        return True

    # 整首不是英文 → 移除
    try:
        if detect(cleaned_text) != "en":
            return True
    except LangDetectException:
        return True

    return False


# ===============================================================
# Load Billboard/Genius scraped lyrics
# ===============================================================
def load_billboard_genius(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []

    for key, item in raw.items():
        artist = item.get("artist") or "Unknown Artist"
        title = item.get("song") or key.split("_", 1)[-1]
        lyrics_raw = item.get("lyrics", "")

        # clean
        lyrics_clean = clean_lyrics(lyrics_raw)

        # 歌詞清完後太短 → skip
        if should_discard_song(lyrics_clean):
            continue

        song_id = "scraped_" + normalize_id(f"{artist}_{title}")

        data.append({
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "year": None,
            "lyrics": lyrics_clean
        })

    print(f"[Billboard/Genius] Loaded {len(data)} cleaned English songs.")
    return data


# ===============================================================
# Load HuggingFace dataset
# ===============================================================
def load_hf_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []

    for idx, item in enumerate(data):
        title = item.get("title") or f"unknown_title_hf_{idx+1}"
        artist = item.get("artist") or "Unknown Artist"
        lyrics_raw = item.get("lyrics", "")
        year = item.get("year")
        song_id = item.get("song_id") or f"hf_{idx+1:05d}"

        lyrics_clean = clean_lyrics(lyrics_raw)

        if should_discard_song(lyrics_clean):
            continue

        cleaned.append({
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "year": year,
            "lyrics": lyrics_clean
        })

    print(f"[HF Dataset] Loaded {len(cleaned)} cleaned English songs.")
    return cleaned


# ===============================================================
# Merge datasets
# ===============================================================
def merge_datasets(list1, list2):
    merged = []
    seen = set()

    for item in list1 + list2:
        artist = safe_lower(item["artist"])
        title = safe_lower(item["title"])

        key = (artist, title)
        if key in seen:
            continue

        seen.add(key)
        merged.append(item)

    print(f"[Merged] Total unique cleaned English songs: {len(merged)}")
    return merged


# ===============================================================
# Save output
# ===============================================================
def save_output(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {path}")


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    BILLBOARD_PATH = "data/raw/lyrics/latest_lyrics_raw.json"
    HF_PATH        = "data/raw/lyrics/lyrics_raw.json"
    OUTPUT_PATH    = "data/raw/lyrics/all_lyrics_raw.json"

    data_scraped = load_billboard_genius(BILLBOARD_PATH)
    data_hf      = load_hf_dataset(HF_PATH)

    merged = merge_datasets(data_scraped, data_hf)

    save_output(merged, OUTPUT_PATH)


# """
# Merge:
# 1. Billboard/Genius scraped lyrics (dict format)
# 2. HuggingFace 30k lyrics dataset (list format)

# Output:
# data/raw/lyrics/all_lyrics_raw.json
# """

# import json
# import os
# import re
# from langdetect import detect


# # ===============================================================
# # Utility
# # ===============================================================
# def safe_lower(s):
#     return s.lower().strip() if isinstance(s, str) else ""


# def normalize_id(s):
#     """Convert arbitrary key to a safe ID."""
#     if not s:
#         return "unknown"
#     s = s.lower()
#     s = re.sub(r"[^a-z0-9]+", "_", s)
#     return s.strip("_")


# # ===============================================================
# # Line-level English filter
# # ===============================================================
# def is_english_line(line):
#     if not line.strip():
#         return False

#     # 英文字符比例
#     english_chars = re.findall(r"[A-Za-z]", line)
#     ratio = len(english_chars) / max(1, len(line))

#     return ratio >= 0.6


# # ===============================================================
# # Clean single song lyrics
# # ===============================================================
# import re
# from langdetect import detect, LangDetectException

# def clean_lyrics(text):
#     """
#     中等強度 lyrics cleaner：
#     - 移除 Read More / Contributors / Translations 等資訊行
#     - 移除開頭的多餘非英文說明（若英文比例太低 → 刪掉）
#     - 移除開頭過長的敘述句子（>120 chars）
#     - 移除開頭括號敘述 (Not lyrics)
#     - 保留所有疑似歌詞的內容
#     """

#     if not text:
#         return ""

#     # Normalize newlines
#     lines = [l.strip() for l in text.split("\n") if l.strip()]

#     cleaned = []

#     # Patterns representing obvious metadata noise
#     BAD_PATTERNS = [
#         r"read more", r"contributors", r"translation", r"transliteration",
#         r"lyrics? provided", r"produced by", r"written by",
#         r"copyright", r"all rights reserved"
#     ]

#     METADATA_RE = re.compile("|".join(BAD_PATTERNS), re.IGNORECASE)

#     # Helper to detect if a line is mostly non-English
#     def is_mostly_non_english(line):
#         try:
#             lang = detect(line)
#             return lang != "en"
#         except LangDetectException:
#             return True  # treat as non-English if detection fails

#     # Step 1: Drop obvious metadata lines
#     filtered = []
#     for line in lines:
#         low = line.lower()

#         # Remove junk lines
#         if METADATA_RE.search(low):
#             continue

#         # Remove "(Prod. by ...)", "(Tony Coles)" etc.
#         if re.match(r"^\([^)]{2,30}\)$", line):
#             continue

#         filtered.append(line)

#     # Step 2: Remove early non-English block
#     # If前 2–3 行都是非英文 → 刪到遇到英文才開始算歌詞
#     started = False
#     for line in filtered:
#         if not started:
#             if is_mostly_non_english(line):
#                 # keep skipping until English shows up
#                 continue
#             # Found likely-English lyric start
#             started = True

#         cleaned.append(line)

#     # Step 3: If nothing left (rare) fallback to filtered
#     if not cleaned:
#         cleaned = filtered

#     # Step 4: Remove first line if it's too long & looks like description
#     if cleaned and len(cleaned[0]) > 120:
#         cleaned.pop(0)

#     return "\n".join(cleaned).strip()






# # ===============================================================
# # Song-level language detection
# # ===============================================================
# def is_english_lyrics(text):
#     try:
#         lang = detect(text)
#         return lang == "en"
#     except:
#         return False


# # ===============================================================
# # Load Billboard/Genius scraped JSON
# # ===============================================================
# def load_billboard_genius(path):
#     with open(path, "r", encoding="utf-8") as f:
#         raw = json.load(f)

#     data = []

#     for key, item in raw.items():
#         artist = item.get("artist") or "Unknown Artist"
#         title = item.get("song") or key.split("_", 1)[-1]
#         lyrics_raw = item.get("lyrics", "")

#         lyrics_clean = clean_lyrics(lyrics_raw)

#         if not lyrics_clean:
#             continue

#         # 整首歌不是英文 → 跳過
#         if not is_english_lyrics(lyrics_clean):
#             continue

#         song_id = "scraped_" + normalize_id(f"{artist}_{title}")

#         data.append({
#             "song_id": song_id,
#             "title": title,
#             "artist": artist,
#             "year": None,
#             "lyrics": lyrics_clean
#         })

#     print(f"[Billboard/Genius] Loaded {len(data)} English songs.")
#     return data


# # ===============================================================
# # Load HuggingFace dataset
# # ===============================================================
# def load_hf_dataset(path):
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     cleaned = []

#     for idx, item in enumerate(data):
#         title = item.get("title") or f"unknown_title_hf_{idx+1}"
#         artist = item.get("artist") or "Unknown Artist"
#         lyrics_raw = item.get("lyrics", "")
#         year = item.get("year")
#         song_id = item.get("song_id") or f"hf_{idx+1:05d}"

#         lyrics_clean = clean_lyrics(lyrics_raw)

#         if not lyrics_clean:
#             continue

#         # 整首不是英文 → 删除
#         if not is_english_lyrics(lyrics_clean):
#             continue

#         cleaned.append({
#             "song_id": song_id,
#             "title": title,
#             "artist": artist,
#             "year": year,
#             "lyrics": lyrics_clean
#         })

#     print(f"[HF Dataset] Loaded {len(cleaned)} English songs.")
#     return cleaned


# # ===============================================================
# # Merge datasets
# # ===============================================================
# def merge_datasets(list1, list2):
#     merged = []
#     seen = set()

#     for item in list1 + list2:
#         artist = safe_lower(item.get("artist"))
#         title = safe_lower(item.get("title"))

#         if not artist or not title:
#             continue

#         key = (artist, title)

#         if key in seen:
#             continue
#         seen.add(key)

#         merged.append(item)

#     print(f"[Merged] Total unique English songs: {len(merged)}")
#     return merged


# # ===============================================================
# # Save output
# # ===============================================================
# def save_output(data, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#     print(f"[Saved] {path}")


# # ===============================================================
# # MAIN
# # ===============================================================
# if __name__ == "__main__":
#     BILLBOARD_PATH = "data/raw/lyrics/latest_lyrics_raw.json"
#     HF_PATH        = "data/raw/lyrics/lyrics_raw.json"
#     OUTPUT_PATH    = "data/raw/lyrics/all_lyrics_raw.json"

#     data_scraped = load_billboard_genius(BILLBOARD_PATH)
#     data_hf      = load_hf_dataset(HF_PATH)

#     merged = merge_datasets(data_scraped, data_hf)

#     save_output(merged, OUTPUT_PATH)
