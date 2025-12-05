"""
Full Lyrics Preprocessing Pipeline (with fuzzy duplicate removal)
Input : all_lyrics_raw.json
Output:
  - clean_lyrics.json
  - lyrics_tokens.csv
  - vocabulary.json
"""

import json
import os
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from difflib import SequenceMatcher


# ================================================================
# NLTK resources
# ================================================================
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
TOKENIZER = TreebankWordTokenizer()


# ================================================================
# Utility
# ================================================================
def safe_lower(s):
    return s.lower().strip() if isinstance(s, str) else ""


def normalize_title(s):
    """Remove punctuation, extra spaces and lowercase."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# ================================================================
# STEP 1 — Fuzzy Duplicate Removal
# ================================================================
def deduplicate_fuzzy(songs, threshold=0.85):
    """
    Remove:
      1. Exact duplicates
      2. Fuzzy duplicates (artist same + title similarity >= threshold)
    """
    result = []
    seen_exact = set()

    for song in songs:
        artist = safe_lower(song["artist"])
        raw_title = song["title"]
        norm_title = normalize_title(raw_title)

        exact_key = (artist, norm_title)

        # exact duplicate
        if exact_key in seen_exact:
            continue

        is_fuzzy_dup = False

        # check fuzzy duplicates
        for kept in result:
            kept_artist = safe_lower(kept["artist"])
            kept_title = normalize_title(kept["title"])

            if kept_artist != artist:
                continue

            sim = similarity(norm_title, kept_title)
            if sim >= threshold:
                is_fuzzy_dup = True
                break

        if not is_fuzzy_dup:
            result.append(song)
            seen_exact.add(exact_key)

    print(f"[Dedup] {len(songs)} → {len(result)} songs after fuzzy + exact removal.")
    return result


# ================================================================
# STEP 2 — Basic text cleaning
# ================================================================
def clean_text_basic(text):
    """Lowercase + remove URLs + remove punctuation + remove numbers."""
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"http[s]?://\S+", "", text)   # URLs
    text = re.sub(r"[^a-z\s]", " ", text)        # punctuation & numbers
    text = re.sub(r"\s+", " ", text)             # multiple spaces
    return text.strip()


# ================================================================
# STEP 3 — Tokenize + Stopwords + Stemming
# ================================================================
def tokenize_and_stem(songs):
    vocab = defaultdict(int)
    token_rows = []

    for song in songs:
        clean = clean_text_basic(song["lyrics"])
        tokens = TOKENIZER.tokenize(clean)

        tokens_clean = []
        for tok in tokens:
            if tok in STOPWORDS:
                continue

            stem = STEMMER.stem(tok)
            tokens_clean.append(stem)
            vocab[stem] += 1

        token_rows.append({
            "song_id": song["song_id"],
            "tokens": tokens_clean
        })

    return token_rows, vocab


# ================================================================
# I/O
# ================================================================
def load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {path}")


def save_tokens_csv(token_rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("song_id,tokens\n")
        for row in token_rows:
            joined = " ".join(row["tokens"])
            f.write(f"{row['song_id']},{joined}\n")
    print(f"[Saved] {path}")


# ================================================================
# MAIN PIPELINE
# ================================================================
if __name__ == "__main__":
    RAW_PATH = "data/raw/lyrics/all_lyrics_raw.json"

    OUT_CLEAN = "data/processed/clean_lyrics.json"
    OUT_TOKENS = "data/processed/lyrics_tokens.csv"
    OUT_VOCAB = "data/processed/vocabulary.json"

    # 1. Load raw songs
    raw_songs = load_raw(RAW_PATH)
    print(f"[Load] Loaded {len(raw_songs)} raw songs.")

    # 2. Fuzzy + Exact Duplicate Removal
    deduped = deduplicate_fuzzy(raw_songs)

    # 3. Tokenization
    token_rows, vocab = tokenize_and_stem(deduped)

    # 4. Save processed outputs
    save_json(deduped, OUT_CLEAN)
    save_tokens_csv(token_rows, OUT_TOKENS)
    save_json(vocab, OUT_VOCAB)

    print("\n=== ALL DONE ===")
