"""
Full Lyrics Preprocessing Pipeline (v3) — cleaner & more robust

Fixes requested:
A) Stronger "section marker" removal (not only inside [ ... ]) to avoid vocab noise like chorus/verse.
B) Contractions normalization/expansion BEFORE punctuation stripping to prevent fragments like "t" from can't.
C) Drop very short tokens (len<=1) after tokenization/stemming.

Pipeline:
1) Load raw songs (JSON array) from RAW_PATH (default: data/raw/lyrics/all_lyrics_raw.json)
2) Deduplicate (exact + fuzzy title similarity within artist)
3) Clean lyrics:
   - remove [ ... ] blocks (optional)
   - remove lines that are section markers even without brackets (e.g., "Chorus:", "Verse 2", "Bridge - Artist")
   - normalize apostrophes (’ -> ')
   - lowercase
   - expand common English contractions
   - remove URLs
   - keep only [a-z] + whitespace
   - collapse whitespace
4) Tokenize + stopwords removal + stemming
   - drop tokens length <= 1 (pre and post stemming)
Outputs (default under data/processed/lyrics/):
  - clean_lyrics.json            (deduped + cleaned lyrics text)
  - lyrics_tokens.csv            (song_id,tokens)
  - vocabulary.json              (stem -> count)
  - bracket_contents_top200.tsv  (optional diagnostic)
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer


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
# Regex & constants
# ================================================================
BRACKET_BLOCK_RE = re.compile(r"\[[^\[\]]+\]")           # non-nested [ ... ]
URL_RE = re.compile(r"http[s]?://\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")                   # keep a-z + whitespace only
MULTISPACE_RE = re.compile(r"\s+")
APOSTROPHE_RE = re.compile(r"[’`´]")                     # normalize to '

# Section marker line detection (after stripping)
# Matches:
#   Chorus, Chorus:, Chorus x2, Chorus [x2], Chorus - Artist
#   Verse 1, Verse 2:, Pre-Chorus, Post-Chorus, Bridge, Intro, Outro, Hook, Refrain, Interlude, Break, Skit, Spoken
SECTION_WORDS = (
    "chorus", "verse", "bridge", "intro", "outro", "pre-chorus", "pre chorus",
    "post-chorus", "post chorus", "hook", "refrain", "interlude", "break",
    "instrumental", "skit", "spoken", "drop", "solo"
)
SECTION_LINE_RE = re.compile(
    r"^\s*(?:"
    + "|".join(re.escape(w) for w in SECTION_WORDS)
    + r")"
    r"(?:\s*(?:\d+|[ivx]+))?"          # optional verse number (1,2,III)
    r"(?:\s*(?:x\s*\d+|\d+\s*x))?"     # optional x7 / 7x
    r"(?:\s*[:\-–—]\s*.*)?\s*$",       # optional ": Artist" or "- Artist"
    flags=re.IGNORECASE,
)

# Also remove standalone repetition markers lines like "[x7]" already handled by brackets,
# but sometimes appears as "x7" or "x 7" alone
REPEAT_LINE_RE = re.compile(r"^\s*(?:x\s*\d+|\d+\s*x|repeat\s*x?\s*\d+)\s*$", re.IGNORECASE)


# ================================================================
# Contractions expansion (lightweight, no external dependency)
# ================================================================
def expand_contractions(text: str) -> str:
    """
    Expand common English contractions in lowercase text.
    This is intentionally lightweight (covers the common ones that cause 't' fragments).
    """
    if not text:
        return ""

    # Special cases first
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "cannot", text)
    text = re.sub(r"\bain't\b", "is not", text)

    # General patterns
    text = re.sub(r"n\'t\b", " not", text)     # didn't -> did not
    text = re.sub(r"\'re\b", " are", text)     # you're -> you are
    text = re.sub(r"\'m\b", " am", text)       # i'm -> i am
    text = re.sub(r"\'ll\b", " will", text)    # i'll -> i will
    text = re.sub(r"\'ve\b", " have", text)    # i've -> i have
    text = re.sub(r"\'d\b", " would", text)    # i'd -> i would (sometimes had; acceptable)

    # it's -> it is (can be possessive; acceptable for retrieval)
    text = re.sub(r"\bit\'s\b", "it is", text)

    return text


# ================================================================
# Helpers
# ================================================================
def normalize_title(s: str) -> str:
    """Normalize title for de-dup: lowercase, keep a-z0-9 and spaces, collapse spaces."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ================================================================
# STEP 1 — Dedup
# ================================================================
def deduplicate_fuzzy(songs: List[Dict[str, Any]], threshold: float = 0.85) -> List[Dict[str, Any]]:
    """
    Remove duplicates in two stages:
    1) Exact duplicate by (artist, normalize(title))
    2) Fuzzy duplicate by title similarity within same artist (SequenceMatcher ratio)
    """
    by_artist = defaultdict(list)
    for s in songs:
        artist = (s.get("artist") or "").strip().lower()
        by_artist[artist].append(s)

    result = []
    seen_exact = set()

    for artist, items in by_artist.items():
        kept_titles = []
        for song in items:
            title = song.get("title") or ""
            key = (artist, normalize_title(title))

            if key in seen_exact:
                continue

            norm_title = normalize_title(title)
            is_fuzzy_dup = any(similarity(norm_title, t) >= threshold for t in kept_titles)
            if is_fuzzy_dup:
                continue

            result.append(song)
            seen_exact.add(key)
            kept_titles.append(norm_title)

    print(f"[Dedup] {len(songs)} → {len(result)} songs after fuzzy + exact removal.")
    return result


# ================================================================
# STEP 2 — Bracket removal + section-marker removal + cleaning
# ================================================================
def remove_square_brackets(text: str) -> str:
    """Remove bracketed blocks like [Verse 1], [Chorus], [x7], [Artist]."""
    if not text:
        return ""
    return BRACKET_BLOCK_RE.sub(" ", text)


def remove_section_marker_lines(text: str) -> str:
    """
    Remove lines that look like section markers (even without brackets),
    e.g., 'Chorus', 'Verse 1:', 'Bridge - Artist', 'Pre-Chorus', 'x7'.
    This prevents chorus/verse tokens from polluting vocabulary.
    """
    if not text:
        return ""
    out_lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if SECTION_LINE_RE.match(s):
            continue
        if REPEAT_LINE_RE.match(s):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines)


def clean_text_basic(text: str, *, remove_brackets: bool = True, remove_section_lines: bool = True) -> str:
    """
    Cleaning steps:
    - optional: remove [ ... ] blocks entirely
    - optional: remove section marker lines even without brackets
    - normalize apostrophes
    - lowercase
    - expand contractions
    - remove URLs
    - keep only [a-z] and whitespace
    - collapse multiple spaces
    """
    if not text:
        return ""

    if remove_brackets:
        text = remove_square_brackets(text)

    if remove_section_lines:
        text = remove_section_marker_lines(text)

    # normalize apostrophes before lower/expand
    text = APOSTROPHE_RE.sub("'", text)

    text = text.lower()
    text = expand_contractions(text)
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def apply_cleaning_to_songs(
    songs: List[Dict[str, Any]],
    *,
    remove_brackets: bool = True,
    remove_section_lines: bool = True
) -> Tuple[List[Dict[str, Any]], Counter]:
    """
    Apply cleaning to each song's lyrics field (returns new list).
    Also returns a counter of bracket contents (diagnostic).
    """
    bracket_content_re = re.compile(r"\[([^\[\]]+)\]")
    bracket_counter = Counter()

    cleaned_songs = []
    for s in songs:
        lyrics = s.get("lyrics") or ""
        for m in bracket_content_re.findall(lyrics):
            bracket_counter[m.strip()] += 1

        cleaned = clean_text_basic(
            lyrics,
            remove_brackets=remove_brackets,
            remove_section_lines=remove_section_lines
        )

        s2 = dict(s)
        s2["lyrics"] = cleaned
        cleaned_songs.append(s2)

    return cleaned_songs, bracket_counter


# ================================================================
# STEP 3 — Tokenize + stopwords + stemming (+ short token filter)
# ================================================================
def tokenize_and_stem(songs: List[Dict[str, Any]]):
    """
    Tokenize cleaned lyrics.
    Input expects song["lyrics"] contains lowercase a-z + spaces only.
    """
    token_rows = []
    vocab = Counter()

    for song in songs:
        clean = song.get("lyrics") or ""
        song_id = song.get("song_id", "")

        if not clean:
            token_rows.append({"song_id": song_id, "tokens": []})
            continue

        tokens = TOKENIZER.tokenize(clean)
        tokens_clean = []

        for tok in tokens:
            if not tok:
                continue
            if len(tok) <= 1:          # (C) drop 1-char tokens early
                continue
            if tok in STOPWORDS:
                continue

            stem = STEMMER.stem(tok)
            if not stem or len(stem) <= 1:  # (C) drop 1-char stems too
                continue

            tokens_clean.append(stem)
            vocab[stem] += 1

        token_rows.append({"song_id": song_id, "tokens": tokens_clean})

    return token_rows, dict(vocab)


# ================================================================
# I/O
# ================================================================
def load_raw(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {path}")


def save_tokens_csv(token_rows, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["song_id", "tokens"])
        for row in token_rows:
            joined = " ".join(row["tokens"])
            writer.writerow([row.get("song_id", ""), joined])
    print(f"[Saved] {path}")


def save_bracket_report(bracket_counter: Counter, out_path: str, top_k: int = 200):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("bracket_text\tcount\n")
        for txt, c in bracket_counter.most_common(top_k):
            f.write(f"{txt}\t{c}\n")
    print(f"[Saved] {out_path}")


# ================================================================
# MAIN
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/lyrics/all_lyrics_raw.json", help="Raw lyrics JSON path")
    ap.add_argument("--outdir", default="data/processed/lyrics", help="Output directory")
    ap.add_argument("--dedup_threshold", type=float, default=0.85, help="Fuzzy title similarity threshold")

    ap.add_argument("--keep_brackets", action="store_true",
                    help="Do NOT remove [ ... ] blocks (not recommended)")
    ap.add_argument("--keep_section_lines", action="store_true",
                    help="Do NOT remove section marker lines like 'Chorus', 'Verse 1' (not recommended)")

    ap.add_argument("--write_bracket_report", action="store_true",
                    help="Write a bracket contents frequency report (top 200)")

    args = ap.parse_args()

    raw_path = args.raw
    outdir = args.outdir.rstrip("/\\")
    remove_brackets = not args.keep_brackets
    remove_section_lines = not args.keep_section_lines

    out_clean = os.path.join(outdir, "clean_lyrics.json")
    out_tokens = os.path.join(outdir, "lyrics_tokens.csv")
    out_vocab = os.path.join(outdir, "vocabulary.json")
    out_bracket_report = os.path.join(outdir, "bracket_contents_top200.tsv")

    # 1) Load
    raw_songs = load_raw(raw_path)
    print(f"[Load] Loaded {len(raw_songs)} raw songs from {raw_path}")

    # 2) Dedup
    deduped = deduplicate_fuzzy(raw_songs, threshold=args.dedup_threshold)

    # 3) Clean
    cleaned_songs, bracket_counter = apply_cleaning_to_songs(
        deduped,
        remove_brackets=remove_brackets,
        remove_section_lines=remove_section_lines,
    )
    print(f"[Clean] remove_brackets={remove_brackets}, remove_section_lines={remove_section_lines}")

    # 4) Tokenization
    token_rows, vocab = tokenize_and_stem(cleaned_songs)

    # 5) Save
    save_json(cleaned_songs, out_clean)
    save_tokens_csv(token_rows, out_tokens)
    save_json(vocab, out_vocab)

    if args.write_bracket_report:
        save_bracket_report(bracket_counter, out_bracket_report, top_k=200)

    print("\n=== ALL DONE (lyrics pipeline v3) ===")


if __name__ == "__main__":
    main()
