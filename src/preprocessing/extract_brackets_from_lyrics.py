import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")  # non-nested brackets

def load_json_any(path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
    - JSON array: [ {...}, {...} ]
    - JSONL: one json object per line
    """
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    # assume jsonl
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out

def normalize_phrase(s: str) -> str:
    # normalize spacing & casing for counting
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def clean_lyrics_remove_brackets(lyrics: str) -> str:
    """
    Remove bracketed annotations like [Verse 1], [Chorus], [x7], [Producer], etc.
    Keep the rest; also tidy whitespace.
    """
    # remove bracket blocks
    cleaned = BRACKET_RE.sub("", lyrics)

    # remove empty bracket-only lines (after removal they become blank)
    lines = [ln.strip() for ln in cleaned.splitlines()]
    lines = [ln for ln in lines if ln]  # drop empty

    # re-join with newline
    return "\n".join(lines).strip()

def classify_phrase(phrase: str) -> str:
    """
    Simple heuristic classification to help you inspect what's inside [].
    """
    p = phrase.lower().strip()

    # repetition markers: [x7], [x 7], [2x], [repeat x4]
    if re.fullmatch(r"(x\s*\d+|\d+\s*x|repeat\s*x\s*\d+)", p):
        return "repeat_marker"

    # common section headers
    section_keywords = [
        "verse", "chorus", "hook", "bridge", "intro", "outro", "pre-chorus",
        "post-chorus", "refrain", "interlude", "break", "drop", "instrumental",
        "spoken", "skit", "solo"
    ]
    if any(k in p for k in section_keywords):
        return "section_marker"

    # credits / metadata
    meta_keywords = ["produced by", "prod.", "written by", "translated by", "genius", "remix"]
    if any(k in p for k in meta_keywords):
        return "metadata"

    # likely speaker label: just a name, or "Name:" / "Name -"
    if re.fullmatch(r"[a-z0-9 .,&'’-]{2,40}", p) and (" " not in p or p.count(" ") <= 3):
        return "speaker_or_tag"

    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to cleaned_lyrics.json or jsonl")
    ap.add_argument("--outdir", default="lyrics_bracket_reports", help="Output directory")
    ap.add_argument("--write_cleaned", action="store_true", help="Also write cleaned lyrics with [] removed")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    songs = load_json_any(in_path)

    phrase_counter = Counter()
    phrase_class_counter = Counter()
    per_song_rows: List[Tuple[str, str, str, str, str]] = []  # song_id,title,artist,class,phrase
    phrase_examples = defaultdict(list)  # phrase -> list of song_id

    cleaned_songs = []

    for s in songs:
        song_id = str(s.get("song_id", ""))
        title = str(s.get("title", ""))
        artist = str(s.get("artist", ""))
        lyrics = str(s.get("lyrics", ""))

        phrases = BRACKET_RE.findall(lyrics)
        for ph in phrases:
            ph_norm = normalize_phrase(ph)
            cls = classify_phrase(ph_norm)
            phrase_counter[ph_norm] += 1
            phrase_class_counter[cls] += 1
            per_song_rows.append((song_id, title, artist, cls, ph_norm))
            if len(phrase_examples[ph_norm]) < 5:
                phrase_examples[ph_norm].append(song_id)

        if args.write_cleaned:
            cleaned = clean_lyrics_remove_brackets(lyrics)
            s2 = dict(s)
            s2["lyrics"] = cleaned
            cleaned_songs.append(s2)

    # (1) per-song list
    per_song_path = outdir / "brackets_per_song.tsv"
    with per_song_path.open("w", encoding="utf-8") as f:
        f.write("song_id\ttitle\tartist\tclass\tbracket_text\n")
        for row in per_song_rows:
            f.write("\t".join(x.replace("\t", " ").replace("\n", " ") for x in row) + "\n")

    # (2) global counts
    counts_path = outdir / "brackets_counts.tsv"
    with counts_path.open("w", encoding="utf-8") as f:
        f.write("bracket_text\tcount\tclass\texample_song_ids\n")
        for ph, c in phrase_counter.most_common():
            cls = classify_phrase(ph)
            ex = ",".join(phrase_examples[ph])
            f.write(f"{ph}\t{c}\t{cls}\t{ex}\n")

    # (3) class summary
    class_path = outdir / "brackets_class_summary.tsv"
    with class_path.open("w", encoding="utf-8") as f:
        f.write("class\tcount\n")
        for cls, c in phrase_class_counter.most_common():
            f.write(f"{cls}\t{c}\n")

    # (4) optional cleaned lyrics output
    if args.write_cleaned:
        cleaned_path = outdir / "cleaned_lyrics_no_brackets.json"
        cleaned_path.write_text(json.dumps(cleaned_songs, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"- Per-song brackets: {per_song_path}")
    print(f"- Bracket counts:   {counts_path}")
    print(f"- Class summary:    {class_path}")
    if args.write_cleaned:
        print(f"- Cleaned lyrics:   {outdir / 'cleaned_lyrics_no_brackets.json'}")

if __name__ == "__main__":
    main()
