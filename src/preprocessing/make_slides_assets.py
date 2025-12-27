# -*- coding: utf-8 -*-
"""
Make slide assets for preprocessing (Lyrics + Posts).

Outputs clean, slide-friendly PNGs:
- stats_summary.png
- lyrics_example.png
- posts_examples.png
- emotion_distribution.png
- expansion_ratio_hist.png
- token_length_hist.png
- top_tokens.png (optional, with censor)

Compatible with Python 3.8+ (no PEP604 | None, no dict[str,...] syntax).
Run from repo root:
  python src\preprocessing\make_slide_assets.py
"""

import argparse
import json
import math
import random
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# Global font settings (>= 11pt)
# -----------------------------
MIN_FONTSIZE = 40
plt.rcParams.update({
    "font.size": MIN_FONTSIZE,
    "axes.labelsize": MIN_FONTSIZE,
    "axes.titlesize": MIN_FONTSIZE,   # 你有手動 fontsize 的 title 不受影響
    "xtick.labelsize": MIN_FONTSIZE,
    "ytick.labelsize": MIN_FONTSIZE,
    "legend.fontsize": MIN_FONTSIZE,
})


# -----------------------------
# Defaults (run from repo root)
# -----------------------------
DEFAULT_LYRICS_JSON = Path("data/processed/lyrics/clean_lyrics.json")
DEFAULT_LYRICS_TOKENS_CSV = Path("data/processed/lyrics/lyrics_tokens.csv")
DEFAULT_VOCAB_JSON = Path("data/processed/lyrics/vocabulary.json")
DEFAULT_POSTS_JSONL = Path("data/processed/posts/posts_clean_expanded.jsonl")
DEFAULT_OUT_DIR = Path("docs/slide_assets")


# A small "presentation safe" censor list (you can expand it)
CENSOR_TOKENS = {
    "fuck", "fucking", "shit", "bitch", "nigga", "nigger", "cunt", "dick", "pussy",
}

# Some extra "domain stopwords" that often dominate lyrics after stemming
DOMAIN_STOPWORDS = {
    "choru", "vers", "intro", "outro", "yeah", "oh", "uh", "na", "la", "doo", "da",
    "repeat", "refrain",
}


def die(msg: str) -> None:
    raise SystemExit(msg)


def ensure_exists(p: Path, label: str) -> None:
    if not p.exists():
        # help user debug Windows relative path problems
        cwd = Path.cwd()
        die(
            f"[FileNotFound] {label} not found:\n"
            f"  given:    {p}\n"
            f"  resolved: {p.resolve()}\n\n"
            f"Current working dir:\n  {cwd}\n\n"
            f"Tip:\n"
            f"  - Please run this script from repo root (where 'data/' exists)\n"
            f"  - Or pass --lyrics_json / --posts_jsonl with correct paths\n"
        )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and (i + 1) >= limit:
                break
    return rows


def wrap_block(text: str, width: int, max_lines: int) -> str:
    """
    Wrap text to a fixed width and truncate to max_lines with ellipsis.
    Keeps newlines if present, but wraps each paragraph.
    """
    parts = text.splitlines() if "\n" in text else [text]
    wrapped_lines: List[str] = []
    for part in parts:
        if not part.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(part, width=width, break_long_words=False, replace_whitespace=False))
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines]
        if wrapped_lines:
            wrapped_lines[-1] = wrapped_lines[-1] + "  ..."
    return "\n".join(wrapped_lines)


def join_tokens(tokens: Any) -> str:
    if isinstance(tokens, list):
        return " ".join(str(t) for t in tokens)
    if isinstance(tokens, str):
        return tokens
    return str(tokens)


def load_lyrics_tokens(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "song_id" not in df.columns or "tokens" not in df.columns:
        die(f"[Bad CSV] Expect columns: song_id,tokens in {csv_path}")
    df["token_count"] = df["tokens"].fillna("").astype(str).apply(lambda s: len(s.split()))
    return df


def vocab_top_tokens(
    vocab: Dict[str, int],
    top_k: int,
    remove_tokens: Iterable[str],
    censor: bool
) -> Tuple[List[str], List[int]]:
    remove = set(remove_tokens)
    items = [(t, c) for t, c in vocab.items() if t not in remove]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:top_k]

    toks: List[str] = []
    cnts: List[int] = []
    for t, c in items:
        if censor and t.lower() in CENSOR_TOKENS:
            toks.append("***")
        else:
            toks.append(t)
        cnts.append(int(c))
    return toks, cnts


def save_stats_summary(
    out_path: Path,
    n_songs: int,
    n_posts: int,
    vocab_size: int,
    avg_song_tokens: float,
    avg_post_clean: float,
    avg_expand_ratio: float,
) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=300)
    plt.axis("off")

    lines = [
        "Dataset & Preprocessing Stats",
        "",
        f"#Songs: {n_songs}",
        f"#Posts: {n_posts}",
        f"Vocab size: {vocab_size}",
        f"Avg tokens / song: {avg_song_tokens:.1f}",
        f"Avg clean tokens / post: {avg_post_clean:.1f}",
        f"Avg expansion ratio: {avg_expand_ratio:.2f} (expanded/clean)",
    ]
    text = "\n".join(lines)

    fig.text(
        0.06, 0.88, text,
        fontsize=28,
        family="DejaVu Sans Mono",
        va="top"
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_lyrics_example(
    out_path: Path,
    lyrics_rows: List[Dict[str, Any]],
    lyrics_tokens_df: pd.DataFrame,
    seed: int = 59
) -> None:
    random.seed(seed)

    # choose a song that exists in tokens csv
    token_map = dict(zip(lyrics_tokens_df["song_id"].astype(str), lyrics_tokens_df["tokens"].astype(str)))
    candidates = [r for r in lyrics_rows if str(r.get("song_id", "")) in token_map]
    if not candidates:
        die("[Lyrics Example] No overlapping song_id between clean_lyrics.json and lyrics_tokens.csv")

    r = random.choice(candidates)
    song_id = str(r.get("song_id", ""))
    title = str(r.get("title", ""))
    artist = str(r.get("artist", ""))

    raw = str(r.get("lyrics", ""))
    raw_excerpt = "\n".join(raw.splitlines()[:16])  # take first N lines
    tokens = token_map.get(song_id, "")
    token_excerpt = " ".join(tokens.split()[:170])  # take first N tokens

    raw_box = wrap_block(raw_excerpt, width=52, max_lines=18)
    token_box = wrap_block(token_excerpt, width=62, max_lines=18)

    fig = plt.figure(figsize=(16, 9), dpi=200)
    plt.axis("off")

    fig.text(0.05, 0.93, "Lyrics Preprocessing Example (Raw → Tokens)", fontsize=30, weight="bold")
    fig.text(0.05, 0.88, f"song_id: {song_id}", fontsize=30)
    fig.text(0.05, 0.85, f"title: {title} — {artist}", fontsize=30)

    # Left (raw)
    fig.text(0.05, 0.80, "Raw lyrics (excerpt)", fontsize=30, weight="bold")
    fig.text(
        0.05, 0.76, raw_box,
        fontsize=14,
        family="DejaVu Sans Mono",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.2)
    )

    # Right (tokens)
    fig.text(0.55, 0.80, "Tokens (normalized + stopwords + stemming)", fontsize=30, weight="bold")
    fig.text(
        0.55, 0.76, token_box,
        fontsize=14,
        family="DejaVu Sans Mono",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.2)
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_posts_examples(
    out_path: Path,
    posts_rows: List[Dict[str, Any]],
    seed: int = 46,
    n_examples: int = 2
) -> None:
    random.seed(seed)
    if not posts_rows:
        die("[Posts Example] posts_clean_expanded.jsonl is empty?")

    # try to pick diverse emotions
    by_emotion: Dict[str, List[Dict[str, Any]]] = {}
    for r in posts_rows:
        emo = str(r.get("emotion", "unknown"))
        by_emotion.setdefault(emo, []).append(r)

    emotions = list(by_emotion.keys())
    random.shuffle(emotions)

    picked: List[Dict[str, Any]] = []
    for emo in emotions:
        if by_emotion[emo]:
            picked.append(random.choice(by_emotion[emo]))
        if len(picked) >= n_examples:
            break
    if len(picked) < n_examples:
        picked = random.sample(posts_rows, k=min(n_examples, len(posts_rows)))

    fig = plt.figure(figsize=(16, 9), dpi=300)
    plt.axis("off")
    fig.text(0.05, 0.93, "Posts Preprocessing Example (Raw → clean_tokens → expanded_tokens)", fontsize=30, weight="bold")

    y = 0.86
    for i, r in enumerate(picked, start=1):
        raw = str(r.get("raw_text", ""))
        clean = join_tokens(r.get("clean_tokens", []))
        expanded = join_tokens(r.get("expanded_tokens", []))

        emo = str(r.get("emotion", ""))
        strength = str(r.get("strength", ""))

        n_clean = len(clean.split())
        n_exp = len(expanded.split())
        ratio = (n_exp / n_clean) if n_clean > 0 else float("nan")

        # truncate expanded tokens for readability
        expanded_trunc = " ".join(expanded.split()[:120])

        raw_w = wrap_block(raw, width=115, max_lines=3)
        clean_w = wrap_block(clean, width=115, max_lines=3)
        exp_w = wrap_block(expanded_trunc, width=115, max_lines=5)

        header = f"Example {i} | emotion={emo}  strength={strength} | n_clean={n_clean}  n_expanded={n_exp}  ratio={ratio:.2f}"
        fig.text(0.05, y, header, fontsize=16, weight="bold")
        y -= 0.04

        fig.text(0.05, y, "RAW:", fontsize=14, weight="bold")
        fig.text(0.12, y, raw_w, fontsize=13, family="DejaVu Sans Mono", va="top")
        y -= 0.07

        fig.text(0.05, y, "CLEAN:", fontsize=14, weight="bold")
        fig.text(0.12, y, clean_w, fontsize=13, family="DejaVu Sans Mono", va="top")
        y -= 0.07

        fig.text(0.05, y, "EXPANDED:", fontsize=14, weight="bold")
        fig.text(0.15, y, exp_w, fontsize=13, family="DejaVu Sans Mono", va="top")
        y -= 0.12

        if i < n_examples:
            fig.text(0.05, y + 0.02, "—" * 90, fontsize=12)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_emotion_distribution(out_path: Path, posts_df: pd.DataFrame) -> None:
    if "emotion" not in posts_df.columns:
        die("[Posts DF] missing emotion column")

    counts = posts_df["emotion"].astype(str).value_counts().sort_index()
    fig = plt.figure(figsize=(16, 9), dpi=350)
    plt.bar(counts.index.tolist(), counts.values.tolist())
    plt.title("Emotion Label Distribution (Posts)", fontsize=30)
    plt.xlabel("Emotion", fontsize=30)
    plt.ylabel("Count", fontsize=30)
    plt.xticks(rotation=60, fontsize=30)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_expansion_ratio_hist(out_path: Path, posts_df: pd.DataFrame) -> None:
    ratios = posts_df["expansion_ratio"].dropna().values
    fig = plt.figure(figsize=(16, 9), dpi=350)
    plt.hist(ratios, bins=20)
    plt.title("Expansion Ratio Distribution (expanded_tokens / clean_tokens)", fontsize=30)
    plt.xlabel("Ratio", fontsize=30)
    plt.ylabel("Num posts", fontsize=30)

    if len(ratios) > 0:
        mean_r = float(np.mean(ratios))
        p50 = float(np.percentile(ratios, 50))
        p90 = float(np.percentile(ratios, 90))
        plt.axvline(mean_r, linestyle="--")
        plt.text(mean_r, plt.ylim()[1] * 0.9, f"mean={mean_r:.2f}", fontsize=30)
        plt.text(p50, plt.ylim()[1] * 0.8, f"p50={p50:.2f}", fontsize=30)
        plt.text(p90, plt.ylim()[1] * 0.7, f"p90={p90:.2f}", fontsize=30)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_token_length_hist_logx(
    out_path: Path,
    song_lengths: np.ndarray,
    post_lengths: np.ndarray
) -> None:
    """
    Fix the "posts disappear" issue by using log x-axis and log-spaced bins.
    """
    song_lengths = song_lengths[song_lengths > 0]
    post_lengths = post_lengths[post_lengths > 0]
    max_len = int(max(song_lengths.max(initial=1), post_lengths.max(initial=1)))

    # log-spaced bins (1..max_len)
    # ensure at least 30 bins for smoothness
    n_bins = 35
    bins = np.unique(np.logspace(0, math.log10(max_len + 1), n_bins).astype(int))
    bins = np.clip(bins, 1, max_len + 1)

    fig = plt.figure(figsize=(16, 9), dpi=300)
    plt.hist(song_lengths, bins=bins, alpha=0.7, label="song tokens")
    plt.hist(post_lengths, bins=bins, alpha=0.7, label="post clean tokens")

    plt.xscale("log")
    plt.title("Token Length Distribution (log x-axis)", fontsize=30)
    plt.xlabel("Token count (log scale)", fontsize=30)
    plt.ylabel("Frequency", fontsize=30)
    plt.legend(fontsize=30)

    # annotate medians
    sm = float(np.median(song_lengths)) if len(song_lengths) else float("nan")
    pm = float(np.median(post_lengths)) if len(post_lengths) else float("nan")
    # plt.text(0.02, 0.95, f"median(song)={sm:.1f}\nmedian(post)={pm:.1f}", transform=plt.gca().transAxes, fontsize=30, va="top")

    ax = plt.gca()
    # median 文字移到左下，並加背景框避免壓到柱子
    ax.text(
        0.98, 0.95,  # 右上角（留一點邊界）
        f"median(song)={sm:.1f}\nmedian(post)={pm:.1f}",
        transform=ax.transAxes,
        fontsize=30,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=6),
    )

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_top_tokens(
    out_path: Path,
    vocab: Dict[str, int],
    top_k: int = 20,
    censor: bool = True
) -> None:
    # remove very common / not informative tokens
    remove = set(DOMAIN_STOPWORDS)  # you can also merge nltk stopwords if you want
    toks, cnts = vocab_top_tokens(vocab, top_k=top_k, remove_tokens=remove, censor=censor)

    fig = plt.figure(figsize=(16, 9), dpi=300)
    y = np.arange(len(toks))[::-1]
    plt.barh(y, cnts[::-1])
    plt.yticks(y, toks[::-1], fontsize=12)
    plt.title("Top Tokens (Filtered)", fontsize=22)
    plt.xlabel("Count", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lyrics_json", type=str, default=str(DEFAULT_LYRICS_JSON))
    ap.add_argument("--lyrics_tokens_csv", type=str, default=str(DEFAULT_LYRICS_TOKENS_CSV))
    ap.add_argument("--vocab_json", type=str, default=str(DEFAULT_VOCAB_JSON))
    ap.add_argument("--posts_jsonl", type=str, default=str(DEFAULT_POSTS_JSONL))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))

    ap.add_argument("--posts_limit", type=int, default=None, help="limit number of posts loaded (debug)")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--top_k_tokens", type=int, default=20)
    ap.add_argument("--save_top_tokens", action="store_true", help="also export top_tokens.png")
    ap.add_argument("--no_censor", action="store_true", help="do not censor sensitive tokens in top tokens")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    lyrics_json = Path(args.lyrics_json)
    lyrics_tokens_csv = Path(args.lyrics_tokens_csv)
    vocab_json = Path(args.vocab_json)
    posts_jsonl = Path(args.posts_jsonl)
    out_dir = Path(args.out_dir)

    ensure_exists(lyrics_json, "lyrics_json")
    ensure_exists(lyrics_tokens_csv, "lyrics_tokens_csv")
    ensure_exists(vocab_json, "vocab_json")
    ensure_exists(posts_jsonl, "posts_jsonl")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    lyrics_rows = read_json(lyrics_json)
    if not isinstance(lyrics_rows, list):
        die("[Lyrics JSON] expected a list of objects in clean_lyrics.json")

    lyrics_tokens_df = load_lyrics_tokens(lyrics_tokens_csv)

    vocab = read_json(vocab_json)
    if not isinstance(vocab, dict):
        die("[Vocab JSON] expected {token: count} dict")

    posts_rows = read_jsonl(posts_jsonl, limit=args.posts_limit)
    posts_df = pd.DataFrame(posts_rows)

    # Defensive columns
    if "clean_tokens" in posts_df.columns:
        posts_df["n_clean"] = posts_df["clean_tokens"].apply(lambda x: len(x) if isinstance(x, list) else len(str(x).split()))
    else:
        die("[Posts JSONL] missing clean_tokens")

    if "expanded_tokens" in posts_df.columns:
        posts_df["n_expanded"] = posts_df["expanded_tokens"].apply(lambda x: len(x) if isinstance(x, list) else len(str(x).split()))
    else:
        die("[Posts JSONL] missing expanded_tokens")

    posts_df["expansion_ratio"] = posts_df.apply(
        lambda r: (float(r["n_expanded"]) / float(r["n_clean"])) if float(r["n_clean"]) > 0 else np.nan,
        axis=1
    )

    # Stats
    n_songs = int(len(lyrics_tokens_df))
    n_posts = int(len(posts_df))
    vocab_size = int(len(vocab))
    avg_song_tokens = float(lyrics_tokens_df["token_count"].mean())
    avg_post_clean = float(posts_df["n_clean"].mean())
    avg_expand_ratio = float(posts_df["expansion_ratio"].mean())

    # Export images
    save_stats_summary(
        out_dir / "stats_summary.png",
        n_songs=n_songs,
        n_posts=n_posts,
        vocab_size=vocab_size,
        avg_song_tokens=avg_song_tokens,
        avg_post_clean=avg_post_clean,
        avg_expand_ratio=avg_expand_ratio,
    )

    save_lyrics_example(out_dir / "lyrics_example.png", lyrics_rows, lyrics_tokens_df, seed=args.seed)
    save_posts_examples(out_dir / "posts_examples.png", posts_rows, seed=args.seed, n_examples=2)

    save_emotion_distribution(out_dir / "emotion_distribution.png", posts_df)
    save_expansion_ratio_hist(out_dir / "expansion_ratio_hist.png", posts_df)

    song_lens = lyrics_tokens_df["token_count"].to_numpy(dtype=int)
    post_lens = posts_df["n_clean"].to_numpy(dtype=int)
    save_token_length_hist_logx(out_dir / "token_length_hist.png", song_lens, post_lens)

    if args.save_top_tokens:
        save_top_tokens(
            out_dir / "top_tokens.png",
            vocab=vocab,
            top_k=args.top_k_tokens,
            censor=(not args.no_censor)
        )

    print("✅ Slide assets saved to:", out_dir.resolve())
    print("  - stats_summary.png")
    print("  - lyrics_example.png")
    print("  - posts_examples.png")
    print("  - emotion_distribution.png")
    print("  - expansion_ratio_hist.png")
    print("  - token_length_hist.png")
    if args.save_top_tokens:
        print("  - top_tokens.png")


if __name__ == "__main__":
    main()
