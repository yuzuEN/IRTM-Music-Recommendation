# IRTM Music Recommendation — Data Preprocessing (Lyrics + Posts)

This folder contains the **end-to-end preprocessing pipeline** for:
- **Lyrics corpus** (retrieval corpus; used for BM25 indexing / graph construction)
- **Social-media posts** (queries; cleaned + hashtag normalization + WordNet-based QE with drift control)

Our core design goal is:  
> **Make posts and lyrics land in the same normalization space** (tokenization + stopwords + stemming) so lexical retrieval (BM25) and downstream graph methods can operate consistently.

---

## 1) Repository Structure (preprocessing-related)

```
src/preprocessing/
  download_lyrics_dataset.py        # Download HuggingFace lyrics dataset → lyrics_raw.json
  crawl_latest_lyrics.py            # Crawl Billboard 2024–2025 + Genius → latest_lyrics_raw.json
  merge_lyrics_corpus.py            # Merge + quality filter → all_lyrics_raw.json
  preprocess_lyrics.py              # Dedup + cleaning + tokenize+stopwords+stem → clean_lyrics/tokens/vocab (+ bracket report)

  preprocess_post.py                # Posts cleaning + hashtag normalization + WordNet QE (+ drift control)
  merge_txt.py                      # merge IG captions txt into a single file for style/reference
  extract_brackets_from_lyrics.py   # bracket diagnostics (a standalone analyzer)
  duplicate_checker.py              # debug duplicate titles/similarity behavior

  make_slides_assets.py             # generate figures (flowchart, histograms) for report/slides
```

<!-- > Note: Older README mentioned `preprocess_lyrics_full.py`, but the current pipeline is in `preprocess_lyrics.py`. -->

---

## 2) Data Layout

### 2.1 Raw inputs

```
data/raw/lyrics/
  lyrics_raw.json                   # from HuggingFace dataset
  latest_lyrics_raw.json            # from Billboard 2024–2025 + Genius crawl
  all_lyrics_raw.json               # merged + filtered master raw corpus

data/raw/posts/
  generated_social_posts_10k.json   # IG-style posts (generated/aligned dataset) → main input to preprocess_post.py
```

### 2.2 Processed outputs

```
data/processed/lyrics/
  clean_lyrics.json                 # cleaned lyrics text + metadata
  lyrics_tokens.csv                 # BM25-ready tokens per song (stemmed)
  vocabulary.json                   # corpus vocabulary (stem → count)
  bracket_contents_top200.tsv       # diagnostics: top bracket contents (optional but recommended)

data/processed/posts/
  posts_clean_expanded.jsonl        # one post per line (JSONL): raw_text + meta + tokens (clean + expanded)
  qe_high_freq_tokens.txt           # corpus-level gate (high-DF source tokens)
  qe_newterm_gate.txt               # corpus-level gate (too-common new terms)
  qe_report_top50.tsv               # new-term statistics
  qe_blocked_report_top50.tsv       # blocked candidates + reasons
  qe_skipped_sources.tsv            # why a source token did not expand
```

---

## 3) Datasets & Resources

### Lyrics sources
- HuggingFace dataset: **mrYou/lyrics-dataset**  
  - Link (for report): https://huggingface.co/datasets/mrYou/lyrics-dataset
- Billboard Hot 100 (2024–2025 weekly lists): used to build “recent popular songs” candidate set
- Genius lyrics: https://genius.com/, used to fetch lyrics for the Billboard-derived list

### Posts sources
- Main input: `generated_social_posts_10k.json` (IG-style posts generated/aligned using existing style/emotion sources + LLM generation)
- External references (not required by pipeline):
  - TweetEval (for emotion/emoji-style reference)
  - IG caption datasets (for style reference)

---

## 4) Quickstart (End-to-end)

### 4.1 Setup

Install dependencies (typical):
```bash
pip install datasets lyricsgenius beautifulsoup4 requests tqdm langdetect nltk contractions
```

Download NLTK resources (recommended):
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 5) Lyrics Pipeline

### Step 1 — Download HuggingFace lyrics dataset
```bash
python src/preprocessing/download_lyrics_dataset.py
```
Output:
- `data/raw/lyrics/lyrics_raw.json`

### Step 2 — Crawl 2024–2025 Billboard + Genius lyrics (optional but recommended)
Set your Genius token first:
```bash
export GENIUS_ACCESS_TOKEN="YOUR_TOKEN"
python src/preprocessing/crawl_latest_lyrics.py
```
Output:
- `data/raw/lyrics/latest_lyrics_raw.json`

### Step 3 — Merge + quality filter
```bash
python src/preprocessing/merge_lyrics_corpus.py
```
Output:
- `data/raw/lyrics/all_lyrics_raw.json`

### Step 4 — Full lyrics preprocessing
```bash
python src/preprocessing/preprocess_lyrics.py
```
Outputs (under `data/processed/lyrics/`):
- `clean_lyrics.json`
- `lyrics_tokens.csv`
- `vocabulary.json`
- `bracket_contents_top200.tsv` (diagnostics)

---

## 6) Posts Pipeline (Cleaning + Hashtag normalization + QE)

Run:
```bash
python src/preprocessing/preprocess_post.py
```

### 6.1 What it produces (important)

Each JSONL line keeps:
- `raw_text` (+ metadata such as emotion/strength)
- **two parallel token views**:
  - `clean_tokens_raw` / `expanded_tokens_raw`  
    - **not stemmed** (more interpretable; good for debugging / analysis)
  - `clean_tokens` / `expanded_tokens`  
    - **BM25-aligned**: stopwords removed + Porter stemming  
    - designed to match the lyrics token space for retrieval alignment

This “raw + aligned” dual view is intentional:  
> **we keep raw tokens for interpretability, but still provide stemmed tokens for retrieval alignment.**

### 6.2 Tokenization & cleaning summary (posts)
- remove URLs / @mentions
- normalize emoji-related artifacts (avoid emoji sticking to words)
- contractions expansion (e.g., can't → cannot)
- hashtag normalization:
  - CamelCase split if possible
  - DP-based word break + WordNet lemma help for all-lowercase hashtags
- stopwords removal + Porter stemming applied **only** to the BM25-aligned token fields

### 6.3 Query Expansion (WordNet) + drift control
- WordNet-based candidate generation
- Lesk-style lightweight sense selection (gloss overlap)
- drift prevention:
  - per-token limits, per-post expansion cap
  - POS constraints + blocklist
  - corpus-level gates:
    - high-frequency source gate (do not expand extremely common source tokens)
    - new-term DF gate (ban expansion terms that become too universal across posts)

Outputs include multiple QE diagnostic reports to audit drift and “over-general expansion”.

---

## 7) FAQ / Common Confusions

### Q1: “Do we stem posts or not?”
We do **both**:
- Keep **raw (non-stemmed)** tokens for interpretability: `*_tokens_raw`
- Provide **stemmed BM25-aligned** tokens for retrieval consistency: `clean_tokens`, `expanded_tokens`

So the correct statement is:
> “Posts are stemmed **only for the retrieval-aligned view**, while raw tokens are retained for analysis/debugging.”

### Q2: Where are the QE reports?
Under `data/processed/posts/`:
- `qe_report_top50.tsv`, `qe_blocked_report_top50.tsv`, `qe_skipped_sources.tsv`, etc.

<!-- ---

## 8) Notes for Report Writing
If you include citations in your paper/report:
- HuggingFace dataset (mrYou/lyrics-dataset)
- WordNet-based QE + query drift prevention literature
- Lesk / extended gloss overlap (for sense selection) -->
