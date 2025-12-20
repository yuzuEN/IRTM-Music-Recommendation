import json
import os
import re
from collections import defaultdict
import numpy as np

# Unified 8-class emotion space (shared across all modules)
EMOTIONS = [
    "joy",
    "anger",
    "fear",
    "sadness",
    "surprise",
    "disgust",
    "excitement",
    "neutral",
]
EMOTION2IDX = {e: i for i, e in enumerate(EMOTIONS)}

def load_nrc_lexicon(path: str):
    """
    Load NRC Word-Emotion Association Lexicon.
    Expected format per line:
        word<TAB>emotion<TAB>0/1

    Returns:
        dict[str, set[str]] : word -> set of NRC labels with value == 1
    """
    lexicon = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            word, emotion, value = parts
            if value == "1":
                lexicon[word].add(emotion)
    return lexicon

def map_nrc_to_8(nrc_labels):
    """
    Map NRC labels (8 emotions + positive/negative) into our 8-class space.

    Mapping (final decision):
      joy          -> joy
      anger        -> anger
      fear         -> fear
      sadness      -> sadness
      surprise     -> surprise
      disgust      -> disgust
      anticipation -> excitement
      trust        -> neutral
      positive     -> joy      (auxiliary, weaker)
      negative     -> sadness  (auxiliary, weaker)

    Output:
      list of (target_emotion, weight)
    """
    mapping = {
        "joy": [("joy", 1.0)],
        "anger": [("anger", 1.0)],
        "fear": [("fear", 1.0)],
        "sadness": [("sadness", 1.0)],
        "surprise": [("surprise", 1.0)],
        "disgust": [("disgust", 1.0)],
        "anticipation": [("excitement", 1.0)],
        "trust": [("neutral", 1.0)],
        # Auxiliary sentiment signals (weaker weights)
        "positive": [("joy", 0.5)],
        "negative": [("sadness", 0.5)],
    }

    mapped = []
    for lab in nrc_labels:
        if lab in mapping:
            mapped.extend(mapping[lab])
    return mapped

def tokenize(text: str):
    """
    Simple tokenizer for NRC lookup:
    - lowercase
    - keep only a-z and whitespace
    - split by whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()

def compute_lyrics_emotion(lyrics: str, nrc_lexicon):
    """
    Compute 8D emotion distribution vector for a lyric string.
    - Count mapped emotions from NRC hits
    - L1 normalize
    - Fallback: if no NRC hits, set neutral = 1
    """
    vec = np.zeros(len(EMOTIONS), dtype=np.float32)

    for w in tokenize(lyrics):
        if w in nrc_lexicon:
            for emo, wgt in map_nrc_to_8(nrc_lexicon[w]):
                vec[EMOTION2IDX[emo]] += wgt

    s = float(vec.sum())
    if s > 0:
        vec /= s
    else:
        vec[EMOTION2IDX["neutral"]] = 1.0  # fallback when no emotion words matched
    return vec

def main():
    lyrics_path = "data/processed/lyrics/clean_lyrics.json"
    nrc_path = "data/raw/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    output_path = "outputs/emotion_vectors/EmotionVec_lyrics.npy"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(lyrics_path, "r", encoding="utf-8") as f:
        songs = json.load(f)

    nrc_lexicon = load_nrc_lexicon(nrc_path)

    emotion_vectors = {}
    for song in songs:
        song_id = song.get("song_id")
        lyrics = song.get("lyrics", "")
        if not song_id:
            continue
        emotion_vectors[song_id] = compute_lyrics_emotion(lyrics, nrc_lexicon)

    np.save(output_path, emotion_vectors, allow_pickle=True)
    print(f"Saved lyrics emotion vectors to {output_path} (n={len(emotion_vectors)})")

if __name__ == "__main__":
    main()
