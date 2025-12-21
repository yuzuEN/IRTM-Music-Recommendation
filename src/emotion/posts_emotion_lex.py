import json
import os
from collections import defaultdict
import numpy as np

# ===============================
# Unified 8-class emotion space
# ===============================
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

# ===============================
# NRC Lexicon
# ===============================
def load_nrc_lexicon(path: str):
    """
    NRC format:
        word<TAB>emotion<TAB>0/1
    """
    lexicon = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emotion, value = parts
            if value == "1":
                lexicon[word].add(emotion)
    return lexicon


def map_nrc_to_8(nrc_labels):
    """
    NRC -> unified 8-class mapping
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
        # auxiliary sentiment
        "positive": [("joy", 0.5)],
        "negative": [("sadness", 0.5)],
    }

    mapped = []
    for lab in nrc_labels:
        if lab in mapping:
            mapped.extend(mapping[lab])
    return mapped


# ===============================
# Core: Post Emotion (Lexicon)
# ===============================
def compute_post_emotion(tokens, nrc_lexicon):
    """
    tokens: list[str]  (expanded_tokens)
    return: np.ndarray shape (8,)
    """
    vec = np.zeros(len(EMOTIONS), dtype=np.float32)

    for tok in tokens:
        tok = tok.lower()
        if tok in nrc_lexicon:
            for emo, wgt in map_nrc_to_8(nrc_lexicon[tok]):
                vec[EMOTION2IDX[emo]] += wgt

    s = float(vec.sum())
    if s > 0:
        vec /= s
    else:
        vec[EMOTION2IDX["neutral"]] = 1.0

    return vec


# ===============================
# Main
# ===============================
def main():
    post_path = r"C:\Users\tinti\Desktop\IRTM_music\data\processed\posts\posts_clean_expanded.jsonl"
    nrc_path = "data/raw/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    output_path = "outputs/emotion_vectors/EmotionVec_posts_lex.npy"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    nrc_lexicon = load_nrc_lexicon(nrc_path)

    emotion_vectors = []

    with open(post_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tokens = obj.get("expanded_tokens", [])
            vec = compute_post_emotion(tokens, nrc_lexicon)
            emotion_vectors.append(vec)

    emotion_vectors = np.stack(emotion_vectors, axis=0)
    np.save(output_path, emotion_vectors)

    print(f"[OK] Saved Post Emotion (Lexicon) to {output_path}")
    print(f"Shape: {emotion_vectors.shape}")
    
    nonzero = (emotion_vectors.sum(axis=1) > 0).mean()
    print(f"Non-empty emotion ratio: {nonzero:.3f}")



if __name__ == "__main__":
    main()
