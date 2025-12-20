import json
import os
import re
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
# NRC Lexicon (same mapping as yours)
# ===============================
def load_nrc_lexicon(path: str):
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
    mapping = {
        "joy": [("joy", 1.0)],
        "anger": [("anger", 1.0)],
        "fear": [("fear", 1.0)],
        "sadness": [("sadness", 1.0)],
        "surprise": [("surprise", 1.0)],
        "disgust": [("disgust", 1.0)],
        "anticipation": [("excitement", 1.0)],
        "trust": [("neutral", 1.0)],
        "positive": [("joy", 0.5)],
        "negative": [("sadness", 0.5)],
    }
    mapped = []
    for lab in nrc_labels:
        if lab in mapping:
            mapped.extend(mapping[lab])
    return mapped

WORD_RE = re.compile(r"[a-z]+")

def phrase_to_words(phrase: str):
    return WORD_RE.findall((phrase or "").lower())

# ===============================
# Load emoji phrase map: phrase<TAB>emoji
# ===============================
def load_emoji_joined(path: str):
    """
    Format per line:
      phrase<TAB>emoji
    Returns:
      dict[emoji] -> list[phrase]
    """
    emo2phrases = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            phrase, emoji = parts[0].strip(), parts[1].strip()
            if phrase and emoji:
                emo2phrases[emoji].append(phrase)
    return emo2phrases

# ===============================
# Build emoji -> 8D emotion vector (via NRC on phrases)
# ===============================
def build_emoji_emotion_table(emo2phrases, nrc_lexicon):
    emoji_table = {}
    for emoji, phrases in emo2phrases.items():
        vec = np.zeros(len(EMOTIONS), dtype=np.float32)

        for ph in phrases:
            for w in phrase_to_words(ph):
                if w in nrc_lexicon:
                    for emo, wgt in map_nrc_to_8(nrc_lexicon[w]):
                        vec[EMOTION2IDX[emo]] += wgt

        s = float(vec.sum())
        if s > 0:
            vec /= s
        else:
            vec[EMOTION2IDX["neutral"]] = 1.0

        emoji_table[emoji] = vec

    return emoji_table

# ===============================
# Extract emojis from text (covers most emoji blocks)
# ===============================
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)

def extract_emojis(text: str):
    if not text:
        return []
    chunks = EMOJI_RE.findall(text)
    out = []
    for ch in chunks:
        out.extend(list(ch))
    return out

# ===============================
# Post emoji emotion: sum emoji vectors
# ===============================
def post_emoji_emotion(raw_text: str, emoji_table: dict):
    vec = np.zeros(len(EMOTIONS), dtype=np.float32)
    emojis = extract_emojis(raw_text)

    used = 0
    for e in emojis:
        if e in emoji_table:
            vec += emoji_table[e]
            used += 1

    s = float(vec.sum())
    if s > 0:
        vec /= s
    else:
        vec[EMOTION2IDX["neutral"]] = 1.0

    return vec, used, len(emojis)

# ===============================
# Main
# ===============================
def main():
    post_path = r"C:\Users\tinti\Desktop\IRTM_music\data\processed\posts\posts_clean_expanded.jsonl"
    emoji_joined_path = r"C:\Users\tinti\Desktop\IRTM_music\data\processed\posts\emoji_joined.txt"
    nrc_path = "data/raw/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

    # outputs
    emoji_table_path = "outputs/emotion_vectors/EmojiEmotionTable.npy"
    post_emoji_vec_path = "outputs/emotion_vectors/EmotionVec_posts_emoji.npy"

    os.makedirs(os.path.dirname(post_emoji_vec_path), exist_ok=True)

    # 1) build emoji table
    nrc = load_nrc_lexicon(nrc_path)
    emo2phrases = load_emoji_joined(emoji_joined_path)
    emoji_table = build_emoji_emotion_table(emo2phrases, nrc)

    np.save(emoji_table_path, emoji_table, allow_pickle=True)
    print(f"[OK] Saved emoji emotion table to {emoji_table_path} (n={len(emoji_table)})")

    # 2) per-post emoji emotion vectors
    vecs = []
    total = 0
    has_emoji = 0
    covered_posts = 0

    with open(post_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            raw = obj.get("raw_text", "")
            v, used, seen = post_emoji_emotion(raw, emoji_table)

            vecs.append(v)
            total += 1
            if seen > 0:
                has_emoji += 1
            if used > 0:
                covered_posts += 1

    vecs = np.stack(vecs, axis=0)
    np.save(post_emoji_vec_path, vecs)

    print(f"[OK] Saved Post Emotion (Emoji->phrase->NRC) to {post_emoji_vec_path}")
    print(f"Shape: {vecs.shape}")
    print(f"Posts with >=1 emoji in raw_text: {has_emoji}/{total} = {has_emoji/total:.3f}")
    print(f"Posts with >=1 emoji covered by emoji_joined: {covered_posts}/{total} = {covered_posts/total:.3f}")

if __name__ == "__main__":
    main()
