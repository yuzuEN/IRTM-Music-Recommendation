"""
組員 A — 貼文前處理 + Query Expansion（完整版本）
=================================================

功能：
1. 貼文前處理：
   - hashtag 分詞 (#NotOkay → not okay)
   - slang 正規化
   - 重複字壓縮
   - @username / URL 過濾
   - emoji 分離
   - tokenization

2. Query Expansion：
   - WordNet 同義詞擴展
   - Emotion Lexicon 擴展（可選）
   - Pseudo Relevance Feedback（可選）

輸入：
  /data/raw/posts/generated_social_posts.json

輸出：
  /data/processed/posts/posts_clean_expanded.jsonl
"""

import os
import re
import json
import string
import pathlib
from typing import List, Dict, Set

from nltk.corpus import wordnet as wn

############################################################
# 0. PATH CONFIGURATION
############################################################

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RAW_POST_DIR = PROJECT_ROOT / "data" / "raw" / "posts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "posts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


############################################################
# 1. 貼文前處理 MODULE
############################################################

class PostPreprocessor:
    """
    貼文前處理功能：
    1. hashtag 分詞：#NotOkay → "not okay"
    2. slang 修正
    3. 重複字壓縮："soooo" → "soo"
    4. 移除 URL
    5. 移除 @username
    6. emoji 分離（embedding 不是 A 的責任）
    7. tokenization
    """

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    USER_PATTERN = re.compile(r"@\w+")
    REPEAT_PATTERN = re.compile(r"(.)\1{2,}")

    SLANG = {
        "idk": "i don't know",
        "imo": "in my opinion",
        "imho": "in my humble opinion",
        "omg": "oh my god",
        "lmao": "laughing",
        "lol": "laughing",
        "u": "you",
        "ur": "your",
        "thx": "thanks",
        "pls": "please",
    }

    def split_hashtag(self, tag: str) -> str:
        parts = re.findall(r"[A-Z][a-z]+|[a-z]+|\d+", tag)
        return " ".join(parts) if parts else tag

    def preprocess_text(self, text: str) -> List[str]:

        # 1. 移除 URL
        text = self.URL_PATTERN.sub(" ", text)

        # 2. 移除 @username
        text = self.USER_PATTERN.sub(" ", text)

        # 3. 處理 hashtag → 分詞
        text = re.sub(r"#(\w+)", lambda m: self.split_hashtag(m.group(1)), text)

        # 4. emoji 分離
        text = "".join(f" {ch} " if ord(ch) > 10000 else ch for ch in text)

        # 5. lowercase
        text = text.lower()

        # 6. tokenization
        tokens = re.findall(r"\w+|[^\w\s]", text)

        # 7. slang & 重複字壓縮
        processed = []
        for t in tokens:
            # 重複字壓縮：soooo → soo
            t = self.REPEAT_PATTERN.sub(r"\1\1", t)

            # slang 正規化
            if t in self.SLANG:
                t = self.SLANG[t]

            processed.extend(t.split())

        return processed


############################################################
# 2. Query Expansion MODULE
############################################################

class QueryExpander:
    """
    支援：
    - WordNet 同義詞擴展
    - Emotion lexicon 擴展（可選）
    - PRF（可選）
    """

    def __init__(self, lexicon_path=None):
        self.lexicon = {}
        if lexicon_path and os.path.exists(lexicon_path):
            with open(lexicon_path, "r", encoding="utf-8") as f:
                for line in f:
                    w, emo = line.strip().split(",")
                    self.lexicon.setdefault(w, set()).add(emo)

    # 1. WordNet 擴展
    def expand_wordnet(self, tokens: List[str]) -> Set[str]:
        expanded = set(tokens)
        for t in tokens:
            for syn in wn.synsets(t):
                for lemma in syn.lemmas()[:2]:
                    name = lemma.name().replace("_", " ").lower()
                    expanded.add(name)
        return expanded

    # 2. Emotion lexicon 擴展
    def expand_lexicon(self, tokens: List[str]) -> Set[str]:
        if not self.lexicon:
            return set(tokens)

        expanded = set(tokens)
        emos = set()

        for t in tokens:
            if t in self.lexicon:
                emos.update(self.lexicon[t])

        for w, e in self.lexicon.items():
            if emos & e:
                expanded.add(w)

        return expanded

    # 3. Pseudo Relevance Feedback (optional)
    def pseudo_relevance_feedback(
        self,
        query_tokens: List[str],
        top_docs_tokens: List[List[str]],
        stopwords: Set[str] = None,
        top_k: int = 8
    ) -> Set[str]:

        if stopwords is None:
            stopwords = {
                "i", "am", "is", "are", "was", "the", "a", "an",
                "in", "on", "for", "to", "with", "and", "or"
            }

        freq = {}
        for doc in top_docs_tokens:
            for t in doc:
                if t not in stopwords:
                    freq[t] = freq.get(t, 0) + 1

        prf_terms = sorted(freq, key=freq.get, reverse=True)[:top_k]

        return set(query_tokens) | set(prf_terms)


############################################################
# 3. 遍歷 /data/raw/posts/*.json → 前處理 + QE
############################################################

def process_posts():
    pre = PostPreprocessor()
    qe = QueryExpander()  # 如需 lexicon 可傳入 lexicon_path

    all_files = list(RAW_POST_DIR.glob("generated_social_posts_10k.json"))
    print(f"找到 {len(all_files)} 個貼文檔案")

    output_path = OUTPUT_DIR / "posts_clean_expanded.jsonl"
    fout = open(output_path, "w", encoding="utf-8")

    for fp in all_files:
        print(f"處理：{fp.name}")

        data = json.load(open(fp, "r", encoding="utf-8"))

        for item in data:
            raw_text = item["text"]

            clean_tokens = pre.preprocess_text(raw_text)

            expanded_tokens = qe.expand_wordnet(clean_tokens)

            obj = {
                "raw_text": raw_text,
                "clean_tokens": clean_tokens,
                "expanded_tokens": list(expanded_tokens),
                "emotion": item["emotion"],
                "strength": item["strength"]
            }

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    fout.close()
    print(f"\n已輸出前處理＋QE 結果至：{output_path}")


############################################################
if __name__ == "__main__":
    process_posts()
