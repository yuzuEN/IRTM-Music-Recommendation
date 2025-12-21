"""
貼文主題編碼器（Post Topic Encoder）
====================================

功能：
將貼文文本轉換成與歌曲對齊的 20 維主題向量。

這是一個專門的 encoder，負責：
1. 載入歌曲的 KMeans 模型（20 維，已合併）
2. 載入歌曲的 TF-IDF vectorizer（已保存，直接載入）
3. 對貼文做 TF-IDF 轉換（使用歌曲的 vectorizer）
4. 用歌曲的 KMeans 模型預測貼文主題
5. 輸出 20 維 one-hot 主題向量（與歌曲對齊）

輸入：
  data/processed/posts/posts_clean_expanded.jsonl
  outputs/topic_vectors/lyrics_kmeans_model.joblib (歌曲 KMeans 模型)
  outputs/topic_vectors/lyrics_tfidf_vectorizer.joblib (歌曲 TF-IDF vectorizer，已 fit)

輸出：
  outputs/topic_vectors/post_topic_20d.npy (20 維主題向量，與歌曲對齊)
  outputs/topic_vectors/post_topic_20d_assignments.jsonl (主題分配)
  outputs/topic_vectors/post_topic_20d_meta.json (元資料)

依賴：
  需要先執行 lyrics_topic_kmeans_scanK_merge.py 來生成模型和 vectorizer。
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ✅ 使用與歌曲相同的 stopwords
LYRICS_TOPIC_STOPWORDS = {
    "yeah", "oh", "uh", "na", "la", "ya",
    "choru", "chorus", "verse", "hook",
    "da", "ah", "woah", "ooh",
    "hey", "huh", "whoa", "mmm", "mm",
    "choru", "chorus", "verse", "vers", "bridg", "outro", "hook",
    "got", "get", "gotta", "got ta",
    "wan", "want", "gon", "gonna",
    "im", "em", "ta",
    "say", "said", "tell", "look", "come", "let", "make",
    "fuck", "shit", "bitch", "nigga", "ass", "damn", "hell", "yo",
}

def build_stopwords_for_lyrics_topic() -> List[str]:
    return sorted(set(ENGLISH_STOP_WORDS) | set(LYRICS_TOPIC_STOPWORDS))


############################################################
# 0. PATH CONFIGURATION
############################################################

POSTS_PATH = r"C:\Users\tinti\Desktop\IRTM_music\data\processed\posts\posts_clean_expanded.jsonl"

OUT_DIR = "outputs/topic_vectors"
os.makedirs(OUT_DIR, exist_ok=True)

# 輸入檔案（歌曲的模型和參數）
LYRICS_KMEANS_MODEL_PATH = os.path.join(OUT_DIR, "lyrics_kmeans_model.joblib")
LYRICS_VECTORIZER_PATH = os.path.join(OUT_DIR, "lyrics_tfidf_vectorizer.joblib")  # ✅ 載入保存的 vectorizer
LYRICS_TFIDF_PATH = os.path.join(OUT_DIR, "lyrics_tfidf.npz")
LYRICS_TFIDF_META_PATH = os.path.join(OUT_DIR, "lyrics_tfidf_meta.json")

# 輸出檔案
POST_TOPIC_20D_PATH = os.path.join(OUT_DIR, "post_topic_20d.npy")
POST_TOPIC_20D_ASSIGN_PATH = os.path.join(OUT_DIR, "post_topic_20d_assignments.jsonl")
POST_TOPIC_20D_META_PATH = os.path.join(OUT_DIR, "post_topic_20d_meta.json")


############################################################
# 1. 載入貼文
############################################################

def tokens_to_text(tokens: Any) -> str:
    if isinstance(tokens, list):
        return " ".join([str(t) for t in tokens if str(t).strip()])
    if isinstance(tokens, str):
        return tokens
    return ""


def load_posts(posts_jsonl_path: str, prefer: str = "expanded_tokens") -> Tuple[List[str], List[Dict[str, Any]]]:
    """載入貼文，轉換成文本列表"""
    texts: List[str] = []
    raws: List[Dict[str, Any]] = []

    with open(posts_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if prefer == "raw_text":
                text = obj.get("raw_text", "")
            else:
                tokens = obj.get(prefer)
                if tokens is None:
                    tokens = obj.get("expanded_tokens") or obj.get("clean_tokens") or obj.get("raw_text")
                text = tokens_to_text(tokens)

            texts.append(text if isinstance(text, str) else "")
            raws.append(obj)

    return texts, raws


############################################################
# 2. 載入歌曲的 TF-IDF Vectorizer
############################################################

def load_lyrics_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """
    載入已保存的歌曲 TF-IDF vectorizer。
    
    這個 vectorizer 已經在歌曲文本上 fit 過，可以直接用來 transform 貼文。
    """
    vectorizer = joblib.load(vectorizer_path)
    print(f"[Load] Loaded lyrics TF-IDF vectorizer from {vectorizer_path}")
    print(f"[Load] Vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer


############################################################
# 3. 載入歌曲的 KMeans 模型
############################################################

def load_lyrics_kmeans_model(model_path: str):
    """載入歌曲的 KMeans 模型"""
    kmeans = joblib.load(model_path)
    print(f"[Load] Loaded lyrics KMeans model from {model_path}")
    print(f"[Load] Number of clusters: {kmeans.n_clusters}")
    return kmeans


############################################################
# 4. 對齊方法：用歌曲的模型預測貼文主題
############################################################

def load_cluster_mapping() -> Tuple[Dict[int, int], int]:
    """
    載入 cluster merge mapping（將原始 29 個 clusters 映射到合併後的 20 個）。
    
    返回：
        mapping: Dict[original_cluster_id, merged_cluster_id]
        K_merged: 合併後的 cluster 數量
    """
    merge_info_path = os.path.join(OUT_DIR, "lyrics_cluster_merge_info.json")
    
    if not os.path.exists(merge_info_path):
        # 如果沒有 merge info，假設沒有合併（1-to-1 mapping）
        print(f"[Warn] Merge info not found: {merge_info_path}, assuming no merge (K=29)")
        return {i: i for i in range(29)}, 29
    
    with open(merge_info_path, "r", encoding="utf-8") as f:
        merge_info = json.load(f)
    
    K_merged = merge_info.get("K_merged", 20)
    groups = merge_info.get("groups", [])
    
    # 建立 mapping：original_cluster_id -> merged_cluster_id
    mapping = {}
    for merged_id, group in enumerate(groups):
        for original_id in group:
            mapping[original_id] = merged_id
    
    print(f"[Load] Cluster mapping: {len(mapping)} original clusters -> {K_merged} merged clusters")
    return mapping, K_merged


def align_posts_to_lyrics_topics(
    post_texts: List[str],
    lyrics_vectorizer: TfidfVectorizer,
    lyrics_kmeans,
    lyrics_tfidf_meta: dict,
    cluster_mapping: Dict[int, int],
    K_merged: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用歌曲的 TF-IDF vectorizer 和 KMeans 模型預測貼文主題，並映射到合併後的 20 維。
    
    輸入：
        post_texts: List[str] - 貼文文本列表
        lyrics_vectorizer: TfidfVectorizer - 歌曲的 TF-IDF vectorizer（已 fit）
        lyrics_kmeans: KMeans - 歌曲的 KMeans 模型（29 維，原始）
        cluster_mapping: Dict[int, int] - 原始 cluster_id 到合併 cluster_id 的映射
        K_merged: int - 合併後的 cluster 數量（20）
        
    輸出：
        topic_labels: np.ndarray - 主題標籤 (num_posts,) - 合併後的標籤（0-19）
        topic_vectors: np.ndarray - 20 維 one-hot 向量 (num_posts, 20)
    """
    print(f"[Align] Transforming posts with lyrics TF-IDF vectorizer...")
    
    # 對貼文做 TF-IDF 轉換（使用歌曲的 vectorizer）
    X_posts = lyrics_vectorizer.transform(post_texts)
    print(f"[Align] Post TF-IDF shape: {X_posts.shape}")
    
    # 用歌曲的 KMeans 模型預測貼文主題（原始 29 維）
    print(f"[Align] Predicting topics with lyrics KMeans model (original 29D)...")
    topic_labels_original = lyrics_kmeans.predict(X_posts)
    print(f"[Align] Original topic labels shape: {topic_labels_original.shape}")
    print(f"[Align] Original topic distribution: {np.bincount(topic_labels_original, minlength=lyrics_kmeans.n_clusters)}")
    
    # 映射到合併後的 cluster_id（20 維）
    print(f"[Align] Mapping to merged clusters (20D)...")
    topic_labels = np.array([cluster_mapping.get(int(tid), 0) for tid in topic_labels_original])
    print(f"[Align] Merged topic labels shape: {topic_labels.shape}")
    print(f"[Align] Merged topic distribution: {np.bincount(topic_labels, minlength=K_merged)}")
    
    # 轉換成 one-hot 向量（20 維）
    num_posts = len(post_texts)
    topic_vectors = np.zeros((num_posts, K_merged), dtype=np.float32)
    topic_vectors[np.arange(num_posts), topic_labels] = 1.0
    
    print(f"[Align] Topic vectors shape: {topic_vectors.shape} (should be [{num_posts}, {K_merged}])")
    
    return topic_labels, topic_vectors


############################################################
# 5. 主流程（需要先 fit vectorizer）
############################################################

def main():
    """
    貼文主題編碼器（Encoder）
    ===========================
    
    功能：將貼文文本轉換成與歌曲對齊的 20 維主題向量。
    
    流程：
    1. 載入歌曲的 KMeans 模型（20 維）
    2. 載入歌曲的 TF-IDF vectorizer（已 fit）
    3. 對貼文做 TF-IDF 轉換
    4. 用 KMeans 模型預測貼文主題
    5. 輸出 20 維 one-hot 主題向量
    
    注意：需要先執行 lyrics_topic_kmeans_scanK_merge.py 來生成模型和 vectorizer。
    """
    print("=" * 60)
    print("Post Topic Alignment to Lyrics 20D Topic Space")
    print("=" * 60)
    
    # 檢查輸入檔案
    if not os.path.exists(LYRICS_KMEANS_MODEL_PATH):
        raise FileNotFoundError(f"Lyrics KMeans model not found: {LYRICS_KMEANS_MODEL_PATH}")
    if not os.path.exists(LYRICS_VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Lyrics TF-IDF vectorizer not found: {LYRICS_VECTORIZER_PATH}\n"
            f"Please run lyrics_topic_kmeans_scanK_merge.py first to generate the vectorizer."
        )
    
    # 1. 載入歌曲的 KMeans 模型
    lyrics_kmeans = load_lyrics_kmeans_model(LYRICS_KMEANS_MODEL_PATH)
    K = lyrics_kmeans.n_clusters
    print(f"[Info] Lyrics KMeans has {K} clusters (should be 20 after merge)")
    
    # 2. 載入歌曲的 TF-IDF vectorizer（已 fit，可直接使用）
    print(f"\n[Load] Loading lyrics TF-IDF vectorizer...")
    lyrics_vectorizer = load_lyrics_vectorizer(LYRICS_VECTORIZER_PATH)
    
    # 3. 載入 cluster mapping（將 29 維映射到 20 維）
    print(f"\n[Load] Loading cluster mapping...")
    cluster_mapping, K_merged = load_cluster_mapping()
    print(f"[Info] Will map from {K} original clusters to {K_merged} merged clusters")
    
    # 4. 載入貼文
    print(f"\n[Load] Loading posts...")
    post_texts, post_raws = load_posts(POSTS_PATH, prefer="expanded_tokens")
    n_posts = len(post_texts)
    print(f"[Load] Loaded {n_posts} posts")
    
    # 載入 metadata（用於記錄）
    with open(LYRICS_TFIDF_META_PATH, "r", encoding="utf-8") as f:
        lyrics_tfidf_meta = json.load(f)
    
    # 5. 對齊貼文主題（預測後映射到 20 維）
    topic_labels, topic_vectors = align_posts_to_lyrics_topics(
        post_texts,
        lyrics_vectorizer,
        lyrics_kmeans,
        lyrics_tfidf_meta,
        cluster_mapping,
        K_merged
    )
    
    # 6. 儲存結果
    print(f"\n[Save] Saving aligned topic vectors...")
    np.save(POST_TOPIC_20D_PATH, topic_vectors)
    print(f"[Save] Saved to {POST_TOPIC_20D_PATH}")
    print(f"[Save] Shape: {topic_vectors.shape}")
    
    # 儲存 assignments
    with open(POST_TOPIC_20D_ASSIGN_PATH, "w", encoding="utf-8") as f:
        for i in range(n_posts):
            record = {
                "idx": i,
                "topic_id": int(topic_labels[i]),
                "raw_text": post_raws[i].get("raw_text", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[Save] Saved assignments to {POST_TOPIC_20D_ASSIGN_PATH}")
    
    # 儲存 metadata
    meta = {
        "type": "aligned_to_lyrics_20d",
        "K_original": int(K),
        "K_merged": int(K_merged),
        "shape": list(topic_vectors.shape),
        "lyrics_kmeans_model_path": LYRICS_KMEANS_MODEL_PATH,
        "lyrics_tfidf_meta_path": LYRICS_TFIDF_META_PATH,
        "posts_path": POSTS_PATH,
        "topic_distribution": {
            int(tid): int(count) 
            for tid, count in enumerate(np.bincount(topic_labels, minlength=K_merged))
        }
    }
    with open(POST_TOPIC_20D_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Save] Saved metadata to {POST_TOPIC_20D_META_PATH}")
    
    print("\n" + "=" * 60)
    print("✅ Post Topic Alignment Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {POST_TOPIC_20D_PATH} (20D topic vectors, aligned with lyrics)")
    print(f"  - {POST_TOPIC_20D_ASSIGN_PATH}")
    print(f"  - {POST_TOPIC_20D_META_PATH}")


if __name__ == "__main__":
    main()

