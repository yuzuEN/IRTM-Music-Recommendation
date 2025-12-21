"""
單次查詢介面
=================================================

功能：
輸入一篇貼文，輸出推薦歌曲列表。

流程：
1. 接收貼文文本
2. 預處理（tokenization）
3. 計算情緒向量（融合 lexicon/emoji/classifier）
4. 計算主題向量（使用歌曲的 KMeans 模型）
5. 執行 BM25 檢索（top-K 候選）
6. 執行 Personalized PageRank 推薦
7. 輸出推薦結果

使用方式：
  python pipeline/query_once.py --post "I feel lonely today"
  或
  python pipeline/query_once.py  # 交互式輸入
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
import pathlib

# 添加項目根目錄到路徑
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# 導入必要的模組
from src.emotion.posts_emotion_lex import compute_post_emotion, load_nrc_lexicon, EMOTIONS
from src.emotion.posts_emotion_emoji import post_emoji_emotion, build_emoji_emotion_table, load_emoji_joined
from src.emotion.posts_emotion_ml import load_posts_as_texts
from src.topic.posts_topic_align20 import load_lyrics_vectorizer, load_lyrics_kmeans_model
from src.bm25.compute_bm25 import compute_bm25_score_for_query
from src.graph_ppr.personalized_pagerank import (
    create_post_song_similarity_teleportation,
    personalized_pagerank
)
from src.preprocessing.preprocess_post import PostPreprocessor

############################################################
# 0. PATH CONFIGURATION
############################################################

# 模型和向量路徑
NRC_LEXICON_PATH = PROJECT_ROOT / "data" / "raw" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
EMOJI_JOINED_PATH = PROJECT_ROOT / "data" / "processed" / "posts" / "emoji_joined.txt"
EMOJI_TABLE_PATH = PROJECT_ROOT / "outputs" / "emotion_vectors" / "EmojiEmotionTable.npy"
EMOTION_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "post_emotion_lr.joblib"

LYRICS_KMEANS_MODEL_PATH = PROJECT_ROOT / "outputs" / "topic_vectors" / "lyrics_kmeans_model.joblib"
LYRICS_VECTORIZER_PATH = PROJECT_ROOT / "outputs" / "topic_vectors" / "lyrics_tfidf_vectorizer.joblib"
MERGE_INFO_PATH = PROJECT_ROOT / "outputs" / "topic_vectors" / "lyrics_cluster_merge_info.json"

# BM25 相關
BM25_MATRIX_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "bm25_matrix.npz"
BM25_METADATA_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "metadata.json"
BM25_VOCAB_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "vocabulary.json"
BM25_IDF_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "idf.json"
SONG_IDS_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "song_ids.json"

# 圖譜和歌曲向量
GRAPH_PATH = PROJECT_ROOT / "outputs" / "graph" / "song_graph.npz"
SONG_EMOTION_PATH = PROJECT_ROOT / "outputs" / "emotion_vectors" / "EmotionVec_lyrics.npy"
SONG_TOPIC_PATH = PROJECT_ROOT / "outputs" / "topic_vectors" / "TopicVec_lyrics_kmeans.npy"

# 歌曲 metadata（歌名、歌手、歌詞）
SONG_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "lyrics" / "clean_lyrics.json"

# 歌曲 metadata（歌名、歌手、歌詞）
SONG_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "lyrics" / "clean_lyrics.json"

############################################################
# 1. 預處理貼文
############################################################

def preprocess_post_text(post_text: str) -> Tuple[List[str], str]:
    """
    預處理貼文文本（使用完整的 preprocessing 模組）。
    
    返回：
        tokens: List[str] - 清理後的 tokens（expanded_tokens）
        raw_text: str - 原始文本
    """
    # 使用完整的 preprocessing 模組
    preprocessor = PostPreprocessor()
    clean_tokens = preprocessor.preprocess_text(post_text)
    
    # 簡單的 query expansion（可選，這裡先不使用）
    # 如果需要，可以導入 QueryExpander
    
    return clean_tokens, post_text

############################################################
# 2. 計算貼文情緒向量（融合）
############################################################

def compute_query_emotion(
    post_text: str,
    tokens: List[str],
    nrc_lexicon: dict,
    emoji_table: dict,
    emotion_model
) -> np.ndarray:
    """
    計算貼文的情緒向量（融合 lexicon + emoji + classifier）。
    
    返回：
        emotion_vec: (8,) - 融合後的情緒向量
    """
    # 1. Lexicon 情緒
    lex_emotion = compute_post_emotion(tokens, nrc_lexicon)
    
    # 2. Emoji 情緒
    emoji_emotion, _, _ = post_emoji_emotion(post_text, emoji_table)
    
    # 3. ML 模型情緒
    # 需要將 tokens 轉換成文本
    post_text_for_model = " ".join(tokens)
    model_probs = emotion_model.predict_proba([post_text_for_model])[0]
    model_emotion = model_probs.astype(np.float32)
    
    # 4. 融合（使用與 posts_emotion_fusion.py 相同的權重）
    WEIGHT_LEXICON = 0.3
    WEIGHT_EMOJI = 0.2
    WEIGHT_CLASSIFIER = 0.5
    
    fused = (
        WEIGHT_LEXICON * lex_emotion +
        WEIGHT_EMOJI * emoji_emotion +
        WEIGHT_CLASSIFIER * model_emotion
    )
    
    # 正規化
    s = fused.sum()
    if s > 0:
        fused = fused / s
    else:
        fused = np.zeros(8)
        fused[7] = 1.0  # neutral
    
    return fused

############################################################
# 3. 計算貼文主題向量
############################################################

def compute_query_topic(
    tokens: List[str],
    lyrics_vectorizer: TfidfVectorizer,
    lyrics_kmeans,
    cluster_mapping: Dict[int, int],
    K_merged: int
) -> np.ndarray:
    """
    計算貼文的主題向量（使用歌曲的 KMeans 模型）。
    
    返回：
        topic_vec: (K_merged,) - 主題向量（one-hot）
    """
    # 將 tokens 轉換成文本
    post_text = " ".join(tokens)
    
    # TF-IDF 轉換
    X_post = lyrics_vectorizer.transform([post_text])
    
    # 預測主題（原始 cluster_id）
    topic_label_original = lyrics_kmeans.predict(X_post)[0]
    
    # 映射到合併後的 cluster_id
    topic_label_merged = cluster_mapping.get(int(topic_label_original), 0)
    
    # 轉換成 one-hot 向量
    topic_vec = np.zeros(K_merged, dtype=np.float32)
    if 0 <= topic_label_merged < K_merged:
        topic_vec[topic_label_merged] = 1.0
    
    return topic_vec

############################################################
# 4. 載入所有必要的模型和資料
############################################################

def load_song_metadata(metadata_path: pathlib.Path) -> Dict[str, Dict]:
    """
    載入歌曲 metadata（title, artist, lyrics），建立 song_id → metadata 的 mapping。
    
    返回：
        song_metadata: Dict[str, dict] - song_id → {"title": str, "artist": str, "lyrics": str}
    """
    print("[Load] Loading song metadata...")
    
    if not metadata_path.exists():
        print(f"[Warn] Song metadata not found: {metadata_path}")
        return {}
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        songs = json.load(f)
    
    metadata = {}
    for song in songs:
        song_id = song.get("song_id")
        if song_id:
            metadata[song_id] = {
                "title": song.get("title", "Unknown"),
                "artist": song.get("artist", "Unknown"),
                "lyrics": song.get("lyrics", "")
            }
    
    print(f"[Load] Loaded metadata for {len(metadata)} songs")
    return metadata

def load_all_models_and_data():
    """
    載入所有必要的模型、向量和資料。
    
    返回：
        models_and_data: Dict - 包含所有載入的模型和資料
    """
    print("[Load] Loading all models and data...")
    
    # 1. 情緒相關
    print("[Load] Loading emotion models...")
    nrc_lexicon = load_nrc_lexicon(str(NRC_LEXICON_PATH))
    
    # 載入 emoji table
    if EMOJI_TABLE_PATH.exists():
        emoji_table = np.load(str(EMOJI_TABLE_PATH), allow_pickle=True).item()
    else:
        # 如果沒有，需要建立
        emo2phrases = load_emoji_joined(str(EMOJI_JOINED_PATH))
        emoji_table = build_emoji_emotion_table(emo2phrases, nrc_lexicon)
    
    # 載入 ML 模型
    emotion_model = joblib.load(str(EMOTION_MODEL_PATH))
    
    # 2. 主題相關
    print("[Load] Loading topic models...")
    lyrics_kmeans = load_lyrics_kmeans_model(str(LYRICS_KMEANS_MODEL_PATH))
    lyrics_vectorizer = load_lyrics_vectorizer(str(LYRICS_VECTORIZER_PATH))
    cluster_mapping, K_merged = load_cluster_mapping_local()
    
    # 3. BM25 相關
    print("[Load] Loading BM25 artifacts...")
    with open(BM25_METADATA_PATH, "r", encoding="utf-8") as f:
        bm25_metadata = json.load(f)
    bm25_matrix = load_npz(str(BM25_MATRIX_PATH))
    
    # 載入 vocabulary 和 idf（分別儲存在不同檔案中）
    with open(BM25_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(BM25_IDF_PATH, "r", encoding="utf-8") as f:
        idf = json.load(f)
    
    with open(SONG_IDS_PATH, "r", encoding="utf-8") as f:
        song_ids = json.load(f)
    
    # 4. 圖譜和歌曲向量
    print("[Load] Loading graph and song vectors...")
    graph = load_npz(str(GRAPH_PATH))
    
    song_emotion = np.load(str(SONG_EMOTION_PATH), allow_pickle=True)
    if song_emotion.dtype == object:
        emotion_dict = song_emotion.item() if song_emotion.ndim == 0 else song_emotion
        song_emotion = np.array([emotion_dict.get(sid, np.zeros(8)) for sid in song_ids])
    
    song_topic = np.load(str(SONG_TOPIC_PATH), allow_pickle=True)
    
    # 5. 載入歌曲 metadata
    print("[Load] Loading song metadata...")
    song_metadata = load_song_metadata(SONG_METADATA_PATH)
    
    print("[Load] All models and data loaded!")
    
    return {
        "nrc_lexicon": nrc_lexicon,
        "emoji_table": emoji_table,
        "emotion_model": emotion_model,
        "lyrics_kmeans": lyrics_kmeans,
        "lyrics_vectorizer": lyrics_vectorizer,
        "cluster_mapping": cluster_mapping,
        "K_merged": K_merged,
        "bm25_matrix": bm25_matrix,
        "bm25_metadata": bm25_metadata,
        "vocab": vocab,
        "idf": idf,
        "song_ids": song_ids,
        "graph": graph,
        "song_emotion": song_emotion,
        "song_topic": song_topic,
        "song_metadata": song_metadata,
    }

############################################################
# 5. 推薦流程
############################################################

def recommend_songs_for_post(
    post_text: str,
    models_and_data: dict,
    top_k: int = 10
) -> List[Dict]:
    """
    為單一貼文產生推薦歌曲。
    
    參數：
        post_text: str - 貼文文本
        models_and_data: dict - 所有載入的模型和資料
        top_k: int - 最終返回的推薦數量
    
    返回：
        recommendations: List[Dict] - 推薦結果列表
    """
    print(f"\n[Query] Processing post: \"{post_text[:50]}...\"")
    
    # 1. 預處理
    tokens, raw_text = preprocess_post_text(post_text)
    print(f"[Query] Preprocessed tokens: {len(tokens)} tokens")
    
    # 2. 計算情緒向量
    print("[Query] Computing emotion vector...")
    query_emotion = compute_query_emotion(
        raw_text,
        tokens,
        models_and_data["nrc_lexicon"],
        models_and_data["emoji_table"],
        models_and_data["emotion_model"]
    )
    print(f"[Query] Emotion vector: {query_emotion}")
    
    # 3. 計算主題向量
    print("[Query] Computing topic vector...")
    query_topic = compute_query_topic(
        tokens,
        models_and_data["lyrics_vectorizer"],
        models_and_data["lyrics_kmeans"],
        models_and_data["cluster_mapping"],
        models_and_data["K_merged"]
    )
    print(f"[Query] Topic vector: {query_topic}")
    
    # 4. BM25 檢索（Stage 1: 找出 top-1000 候選）
    print("[Query] Stage 1: BM25 retrieval (top-1000 candidates)...")
    bm25_scores = compute_bm25_score_for_query(
        tokens,
        models_and_data["bm25_matrix"],
        models_and_data["vocab"],
        models_and_data["idf"],
        models_and_data["bm25_metadata"],
        models_and_data["song_ids"]
    )
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:1000]
    print(f"[Query] BM25 top-1000 candidates found")
    print(f"[Query] BM25 score range: min={bm25_scores[bm25_top_indices[-1]]:.4f}, max={bm25_scores[bm25_top_indices[0]]:.4f}")
    
    # 5. Stage 2: 在候選內計算 Emotion + Topic 相似度（Reranking）
    print("[Query] Stage 2: Computing Emotion + Topic similarity within candidates...")
    
    # 提取候選歌曲的向量
    candidate_emotion = models_and_data["song_emotion"][bm25_top_indices]  # (1000, 8)
    candidate_topic = models_and_data["song_topic"][bm25_top_indices]      # (1000, 21)
    
    # 計算相似度（只對候選）
    from sklearn.metrics.pairwise import cosine_similarity
    emotion_sim = cosine_similarity(
        query_emotion.reshape(1, -1),
        candidate_emotion
    )[0]  # (1000,)
    
    topic_sim = cosine_similarity(
        query_topic.reshape(1, -1),
        candidate_topic
    )[0]  # (1000,)
    
    print(f"[Query] Emotion similarity stats (within candidates): min={emotion_sim.min():.6f}, max={emotion_sim.max():.6f}, mean={emotion_sim.mean():.6f}, std={emotion_sim.std():.6f}")
    print(f"[Query] Topic similarity stats (within candidates): min={topic_sim.min():.6f}, max={topic_sim.max():.6f}, mean={topic_sim.mean():.6f}, std={topic_sim.std():.6f}")
    
    # 組合相似度
    combined_sim = 0.6 * emotion_sim + 0.4 * topic_sim  # (1000,)
    print(f"[Query] Combined similarity stats: min={combined_sim.min():.6f}, max={combined_sim.max():.6f}, mean={combined_sim.mean():.6f}, std={combined_sim.std():.6f}")
    
    # 6. 建立 teleportation vector（只對候選賦值）
    print("[Query] Building teleportation vector (only candidates have scores)...")
    teleportation_full = np.zeros(len(models_and_data["song_ids"]))
    teleportation_full[bm25_top_indices] = combined_sim
    
    # 正規化為機率分布
    sim_sum = teleportation_full.sum()
    if sim_sum > 0:
        teleportation_full = teleportation_full / sim_sum
    else:
        # Fallback: 如果所有相似度都是 0，使用均勻分布（只對候選）
        teleportation_full[bm25_top_indices] = 1.0 / len(bm25_top_indices)
    
    print(f"[Query] Teleportation vector: {len(bm25_top_indices)} candidates with non-zero scores")
    print(f"[Query] Teleportation stats: min={teleportation_full.min():.10f}, max={teleportation_full.max():.10f}, mean={teleportation_full.mean():.10f}, std={teleportation_full.std():.10f}")
    
    # 7. Stage 3: Personalized PageRank（在整個圖上擴散）
    print("[Query] Running Personalized PageRank...")
    ppr_scores = personalized_pagerank(
        models_and_data["graph"],
        teleportation_full,
        alpha=0.3  # Random walk: 70%, Teleportation: 30%
    )
    print(f"[Query] PPR stats: min={ppr_scores.min():.6f}, max={ppr_scores.max():.6f}, mean={ppr_scores.mean():.6f}, std={ppr_scores.std():.6f}")
    
    # 8. 取得 top-K 推薦
    top_indices = np.argsort(ppr_scores)[::-1][:top_k]
    print(f"[Query] Top-{top_k} PPR scores: {ppr_scores[top_indices]}")
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        # 檢查是否在 BM25 候選中
        is_candidate = idx in bm25_top_indices
        bm25_idx_in_candidates = np.where(bm25_top_indices == idx)[0]
        
        recommendations.append({
            "rank": rank,
            "song_id": models_and_data["song_ids"][idx],
            "ppr_score": float(ppr_scores[idx]),
            "teleportation_score": float(teleportation_full[idx]),
            "is_bm25_candidate": bool(is_candidate),
            "bm25_score": float(bm25_scores[idx]) if is_candidate else 0.0,
            "combined_sim": float(combined_sim[bm25_idx_in_candidates[0]]) if len(bm25_idx_in_candidates) > 0 else 0.0,
        })
    
    return recommendations

############################################################
# 6. Main
############################################################

def load_cluster_mapping_local() -> Tuple[Dict[int, int], int]:
    """載入 cluster mapping（從 posts_topic_align20.py 複製）"""
    if not MERGE_INFO_PATH.exists():
        return {i: i for i in range(29)}, 29
    
    with open(MERGE_INFO_PATH, "r", encoding="utf-8") as f:
        merge_info = json.load(f)
    
    K_merged = merge_info.get("K_merged", 21)
    groups = merge_info.get("groups", [])
    
    mapping = {}
    for merged_id, group in enumerate(groups):
        for original_id in group:
            mapping[original_id] = merged_id
    
    return mapping, K_merged

def main():
    """
    主流程：單次查詢介面。
    """
    parser = argparse.ArgumentParser(description="Query music recommendation for a single post")
    parser.add_argument("--post", type=str, help="Post text to query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return")
    
    args = parser.parse_args()
    
    # 獲取貼文文本
    if args.post:
        post_text = args.post
    else:
        # 交互式輸入
        print("=" * 60)
        print("Music Recommendation Query Interface")
        print("=" * 60)
        post_text = input("\n請輸入貼文文本: ").strip()
        if not post_text:
            print("錯誤：貼文文本不能為空")
            return
    
    # 檢查必要檔案
    required_files = [
        NRC_LEXICON_PATH, EMOJI_JOINED_PATH, EMOTION_MODEL_PATH,
        LYRICS_KMEANS_MODEL_PATH, LYRICS_VECTORIZER_PATH,
        BM25_MATRIX_PATH, BM25_METADATA_PATH, BM25_VOCAB_PATH, BM25_IDF_PATH, SONG_IDS_PATH,
        GRAPH_PATH, SONG_EMOTION_PATH, SONG_TOPIC_PATH
        # SONG_METADATA_PATH 是可選的，如果不存在會顯示警告但不會中斷
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("錯誤：以下必要檔案不存在：")
        for f in missing_files:
            print(f"  - {f}")
        print("\n請先執行完整的 pipeline 來生成這些檔案。")
        return
    
    # 載入所有模型和資料
    models_and_data = load_all_models_and_data()
    
    # 產生推薦
    recommendations = recommend_songs_for_post(
        post_text,
        models_and_data,
        top_k=args.top_k
    )
    
    # 輸出結果
    print("\n" + "=" * 60)
    print("推薦結果")
    print("=" * 60)
    print(f"\n貼文: \"{post_text}\"")
    print(f"\nTop {len(recommendations)} 推薦歌曲：\n")
    
    for rec in recommendations:
        candidate_mark = "★" if rec['is_bm25_candidate'] else " "
        song_id = rec['song_id']
        
        # 取得歌曲 metadata
        metadata = models_and_data["song_metadata"].get(song_id, {})
        title = metadata.get("title", "Unknown")
        artist = metadata.get("artist", "Unknown")
        lyrics = metadata.get("lyrics", "")
        
        # 截取歌詞前 150 字元（避免輸出太長）
        lyrics_preview = lyrics[:150] + "..." if len(lyrics) > 150 else lyrics
        # 清理歌詞中的換行符，用空格代替
        if lyrics_preview:
            lyrics_preview = lyrics_preview.replace("\n", " ").replace("\r", " ")
        
        print(f"\n{rec['rank']:2d}. {candidate_mark} {title} - {artist}")
        print(f"    Song ID: {song_id}")
        print(f"    PPR: {rec['ppr_score']:.10f} | Teleport: {rec['teleportation_score']:.10f} | "
              f"BM25: {rec['bm25_score']:.4f} | 29D Sim: {rec['combined_sim']:.4f}")
        if lyrics_preview:
            print(f"    Lyrics: {lyrics_preview}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

