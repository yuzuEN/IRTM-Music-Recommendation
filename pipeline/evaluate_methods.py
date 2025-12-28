"""
推薦方法對照評估腳本
=================================================

功能：
對同一篇貼文，使用不同的推薦方法組合產生推薦結果，方便對照比較。

支援的方法組合：
1. BM25 only
2. BM25 + Emotion only reranking
3. BM25 + Topic only reranking
4. BM25 + Emotion + Topic reranking
5. BM25 + Emotion + Topic + PPR (完整流程)
6. Emotion + Topic only (no BM25)
7. BM25 + PPR (no reranking)
8. PPR only (uniform teleportation)
9. BM25 + Emotion + Topic + PPR (不同權重)

使用方式：
  python pipeline/evaluate_methods.py --post "I feel lonely today" --methods 1,2,3,5
  python pipeline/evaluate_methods.py --post "I feel lonely today" --methods all
  python pipeline/evaluate_methods.py --post "I feel lonely today" --methods 1,2,3,5 --output results.json
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional
import pathlib
from enum import Enum

# 添加項目根目錄到路徑
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# 導入必要的模組
from src.emotion.posts_emotion_lex import compute_post_emotion, load_nrc_lexicon, EMOTIONS
from src.emotion.posts_emotion_emoji import post_emoji_emotion, build_emoji_emotion_table, load_emoji_joined
from src.topic.posts_topic_align20 import load_lyrics_vectorizer, load_lyrics_kmeans_model, load_cluster_mapping
from src.bm25.compute_bm25 import compute_bm25_score_for_query
from src.graph_ppr.personalized_pagerank import personalized_pagerank
from src.preprocessing.preprocess_post import PostPreprocessor

############################################################
# 0. PATH CONFIGURATION
############################################################

NRC_LEXICON_PATH = PROJECT_ROOT / "data" / "raw" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
EMOJI_JOINED_PATH = PROJECT_ROOT / "data" / "processed" / "posts" / "emoji_joined.txt"
EMOJI_TABLE_PATH = PROJECT_ROOT / "outputs" / "emotion_vectors" / "EmojiEmotionTable.npy"
EMOTION_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "post_emotion_lr.joblib"

LYRICS_KMEANS_MODEL_PATH = PROJECT_ROOT / "outputs" / "topic_vectors" / "lyrics_kmeans_model.joblib"
LYRICS_VECTORIZER_PATH = PROJECT_ROOT / "outputs" / "topic_vectors" / "lyrics_tfidf_vectorizer.joblib"
MERGE_INFO_PATH = PROJECT_ROOT / "outputs" / "topic_vectors" / "lyrics_cluster_merge_info.json"

BM25_MATRIX_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "bm25_matrix.npz"
BM25_METADATA_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "metadata.json"
BM25_VOCAB_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "vocabulary.json"
BM25_IDF_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "idf.json"
SONG_IDS_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "song_ids.json"

GRAPH_PATH = PROJECT_ROOT / "outputs" / "graph" / "song_graph.npz"
SONG_EMOTION_PATH = PROJECT_ROOT / "outputs" / "emotion_vectors" / "EmotionVec_lyrics.npy"
SONG_TOPIC_PATH = PROJECT_ROOT / "outputs" / "topic_vectors" / "TopicVec_lyrics_kmeans.npy"

SONG_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "lyrics" / "clean_lyrics.json"

############################################################
# 1. 方法組合定義
############################################################

class MethodType(Enum):
    BM25_ONLY = 1
    BM25_EMOTION_ONLY = 2
    BM25_TOPIC_ONLY = 3
    BM25_EMOTION_TOPIC = 4
    BM25_EMOTION_TOPIC_PPR = 5
    EMOTION_TOPIC_ONLY = 6
    BM25_PPR = 7
    PPR_ONLY = 8
    BM25_EMOTION_TOPIC_PPR_WEIGHTED = 9

METHOD_NAMES = {
    MethodType.BM25_ONLY: "BM25 only",
    MethodType.BM25_EMOTION_ONLY: "BM25 + Emotion reranking",
    MethodType.BM25_TOPIC_ONLY: "BM25 + Topic reranking",
    MethodType.BM25_EMOTION_TOPIC: "BM25 + Emotion + Topic reranking",
    MethodType.BM25_EMOTION_TOPIC_PPR: "BM25 + Emotion + Topic + PPR",
    MethodType.EMOTION_TOPIC_ONLY: "Emotion + Topic only (no BM25)",
    MethodType.BM25_PPR: "BM25 + PPR (no reranking)",
    MethodType.PPR_ONLY: "PPR only (uniform)",
    MethodType.BM25_EMOTION_TOPIC_PPR_WEIGHTED: "BM25 + Emotion + Topic + PPR (weighted)",
}

############################################################
# 2. 預處理和向量計算
############################################################

def preprocess_post_text(post_text: str) -> Tuple[List[str], str]:
    """預處理貼文文本"""
    preprocessor = PostPreprocessor()
    clean_tokens = preprocessor.preprocess_text(post_text)
    return clean_tokens, post_text

def compute_query_emotion(
    post_text: str,
    tokens: List[str],
    nrc_lexicon: dict,
    emoji_table: dict,
    emotion_model,
    weight_lexicon: float = 0.04,
    weight_emoji: float = 0.06,
    weight_classifier: float = 0.90,
    conditional_fusion: bool = True
) -> np.ndarray:
    """
    計算貼文的情緒向量（融合）
    
    Args:
        conditional_fusion: 如果 True，當沒有 emoji 時會排除 emoji 信號並重新分配權重
    """
    # 1. Lexicon 情緒
    from src.emotion.posts_emotion_lex import compute_post_emotion, EMOTION2IDX
    lex_emotion = compute_post_emotion(tokens, nrc_lexicon)
    
    # 2. Emoji 情緒
    emoji_emotion, emoji_used, _ = post_emoji_emotion(post_text, emoji_table)
    
    # 3. ML 模型情緒
    post_text_for_model = " ".join(tokens)
    model_probs = emotion_model.predict_proba([post_text_for_model])[0]
    model_emotion = model_probs.astype(np.float32)
    
    # 4. 檢查是否有實際信號（排除 neutral fallback）
    # Lexicon: 檢查是否為全 neutral（即 sum == 0 後的 fallback）
    neutral_idx = EMOTION2IDX["neutral"]  # neutral 是索引 7
    has_lexicon_signal = not (lex_emotion[neutral_idx] == 1.0 and lex_emotion.sum() == 1.0)
    
    # Emoji: 檢查是否有 emoji 且被 emoji_table 覆蓋
    has_emoji_signal = emoji_used > 0
    
    # 5. 條件融合：排除 fallback 情況
    if conditional_fusion:
        # 收集有效的信號和對應的權重
        valid_signals = []
        valid_weights = []
        
        if has_lexicon_signal:
            valid_signals.append(lex_emotion)
            valid_weights.append(weight_lexicon)
        
        if has_emoji_signal:
            valid_signals.append(emoji_emotion)
            valid_weights.append(weight_emoji)
        
        # ML 模型總是有效（因為它總是有預測）
        valid_signals.append(model_emotion)
        valid_weights.append(weight_classifier)
        
        # 重新正規化權重
        total_weight = sum(valid_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in valid_weights]
            # 加權融合
            fused = np.zeros(len(EMOTIONS), dtype=np.float32)
            for signal, weight in zip(valid_signals, normalized_weights):
                fused += weight * signal
        else:
            # 如果所有權重都是 0（理論上不應該發生），只用 ML
            fused = model_emotion
    else:
        # 不需要條件融合時，使用原始權重
        fused = (
            weight_lexicon * lex_emotion +
            weight_emoji * emoji_emotion +
            weight_classifier * model_emotion
        )
    
    # 正規化
    s = fused.sum()
    if s > 0:
        fused = fused / s
    else:
        fused = np.zeros(8)
        fused[7] = 1.0  # neutral
    
    return fused

def compute_query_topic(
    tokens: List[str],
    lyrics_vectorizer,
    lyrics_kmeans,
    cluster_mapping: Dict[int, int],
    K_merged: int
) -> np.ndarray:
    """計算貼文的主題向量"""
    post_text = " ".join(tokens)
    X_post = lyrics_vectorizer.transform([post_text])
    topic_label_original = lyrics_kmeans.predict(X_post)[0]
    topic_label_merged = cluster_mapping.get(int(topic_label_original), 0)
    
    topic_vec = np.zeros(K_merged, dtype=np.float32)
    if 0 <= topic_label_merged < K_merged:
        topic_vec[topic_label_merged] = 1.0
    
    return topic_vec

############################################################
# 3. 載入所有模型和資料
############################################################

def load_song_metadata(metadata_path: pathlib.Path) -> Dict[str, Dict]:
    """載入歌曲 metadata"""
    if not metadata_path.exists():
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
    return metadata

def load_all_models_and_data():
    """載入所有必要的模型和資料"""
    print("[Load] Loading all models and data...")
    
    # 1. 情緒相關
    nrc_lexicon = load_nrc_lexicon(str(NRC_LEXICON_PATH))
    
    if EMOJI_TABLE_PATH.exists():
        emoji_table = np.load(str(EMOJI_TABLE_PATH), allow_pickle=True).item()
    else:
        from src.emotion.posts_emotion_emoji import load_emoji_joined
        emo2phrases = load_emoji_joined(str(EMOJI_JOINED_PATH))
        from src.emotion.posts_emotion_emoji import build_emoji_emotion_table
        emoji_table = build_emoji_emotion_table(emo2phrases, nrc_lexicon)
    
    emotion_model = joblib.load(str(EMOTION_MODEL_PATH))
    
    # 2. 主題相關
    lyrics_kmeans = load_lyrics_kmeans_model(str(LYRICS_KMEANS_MODEL_PATH))
    lyrics_vectorizer = load_lyrics_vectorizer(str(LYRICS_VECTORIZER_PATH))
    
    # 載入 cluster mapping
    if not MERGE_INFO_PATH.exists():
        cluster_mapping = {i: i for i in range(29)}
        K_merged = 29
    else:
        with open(MERGE_INFO_PATH, "r", encoding="utf-8") as f:
            merge_info = json.load(f)
        K_merged = merge_info.get("K_merged", 22)  # ✅ 默认值更新为22（当前合并后的主题数）
        groups = merge_info.get("groups", [])
        cluster_mapping = {}
        for merged_id, group in enumerate(groups):
            for original_id in group:
                cluster_mapping[original_id] = merged_id
    
    # 3. BM25 相關
    with open(BM25_METADATA_PATH, "r", encoding="utf-8") as f:
        bm25_metadata = json.load(f)
    bm25_matrix = load_npz(str(BM25_MATRIX_PATH))
    
    with open(BM25_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(BM25_IDF_PATH, "r", encoding="utf-8") as f:
        idf = json.load(f)
    
    with open(SONG_IDS_PATH, "r", encoding="utf-8") as f:
        song_ids = json.load(f)
    
    # 4. 圖譜和歌曲向量
    graph = load_npz(str(GRAPH_PATH))
    
    song_emotion = np.load(str(SONG_EMOTION_PATH), allow_pickle=True)
    if song_emotion.dtype == object:
        emotion_dict = song_emotion.item() if song_emotion.ndim == 0 else song_emotion
        song_emotion = np.array([emotion_dict.get(sid, np.zeros(8)) for sid in song_ids])
    
    song_topic = np.load(str(SONG_TOPIC_PATH), allow_pickle=True)
    
    # 5. 歌曲 metadata
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
# 4. 各種推薦方法實作
############################################################

def method_bm25_only(
    tokens: List[str],
    models_and_data: dict,
    top_k: int = 10,
    bm25_top_k: int = 1000
) -> List[Dict]:
    """方法 1: BM25 only"""
    bm25_scores = compute_bm25_score_for_query(
        tokens,
        models_and_data["bm25_matrix"],
        models_and_data["vocab"],
        models_and_data["idf"],
        models_and_data["bm25_metadata"],
        models_and_data["song_ids"]
    )
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        recommendations.append({
            "rank": rank,
            "song_id": models_and_data["song_ids"][idx],
            "bm25_score": float(bm25_scores[idx]),
            "method": "BM25 only",
        })
    return recommendations

def method_bm25_emotion_only(
    tokens: List[str],
    query_emotion: np.ndarray,
    models_and_data: dict,
    top_k: int = 10,
    bm25_top_k: int = 1000
) -> List[Dict]:
    """方法 2: BM25 + Emotion only reranking"""
    # BM25 篩選 top-1000 候選
    bm25_scores = compute_bm25_score_for_query(
        tokens,
        models_and_data["bm25_matrix"],
        models_and_data["vocab"],
        models_and_data["idf"],
        models_and_data["bm25_metadata"],
        models_and_data["song_ids"]
    )
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k]
    
    # 在候選內計算 Emotion 相似度
    candidate_emotion = models_and_data["song_emotion"][bm25_top_indices]
    emotion_sim = cosine_similarity(
        query_emotion.reshape(1, -1),
        candidate_emotion
    )[0]
    
    # 只按相似度排序（不混合 BM25 分數）
    top_indices_in_candidates = np.argsort(emotion_sim)[::-1][:top_k]
    top_indices = bm25_top_indices[top_indices_in_candidates]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        candidate_idx = np.where(bm25_top_indices == idx)[0][0]
        recommendations.append({
            "rank": rank,
            "song_id": models_and_data["song_ids"][idx],
            "bm25_score": float(bm25_scores[idx]),
            "emotion_sim": float(emotion_sim[candidate_idx]),
            "method": "BM25 + Emotion reranking",
        })
    return recommendations

def method_bm25_topic_only(
    tokens: List[str],
    query_topic: np.ndarray,
    models_and_data: dict,
    top_k: int = 10,
    bm25_top_k: int = 1000
) -> List[Dict]:
    """方法 3: BM25 + Topic only reranking"""
    # BM25 篩選 top-1000 候選
    bm25_scores = compute_bm25_score_for_query(
        tokens,
        models_and_data["bm25_matrix"],
        models_and_data["vocab"],
        models_and_data["idf"],
        models_and_data["bm25_metadata"],
        models_and_data["song_ids"]
    )
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k]
    
    # 在候選內計算 Topic 相似度
    candidate_topic = models_and_data["song_topic"][bm25_top_indices]
    topic_sim = cosine_similarity(
        query_topic.reshape(1, -1),
        candidate_topic
    )[0]
    
    # 只按相似度排序（不混合 BM25 分數）
    top_indices_in_candidates = np.argsort(topic_sim)[::-1][:top_k]
    top_indices = bm25_top_indices[top_indices_in_candidates]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        candidate_idx = np.where(bm25_top_indices == idx)[0][0]
        recommendations.append({
            "rank": rank,
            "song_id": models_and_data["song_ids"][idx],
            "bm25_score": float(bm25_scores[idx]),
            "topic_sim": float(topic_sim[candidate_idx]),
            "method": "BM25 + Topic reranking",
        })
    return recommendations

def method_bm25_emotion_topic(
    tokens: List[str],
    query_emotion: np.ndarray,
    query_topic: np.ndarray,
    models_and_data: dict,
    top_k: int = 10,
    bm25_top_k: int = 1000,
    emotion_weight: float = 0.6,
    topic_weight: float = 0.4
) -> List[Dict]:
    """方法 4: BM25 + Emotion + Topic reranking"""
    # BM25 篩選
    bm25_scores = compute_bm25_score_for_query(
        tokens,
        models_and_data["bm25_matrix"],
        models_and_data["vocab"],
        models_and_data["idf"],
        models_and_data["bm25_metadata"],
        models_and_data["song_ids"]
    )
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k]
    
    # Emotion + Topic similarity reranking
    candidate_emotion = models_and_data["song_emotion"][bm25_top_indices]
    candidate_topic = models_and_data["song_topic"][bm25_top_indices]
    
    emotion_sim = cosine_similarity(
        query_emotion.reshape(1, -1),
        candidate_emotion
    )[0]
    
    topic_sim = cosine_similarity(
        query_topic.reshape(1, -1),
        candidate_topic
    )[0]
    
    # 組合相似度
    combined_sim = emotion_weight * emotion_sim + topic_weight * topic_sim
    
    top_indices_in_candidates = np.argsort(combined_sim)[::-1][:top_k]
    top_indices = bm25_top_indices[top_indices_in_candidates]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        candidate_idx = np.where(bm25_top_indices == idx)[0][0]
        recommendations.append({
            "rank": rank,
            "song_id": models_and_data["song_ids"][idx],
            "bm25_score": float(bm25_scores[idx]),
            "emotion_sim": float(emotion_sim[candidate_idx]),
            "topic_sim": float(topic_sim[candidate_idx]),
            "combined_sim": float(combined_sim[candidate_idx]),
            "method": "BM25 + Emotion + Topic reranking",
        })
    return recommendations

def method_bm25_emotion_topic_ppr(
    tokens: List[str],
    query_emotion: np.ndarray,
    query_topic: np.ndarray,
    models_and_data: dict,
    top_k: int = 10,
    bm25_top_k: int = 1000,
    emotion_weight: float = 0.6,
    topic_weight: float = 0.4,
    alpha: float = 0.3
) -> List[Dict]:
    """方法 5: BM25 + Emotion + Topic + PPR (完整流程)"""
    # BM25 篩選
    bm25_scores = compute_bm25_score_for_query(
        tokens,
        models_and_data["bm25_matrix"],
        models_and_data["vocab"],
        models_and_data["idf"],
        models_and_data["bm25_metadata"],
        models_and_data["song_ids"]
    )
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k]
    
    # Emotion + Topic similarity reranking
    candidate_emotion = models_and_data["song_emotion"][bm25_top_indices]
    candidate_topic = models_and_data["song_topic"][bm25_top_indices]
    
    emotion_sim = cosine_similarity(
        query_emotion.reshape(1, -1),
        candidate_emotion
    )[0]
    
    topic_sim = cosine_similarity(
        query_topic.reshape(1, -1),
        candidate_topic
    )[0]
    
    combined_sim = emotion_weight * emotion_sim + topic_weight * topic_sim
    
    # 建立 teleportation vector
    teleportation_full = np.zeros(len(models_and_data["song_ids"]))
    teleportation_full[bm25_top_indices] = combined_sim
    
    sim_sum = teleportation_full.sum()
    if sim_sum > 0:
        teleportation_full = teleportation_full / sim_sum
    else:
        teleportation_full[bm25_top_indices] = 1.0 / len(bm25_top_indices)
    
    # PPR
    ppr_scores = personalized_pagerank(
        models_and_data["graph"],
        teleportation_full,
        alpha=alpha
    )
    
    top_indices = np.argsort(ppr_scores)[::-1][:top_k]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        is_candidate = idx in bm25_top_indices
        bm25_idx_in_candidates = np.where(bm25_top_indices == idx)[0]
        
        recommendations.append({
            "rank": rank,
            "song_id": models_and_data["song_ids"][idx],
            "ppr_score": float(ppr_scores[idx]),
            "teleportation_score": float(teleportation_full[idx]),
            "bm25_score": float(bm25_scores[idx]) if is_candidate else 0.0,
            "emotion_sim": float(emotion_sim[bm25_idx_in_candidates[0]]) if len(bm25_idx_in_candidates) > 0 else 0.0,
            "topic_sim": float(topic_sim[bm25_idx_in_candidates[0]]) if len(bm25_idx_in_candidates) > 0 else 0.0,
            "combined_sim": float(combined_sim[bm25_idx_in_candidates[0]]) if len(bm25_idx_in_candidates) > 0 else 0.0,
            "is_bm25_candidate": bool(is_candidate),
            "method": "BM25 + Emotion + Topic + PPR",
        })
    return recommendations

def method_emotion_topic_only(
    query_emotion: np.ndarray,
    query_topic: np.ndarray,
    models_and_data: dict,
    top_k: int = 10,
    emotion_weight: float = 0.6,
    topic_weight: float = 0.4
) -> List[Dict]:
    """方法 6: Emotion + Topic only (no BM25)"""
    # 對全部歌曲計算相似度
    emotion_sim = cosine_similarity(
        query_emotion.reshape(1, -1),
        models_and_data["song_emotion"]
    )[0]
    
    topic_sim = cosine_similarity(
        query_topic.reshape(1, -1),
        models_and_data["song_topic"]
    )[0]
    
    combined_sim = emotion_weight * emotion_sim + topic_weight * topic_sim
    
    top_indices = np.argsort(combined_sim)[::-1][:top_k]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        recommendations.append({
            "rank": rank,
            "song_id": models_and_data["song_ids"][idx],
            "emotion_sim": float(emotion_sim[idx]),
            "topic_sim": float(topic_sim[idx]),
            "combined_sim": float(combined_sim[idx]),
            "method": "Emotion + Topic only (no BM25)",
        })
    return recommendations

def method_bm25_ppr(
    tokens: List[str],
    models_and_data: dict,
    top_k: int = 10,
    bm25_top_k: int = 1000,
    alpha: float = 0.3
) -> List[Dict]:
    """方法 7: BM25 + PPR (no reranking)"""
    # BM25 篩選
    bm25_scores = compute_bm25_score_for_query(
        tokens,
        models_and_data["bm25_matrix"],
        models_and_data["vocab"],
        models_and_data["idf"],
        models_and_data["bm25_metadata"],
        models_and_data["song_ids"]
    )
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k]
    
    # 直接用 BM25 分數作為 teleportation
    teleportation_full = np.zeros(len(models_and_data["song_ids"]))
    teleportation_full[bm25_top_indices] = bm25_scores[bm25_top_indices]
    
    score_sum = teleportation_full.sum()
    if score_sum > 0:
        teleportation_full = teleportation_full / score_sum
    else:
        teleportation_full[bm25_top_indices] = 1.0 / len(bm25_top_indices)
    
    # PPR
    ppr_scores = personalized_pagerank(
        models_and_data["graph"],
        teleportation_full,
        alpha=alpha
    )
    
    top_indices = np.argsort(ppr_scores)[::-1][:top_k]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        is_candidate = idx in bm25_top_indices
        recommendations.append({
            "rank": rank,
            "song_id": models_and_data["song_ids"][idx],
            "ppr_score": float(ppr_scores[idx]),
            "teleportation_score": float(teleportation_full[idx]),
            "bm25_score": float(bm25_scores[idx]) if is_candidate else 0.0,
            "is_bm25_candidate": bool(is_candidate),
            "method": "BM25 + PPR (no reranking)",
        })
    return recommendations

def method_ppr_only(
    query_emotion: np.ndarray,
    query_topic: np.ndarray,
    models_and_data: dict,
    top_k: int = 10,
    alpha: float = 0.3,
    uniform: bool = False
) -> List[Dict]:
    """方法 8: PPR only"""
    if uniform:
        # 均勻 teleportation
        teleportation_full = np.ones(len(models_and_data["song_ids"])) / len(models_and_data["song_ids"])
    else:
        # 基於 Emotion + Topic 相似度對全部歌曲
        emotion_sim = cosine_similarity(
            query_emotion.reshape(1, -1),
            models_and_data["song_emotion"]
        )[0]
        
        topic_sim = cosine_similarity(
            query_topic.reshape(1, -1),
            models_and_data["song_topic"]
        )[0]
        
        combined_sim = 0.6 * emotion_sim + 0.4 * topic_sim
        teleportation_full = combined_sim / combined_sim.sum()
    
    # PPR
    ppr_scores = personalized_pagerank(
        models_and_data["graph"],
        teleportation_full,
        alpha=alpha
    )
    
    top_indices = np.argsort(ppr_scores)[::-1][:top_k]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        recommendations.append({
            "rank": rank,
            "song_id": models_and_data["song_ids"][idx],
            "ppr_score": float(ppr_scores[idx]),
            "teleportation_score": float(teleportation_full[idx]),
            "method": "PPR only (uniform)" if uniform else "PPR only (similarity-based)",
        })
    return recommendations

############################################################
# 5. 主函數
############################################################

def evaluate_methods(
    post_text: str,
    method_types: List[MethodType],
    models_and_data: dict,
    top_k: int = 10,
    bm25_top_k: int = 1000,
    emotion_weight: float = 0.6,
    topic_weight: float = 0.4,
    alpha: float = 0.3
) -> Dict[str, List[Dict]]:
    """評估多種方法"""
    print(f"\n[Query] Processing post: \"{post_text[:50]}...\"")
    
    # 預處理
    tokens, raw_text = preprocess_post_text(post_text)
    
    # 計算向量
    query_emotion = compute_query_emotion(
        raw_text,
        tokens,
        models_and_data["nrc_lexicon"],
        models_and_data["emoji_table"],
        models_and_data["emotion_model"]
    )
    
    query_topic = compute_query_topic(
        tokens,
        models_and_data["lyrics_vectorizer"],
        models_and_data["lyrics_kmeans"],
        models_and_data["cluster_mapping"],
        models_and_data["K_merged"]
    )
    
    results = {}
    
    for method_type in method_types:
        method_name = METHOD_NAMES[method_type]
        print(f"\n[Method] {method_name}...")
        
        try:
            if method_type == MethodType.BM25_ONLY:
                recs = method_bm25_only(tokens, models_and_data, top_k, bm25_top_k)
            elif method_type == MethodType.BM25_EMOTION_ONLY:
                recs = method_bm25_emotion_only(tokens, query_emotion, models_and_data, top_k, bm25_top_k)
            elif method_type == MethodType.BM25_TOPIC_ONLY:
                recs = method_bm25_topic_only(tokens, query_topic, models_and_data, top_k, bm25_top_k)
            elif method_type == MethodType.BM25_EMOTION_TOPIC:
                recs = method_bm25_emotion_topic(tokens, query_emotion, query_topic, models_and_data, top_k, bm25_top_k, emotion_weight, topic_weight)
            elif method_type == MethodType.BM25_EMOTION_TOPIC_PPR:
                recs = method_bm25_emotion_topic_ppr(tokens, query_emotion, query_topic, models_and_data, top_k, bm25_top_k, emotion_weight, topic_weight, alpha)
            elif method_type == MethodType.EMOTION_TOPIC_ONLY:
                recs = method_emotion_topic_only(query_emotion, query_topic, models_and_data, top_k, emotion_weight, topic_weight)
            elif method_type == MethodType.BM25_PPR:
                recs = method_bm25_ppr(tokens, models_and_data, top_k, bm25_top_k, alpha)
            elif method_type == MethodType.PPR_ONLY:
                recs = method_ppr_only(query_emotion, query_topic, models_and_data, top_k, alpha, uniform=False)
            elif method_type == MethodType.BM25_EMOTION_TOPIC_PPR_WEIGHTED:
                # 測試不同權重
                recs = method_bm25_emotion_topic_ppr(tokens, query_emotion, query_topic, models_and_data, top_k, bm25_top_k, 0.8, 0.2, alpha)
                recs[0]["method"] = "BM25 + Emotion + Topic + PPR (0.8:0.2)"
            else:
                continue
            
            results[method_name] = recs
            print(f"[OK] {method_name}: {len(recs)} recommendations")
            
        except Exception as e:
            print(f"[Error] {method_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def merge_and_deduplicate_results(
    results: Dict[str, List[Dict]],
    top_k_per_method: int = 5
) -> List[Dict]:
    """
    合併所有方法的結果，去重，並標註每首歌來自哪些方法。
    
    返回：
        merged_recs: List[Dict] - 每首歌包含：
            - song_id
            - title, artist, lyrics (from metadata)
            - methods: List[str] - 哪些方法推薦了這首歌
            - method_details: Dict[str, Dict] - 每個方法的詳細資訊（rank, scores等）
    """
    song_to_info = {}  # song_id -> {methods: [], method_details: {}, metadata: {}}
    
    # 收集所有方法的結果
    for method_name, recs in results.items():
        for rec in recs[:top_k_per_method]:  # 只取 top-K
            song_id = rec['song_id']
            
            if song_id not in song_to_info:
                song_to_info[song_id] = {
                    "song_id": song_id,
                    "methods": [],
                    "method_details": {},
                }
            
            # 記錄這個方法推薦了這首歌
            song_to_info[song_id]["methods"].append(method_name)
            
            # 記錄這個方法的詳細資訊
            song_to_info[song_id]["method_details"][method_name] = {
                "rank": rec.get("rank", 0),
                "bm25_score": rec.get("bm25_score", None),
                "emotion_sim": rec.get("emotion_sim", None),
                "topic_sim": rec.get("topic_sim", None),
                "combined_sim": rec.get("combined_sim", None),
                "combined_score": rec.get("combined_score", None),
                "ppr_score": rec.get("ppr_score", None),
                "teleportation_score": rec.get("teleportation_score", None),
            }
    
    # 轉換為列表
    merged_recs = list(song_to_info.values())
    
    # 按照被推薦的次數排序（被越多方法推薦的排在前面）
    merged_recs.sort(key=lambda x: len(x["methods"]), reverse=True)
    
    return merged_recs

def print_results(
    results: Dict[str, List[Dict]], 
    models_and_data: dict, 
    post_text: str,
    top_k_per_method: int = 5
):
    """輸出結果（兩種格式：方法別 + 合併評分列表）"""
    print("\n" + "=" * 80)
    print("推薦結果對照")
    print("=" * 80)
    print(f"\n貼文: \"{post_text}\"")
    
    # 1. 方法別的詳細輸出
    print(f"\n{'=' * 80}")
    print("方法別詳細結果（每種方法 top-{})".format(top_k_per_method))
    print(f"{'=' * 80}")
    
    for method_name, recs in results.items():
        print(f"\n方法: {method_name}")
        print("-" * 80)
        
        for rec in recs[:top_k_per_method]:
            song_id = rec['song_id']
            metadata = models_and_data["song_metadata"].get(song_id, {})
            title = metadata.get("title", "Unknown")
            artist = metadata.get("artist", "Unknown")
            
            print(f"  {rec['rank']:2d}. {title} - {artist} ({song_id})")
            
            # 根據方法顯示不同的分數
            score_parts = []
            if 'ppr_score' in rec:
                score_parts.append(f"PPR: {rec['ppr_score']:.10f}")
            if 'bm25_score' in rec and rec['bm25_score'] > 0:
                score_parts.append(f"BM25: {rec['bm25_score']:.4f}")
            if 'emotion_sim' in rec and rec['emotion_sim'] > 0:
                score_parts.append(f"Emotion: {rec['emotion_sim']:.4f}")
            if 'topic_sim' in rec and rec['topic_sim'] > 0:
                score_parts.append(f"Topic: {rec['topic_sim']:.4f}")
            if 'combined_sim' in rec and rec['combined_sim'] > 0:
                score_parts.append(f"Combined: {rec['combined_sim']:.4f}")
            if 'combined_score' in rec:
                score_parts.append(f"Score: {rec['combined_score']:.4f}")
            
            if score_parts:
                print(f"      {' | '.join(score_parts)}")
    
    # 2. 合併評分列表
    print(f"\n{'=' * 80}")
    print("合併評分列表（所有方法的 top-{} 結果，已去重）".format(top_k_per_method))
    print(f"{'=' * 80}")
    print("\n請對每首歌評分（例如：1-5 分，或 相關/不相關）\n")
    
    merged_recs = merge_and_deduplicate_results(results, top_k_per_method)
    
    for idx, rec in enumerate(merged_recs, 1):
        song_id = rec['song_id']
        metadata = models_and_data["song_metadata"].get(song_id, {})
        title = metadata.get("title", "Unknown")
        artist = metadata.get("artist", "Unknown")
        lyrics = metadata.get("lyrics", "")
        lyrics_preview = lyrics[:100] + "..." if len(lyrics) > 100 else lyrics
        if lyrics_preview:
            lyrics_preview = lyrics_preview.replace("\n", " ").replace("\r", " ")
        
        methods_str = ", ".join(rec["methods"])
        
        print(f"\n{'=' * 80}")
        print(f"歌曲 #{idx}: {title} - {artist}")
        print(f"Song ID: {song_id}")
        print(f"推薦方法: {methods_str} ({len(rec['methods'])} 種方法)")
        
        # 顯示每個方法的排名和分數
        print("\n各方法詳細資訊：")
        for method_name, details in rec["method_details"].items():
            print(f"  - {method_name}:")
            print(f"    排名: {details['rank']}")
            if details['bm25_score'] is not None:
                print(f"    BM25: {details['bm25_score']:.4f}")
            if details['emotion_sim'] is not None:
                print(f"    Emotion: {details['emotion_sim']:.4f}")
            if details['topic_sim'] is not None:
                print(f"    Topic: {details['topic_sim']:.4f}")
            if details['combined_sim'] is not None:
                print(f"    Combined: {details['combined_sim']:.4f}")
            if details['ppr_score'] is not None:
                print(f"    PPR: {details['ppr_score']:.10f}")
        
        if lyrics_preview:
            print(f"\n歌詞預覽: {lyrics_preview}")
        
        print(f"\n評分: _______ (請在此填寫分數)")
    
    print(f"\n{'=' * 80}")
    print(f"總共 {len(merged_recs)} 首不重複的歌曲（來自 {len(results)} 種方法）")
    print(f"{'=' * 80}")

def load_posts_from_file(file_path: str) -> List[Tuple[str, str]]:
    """
    從檔案載入多篇貼文。
    
    支援格式：
    - JSONL: 每行一個 JSON 物件，包含 "raw_text" 或 "text" 欄位
    - TXT: 每行一篇貼文
    
    返回：
        List[Tuple[post_id, post_text]]
    """
    posts = []
    
    if file_path.endswith('.jsonl'):
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # 嘗試多種可能的欄位名稱
                    post_text = obj.get("raw_text") or obj.get("text") or obj.get("content") or obj.get("post_text", "")
                    post_id = obj.get("post_id") or obj.get("id") or f"post_{idx}"
                    if post_text:
                        posts.append((post_id, post_text))
                except json.JSONDecodeError:
                    continue
    elif file_path.endswith('.txt'):
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    posts.append((f"post_{idx}", line))
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .jsonl or .txt")
    
    return posts

def main():
    parser = argparse.ArgumentParser(description="Evaluate different recommendation methods")
    parser.add_argument("--post", type=str, help="Single post text to query")
    parser.add_argument("--posts-file", type=str, help="File containing multiple posts (.jsonl or .txt)")
    parser.add_argument("--methods", type=str, default="1,2,3,4,5", 
                       help="Comma-separated method IDs (1-9) or 'all'")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations per method (default: 5 for evaluation)")
    parser.add_argument("--bm25-top-k", type=int, default=1000, help="BM25 candidate size")
    parser.add_argument("--emotion-weight", type=float, default=0.6, help="Emotion weight in reranking")
    parser.add_argument("--topic-weight", type=float, default=0.4, help="Topic weight in reranking")
    parser.add_argument("--alpha", type=float, default=0.3, help="PPR alpha (teleportation probability)")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--csv", type=str, help="Output CSV file path for rating (will append post_id if multiple posts)")
    parser.add_argument("--output-dir", type=str, help="Output directory for multiple posts (creates separate files)")
    
    args = parser.parse_args()
    
    # 獲取貼文列表
    posts_to_evaluate = []
    
    if args.posts_file:
        # 從檔案載入多篇貼文
        print(f"[Load] Loading posts from {args.posts_file}...")
        posts_to_evaluate = load_posts_from_file(args.posts_file)
        print(f"[Load] Loaded {len(posts_to_evaluate)} posts")
    elif args.post:
        # 單篇貼文
        posts_to_evaluate = [("single_post", args.post)]
    else:
        # 交互式輸入
        print("=" * 80)
        print("推薦方法對照評估")
        print("=" * 80)
        post_text = input("\n請輸入貼文文本: ").strip()
        if not post_text:
            print("錯誤：貼文文本不能為空")
            return
        posts_to_evaluate = [("interactive_post", post_text)]
    
    # 解析方法列表
    if args.methods.lower() == "all":
        method_types = list(MethodType)
    else:
        method_ids = [int(x.strip()) for x in args.methods.split(",")]
        method_types = [MethodType(mid) for mid in method_ids if mid in [m.value for m in MethodType]]
    
    if not method_types:
        print("錯誤：沒有有效的方法 ID")
        return
    
    # 載入模型和資料（只載入一次，所有貼文共用）
    models_and_data = load_all_models_and_data()
    
    # 批量評估
    all_results = {}
    
    for post_id, post_text in posts_to_evaluate:
        print(f"\n{'=' * 80}")
        print(f"評估貼文: {post_id}")
        print(f"{'=' * 80}")
        
        # 評估
        results = evaluate_methods(
            post_text,
            method_types,
            models_and_data,
            top_k=args.top_k,
            bm25_top_k=args.bm25_top_k,
            emotion_weight=args.emotion_weight,
            topic_weight=args.topic_weight,
            alpha=args.alpha
        )
        
        # 輸出結果（只在單篇貼文時顯示詳細輸出）
        if len(posts_to_evaluate) == 1:
            print_results(results, models_and_data, post_text, top_k_per_method=args.top_k)
        
        # 保存結果
        all_results[post_id] = {
            "post_text": post_text,
            "results": results
        }
        
        # 如果指定了輸出目錄，為每篇貼文建立單獨的檔案
        if args.output_dir:
            import os
            os.makedirs(args.output_dir, exist_ok=True)
            
            # JSON 輸出
            if args.output:
                output_path = os.path.join(args.output_dir, f"{post_id}_results.json")
            else:
                output_path = os.path.join(args.output_dir, f"{post_id}_results.json")
            
            merged_recs = merge_and_deduplicate_results(results, top_k_per_method=args.top_k)
            for rec in merged_recs:
                song_id = rec['song_id']
                metadata = models_and_data["song_metadata"].get(song_id, {})
                rec['title'] = metadata.get("title", "Unknown")
                rec['artist'] = metadata.get("artist", "Unknown")
                rec['lyrics'] = metadata.get("lyrics", "")
            
            output_data = {
                "post_id": post_id,
                "post_text": post_text,
                "config": {
                    "top_k_per_method": args.top_k,
                    "bm25_top_k": args.bm25_top_k,
                    "emotion_weight": args.emotion_weight,
                    "topic_weight": args.topic_weight,
                    "alpha": args.alpha,
                },
                "methods_detail": {name: recs for name, recs in results.items()},
                "merged_results": merged_recs,
                "summary": {
                    "num_methods": len(results),
                    "num_unique_songs": len(merged_recs),
                    "methods_used": list(results.keys()),
                }
            }
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Results saved to {output_path}")
            
            # CSV 輸出
            if args.csv:
                csv_path = os.path.join(args.output_dir, f"{post_id}_ratings.csv")
                _save_csv_for_post(post_id, post_text, results, merged_recs, models_and_data, csv_path, args.top_k)
    
    # 如果只有一篇貼文，使用原本的輸出邏輯
    if len(posts_to_evaluate) == 1:
        post_id, post_text = posts_to_evaluate[0]
        results = all_results[post_id]["results"]
        
        # 保存結果
        if args.output and not args.output_dir:
            # 合併結果
            merged_recs = merge_and_deduplicate_results(results, top_k_per_method=args.top_k)
            
            # 為每首歌添加 metadata
            for rec in merged_recs:
                song_id = rec['song_id']
                metadata = models_and_data["song_metadata"].get(song_id, {})
                rec['title'] = metadata.get("title", "Unknown")
                rec['artist'] = metadata.get("artist", "Unknown")
                rec['lyrics'] = metadata.get("lyrics", "")
            
            output_data = {
                "post_text": post_text,
                "config": {
                    "top_k_per_method": args.top_k,
                    "bm25_top_k": args.bm25_top_k,
                    "emotion_weight": args.emotion_weight,
                    "topic_weight": args.topic_weight,
                    "alpha": args.alpha,
                },
                "methods_detail": {name: recs for name, recs in results.items()},
                "merged_results": merged_recs,
                "summary": {
                    "num_methods": len(results),
                    "num_unique_songs": len(merged_recs),
                    "methods_used": list(results.keys()),
                }
            }
            
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n[OK] Results saved to {args.output}")
            print(f"     - {len(results)} 種方法")
            print(f"     - {len(merged_recs)} 首不重複歌曲")
            
            # 輸出 CSV 評分表
            if args.csv and not args.output_dir:
                _save_csv_for_post(post_id, post_text, results, merged_recs, models_and_data, args.csv, args.top_k)
    
    # 多篇貼文的批量輸出
    else:
        print(f"\n{'=' * 80}")
        print(f"批量評估完成：{len(posts_to_evaluate)} 篇貼文")
        print(f"{'=' * 80}")
        
        # 合併所有結果
        if args.output and not args.output_dir:
            all_output_data = {
                "num_posts": len(posts_to_evaluate),
                "config": {
                    "top_k_per_method": args.top_k,
                    "bm25_top_k": args.bm25_top_k,
                    "emotion_weight": args.emotion_weight,
                    "topic_weight": args.topic_weight,
                    "alpha": args.alpha,
                },
                "posts": {}
            }
            
            for post_id, post_text in posts_to_evaluate:
                results = all_results[post_id]["results"]
                merged_recs = merge_and_deduplicate_results(results, top_k_per_method=args.top_k)
                
                for rec in merged_recs:
                    song_id = rec['song_id']
                    metadata = models_and_data["song_metadata"].get(song_id, {})
                    rec['title'] = metadata.get("title", "Unknown")
                    rec['artist'] = metadata.get("artist", "Unknown")
                    rec['lyrics'] = metadata.get("lyrics", "")
                
                all_output_data["posts"][post_id] = {
                    "post_text": post_text,
                    "methods_detail": {name: recs for name, recs in results.items()},
                    "merged_results": merged_recs,
                    "summary": {
                        "num_methods": len(results),
                        "num_unique_songs": len(merged_recs),
                        "methods_used": list(results.keys()),
                    }
                }
            
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_output_data, f, ensure_ascii=False, indent=2)
            print(f"[OK] All results saved to {args.output}")
        
        # CSV 輸出（合併所有貼文到一個 CSV）
        if args.csv and not args.output_dir:
            import csv
            print(f"\n[CSV] Generating combined rating CSV file...")
            
            # 收集所有貼文的結果
            all_csv_rows = []
            headers = [
                "post_id", "song_id", "title", "artist",
                "BM25_only_rank", "BM25_Emotion_rank", "BM25_Topic_rank",
                "BM25_Emotion_Topic_rank", "BM25_Emotion_Topic_PPR_rank",
                "Emotion_Topic_only_rank", "BM25_PPR_rank", "PPR_only_rank",
                "rating_user1", "rating_user2", "rating_user3"
            ]
            
            method_to_header = {
                "BM25 only": "BM25_only_rank",
                "BM25 + Emotion reranking": "BM25_Emotion_rank",
                "BM25 + Topic reranking": "BM25_Topic_rank",
                "BM25 + Emotion + Topic reranking": "BM25_Emotion_Topic_rank",
                "BM25 + Emotion + Topic + PPR": "BM25_Emotion_Topic_PPR_rank",
                "Emotion + Topic only (no BM25)": "Emotion_Topic_only_rank",
                "BM25 + PPR (no reranking)": "BM25_PPR_rank",
                "PPR only (similarity-based)": "PPR_only_rank",
            }
            
            for post_id, post_text in posts_to_evaluate:
                results = all_results[post_id]["results"]
                merged_recs = merge_and_deduplicate_results(results, top_k_per_method=args.top_k)
                
                for rec in merged_recs:
                    song_id = rec['song_id']
                    metadata = models_and_data["song_metadata"].get(song_id, {})
                    title = metadata.get("title", "Unknown")
                    artist = metadata.get("artist", "Unknown")
                    
                    row = {
                        "post_id": post_id,
                        "song_id": song_id,
                        "title": title,
                        "artist": artist,
                    }
                    
                    for header in method_to_header.values():
                        row[header] = ""
                    
                    for method_name, details in rec["method_details"].items():
                        header_key = method_to_header.get(method_name)
                        if header_key:
                            row[header_key] = details["rank"]
                    
                    row["rating_user1"] = ""
                    row["rating_user2"] = ""
                    row["rating_user3"] = ""
                    
                    all_csv_rows.append(row)
            
            with open(args.csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(all_csv_rows)
            
            print(f"[OK] Combined rating CSV saved to {args.csv}")
            print(f"     - {len(posts_to_evaluate)} 篇貼文")
            print(f"     - {len(all_csv_rows)} 行資料（每篇貼文的歌曲）")
            print(f"     - 請在 rating_user1, rating_user2, rating_user3 欄位填入評分（1-5 分）")
            print(f"     - 填完後執行: python pipeline/calculate_scores.py {args.csv}")
        import csv
        csv_path = args.csv
        print(f"\n[CSV] Generating rating CSV file...")
        
        # 合併結果（如果還沒合併）
        if not args.output:
            merged_recs = merge_and_deduplicate_results(results, top_k_per_method=args.top_k)
            for rec in merged_recs:
                song_id = rec['song_id']
                metadata = models_and_data["song_metadata"].get(song_id, {})
                rec['title'] = metadata.get("title", "Unknown")
                rec['artist'] = metadata.get("artist", "Unknown")
        else:
            # 從 output_data 取得 merged_recs
            merged_recs = merge_and_deduplicate_results(results, top_k_per_method=args.top_k)
            for rec in merged_recs:
                song_id = rec['song_id']
                metadata = models_and_data["song_metadata"].get(song_id, {})
                rec['title'] = metadata.get("title", "Unknown")
                rec['artist'] = metadata.get("artist", "Unknown")
        
        # 建立 CSV 資料
        csv_rows = []
        
        # Header: 歌曲資訊 + 各方法的排名 + 評分欄位（可多個使用者）
        headers = [
            "song_id", "title", "artist",
            "BM25_only_rank", "BM25_Emotion_rank", "BM25_Topic_rank",
            "BM25_Emotion_Topic_rank", "BM25_Emotion_Topic_PPR_rank",
            "Emotion_Topic_only_rank", "BM25_PPR_rank", "PPR_only_rank",
            "rating_user1", "rating_user2", "rating_user3"
        ]
        
        # 建立方法名稱到 header 的映射
        method_to_header = {
            "BM25 only": "BM25_only_rank",
            "BM25 + Emotion reranking": "BM25_Emotion_rank",
            "BM25 + Topic reranking": "BM25_Topic_rank",
            "BM25 + Emotion + Topic reranking": "BM25_Emotion_Topic_rank",
            "BM25 + Emotion + Topic + PPR": "BM25_Emotion_Topic_PPR_rank",
            "Emotion + Topic only (no BM25)": "Emotion_Topic_only_rank",
            "BM25 + PPR (no reranking)": "BM25_PPR_rank",
            "PPR only (similarity-based)": "PPR_only_rank",
        }
        
        # 為每首歌曲建立一行
        for rec in merged_recs:
            song_id = rec['song_id']
            title = rec.get('title', 'Unknown')
            artist = rec.get('artist', 'Unknown')
            
            row = {
                "song_id": song_id,
                "title": title,
                "artist": artist,
            }
            
            # 初始化所有方法的排名為空
            for header in method_to_header.values():
                row[header] = ""
            
            # 填入各方法的排名
            for method_name, details in rec["method_details"].items():
                header_key = method_to_header.get(method_name)
                if header_key:
                    row[header_key] = details["rank"]
            
            # 評分欄位（預留 3 個使用者，可手動增加）
            row["rating_user1"] = ""
            row["rating_user2"] = ""
            row["rating_user3"] = ""
            
            csv_rows.append(row)
        
        # 寫入 CSV
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"[OK] Rating CSV saved to {csv_path}")
        print(f"     - {len(csv_rows)} 首歌曲")
        print(f"     - 請在 rating_user1, rating_user2, rating_user3 欄位填入評分（1-5 分）")
        print(f"     - 填完後執行: python pipeline/calculate_scores.py {csv_path}")

if __name__ == "__main__":
    main()

