"""
組員 D — Personalized PageRank 推薦模組
=================================================

功能：
1. 讀取相似度圖譜（song-to-song similarity graph）
2. 讀取查詢（post）的 emotion 和 topic 向量
3. 計算查詢與所有歌曲的 cosine similarity（作為 teleportation vector）
4. 執行 Personalized PageRank
5. 輸出推薦結果

輸入：
  outputs/graph/song_graph.npz (相似度圖譜)
  outputs/emotion_vectors/EmotionVec_lyrics.npy (歌曲情緒向量)
  outputs/emotion_vectors/EmotionVec_posts_model.npy (貼文情緒向量)
  outputs/topic_vectors/TopicVec_lyrics_kmeans.npy (歌曲主題向量)
  outputs/topic_vectors/posts_topic_assignments.jsonl (貼文主題分配)
  outputs/bm25_vectors/song_ids.json (歌曲 ID 列表)
  outputs/retrieval/bm25_topk.jsonl (BM25 候選，用於對齊 query_id)

輸出：
  outputs/recommendations/ppr_recommendations.jsonl (PPR 推薦結果)
"""

import os
import json
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import pathlib


############################################################
# 0. PATH CONFIGURATION
############################################################

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# 輸入檔案
GRAPH_PATH = PROJECT_ROOT / "outputs" / "graph" / "song_graph.npz"
SONG_EMOTION_PATH = PROJECT_ROOT / "outputs" / "emotion_vectors" / "EmotionVec_lyrics.npy"
SONG_TOPIC_PATH = PROJECT_ROOT / "outputs" / "topic_vectors" / "TopicVec_lyrics_kmeans.npy"
POST_EMOTION_PATH = PROJECT_ROOT / "outputs" / "emotion_vectors" / "EmotionVec_posts_model.npy"
POST_TOPIC_PATH = PROJECT_ROOT / "outputs" / "topic_vectors" / "posts_topic_assignments.jsonl"
SONG_IDS_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "song_ids.json"
BM25_TOPK_PATH = PROJECT_ROOT / "outputs" / "retrieval" / "bm25_topk.jsonl"

# 輸出目錄
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "recommendations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RECOMMENDATIONS_OUTPUT_PATH = OUTPUT_DIR / "ppr_recommendations.jsonl"


############################################################
# 1. 載入資料
############################################################

def load_graph_and_vectors():
    """
    載入圖譜和所有向量。
    
    返回：
        graph: csr_matrix - 相似度圖譜
        song_emotion: np.ndarray (N, 8) - 歌曲情緒向量
        song_topic: np.ndarray (N, T) - 歌曲主題向量
        song_ids: List[str] - 歌曲 ID 列表
    """
    print("[Load] Loading graph and vectors...")
    
    # 載入圖譜
    graph = load_npz(str(GRAPH_PATH))
    print(f"[Load] Graph shape: {graph.shape}, edges: {graph.nnz}")
    
    # 載入歌曲 ID
    with open(SONG_IDS_PATH, "r", encoding="utf-8") as f:
        song_ids = json.load(f)
    print(f"[Load] Song IDs: {len(song_ids)} songs")
    
    # 載入歌曲情緒向量
    song_emotion = np.load(str(SONG_EMOTION_PATH), allow_pickle=True)
    if song_emotion.dtype == object:
        # 如果是字典格式，轉換成陣列
        emotion_dict = song_emotion.item() if song_emotion.ndim == 0 else song_emotion
        song_emotion = np.array([emotion_dict.get(sid, np.zeros(8)) for sid in song_ids])
    print(f"[Load] Song emotion vectors: {song_emotion.shape}")
    
    # 載入歌曲主題向量
    song_topic = np.load(str(SONG_TOPIC_PATH), allow_pickle=True)
    print(f"[Load] Song topic vectors: {song_topic.shape}")
    
    # 檢查維度一致性
    num_songs = len(song_ids)
    assert song_emotion.shape[0] == num_songs, f"Emotion vectors shape mismatch: {song_emotion.shape[0]} != {num_songs}"
    assert song_topic.shape[0] == num_songs, f"Topic vectors shape mismatch: {song_topic.shape[0]} != {num_songs}"
    assert graph.shape[0] == num_songs, f"Graph shape mismatch: {graph.shape[0]} != {num_songs}"
    
    return graph, song_emotion, song_topic, song_ids


def load_post_vectors():
    """
    載入貼文的情緒和主題向量。
    
    返回：
        post_emotion: np.ndarray (M, 8) - 貼文情緒向量
        post_topic_assignments: List[Dict] - 貼文主題分配（包含 cluster_id）
    """
    print("[Load] Loading post vectors...")
    
    # 載入貼文情緒向量
    post_emotion = np.load(str(POST_EMOTION_PATH), allow_pickle=True)
    print(f"[Load] Post emotion vectors: {post_emotion.shape}")
    
    # 載入貼文主題分配
    post_topic_assignments = []
    with open(POST_TOPIC_PATH, "r", encoding="utf-8") as f:
        for line in f:
            post_topic_assignments.append(json.loads(line))
    print(f"[Load] Post topic assignments: {len(post_topic_assignments)} posts")
    
    return post_emotion, post_topic_assignments


def load_cluster_mapping() -> Dict[int, int]:
    """
    載入 cluster merge mapping（將原始 29 個 clusters 映射到合併後的 20 個）。
    
    返回：
        mapping: Dict[original_cluster_id, merged_cluster_id]
    """
    merge_info_path = PROJECT_ROOT / "outputs" / "topic_vectors" / "lyrics_cluster_merge_info.json"
    
    if not merge_info_path.exists():
        # 如果沒有 merge info，假設沒有合併（1-to-1 mapping）
        return {i: i for i in range(29)}
    
    with open(merge_info_path, "r", encoding="utf-8") as f:
        merge_info = json.load(f)
    
    # 建立 mapping：original_cluster_id -> merged_cluster_id
    mapping = {}
    for merged_id, group in enumerate(merge_info["groups"]):
        for original_id in group:
            mapping[original_id] = merged_id
    
    return mapping


def cluster_id_to_topic_vector(
    cluster_id: int,
    cluster_mapping: Dict[int, int],
    num_clusters: int = 20
) -> np.ndarray:
    """
    將原始 cluster_id 轉換成合併後的 one-hot 主題向量。
    
    參數：
        cluster_id: 原始主題 cluster ID (0-28)
        cluster_mapping: 原始 cluster_id 到合併 cluster_id 的映射
        num_clusters: 合併後的主題數（預設 20）
    
    返回：
        topic_vector: np.ndarray (num_clusters,) - one-hot 向量
    """
    # 映射到合併後的 cluster_id
    merged_cluster_id = cluster_mapping.get(cluster_id, 0)
    
    # 建立 one-hot 向量
    vec = np.zeros(num_clusters)
    if 0 <= merged_cluster_id < num_clusters:
        vec[merged_cluster_id] = 1.0
    
    return vec


############################################################
# 2. 建立 Teleportation Vector（查詢-歌曲相似度）
############################################################

def create_post_song_similarity_teleportation(
    query_emotion: np.ndarray,
    query_topic: np.ndarray,
    song_emotion: np.ndarray,
    song_topic: np.ndarray,
    emotion_weight: float = 0.6,
    topic_weight: float = 0.4
) -> np.ndarray:
    """
    計算查詢（post）與所有歌曲的相似度，作為 teleportation vector。
    
    參數：
        query_emotion: (8,) - 查詢情緒向量
        query_topic: (T,) - 查詢主題向量
        song_emotion: (N, 8) - 所有歌曲的情緒向量
        song_topic: (N, T) - 所有歌曲的主題向量
        emotion_weight: 情緒相似度權重
        topic_weight: 主題相似度權重
    
    返回：
        v: (N,) - teleportation vector（已正規化為機率分布）
    """
    # 計算情緒相似度
    emotion_sim = cosine_similarity(
        query_emotion.reshape(1, -1),
        song_emotion
    )[0]  # (N,)
    
    # 計算主題相似度
    topic_sim = cosine_similarity(
        query_topic.reshape(1, -1),
        song_topic
    )[0]  # (N,)
    
    # 組合相似度
    combined_sim = emotion_weight * emotion_sim + topic_weight * topic_sim
    
    # 正規化為機率分布
    sim_sum = combined_sim.sum()
    if sim_sum > 0:
        v = combined_sim / sim_sum
    else:
        # Fallback: 如果所有相似度都是 0，使用均勻分布
        v = np.ones(len(combined_sim)) / len(combined_sim)
    
    return v


############################################################
# 3. Personalized PageRank 演算法
############################################################

def personalized_pagerank(
    graph: csr_matrix,
    teleportation_vector: np.ndarray,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> np.ndarray:
    """
    執行 Personalized PageRank。
    
    參數：
        graph: csr_matrix (N, N) - 相似度圖譜
        teleportation_vector: np.ndarray (N,) - teleportation 向量（已正規化）
        alpha: float - damping factor（跳回 teleportation 的機率）
        max_iter: int - 最大迭代次數
        tol: float - 收斂容忍度
    
    返回：
        ppr_scores: np.ndarray (N,) - PPR 分數
    """
    num_nodes = graph.shape[0]
    
    # 正規化圖譜（row-stochastic）
    row_sums = graph.sum(axis=1).A1  # 轉換為 1D array
    row_sums[row_sums == 0] = 1  # 處理孤立節點（沒有出邊）
    graph_normalized = graph.multiply(1.0 / row_sums[:, np.newaxis])
    
    # 初始化 rank vector
    r = np.ones(num_nodes) / num_nodes
    
    # 迭代直到收斂
    for iteration in range(max_iter):
        # PPR 更新：r = (1-α) * M * r + α * v
        r_new = (1 - alpha) * graph_normalized.dot(r) + alpha * teleportation_vector
        
        # 檢查收斂
        diff = np.linalg.norm(r_new - r)
        if diff < tol:
            print(f"[PPR] Converged in {iteration + 1} iterations (diff={diff:.2e})")
            break
        
        r = r_new
    
    if iteration == max_iter - 1:
        print(f"[PPR] Reached max iterations ({max_iter}), final diff={diff:.2e}")
    
    return r


############################################################
# 4. 產生推薦結果
############################################################

def generate_recommendations_for_query(
    query_id: str,
    query_idx: int,
    query_emotion: np.ndarray,
    query_topic: np.ndarray,
    graph: csr_matrix,
    song_emotion: np.ndarray,
    song_topic: np.ndarray,
    song_ids: List[str],
    top_k: int = 10,
    alpha: float = 0.85
) -> Dict:
    """
    為單一查詢產生推薦結果。
    
    參數：
        query_id: 查詢 ID（例如 "post_0"）
        query_idx: 查詢索引（用於對齊 emotion/topic 向量）
        query_emotion: (8,) - 查詢情緒向量
        query_topic: (T,) - 查詢主題向量
        graph: 相似度圖譜
        song_emotion: (N, 8) - 歌曲情緒向量
        song_topic: (N, T) - 歌曲主題向量
        song_ids: 歌曲 ID 列表
        top_k: 返回前 K 個推薦
        alpha: PPR damping factor
    
    返回：
        result: Dict - 包含推薦結果
    """
    # 建立 teleportation vector（查詢-歌曲相似度）
    v = create_post_song_similarity_teleportation(
        query_emotion,
        query_topic,
        song_emotion,
        song_topic
    )
    
    # 執行 PPR
    ppr_scores = personalized_pagerank(
        graph,
        v,
        alpha=alpha
    )
    
    # 取得 top-K 推薦
    top_indices = np.argsort(ppr_scores)[::-1][:top_k]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        recommendations.append({
            "rank": rank,
            "song_id": song_ids[idx],
            "ppr_score": float(ppr_scores[idx])
        })
    
    return {
        "query_id": query_id,
        "query_idx": query_idx,
        "recommendations": recommendations,
        "num_songs": len(song_ids),
        "alpha": alpha
    }


############################################################
# 5. Main Pipeline
############################################################

def main():
    """
    主流程：為所有查詢產生 PPR 推薦結果。
    """
    print("=" * 60)
    print("Personalized PageRank Recommendation")
    print("=" * 60)
    
    # 檢查輸入檔案
    required_files = [
        GRAPH_PATH, SONG_EMOTION_PATH, SONG_TOPIC_PATH,
        POST_EMOTION_PATH, POST_TOPIC_PATH, SONG_IDS_PATH, BM25_TOPK_PATH
    ]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # 1. 載入圖譜和歌曲向量
    graph, song_emotion, song_topic, song_ids = load_graph_and_vectors()
    
    # 2. 載入貼文向量
    post_emotion, post_topic_assignments = load_post_vectors()
    
    # 3. 讀取 BM25 top-K 以取得查詢列表和對齊
    print("[Load] Loading BM25 top-K results for query alignment...")
    query_data_list = []
    with open(BM25_TOPK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            query_data_list.append(json.loads(line))
    print(f"[Load] Found {len(query_data_list)} queries")
    
    # 4. 載入 cluster mapping（將原始 29 clusters 映射到合併後的 20 clusters）
    print("\n[Load] Loading cluster mapping...")
    cluster_mapping = load_cluster_mapping()
    print(f"[Load] Cluster mapping: {len(cluster_mapping)} original clusters -> 20 merged clusters")
    
    # 5. 為每個查詢產生推薦
    print("\n[PPR] Generating recommendations...")
    results = []
    
    num_clusters = 20  # 與 lyrics topic vectors 的維度一致
    
    for query_data in query_data_list:
        query_id = query_data["query_id"]
        
        # 從 query_id 提取索引（例如 "post_0" -> 0）
        try:
            query_idx = int(query_id.split("_")[1])
        except (IndexError, ValueError):
            print(f"[Warning] Could not parse query_id: {query_id}, skipping")
            continue
        
        # 檢查索引範圍
        if query_idx >= len(post_emotion) or query_idx >= len(post_topic_assignments):
            print(f"[Warning] Query index {query_idx} out of range, skipping")
            continue
        
        # 取得查詢情緒向量
        q_emotion = post_emotion[query_idx]
        
        # 取得查詢主題向量（從 cluster_id 轉換）
        cluster_id = post_topic_assignments[query_idx]["cluster_id"]
        q_topic = cluster_id_to_topic_vector(cluster_id, cluster_mapping, num_clusters)
        
        # 產生推薦
        result = generate_recommendations_for_query(
            query_id=query_id,
            query_idx=query_idx,
            query_emotion=q_emotion,
            query_topic=q_topic,
            graph=graph,
            song_emotion=song_emotion,
            song_topic=song_topic,
            song_ids=song_ids,
            top_k=10,
            alpha=0.85
        )
        
        results.append(result)
        
        if len(results) % 100 == 0:
            print(f"[PPR] Processed {len(results)} queries...")
    
    # 6. 儲存結果
    print(f"\n[Save] Saving recommendations to {RECOMMENDATIONS_OUTPUT_PATH}")
    with open(RECOMMENDATIONS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 60)
    print("✅ Personalized PageRank Complete!")
    print("=" * 60)
    print(f"\nOutput file:")
    print(f"  - {RECOMMENDATIONS_OUTPUT_PATH}")
    print(f"\nStatistics:")
    print(f"  - Total queries processed: {len(results)}")
    print(f"  - Recommendations per query: 10")
    print(f"  - Total songs: {len(song_ids)}")


if __name__ == "__main__":
    main()

