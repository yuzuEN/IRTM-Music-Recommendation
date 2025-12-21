"""
組員 D — 歌曲相似度圖譜建構模組
=================================================

功能：
1. 讀取歌曲的 Emotion 向量和 Topic 向量
2. 計算歌曲間的 cosine similarity
3. 建立稀疏圖（每首歌保留 top-M 相似鄰居或相似度 > threshold）
4. 輸出稀疏 adjacency matrix

輸入：
  outputs/emotion_vectors/EmotionVec_lyrics.npy
  outputs/topic_vectors/TopicVec_lyrics_kmeans.npy
  outputs/bm25_vectors/song_ids.json

輸出：
  outputs/graph/song_graph.npz (稀疏 adjacency matrix)
  outputs/graph/graph_metadata.json
"""

import os
import json
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import pathlib


############################################################
# 0. PATH CONFIGURATION
############################################################

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# 輸入檔案
EMOTION_VEC_PATH = PROJECT_ROOT / "outputs" / "emotion_vectors" / "EmotionVec_lyrics.npy"
TOPIC_VEC_PATH = PROJECT_ROOT / "outputs" / "topic_vectors" / "TopicVec_lyrics_kmeans.npy"
SONG_IDS_PATH = PROJECT_ROOT / "outputs" / "bm25_vectors" / "song_ids.json"

# 輸出目錄
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "graph"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRAPH_OUTPUT_PATH = OUTPUT_DIR / "song_graph.npz"
METADATA_OUTPUT_PATH = OUTPUT_DIR / "graph_metadata.json"


############################################################
# 1. 載入向量
############################################################

def load_vectors(
    emotion_path: pathlib.Path,
    topic_path: pathlib.Path,
    song_ids_path: pathlib.Path
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    載入歌曲的 Emotion 和 Topic 向量。
    
    輸入：
        emotion_path: Emotion 向量檔案路徑
        topic_path: Topic 向量檔案路徑
        song_ids_path: 歌曲 ID 列表檔案路徑
        
    輸出：
        emotion_vectors: np.ndarray, shape (num_songs, 8)
        topic_vectors: np.ndarray, shape (num_songs, T)
        song_ids: List[str]
    """
    print(f"[Load] Loading vectors...")
    
    # 載入歌曲 ID 列表（先載入，用於對齊）
    with open(song_ids_path, "r", encoding="utf-8") as f:
        song_ids = json.load(f)
    print(f"[Load] Song IDs: {len(song_ids)} songs")
    
    # 載入 Emotion 向量（可能是字典格式）
    emotion_data = np.load(str(emotion_path), allow_pickle=True)
    
    # 檢查是否是字典格式
    if isinstance(emotion_data, dict) or (isinstance(emotion_data, np.ndarray) and emotion_data.dtype == object):
        # 如果是字典（或 object array 包含字典），需要轉換成陣列
        if isinstance(emotion_data, np.ndarray):
            emotion_dict = emotion_data.item() if emotion_data.ndim == 0 else emotion_data
        else:
            emotion_dict = emotion_data
        
        # 根據 song_ids 順序建立陣列
        emotion_vectors = []
        for song_id in song_ids:
            if song_id in emotion_dict:
                emotion_vectors.append(emotion_dict[song_id])
            else:
                # 如果某首歌沒有 emotion 向量，使用零向量
                emotion_vectors.append(np.zeros(8))
        emotion_vectors = np.array(emotion_vectors)
    else:
        # 已經是陣列格式
        emotion_vectors = emotion_data
    
    print(f"[Load] Emotion vectors: {emotion_vectors.shape}")
    
    # 載入 Topic 向量
    topic_vectors = np.load(str(topic_path), allow_pickle=True)
    print(f"[Load] Topic vectors: {topic_vectors.shape}")
    
    # 檢查維度是否一致
    num_songs = len(song_ids)
    if emotion_vectors.shape[0] != num_songs:
        raise ValueError(
            f"Emotion vectors shape[0] ({emotion_vectors.shape[0]}) "
            f"!= number of songs ({num_songs})"
        )
    if topic_vectors.shape[0] != num_songs:
        raise ValueError(
            f"Topic vectors shape[0] ({topic_vectors.shape[0]}) "
            f"!= number of songs ({num_songs})"
        )
    
    return emotion_vectors, topic_vectors, song_ids


############################################################
# 2. 向量融合（可選）
############################################################

def combine_vectors(
    emotion_vectors: np.ndarray,
    topic_vectors: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    將 Emotion 和 Topic 向量融合成單一向量。
    
    輸入：
        emotion_vectors: (num_songs, 8) - 8 維情緒向量
        topic_vectors: (num_songs, T) - T 維主題向量（例如 20 維）
        normalize: bool - 是否正規化（建議 True，讓兩個向量的權重平衡）
        
    輸出：
        combined_vectors: (num_songs, 8 + T) - 合併後的向量
        例如：如果 topic 是 20 維，則結果是 28 維
    
    說明：
        直接使用 np.hstack() 將兩個向量水平拼接：
        - 前 8 維是 emotion
        - 後 T 維是 topic
        
        如果 normalize=True，會先分別正規化兩個向量（L2 norm），
        讓它們在相似度計算中的權重更平衡。
    """
    if normalize:
        # 正規化每個向量（L2 norm），讓它們在相似度計算中權重平衡
        # 這樣做的好處：避免 emotion (8維) 和 topic (T維) 的維度差異影響相似度
        emotion_norm = emotion_vectors / (np.linalg.norm(emotion_vectors, axis=1, keepdims=True) + 1e-8)
        topic_norm = topic_vectors / (np.linalg.norm(topic_vectors, axis=1, keepdims=True) + 1e-8)
        
        # 拼接正規化後的向量
        combined = np.hstack([emotion_norm, topic_norm])
    else:
        # 直接拼接（不做正規化）
        combined = np.hstack([emotion_vectors, topic_vectors])
    
    return combined


############################################################
# 3. 計算 Cosine Similarity
############################################################

def compute_cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    計算所有歌曲間的 cosine similarity 矩陣。
    
    輸入：
        vectors: np.ndarray, shape (num_songs, D)
        
    輸出：
        similarity_matrix: np.ndarray, shape (num_songs, num_songs)
    """
    print(f"[Similarity] Computing cosine similarity matrix...")
    print(f"  Input shape: {vectors.shape}")
    
    # 使用 sklearn 的 cosine_similarity
    # 這會計算所有 pairs 的相似度
    similarity_matrix = cosine_similarity(vectors)
    
    print(f"[Similarity] Similarity matrix shape: {similarity_matrix.shape}")
    print(f"[Similarity] Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    
    return similarity_matrix


############################################################
# 4. 稀疏化（保留 top-M 鄰居或 threshold）
############################################################

def sparsify_graph(
    similarity_matrix: np.ndarray,
    strategy: str = "top_m",
    top_m: int = 20,
    threshold: float = 0.1
) -> csr_matrix:
    """
    將相似度矩陣稀疏化，避免全連結圖。
    
    輸入：
        similarity_matrix: (num_songs, num_songs) 的相似度矩陣
        strategy: "top_m" 或 "threshold"
        top_m: 每首歌保留前 M 個最相似的鄰居
        threshold: 相似度門檻（只保留 > threshold 的邊）
        
    輸出：
        sparse_graph: csr_matrix，稀疏 adjacency matrix
    """
    num_songs = similarity_matrix.shape[0]
    
    print(f"[Sparsify] Strategy: {strategy}")
    
    if strategy == "top_m":
        print(f"[Sparsify] Keeping top-{top_m} neighbors per song")
        
        # 對每首歌找出 top-M 最相似的（排除自己）
        rows = []
        cols = []
        data = []
        
        for i in range(num_songs):
            # 獲取第 i 首歌與所有歌的相似度
            similarities = similarity_matrix[i, :]
            
            # 排除自己（設為 -1）
            similarities[i] = -1
            
            # 找出 top-M
            top_indices = np.argsort(similarities)[::-1][:top_m]
            
            # 只保留相似度 > 0 的（避免負相關）
            for j in top_indices:
                sim = similarity_matrix[i, j]
                if sim > 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(sim)
    
    elif strategy == "threshold":
        print(f"[Sparsify] Keeping edges with similarity > {threshold}")
        
        # 找出所有 > threshold 的邊（排除對角線）
        mask = (similarity_matrix > threshold) & (np.eye(num_songs) == 0)
        rows, cols = np.where(mask)
        data = similarity_matrix[mask]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 建立稀疏矩陣
    sparse_graph = csr_matrix((data, (rows, cols)), shape=(num_songs, num_songs))
    
    # 計算統計資訊
    num_edges = sparse_graph.nnz
    avg_degree = num_edges / num_songs if num_songs > 0 else 0
    
    print(f"[Sparsify] Sparse graph created:")
    print(f"  Number of edges: {num_edges}")
    print(f"  Average degree: {avg_degree:.2f}")
    print(f"  Sparsity: {(1 - num_edges / (num_songs * num_songs)) * 100:.2f}%")
    
    return sparse_graph


############################################################
# 5. 儲存圖譜
############################################################

def save_graph(
    graph: csr_matrix,
    song_ids: List[str],
    metadata: Dict,
    graph_path: pathlib.Path,
    metadata_path: pathlib.Path
) -> None:
    """
    儲存稀疏圖譜和元資料。
    
    輸入：
        graph: csr_matrix，稀疏圖
        song_ids: List[str]，歌曲 ID 列表
        metadata: Dict，元資料
        graph_path: 圖譜檔案輸出路徑
        metadata_path: 元資料檔案輸出路徑
    """
    print(f"[Save] Saving graph to {graph_path}")
    
    # 儲存稀疏矩陣
    save_npz(str(graph_path), graph)
    
    # 儲存元資料
    metadata_with_ids = {
        **metadata,
        "song_ids": song_ids,
        "num_nodes": graph.shape[0],
        "num_edges": graph.nnz,
        "sparsity": 1 - (graph.nnz / (graph.shape[0] * graph.shape[1]))
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_with_ids, f, ensure_ascii=False, indent=2)
    
    print(f"[Save] Saved metadata to {metadata_path}")


############################################################
# 6. Main Pipeline
############################################################

def main():
    """
    主流程：建立歌曲相似度圖譜。
    """
    print("=" * 60)
    print("Similarity Graph Construction")
    print("=" * 60)
    
    # 檢查輸入檔案
    if not EMOTION_VEC_PATH.exists():
        raise FileNotFoundError(f"Emotion vectors not found: {EMOTION_VEC_PATH}")
    if not TOPIC_VEC_PATH.exists():
        raise FileNotFoundError(f"Topic vectors not found: {TOPIC_VEC_PATH}")
    if not SONG_IDS_PATH.exists():
        raise FileNotFoundError(f"Song IDs not found: {SONG_IDS_PATH}")
    
    # 1. 載入向量
    emotion_vectors, topic_vectors, song_ids = load_vectors(
        EMOTION_VEC_PATH,
        TOPIC_VEC_PATH,
        SONG_IDS_PATH
    )
    
    # 2. 融合向量（可選：可以只用 Emotion 或只用 Topic，或加權融合）
    # 選項 1: 只用 Emotion
    # combined_vectors = emotion_vectors
    
    # 選項 2: 只用 Topic
    # combined_vectors = topic_vectors
    
    # 選項 3: 拼接（推薦）
    # 直接將 emotion (8維) 和 topic (T維，例如20維) 拼接成 (8+T) 維向量
    # normalize=True 會先正規化兩個向量，讓它們在相似度計算中權重平衡
    combined_vectors = combine_vectors(
        emotion_vectors,
        topic_vectors,
        normalize=True  # 建議 True，讓 emotion 和 topic 的權重平衡
    )
    
    print(f"[Combine] Combined vectors shape: {combined_vectors.shape}")
    
    # 3. 計算相似度矩陣
    similarity_matrix = compute_cosine_similarity_matrix(combined_vectors)
    
    # 4. 稀疏化
    # 選項 1: top-M（推薦，效率高）
    sparse_graph = sparsify_graph(
        similarity_matrix,
        strategy="top_m",
        top_m=20  # 每首歌保留前 20 個最相似的鄰居
    )
    
    # 選項 2: threshold
    # sparse_graph = sparsify_graph(
    #     similarity_matrix,
    #     strategy="threshold",
    #     threshold=0.1
    # )
    
    # 5. 儲存
    metadata = {
        "emotion_vector_shape": list(emotion_vectors.shape),
        "topic_vector_shape": list(topic_vectors.shape),
        "combined_vector_shape": list(combined_vectors.shape),
        "sparsify_strategy": "top_m",
        "top_m": 20,
        "similarity_range": [float(similarity_matrix.min()), float(similarity_matrix.max())]
    }
    
    save_graph(
        sparse_graph,
        song_ids,
        metadata,
        GRAPH_OUTPUT_PATH,
        METADATA_OUTPUT_PATH
    )
    
    print("\n" + "=" * 60)
    print("✅ Graph Construction Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {GRAPH_OUTPUT_PATH}")
    print(f"  - {METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

