"""
BM25-only Query Demo (Stage 1: BM25 候選生成)
==============================================

功能：
1. 載入 BM25 artifacts（矩陣、vocabulary、IDF、metadata）
2. 從貼文資料讀取 query（使用 expanded_tokens）
3. 編碼 query 成 BM25 向量
4. 計算 BM25 分數，找出 Top-K 候選歌曲（候選列表 + 分數）
5. 輸出結果供驗證與 baseline 對照

用途：
- 驗證 Stage 1: BM25 語意檢索是否有效
- 提供 D 組員（Stage 2: Reranking + PPR）的 baseline 對照
- 可用於報告中的案例展示

注意：
- 這是「純 BM25」的結果（Stage 1 輸出）
- 後續 D 組員會在此基礎上加入情緒/主題 reranking 和 PPR
"""

import json
import numpy as np
from scipy.sparse import load_npz
# 不再使用 cosine_similarity，改用 BM25 分數（dot product）
import pathlib
from typing import List, Tuple
from collections import Counter

# Import encode_query_tokens from compute_bm25
import sys
sys.path.append(str(pathlib.Path(__file__).parent))
from compute_bm25 import encode_query_tokens


############################################################
# PATH CONFIGURATION
############################################################

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

BM25_DIR = PROJECT_ROOT / "outputs" / "bm25_vectors"
POSTS_PATH = PROJECT_ROOT / "data" / "processed" / "posts" / "posts_clean_expanded.jsonl"
CLEAN_LYRICS_PATH = PROJECT_ROOT / "data" / "processed" / "lyrics" / "clean_lyrics.json"


############################################################
# LOAD SONG METADATA
############################################################

def load_song_metadata(clean_lyrics_path: pathlib.Path) -> dict:
    """
    載入歌曲 metadata（title, artist），建立 song_id → metadata 的 mapping。
    
    輸出：
        song_metadata: Dict[str, dict] - song_id → {"title": str, "artist": str}
    """
    print("[Load] Loading song metadata...")
    
    with open(clean_lyrics_path, "r", encoding="utf-8") as f:
        songs = json.load(f)
    
    metadata = {}
    for song in songs:
        song_id = song["song_id"]
        metadata[song_id] = {
            "title": song.get("title", "Unknown"),
            "artist": song.get("artist", "Unknown")
        }
    
    print(f"[Load] Loaded metadata for {len(metadata)} songs")
    return metadata


############################################################
# LOAD BM25 ARTIFACTS
############################################################

def load_bm25_artifacts(bm25_dir: pathlib.Path) -> dict:
    """
    載入所有 BM25 artifacts。
    
    輸出：
        artifacts: dict 包含
            - song_ids: List[str]
            - vocab: Dict[str, int]
            - idf: Dict[str, float]
            - metadata: dict
            - bm25_matrix: csr_matrix
    """
    print("[Load] Loading BM25 artifacts...")
    
    # 1. song_ids.json
    with open(bm25_dir / "song_ids.json", "r", encoding="utf-8") as f:
        song_ids = json.load(f)
    
    # 2. vocabulary.json
    with open(bm25_dir / "vocabulary.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    # 3. idf.json
    with open(bm25_dir / "idf.json", "r", encoding="utf-8") as f:
        idf = json.load(f)
    
    # 4. metadata.json
    with open(bm25_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # 5. bm25_matrix.npz
    bm25_matrix = load_npz(bm25_dir / "bm25_matrix.npz")
    
    print(f"[Load] Loaded {len(song_ids)} songs, vocab size: {len(vocab)}")
    print(f"[Load] BM25 matrix shape: {bm25_matrix.shape}")
    
    return {
        "song_ids": song_ids,
        "vocab": vocab,
        "idf": idf,
        "metadata": metadata,
        "bm25_matrix": bm25_matrix
    }


############################################################
# LOAD POST QUERIES
############################################################

def load_post_queries(posts_path: pathlib.Path, num_queries: int = 5) -> List[dict]:
    """
    從 posts_clean_expanded.jsonl 載入幾筆貼文當 query。
    
    輸入：
        posts_path: pathlib.Path - posts_clean_expanded.jsonl 路徑
        num_queries: int - 要載入幾筆（預設 5）
        
    輸出：
        queries: List[dict] - 每筆包含 raw_text, expanded_tokens, emotion 等
    """
    queries = []
    
    with open(posts_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_queries:
                break
            query = json.loads(line.strip())
            queries.append(query)
    
    print(f"[Load] Loaded {len(queries)} post queries")
    return queries


############################################################
# BM25 QUERY & RANKING
############################################################

def query_bm25(
    query_tokens: List[str],
    artifacts: dict,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    對 query tokens 做 BM25 檢索，回傳 Top-K 推薦歌曲。
    
    輸入：
        query_tokens: List[str] - query tokens（例如 expanded_tokens）
        artifacts: dict - BM25 artifacts（從 load_bm25_artifacts 取得）
        top_k: int - 要回傳幾首（預設 10）
        
    輸出：
        results: List[Tuple[song_id, score]] - Top-K 歌曲與分數（降序）
    """
    vocab = artifacts["vocab"]
    idf = artifacts["idf"]
    metadata = artifacts["metadata"]
    bm25_matrix = artifacts["bm25_matrix"]
    
    # 編碼 query 成 BM25 向量
    query_vec = encode_query_tokens(
        query_tokens,
        vocab,
        idf,
        metadata["N"],
        metadata["avgdl"],
        k1=metadata["k1"],
        b=metadata["b"],
        mode="bm25"
    )
    
    # 計算 BM25 分數（用 dot product，不是 cosine similarity）
    # bm25_matrix 是 (num_songs, vocab_size)
    # query_vec 是 (vocab_size,)
    # 結果是 (num_songs,)
    scores = bm25_matrix.dot(query_vec)
    
    # 找出 Top-K
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # 組合成結果
    results = [
        (artifacts["song_ids"][idx], float(scores[idx]))
        for idx in top_indices
    ]
    
    return results


############################################################
# DISPLAY RESULTS
############################################################

def display_results(
    query: dict,
    results: List[Tuple[str, float]],
    song_metadata: dict,
    top_k: int = 10
) -> None:
    """
    顯示 query 與推薦結果（包含歌曲 title 和 artist）。
    """
    print("\n" + "=" * 80)
    print("QUERY:")
    print("-" * 80)
    print(f"Raw Text: {query['raw_text']}")
    print(f"Emotion: {query.get('emotion', 'N/A')} ({query.get('strength', 'N/A')})")
    print(f"Tokens (first 20): {query['expanded_tokens'][:20]}...")
    
    print("\n" + "=" * 80)
    print(f"TOP-{top_k} BM25 RECOMMENDATIONS:")
    print("-" * 80)
    
    for rank, (song_id, score) in enumerate(results, 1):
        # 取得歌曲 metadata
        meta = song_metadata.get(song_id, {})
        title = meta.get("title", "Unknown")
        artist = meta.get("artist", "Unknown")
        
        # 顯示格式：rank. title - artist (song_id) | Score
        display_str = f"{title} - {artist}"
        if len(display_str) > 50:
            display_str = display_str[:47] + "..."
        
        print(f"{rank:2d}. {display_str:50s} | Score: {score:.6f}")
        print(f"    ({song_id})")
    
    print("=" * 80)


############################################################
# MAIN DEMO
############################################################

def main():
    """
    主 demo：載入 artifacts、讀取幾筆貼文、做 BM25 檢索、顯示結果。
    """
    print("=" * 80)
    print("BM25-Only Query Demo")
    print("=" * 80)
    
    # 檢查檔案是否存在
    if not CLEAN_LYRICS_PATH.exists():
        raise FileNotFoundError(f"Clean lyrics file not found: {CLEAN_LYRICS_PATH}")
    if not BM25_DIR.exists():
        raise FileNotFoundError(f"BM25 directory not found: {BM25_DIR}")
    if not POSTS_PATH.exists():
        raise FileNotFoundError(f"Posts file not found: {POSTS_PATH}")
    
    # 1. 載入歌曲 metadata
    song_metadata = load_song_metadata(CLEAN_LYRICS_PATH)
    
    # 2. 載入 BM25 artifacts
    artifacts = load_bm25_artifacts(BM25_DIR)
    
    # 3. 載入幾筆貼文當 query
    queries = load_post_queries(POSTS_PATH, num_queries=5)
    
    # 3. 對每筆 query 做檢索
    print("\n" + "=" * 80)
    print("RUNNING BM25 QUERIES...")
    print("=" * 80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n>>> Query {i}/{len(queries)}")
        
        # 使用 expanded_tokens（已做 query expansion）
        query_tokens = query.get("expanded_tokens", query.get("clean_tokens", []))
        
        if not query_tokens:
            print("  [Skip] No tokens available")
            continue
        
        # BM25 檢索
        results = query_bm25(query_tokens, artifacts, top_k=10)
        
        # 顯示結果
        display_results(query, results, song_metadata, top_k=10)
    
    print("\n" + "=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)
    print("\nNote: This is Stage 1: BM25-only baseline (候選生成).")
    print("      D 組員可以此為 baseline，比較 Stage 2 (Reranking + PPR) 的效果。")


if __name__ == "__main__":
    main()

