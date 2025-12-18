"""
組員 C — BM25 語意向量計算（完整版本）
=================================================

功能：
1. 從 A 的前處理結果讀入歌詞 tokens
2. 建立 vocabulary 與統計量（DF、IDF、文件長度）
3. 計算 TF-IDF 與 BM25 向量矩陣
4. 輸出所有 artifacts（矩陣、vocabulary、metadata）
5. 提供 query encoding 函式（給後續 query / FinalVec 使用）

輸入：
  /data/processed/lyrics/lyrics_tokens.csv

輸出：
  /outputs/bm25_vectors/
    - tfidf_matrix.npz
    - bm25_matrix.npz
    - vocabulary.json
    - song_ids.json
    - idf.json
    - metadata.json
"""

import os
import json
import csv
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import pathlib
import math


############################################################
# 0. PATH CONFIGURATION
############################################################

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

LYRICS_TOKENS_PATH = PROJECT_ROOT / "data" / "processed" / "lyrics" / "lyrics_tokens.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bm25_vectors"
os.makedirs(OUTPUT_DIR, exist_ok=True)


############################################################
# 1. 讀取歌詞 tokens
############################################################

def load_lyrics_tokens(csv_path) -> Tuple[List[str], List[List[str]]]:
    """
    從 lyrics_tokens.csv 讀入所有歌詞，轉成文件列表與歌 ID 列表。
    
    輸入：
        csv_path: str 或 pathlib.Path - lyrics_tokens.csv 的路徑
        
    輸出：
        song_ids: List[str] - 歌曲 ID 列表（順序與 docs_tokens 對齊）
        docs_tokens: List[List[str]] - 每首歌的 tokens（list of token lists）
    """
    song_ids = []
    docs_tokens = []
    
    with open(str(csv_path), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = row["song_id"]
            tokens_str = row["tokens"]
            # tokens 是空白分隔的字串，轉成 list
            tokens = tokens_str.split()
            song_ids.append(song_id)
            docs_tokens.append(tokens)
    
    print(f"[Load] Loaded {len(song_ids)} songs from {csv_path}")
    return song_ids, docs_tokens


############################################################
# 2. 建立 vocabulary 與 DF 統計
############################################################

def build_vocabulary(
    docs_tokens: List[List[str]], 
    min_df: int = 5, 
    max_df_ratio: float = 0.5
) -> Tuple[Dict[str, int], Dict[str, int], int]:
    """
    建立 vocabulary（詞彙表）與 DF 統計，並做 pruning。
    
    輸入：
        docs_tokens: List[List[str]] - 所有歌曲的 tokens
        min_df: int - 最小 document frequency（低於此值的詞丟棄）
        max_df_ratio: float - 最大 document frequency 比例（超過此比例的詞丟棄）
        
    輸出：
        vocab: Dict[str, int] - 詞 → column index（0 ~ |V|-1）
        df: Dict[str, int] - 詞 → document frequency
        N: int - 總文件數（歌曲數）
    """
    N = len(docs_tokens)
    
    # 計算每個詞的 document frequency
    term_doc_count = defaultdict(int)
    for doc_tokens in docs_tokens:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            term_doc_count[token] += 1
    
    # Pruning: 過少或過常見的詞丟棄
    max_df = int(N * max_df_ratio)
    pruned_terms = {
        term: count 
        for term, count in term_doc_count.items() 
        if min_df <= count <= max_df
    }
    
    # 建立 vocabulary（詞 → index）
    vocab = {term: idx for idx, term in enumerate(sorted(pruned_terms.keys()))}
    
    # 保留 pruned 後的 df
    df = {term: pruned_terms[term] for term in vocab.keys()}
    
    print(f"[Vocab] Built vocabulary: {len(vocab)} terms (pruned from {len(term_doc_count)})")
    print(f"[Vocab] Pruning: min_df={min_df}, max_df_ratio={max_df_ratio} (max_df={max_df})")
    
    return vocab, df, N


############################################################
# 3. 計算文件長度統計
############################################################

def compute_length_stats(docs_tokens: List[List[str]]) -> Tuple[List[int], float]:
    """
    計算每首歌的長度（token 數）與平均長度。
    
    輸入：
        docs_tokens: List[List[str]] - 所有歌曲的 tokens
        
    輸出：
        doc_lengths: List[int] - 每首歌的 token 數（順序與 docs_tokens 對齊）
        avgdl: float - 所有歌詞長度的平均值
    """
    doc_lengths = [len(tokens) for tokens in docs_tokens]
    avgdl = np.mean(doc_lengths) if doc_lengths else 0.0
    
    print(f"[Length] Average document length: {avgdl:.2f} tokens")
    return doc_lengths, avgdl


############################################################
# 4. 計算 IDF
############################################################

def compute_idf(df: Dict[str, int], N: int, scheme: str = "bm25") -> Dict[str, float]:
    """
    根據 DF 與 N 計算 IDF。
    
    輸入：
        df: Dict[str, int] - 詞 → document frequency
        N: int - 總文件數
        scheme: str - "bm25" 或 "tfidf"
        
    輸出：
        idf: Dict[str, float] - 詞 → idf 值
    """
    idf = {}
    
    if scheme == "bm25":
        # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
        for term, df_val in df.items():
            idf[term] = math.log((N - df_val + 0.5) / (df_val + 0.5) + 1)
    elif scheme == "tfidf":
        # Standard TF-IDF IDF: log(N / (df + 1))
        for term, df_val in df.items():
            idf[term] = math.log(N / (df_val + 1))
    else:
        raise ValueError(f"Unknown IDF scheme: {scheme}")
    
    print(f"[IDF] Computed IDF for {len(idf)} terms (scheme: {scheme})")
    return idf


############################################################
# 5. 建立 TF-IDF 矩陣
############################################################

def build_tfidf_matrix(
    docs_tokens: List[List[str]], 
    vocab: Dict[str, int], 
    idf: Dict[str, float]
) -> csr_matrix:
    """
    建立「歌曲 × 詞彙」的 TF-IDF 稀疏矩陣。
    
    輸入：
        docs_tokens: List[List[str]] - 所有歌曲的 tokens
        vocab: Dict[str, int] - 詞 → column index
        idf: Dict[str, float] - 詞 → idf 值（TF-IDF 版本）
        
    輸出：
        tfidf_matrix: csr_matrix - 形狀 (num_songs, vocab_size)
    """
    num_songs = len(docs_tokens)
    vocab_size = len(vocab)
    
    # 建立稀疏矩陣的 row, col, data
    rows = []
    cols = []
    data = []
    
    for doc_idx, doc_tokens in enumerate(docs_tokens):
        # 計算這首歌的 term frequency
        term_freq = Counter(doc_tokens)
        doc_length = len(doc_tokens)
        
        # 只處理在 vocabulary 中的詞
        for term, tf in term_freq.items():
            if term in vocab:
                col_idx = vocab[term]
                # TF-IDF = (tf / doc_length) * idf
                tf_normalized = tf / doc_length if doc_length > 0 else 0
                tfidf_score = tf_normalized * idf[term]
                
                rows.append(doc_idx)
                cols.append(col_idx)
                data.append(tfidf_score)
    
    # 建立稀疏矩陣
    tfidf_matrix = csr_matrix((data, (rows, cols)), shape=(num_songs, vocab_size))
    
    print(f"[TF-IDF] Built TF-IDF matrix: {tfidf_matrix.shape}")
    return tfidf_matrix


############################################################
# 6. 建立 BM25 矩陣
############################################################

def build_bm25_matrix(
    docs_tokens: List[List[str]], 
    vocab: Dict[str, int], 
    idf: Dict[str, float],
    doc_lengths: List[int],
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75
) -> Tuple[csr_matrix, Dict[str, float]]:
    """
    建立 BM25 向量矩陣（每首歌對每個詞的 BM25 分數）。
    
    輸入：
        docs_tokens: List[List[str]] - 所有歌曲的 tokens
        vocab: Dict[str, int] - 詞 → column index
        idf: Dict[str, float] - 詞 → idf 值（BM25 版本）
        doc_lengths: List[int] - 每首歌的長度
        avgdl: float - 平均文件長度
        k1: float - BM25 參數 k1（預設 1.5）
        b: float - BM25 參數 b（預設 0.75）
        
    輸出：
        bm25_matrix: csr_matrix - 形狀 (num_songs, vocab_size)
        bm25_params: Dict[str, float] - 參數記錄
    """
    num_songs = len(docs_tokens)
    vocab_size = len(vocab)
    
    # 建立稀疏矩陣的 row, col, data
    rows = []
    cols = []
    data = []
    
    for doc_idx, doc_tokens in enumerate(docs_tokens):
        # 計算這首歌的 term frequency
        term_freq = Counter(doc_tokens)
        doc_length = doc_lengths[doc_idx]
        
        # 只處理在 vocabulary 中的詞
        for term, tf in term_freq.items():
            if term in vocab:
                col_idx = vocab[term]
                
                # BM25 formula:
                # BM25(w) = idf(w) * (tf(w) * (k1 + 1)) / (tf(w) + k1 * (1 - b + b * |d| / avgdl))
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_length / avgdl)
                bm25_score = idf[term] * (numerator / denominator)
                
                rows.append(doc_idx)
                cols.append(col_idx)
                data.append(bm25_score)
    
    # 建立稀疏矩陣
    bm25_matrix = csr_matrix((data, (rows, cols)), shape=(num_songs, vocab_size))
    
    bm25_params = {"k1": k1, "b": b}
    
    print(f"[BM25] Built BM25 matrix: {bm25_matrix.shape} (k1={k1}, b={b})")
    return bm25_matrix, bm25_params


############################################################
# 7. 儲存所有 artifacts
############################################################

def save_bm25_artifacts(
    output_dir: pathlib.Path,
    song_ids: List[str],
    vocab: Dict[str, int],
    tfidf_matrix: csr_matrix,
    bm25_matrix: csr_matrix,
    idf: Dict[str, float],
    N: int,
    doc_lengths: List[int],
    avgdl: float,
    k1: float,
    b: float,
    min_df: int,
    max_df_ratio: float
) -> None:
    """
    將所有 BM25 artifacts 存到 output_dir。
    
    輸入：
        output_dir: pathlib.Path - 輸出目錄
        song_ids: List[str] - 歌曲 ID 列表
        vocab: Dict[str, int] - vocabulary
        tfidf_matrix: csr_matrix - TF-IDF 矩陣
        bm25_matrix: csr_matrix - BM25 矩陣
        idf: Dict[str, float] - IDF 值
        N: int - 總文件數
        doc_lengths: List[int] - 每首歌的長度
        avgdl: float - 平均文件長度
        k1, b: float - BM25 參數
        min_df, max_df_ratio: int, float - vocabulary pruning 參數
    """
    # 1. song_ids.json
    song_ids_path = output_dir / "song_ids.json"
    with open(song_ids_path, "w", encoding="utf-8") as f:
        json.dump(song_ids, f, ensure_ascii=False, indent=2)
    print(f"[Save] {song_ids_path}")
    
    # 2. vocabulary.json
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[Save] {vocab_path}")
    
    # 3. tfidf_matrix.npz
    tfidf_path = output_dir / "tfidf_matrix.npz"
    save_npz(tfidf_path, tfidf_matrix)
    print(f"[Save] {tfidf_path}")
    
    # 4. bm25_matrix.npz
    bm25_path = output_dir / "bm25_matrix.npz"
    save_npz(bm25_path, bm25_matrix)
    print(f"[Save] {bm25_path}")
    
    # 5. idf.json
    idf_path = output_dir / "idf.json"
    with open(idf_path, "w", encoding="utf-8") as f:
        json.dump(idf, f, ensure_ascii=False, indent=2)
    print(f"[Save] {idf_path}")
    
    # 6. metadata.json（包含所有統計量與參數）
    metadata = {
        "N": N,
        "avgdl": avgdl,
        "doc_lengths": doc_lengths,
        "vocab_size": len(vocab),
        "k1": k1,
        "b": b,
        "min_df": min_df,
        "max_df_ratio": max_df_ratio,
        "max_df": int(N * max_df_ratio)
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[Save] {metadata_path}")


############################################################
# 8. Query Encoding（給後續 query / FinalVec 使用）
############################################################

def encode_query_tokens(
    tokens: List[str],
    vocab: Dict[str, int],
    idf: Dict[str, float],
    N: int,
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
    mode: str = "bm25"
) -> np.ndarray:
    """
    將 query tokens 轉成向量表示（與歌詞向量同維度）。
    
    輸入：
        tokens: List[str] - query tokens（例如一篇貼文的 expanded_tokens）
        vocab: Dict[str, int] - vocabulary
        idf: Dict[str, float] - IDF 值
        N: int - 總文件數（BM25 用）
        avgdl: float - 平均文件長度（BM25 用）
        k1, b: float - BM25 參數
        mode: str - "tfidf" 或 "bm25"
        
    輸出：
        query_vec: np.ndarray - 形狀 (vocab_size,)，與歌詞向量同維度
    """
    vocab_size = len(vocab)
    query_vec = np.zeros(vocab_size)
    
    if not tokens:
        return query_vec
    
    # 計算 query 的 term frequency
    term_freq = Counter(tokens)
    query_length = len(tokens)
    
    if mode == "tfidf":
        # TF-IDF encoding
        for term, tf in term_freq.items():
            if term in vocab:
                col_idx = vocab[term]
                tf_normalized = tf / query_length if query_length > 0 else 0
                query_vec[col_idx] = tf_normalized * idf[term]
    
    elif mode == "bm25":
        # BM25 encoding（把 query 當作一個短文件）
        for term, tf in term_freq.items():
            if term in vocab:
                col_idx = vocab[term]
                # BM25 formula（query 長度用 query_length）
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * query_length / avgdl)
                query_vec[col_idx] = idf[term] * (numerator / denominator)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return query_vec


############################################################
# 9. Main Pipeline
############################################################

def main():
    """
    主流程：從 lyrics_tokens.csv 建立所有 BM25 artifacts。
    """
    print("=" * 60)
    print("BM25 Vector Generation Pipeline")
    print("=" * 60)
    
    # 1. 讀取歌詞 tokens
    song_ids, docs_tokens = load_lyrics_tokens(LYRICS_TOKENS_PATH)
    
    # 2. 建立 vocabulary 與 DF
    vocab, df, N = build_vocabulary(docs_tokens, min_df=5, max_df_ratio=0.5)
    
    # 3. 計算文件長度統計
    doc_lengths, avgdl = compute_length_stats(docs_tokens)
    
    # 4. 計算 IDF（BM25 版本）
    idf_bm25 = compute_idf(df, N, scheme="bm25")
    
    # 5. 計算 IDF（TF-IDF 版本，給 TF-IDF 矩陣用）
    idf_tfidf = compute_idf(df, N, scheme="tfidf")
    
    # 6. 建立 TF-IDF 矩陣
    tfidf_matrix = build_tfidf_matrix(docs_tokens, vocab, idf_tfidf)
    
    # 7. 建立 BM25 矩陣
    bm25_matrix, bm25_params = build_bm25_matrix(
        docs_tokens, vocab, idf_bm25, doc_lengths, avgdl, k1=1.5, b=0.75
    )
    
    # 8. 儲存所有 artifacts
    save_bm25_artifacts(
        OUTPUT_DIR,
        song_ids,
        vocab,
        tfidf_matrix,
        bm25_matrix,
        idf_bm25,  # 存 BM25 版本的 IDF（query encoding 會用）
        N,
        doc_lengths,
        avgdl,
        bm25_params["k1"],
        bm25_params["b"],
        min_df=5,
        max_df_ratio=0.5
    )
    
    print("\n" + "=" * 60)
    print("✅ BM25 Vector Generation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - song_ids.json")
    print("  - vocabulary.json")
    print("  - tfidf_matrix.npz")
    print("  - bm25_matrix.npz")
    print("  - idf.json")
    print("  - metadata.json")


if __name__ == "__main__":
    main()

