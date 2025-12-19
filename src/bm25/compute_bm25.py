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
VOCABULARY_PATH = PROJECT_ROOT / "data" / "processed" / "lyrics" / "vocabulary.json"
POSTS_PATH = PROJECT_ROOT / "data" / "processed" / "posts" / "posts_clean_expanded.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bm25_vectors"
RETRIEVAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "retrieval"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RETRIEVAL_OUTPUT_DIR, exist_ok=True)


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

def load_existing_vocabulary(vocab_path: pathlib.Path) -> Dict[str, int]:
    """
    從 vocabulary.json 載入現有的詞彙表。
    
    輸入：
        vocab_path: pathlib.Path - vocabulary.json 的路徑
        
    輸出：
        vocab_terms: Dict[str, int] - 詞 → 詞頻（從 vocabulary.json 讀取）
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_terms = json.load(f)
    
    print(f"[Vocab] Loaded existing vocabulary: {len(vocab_terms)} terms from {vocab_path}")
    return vocab_terms


def build_vocabulary(
    docs_tokens: List[List[str]], 
    vocab_path: pathlib.Path = None,
    min_df: int = 5, 
    max_df_ratio: float = 0.5
) -> Tuple[Dict[str, int], Dict[str, int], int]:
    """
    建立 vocabulary（詞彙表）與 DF 統計。
    
    如果提供了 vocab_path，則直接使用現有的 vocabulary.json 詞彙表。
    否則，會從 docs_tokens 重新建立 vocabulary（舊的行為）。
    
    輸入：
        docs_tokens: List[List[str]] - 所有歌曲的 tokens
        vocab_path: pathlib.Path - 可選，現有 vocabulary.json 的路徑
        min_df: int - 最小 document frequency（用於 DF 計算，不影響 vocabulary 選擇）
        max_df_ratio: float - 最大 document frequency 比例（用於 DF 計算，不影響 vocabulary 選擇）
        
    輸出：
        vocab: Dict[str, int] - 詞 → column index（0 ~ |V|-1）
        df: Dict[str, int] - 詞 → document frequency（只包含在 vocabulary 中的詞）
        N: int - 總文件數（歌曲數）
    """
    N = len(docs_tokens)
    
    # 如果提供了 vocab_path，使用現有的 vocabulary
    if vocab_path and vocab_path.exists():
        # 載入現有的詞彙表（詞 → 詞頻）
        vocab_terms = load_existing_vocabulary(vocab_path)
        
        # 建立 vocabulary（詞 → index），按照字母順序排序以保持一致性
        vocab = {term: idx for idx, term in enumerate(sorted(vocab_terms.keys()))}
        
        print(f"[Vocab] Using existing vocabulary: {len(vocab)} terms")
    else:
        # 舊的行為：從 docs_tokens 建立 vocabulary
        print("[Vocab] Building vocabulary from docs_tokens...")
        
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
        
        print(f"[Vocab] Built vocabulary: {len(vocab)} terms (pruned from {len(term_doc_count)})")
        print(f"[Vocab] Pruning: min_df={min_df}, max_df_ratio={max_df_ratio} (max_df={max_df})")
    
    # 計算 DF（document frequency）：只計算在 vocabulary 中的詞
    # 這一步無論是否使用現有 vocabulary 都需要做
    term_doc_count = defaultdict(int)
    for doc_tokens in docs_tokens:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            if token in vocab:  # 只計算在 vocabulary 中的詞
                term_doc_count[token] += 1
    
    # 建立 DF dict（只包含在 vocabulary 中的詞）
    df = {term: term_doc_count.get(term, 0) for term in vocab.keys()}
    
    print(f"[Vocab] Computed DF for {len(df)} terms in vocabulary")
    
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
# 9. BM25 Top-K Candidate Retrieval（對貼文計算候選）
############################################################

def compute_bm25_score_for_query(
    query_tokens: List[str],
    bm25_matrix: csr_matrix,
    vocab: Dict[str, int],
    idf: Dict[str, float],
    metadata: dict,
    song_ids: List[str]
) -> np.ndarray:
    """
    對 query tokens 計算與所有歌曲的 BM25 分數。
    
    輸入：
        query_tokens: List[str] - query tokens
        bm25_matrix: csr_matrix - BM25 矩陣
        vocab: Dict[str, int] - vocabulary
        idf: Dict[str, float] - IDF 值
        metadata: dict - 包含 N, avgdl, k1, b
        song_ids: List[str] - 歌曲 ID 列表
        
    輸出：
        scores: np.ndarray - 形狀 (num_songs,)，每首歌的 BM25 分數
    """
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
    
    # 計算與所有歌曲的 BM25 分數（用 dot product）
    # bm25_matrix 是 (num_songs, vocab_size)
    # query_vec 是 (vocab_size,)
    # 結果是 (num_songs,)
    scores = bm25_matrix.dot(query_vec)
    
    return scores


def compute_bm25_topk_for_posts(
    posts_path: pathlib.Path,
    bm25_matrix: csr_matrix,
    song_ids: List[str],
    vocab: Dict[str, int],
    idf: Dict[str, float],
    metadata: dict,
    top_k: int = 100,
    output_path: pathlib.Path = None
) -> None:
    """
    對所有貼文計算 BM25 top-K 候選，輸出到 outputs/retrieval/bm25_topk.jsonl。
    
    輸入：
        posts_path: pathlib.Path - posts_clean_expanded.jsonl 路徑
        bm25_matrix: csr_matrix - BM25 矩陣
        song_ids: List[str] - 歌曲 ID 列表
        vocab: Dict[str, int] - vocabulary
        idf: Dict[str, float] - IDF 值
        metadata: dict - 統計量
        top_k: int - 要取幾首候選（預設 100）
        output_path: pathlib.Path - 輸出路徑（預設 outputs/retrieval/bm25_topk.jsonl）
    """
    if output_path is None:
        output_path = RETRIEVAL_OUTPUT_DIR / "bm25_topk.jsonl"
    
    print(f"\n[BM25 Retrieval] Computing top-{top_k} candidates for all posts...")
    
    query_count = 0
    
    with open(posts_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            post = json.loads(line.strip())
            query_id = f"post_{query_count}"
            
            # 使用 expanded_tokens（如果有的話），否則用 clean_tokens
            query_tokens = post.get("expanded_tokens", post.get("clean_tokens", []))
            
            if not query_tokens:
                # 如果沒有 tokens，跳過
                continue
            
            # 計算 BM25 分數
            scores = compute_bm25_score_for_query(
                query_tokens,
                bm25_matrix,
                vocab,
                idf,
                metadata,
                song_ids
            )
            
            # 找出 top-K
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # 組合成候選列表
            candidates = [
                {
                    "song_id": song_ids[idx],
                    "bm25_score": float(scores[idx])
                }
                for idx in top_indices
            ]
            
            # 輸出 JSONL 格式
            result = {
                "query_id": query_id,
                "raw_text": post.get("raw_text", ""),
                "emotion": post.get("emotion", "unknown"),
                "top_k": top_k,
                "candidates": candidates
            }
            
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            query_count += 1
            
            if query_count % 1000 == 0:
                print(f"  Processed {query_count} queries...")
    
    print(f"[BM25 Retrieval] ✅ Completed! Processed {query_count} queries")
    print(f"[BM25 Retrieval] Output: {output_path}")


def save_idf_for_b_module(
    idf: Dict[str, float],
    output_path: pathlib.Path = None
) -> None:
    """
    將 IDF 值存到 outputs/retrieval/idf.json，供 B 組員使用。
    
    輸入：
        idf: Dict[str, float] - IDF 值
        output_path: pathlib.Path - 輸出路徑（預設 outputs/retrieval/idf.json）
    """
    if output_path is None:
        output_path = RETRIEVAL_OUTPUT_DIR / "idf.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(idf, f, ensure_ascii=False, indent=2)
    
    print(f"[Save] IDF saved for B module: {output_path}")


############################################################
# 10. Main Pipeline
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
    
    # 2. 建立 vocabulary 與 DF（使用現有的 vocabulary.json）
    vocab, df, N = build_vocabulary(docs_tokens, vocab_path=VOCABULARY_PATH, min_df=5, max_df_ratio=0.5)
    
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
    
    # 9. 建立 metadata dict（給後續函式使用）
    metadata = {
        "N": N,
        "avgdl": avgdl,
        "k1": bm25_params["k1"],
        "b": bm25_params["b"]
    }
    
    # 10. 對所有貼文計算 BM25 top-K 候選（Stage 1: Retrieval）
    if POSTS_PATH.exists():
        compute_bm25_topk_for_posts(
            POSTS_PATH,
            bm25_matrix,
            song_ids,
            vocab,
            idf_bm25,
            metadata,
            top_k=100,
            output_path=RETRIEVAL_OUTPUT_DIR / "bm25_topk.jsonl"
        )
    else:
        print(f"\n[Warning] Posts file not found: {POSTS_PATH}")
        print("  Skipping BM25 top-K candidate generation.")
    
    # 11. 輸出 IDF 給 B 組員使用（可選）
    save_idf_for_b_module(idf_bm25, RETRIEVAL_OUTPUT_DIR / "idf.json")
    
    print("\n" + "=" * 60)
    print("✅ BM25 Vector Generation & Retrieval Complete!")
    print("=" * 60)
    print(f"\nOutput directories:")
    print(f"  - {OUTPUT_DIR}")
    print(f"  - {RETRIEVAL_OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  [BM25 Vectors]")
    print("  - song_ids.json")
    print("  - vocabulary.json")
    print("  - tfidf_matrix.npz")
    print("  - bm25_matrix.npz")
    print("  - idf.json")
    print("  - metadata.json")
    print("\n  [Retrieval]")
    print("  - bm25_topk.jsonl (top-K candidates for each post)")
    print("  - idf.json (for B module)")


if __name__ == "__main__":
    main()

