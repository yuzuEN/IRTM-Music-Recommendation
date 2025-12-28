import os
import csv
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ✅ Lyrics-topic 專用 stopwords（只影響 topic TF-IDF）
LYRICS_TOPIC_STOPWORDS = {
    # 語氣詞和填充詞
    "yeah", "oh", "uh", "na", "la", "ya",
    "da", "ah", "woah", "ooh",
    "hey", "huh", "whoa", "mmm", "mm",
    # 歌曲結構詞
    "choru", "chorus", "verse", "vers", "bridg", "outro", "hook",
    # 口語化詞
    "got", "get", "gotta", "got ta",
    "wan", "want", "gon", "gonna",
    "im", "em", "ta", "babi", "away", "yea", "ayi", "ay", "yuh", "aye", "lil", "tryna",
    # 通用動詞
    "say", "said", "tell", "look", "come", "let", "make",
    # 粗俗詞
    "fuck", "shit", "bitch", "nigga", "ass", "damn", "hell", "yo",
    # # ✅ 新增：過於通用的情感/時間詞（出現在太多主題中，降低區分度）
    # "love", "feel", "time", "way", "caus",  # cause 的詞幹
    # "need", "think", "thing", "day", "night", "life",
    # "heart", "eye", "hand", "arm",
}

def build_stopwords_for_lyrics_topic() -> List[str]:
    return sorted(set(ENGLISH_STOP_WORDS) | set(LYRICS_TOPIC_STOPWORDS))


# ===============================
# Config
# ===============================
LYRICS_TOKENS_PATH = r"data/processed/lyrics/lyrics_tokens.csv"
SONG_IDS_PATH = r"outputs/bm25_vectors/song_ids.json"   # ✅ C 的對齊順序

OUT_DIR = "outputs/topic_vectors"
os.makedirs(OUT_DIR, exist_ok=True)

TFIDF_PATH = os.path.join(OUT_DIR, "lyrics_tfidf.npz")
TFIDF_META_PATH = os.path.join(OUT_DIR, "lyrics_tfidf_meta.json")

# Scan-K outputs
SCAN_TABLE_PATH = os.path.join(OUT_DIR, "lyrics_kmeans_scanK.tsv")
SCAN_JSON_PATH = os.path.join(OUT_DIR, "lyrics_kmeans_scanK.json")

# Final chosen-K outputs
MODEL_PATH = os.path.join(OUT_DIR, "lyrics_kmeans_model.joblib")
VECTORIZER_PATH = os.path.join(OUT_DIR, "lyrics_tfidf_vectorizer.joblib")  # ✅ 新增：保存 vectorizer
ASSIGN_PATH = os.path.join(OUT_DIR, "lyrics_topic_assignments.jsonl")
SUMMARY_PATH = os.path.join(OUT_DIR, "lyrics_topic_summary.json")
EVAL_PATH = os.path.join(OUT_DIR, "lyrics_topic_eval.json")
KEYWORDS_TSV_PATH = os.path.join(OUT_DIR, "lyrics_topic_keywords.tsv")

# ✅ 向量輸出（給 cosine similarity）
TOPIC_VEC_PATH = os.path.join(OUT_DIR, "TopicVec_lyrics_kmeans.npy")
TOPIC_VEC_META_PATH = os.path.join(OUT_DIR, "TopicVec_lyrics_kmeans_meta.json")

# ✅ cluster similarity pairs
CLUSTER_SIM_PAIRS_PATH = os.path.join(OUT_DIR, "lyrics_cluster_sim_pairs.tsv")
CLUSTER_SIM_TOPN_PATH = os.path.join(OUT_DIR, "lyrics_cluster_sim_topN.tsv")
MERGE_INFO_PATH = os.path.join(OUT_DIR, "lyrics_cluster_merge_info.json")

# K range to scan
K_MIN = 29
K_MAX = 29
K_STEP = 1

# Evaluation settings
SIL_SAMPLE_N = 5000
RANDOM_STATE = 42

# Optional: treat tiny clusters as "bad"
MIN_CLUSTER_SIZE = 30  # set 0 to disable

# ✅ merge threshold
MERGE_THRESHOLD = 0.7
TOPN_SIM_PAIRS = 50

# ✅ High-frequency word filtering (自動將高頻詞加入 stopwords)
HIGH_FREQ_TOP_PERCENT = 20  # 出現在超過 20% 文檔中的詞會被加入 stopwords


# ===============================
# Utilities
# ===============================
def load_song_ids(song_ids_path: str) -> List[str]:
    with open(song_ids_path, "r", encoding="utf-8") as f:
        song_ids = json.load(f)
    if not isinstance(song_ids, list) or not song_ids:
        raise ValueError("song_ids.json is empty or not a list.")
    return [str(x) for x in song_ids]

def load_lyrics_tokens_csv(csv_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "song_id" not in reader.fieldnames or "tokens" not in reader.fieldnames:
            raise ValueError("lyrics_tokens.csv must have columns: song_id,tokens")
        for row in reader:
            sid = str(row.get("song_id", "")).strip()
            toks = str(row.get("tokens", "")).strip()
            if sid:
                mapping[sid] = toks
    if not mapping:
        raise ValueError("No rows loaded from lyrics_tokens.csv")
    return mapping

def align_lyrics_to_song_ids(
    song_ids: List[str],
    songid_to_tokens: Dict[str, str],
    missing_policy: str = "drop",  # "drop" or "empty"
) -> Tuple[List[str], List[str], Dict[str, int]]:
    texts: List[str] = []
    used: List[str] = []

    missing = 0
    for sid in song_ids:
        if sid in songid_to_tokens and songid_to_tokens[sid].strip():
            texts.append(songid_to_tokens[sid])
            used.append(sid)
        else:
            missing += 1
            if missing_policy == "empty":
                texts.append("")
                used.append(sid)
            elif missing_policy == "drop":
                continue
            else:
                raise ValueError("missing_policy must be 'drop' or 'empty'")

    stats = {
        "n_song_ids": int(len(song_ids)),
        "n_used": int(len(used)),
        "n_missing": int(missing),
        "missing_policy": missing_policy,
    }
    if len(texts) != len(used):
        raise RuntimeError("Alignment internal error: texts != used.")
    return texts, used, stats

def get_top_terms_per_cluster(
    X_tfidf: sparse.spmatrix,
    labels: np.ndarray,
    feature_names: np.ndarray,
    topn: int = 12,
) -> Dict[int, List[str]]:
    k = int(labels.max()) + 1
    top_terms: Dict[int, List[str]] = {}

    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            top_terms[cid] = []
            continue
        mean_vec = X_tfidf[idx].mean(axis=0)
        mean_vec = np.asarray(mean_vec).ravel()
        top_idx = np.argsort(-mean_vec)[:topn]
        top_terms[cid] = [feature_names[i] for i in top_idx if mean_vec[i] > 0]
    return top_terms

def sampled_silhouette_cosine(
    X: sparse.spmatrix,
    labels: np.ndarray,
    sample_n: int,
    random_state: int
) -> Optional[float]:
    n = X.shape[0]
    if n <= 2:
        return None
    sample_n = min(sample_n, n)
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    return float(silhouette_score(X[sample_idx], labels[sample_idx], metric="cosine"))

def cluster_stats(labels: np.ndarray, K: int) -> Dict[str, Any]:
    sizes = np.bincount(labels, minlength=K)
    mn = int(sizes.min()) if sizes.size else 0
    mx = int(sizes.max()) if sizes.size else 0
    n_small = int((sizes < MIN_CLUSTER_SIZE).sum()) if MIN_CLUSTER_SIZE > 0 else 0
    return {"size_min": mn, "size_max": mx, "n_small_clusters": n_small}

def make_topic_vector_hard(labels: np.ndarray, K: int) -> np.ndarray:
    n = labels.shape[0]
    vec = np.zeros((n, K), dtype=np.float32)
    vec[np.arange(n), labels] = 1.0
    return vec

def to_py(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(to_py(k)): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    return obj


# ===============================
# cluster similarity + merge helpers
# ===============================
def compute_cluster_sim_matrix(centroids: np.ndarray) -> np.ndarray:
    return cosine_similarity(centroids)

def save_cluster_sim_pairs(sim_mat: np.ndarray, threshold: float, out_path: str) -> int:
    K = sim_mat.shape[0]
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            s = float(sim_mat[i, j])
            if s >= threshold:
                pairs.append((i, j, s))
    pairs.sort(key=lambda x: x[2], reverse=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("cluster_i\tcluster_j\tcosine_sim\n")
        for i, j, s in pairs:
            f.write(f"{i}\t{j}\t{s:.6f}\n")
    return len(pairs)

def save_cluster_sim_topN(sim_mat: np.ndarray, topn: int, out_path: str) -> None:
    K = sim_mat.shape[0]
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            pairs.append((i, j, float(sim_mat[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:topn]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("rank\tcluster_i\tcluster_j\tcosine_sim\n")
        for r, (i, j, s) in enumerate(pairs, start=1):
            f.write(f"{r}\t{i}\t{j}\t{s:.6f}\n")

def build_merge_map_by_threshold(sim_mat: np.ndarray, threshold: float) -> Tuple[Dict[int, int], List[List[int]]]:
    K = sim_mat.shape[0]
    visited = [False] * K
    groups: List[List[int]] = []

    for i in range(K):
        if visited[i]:
            continue
        q = deque([i])
        visited[i] = True
        comp = [i]

        while q:
            u = q.popleft()
            nbrs = np.where(sim_mat[u] >= threshold)[0]
            for v in nbrs:
                v = int(v)
                if v == u:
                    continue
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
                    comp.append(v)

        groups.append(sorted(comp))

    # 穩定排序：大群先，其次用最小 id
    groups = sorted(groups, key=lambda g: (len(g), -g[0]), reverse=True)

    old2new: Dict[int, int] = {}
    for new_id, g in enumerate(groups):
        for old_id in g:
            old2new[int(old_id)] = int(new_id)

    return old2new, groups

def remap_labels(labels: np.ndarray, old2new: Dict[int, int]) -> np.ndarray:
    return np.array([old2new[int(x)] for x in labels], dtype=np.int32)

def get_top_terms_per_topic(
    X_tfidf: sparse.spmatrix,
    topic_labels: np.ndarray,
    feature_names: np.ndarray,
    topn: int = 12,
) -> Dict[int, List[str]]:
    Kt = int(topic_labels.max()) + 1
    top_terms: Dict[int, List[str]] = {}
    for tid in range(Kt):
        idx = np.where(topic_labels == tid)[0]
        if len(idx) == 0:
            top_terms[tid] = []
            continue
        mean_vec = X_tfidf[idx].mean(axis=0)
        mean_vec = np.asarray(mean_vec).ravel()
        top_idx = np.argsort(-mean_vec)[:topn]
        top_terms[tid] = [feature_names[i] for i in top_idx if mean_vec[i] > 0]
    return top_terms

def cluster_sizes_from_labels(labels: np.ndarray) -> Dict[int, int]:
    K = int(labels.max()) + 1
    return {int(i): int((labels == i).sum()) for i in range(K)}


# ===============================
# Main
# ===============================
def main():
    # 1) Load + align
    song_ids = load_song_ids(SONG_IDS_PATH)
    songid_to_tokens = load_lyrics_tokens_csv(LYRICS_TOKENS_PATH)

    texts, used_song_ids, align_stats = align_lyrics_to_song_ids(
        song_ids=song_ids,
        songid_to_tokens=songid_to_tokens,
        missing_policy="drop",
    )

    n = len(texts)
    print(f"[INFO] Loaded song_ids: {len(song_ids)}")
    print(f"[INFO] Lyrics docs used: {n}")
    print(f"[INFO] Missing lyrics in song_ids: {align_stats['n_missing']} (policy={align_stats['missing_policy']})")

    if n < 10:
        raise RuntimeError("Too few lyrics documents after alignment; check your files/paths.")

    # 2) TF-IDF（兩階段：先檢測高頻詞，再加入 stopwords）
    stopwords = build_stopwords_for_lyrics_topic()
    
    # 第一階段：先 fit 一次，檢測高頻詞
    print("[TF-IDF] Stage 1: Detecting high-frequency words...")
    vectorizer_temp = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        max_features=50000,
        min_df=8,
        max_df=1.0,  # 先不過濾，讓所有詞都進來
        stop_words=stopwords,
    )
    X_temp = vectorizer_temp.fit_transform(texts)
    feature_names_temp = np.array(vectorizer_temp.get_feature_names_out())
    
    # 計算每個詞的 document frequency（出現在多少個文檔中）
    doc_freq = np.asarray((X_temp > 0).sum(axis=0)).ravel()  # (vocab_size,)
    doc_freq_ratio = doc_freq / X_temp.shape[0]  # 出現在多少比例的文檔中
    
    # 找出高頻詞（例如：出現在超過 TOP_PERCENT% 文檔中的詞）
    high_freq_threshold = HIGH_FREQ_TOP_PERCENT / 100.0
    high_freq_mask = doc_freq_ratio >= high_freq_threshold
    high_freq_words_candidates = set(feature_names_temp[high_freq_mask])
    
    # ✅ 排除已經在 LYRICS_TOPIC_STOPWORDS 中的詞（因為這些已經在 stopwords 中了）
    high_freq_words = high_freq_words_candidates - LYRICS_TOPIC_STOPWORDS
    
    print(f"[TF-IDF] Found {len(high_freq_words_candidates)} high-frequency words (appear in >= {HIGH_FREQ_TOP_PERCENT}% of documents)")
    print(f"[TF-IDF] After excluding LYRICS_TOPIC_STOPWORDS: {len(high_freq_words)} words to add")
    if len(high_freq_words) > 0:
        print(f"[TF-IDF] Top 20 high-frequency words to add: {sorted(list(high_freq_words))[:20]}")
    
    # 將高頻詞加入 stopwords（轉換為 set 以便合併，然後轉回 list）
    stopwords_set = set(stopwords)
    stopwords_extended_set = stopwords_set | high_freq_words
    stopwords_extended = sorted(list(stopwords_extended_set))  # ✅ 轉回 list，因為 TfidfVectorizer 需要 list
    print(f"[TF-IDF] Extended stopwords: {len(stopwords_set)} -> {len(stopwords_extended_set)} (+{len(high_freq_words)})")
    
    # 第二階段：使用擴展後的 stopwords 重新 fit
    print("[TF-IDF] Stage 2: Fitting with extended stopwords...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        max_features=50000,
        min_df=8,
        max_df=0.35,  # 仍然保留這個參數作為額外過濾
        stop_words=stopwords_extended,  # ✅ 現在是 list，符合 TfidfVectorizer 的要求
    )
    X = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # DEBUG A
    term_tfidf_sum = np.asarray(X.sum(axis=0)).ravel()
    order_tfidf = np.argsort(-term_tfidf_sum)
    TOP_N = 200
    print("\n" + "="*60)
    print("[DEBUG A] Top terms by GLOBAL TF-IDF weight")
    print("="*60)
    print("rank\tterm\ttfidf_sum")
    for rank, idx in enumerate(order_tfidf[:TOP_N], start=1):
        print(f"{rank}\t{feature_names[idx]}\t{term_tfidf_sum[idx]:.4f}")

    # DEBUG B
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    df_ratio = df / X.shape[0]
    order_df = np.argsort(-df)
    print("\n" + "="*60)
    print("[DEBUG B] Top terms by DOCUMENT FREQUENCY")
    print("="*60)
    print("rank\tterm\tdf\tdf_ratio")
    for rank, idx in enumerate(order_df[:TOP_N], start=1):
        print(f"{rank}\t{feature_names[idx]}\t{df[idx]}\t{df_ratio[idx]:.2%}")

    # Save TF-IDF + TFIDF meta ✅（這裡不要碰 K/merge）
    sparse.save_npz(TFIDF_PATH, X)
    with open(TFIDF_META_PATH, "w", encoding="utf-8") as f:
        meta = {
            "lyrics_tokens_path": LYRICS_TOKENS_PATH,
            "song_ids_path": SONG_IDS_PATH,
            "align_stats": align_stats,
            "n_songs_used": int(n),
            "vocab_size": int(X.shape[1]),
            "tfidf": {
                "ngram_range": [1, 1],
                "max_features": 50000,
                "min_df": 8,
                "max_df": 0.35,
                "high_freq_top_percent": HIGH_FREQ_TOP_PERCENT,
                "stop_words": {
                    "english": True,
                    "lyrics_topic_stopwords": sorted(list(LYRICS_TOPIC_STOPWORDS)),
                    "auto_detected_high_freq_words": sorted(list(high_freq_words)),  # ✅ 記錄自動檢測的高頻詞
                }
            },
        }
        json.dump(to_py(meta), f, ensure_ascii=False, indent=2)

    print(f"[OK] TF-IDF saved: {TFIDF_PATH}")
    print(f"[INFO] TF-IDF shape: {X.shape}")

    # 3) Scan K
    scan_rows: List[Dict[str, Any]] = []
    best_k = None
    best_sil = -1e9

    for K in range(K_MIN, K_MAX + 1, K_STEP):
        print(f"[SCAN] K={K} ...")
        kmeans = KMeans(
            n_clusters=K,
            random_state=RANDOM_STATE,
            n_init=10,
            max_iter=300,
        )
        labels = kmeans.fit_predict(X)

        sil = sampled_silhouette_cosine(X, labels, sample_n=SIL_SAMPLE_N, random_state=RANDOM_STATE)
        stats = cluster_stats(labels, K)

        row = {
            "K": int(K),
            "inertia_rss": float(kmeans.inertia_),
            "silhouette_cosine_sampled": (None if sil is None else float(sil)),
            "silhouette_sampled_n": int(min(SIL_SAMPLE_N, X.shape[0])),
            **stats,
        }
        scan_rows.append(row)

        if sil is not None and sil > best_sil:
            best_sil = sil
            best_k = K

    with open(SCAN_TABLE_PATH, "w", encoding="utf-8", newline="") as f:
        cols = [
            "K", "inertia_rss", "silhouette_cosine_sampled", "silhouette_sampled_n",
            "size_min", "size_max", "n_small_clusters"
        ]
        f.write("\t".join(cols) + "\n")
        for r in scan_rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    scan_summary = {
        "lyrics_tokens_path": LYRICS_TOKENS_PATH,
        "song_ids_path": SONG_IDS_PATH,
        "tfidf_path": TFIDF_PATH,
        "align_stats": align_stats,
        "K_range": {"min": K_MIN, "max": K_MAX, "step": K_STEP},
        "silhouette": {"metric": "cosine", "sample_n": int(min(SIL_SAMPLE_N, X.shape[0]))},
        "min_cluster_size": int(MIN_CLUSTER_SIZE),
        "best_by_silhouette": {
            "K": int(best_k) if best_k is not None else None,
            "silhouette": float(best_sil) if best_k is not None else None
        },
        "rows": scan_rows,
    }
    with open(SCAN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(to_py(scan_summary), f, ensure_ascii=False, indent=2)

    print(f"[OK] Scan table saved: {SCAN_TABLE_PATH}")
    print(f"[OK] Scan json saved : {SCAN_JSON_PATH}")
    print(f"[INFO] Best K by silhouette: {best_k} (sil={best_sil:.4f})")

    # 4) Final KMeans
    if best_k is None:
        raise RuntimeError("No valid silhouette score produced; cannot choose K automatically.")

    K = int(best_k)
    kmeans = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init=5,  # ✅ 減少初始化次數以節省記憶體
        max_iter=300,
        init='random',  # ✅ 使用 random 初始化避免稀疏矩陣記憶體問題
    )
    labels = kmeans.fit_predict(X)
    joblib.dump(kmeans, MODEL_PATH)
    
    # ✅ 保存 vectorizer（供 posts_topic_align20.py 使用）
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"[OK] Final KMeans model saved: {MODEL_PATH}")
    print(f"[OK] TF-IDF vectorizer saved: {VECTORIZER_PATH}")
    print(f"[INFO] Final KMeans inertia (RSS): {kmeans.inertia_:.2f}")

    # 5) cluster centroid similarity
    sim_mat = compute_cluster_sim_matrix(kmeans.cluster_centers_)
    n_pairs = save_cluster_sim_pairs(sim_mat, MERGE_THRESHOLD, CLUSTER_SIM_PAIRS_PATH)
    save_cluster_sim_topN(sim_mat, TOPN_SIM_PAIRS, CLUSTER_SIM_TOPN_PATH)

    print(f"[OK] cluster_sim_pairs.tsv saved: {CLUSTER_SIM_PAIRS_PATH} (pairs >= {MERGE_THRESHOLD}: {n_pairs})")
    print(f"[OK] cluster_sim_topN.tsv  saved: {CLUSTER_SIM_TOPN_PATH} (topN={TOPN_SIM_PAIRS})")
    if n_pairs == 0:
        print(
            f"[WARN] No cluster pairs with cosine_sim >= {MERGE_THRESHOLD}. "
            f"Try lowering MERGE_THRESHOLD and consult cluster_sim_topN.tsv."
        )

    # 6) Merge clusters (first pass: threshold-based)
    old2new, groups = build_merge_map_by_threshold(sim_mat, MERGE_THRESHOLD)
    merged_labels_temp = remap_labels(labels, old2new)
    K_merged_temp = int(merged_labels_temp.max()) + 1
    
    print(f"[INFO] First pass merge: Original K={K} -> Merged K={K_merged_temp} (threshold={MERGE_THRESHOLD})")
    
    # ✅ 手動合併額外的主題對（基於合併後的主題）
    # 策略：先基於阈值合併得到25個主題，然後基於合併後主題的關鍵詞來匹配和合併
    
    def merge_groups_manually_by_merged_topic_ids(groups: List[List[int]], merged_topic_pairs: List[Tuple[int, int]], old2new_map: Dict[int, int]) -> List[List[int]]:
        """基於合併後的主題ID來合併groups"""
        # 建立合併後topic_id到groups索引的映射
        merged_topic_to_group_idx = {}
        for g_idx, group in enumerate(groups):
            # 找出這個group對應的合併後topic_id（取group中第一個元素的映射）
            if group:
                first_orig_id = group[0]
                merged_topic_id = old2new_map.get(first_orig_id)
                if merged_topic_id is not None:
                    merged_topic_to_group_idx[merged_topic_id] = g_idx
        
        # 合併指定的合併後主題對（需要動態更新映射，因為合併後索引會變化）
        for merged_id1, merged_id2 in merged_topic_pairs:
            # 重新建立映射（因為前面的合併可能改變了索引）
            merged_topic_to_group_idx = {}
            for g_idx, group in enumerate(groups):
                if group is not None and group:  # 只處理非None的組
                    first_orig_id = group[0]
                    merged_topic_id = old2new_map.get(first_orig_id)
                    if merged_topic_id is not None:
                        merged_topic_to_group_idx[merged_topic_id] = g_idx
            
            if merged_id1 not in merged_topic_to_group_idx or merged_id2 not in merged_topic_to_group_idx:
                print(f"[Warn] Cannot merge merged topic {merged_id1} and {merged_id2}: one or both not found")
                continue
            
            idx1 = merged_topic_to_group_idx[merged_id1]
            idx2 = merged_topic_to_group_idx[merged_id2]
            
            if idx1 == idx2:
                print(f"[Info] Merged topic {merged_id1} and {merged_id2} already in the same group")
                continue
            
            # 檢查是否為None（已被合併）
            if groups[idx1] is None or groups[idx2] is None:
                print(f"[Warn] Cannot merge merged topic {merged_id1} and {merged_id2}: one or both groups are None")
                continue
            
            # 合併兩個組
            groups[idx1].extend(groups[idx2])
            groups[idx1] = sorted(list(set(groups[idx1])))  # 去重並排序
            
            # 移除idx2的組（設為None，稍後過濾）
            groups[idx2] = None
        
        # 過濾None的組並重新排序
        groups = [g for g in groups if g is not None]
        groups = sorted(groups, key=lambda g: (len(g), -g[0]), reverse=True)
        
        return groups
    
    # ✅ 手動合併指定的合併後主題對（K=25的主題ID）
    # 策略：直接使用合併後的主題ID（15, 18, 3, 4, 7, 8），這些是第一次threshold合併後的主題ID
    print(f"[Merge] Preparing to merge merged topic pairs...")
    
    # 定義要合併的合併後主題對（這些是第一次threshold合併後的K=25主題ID）
    manual_merge_merged_pairs = [
        (15, 18),  # 合併後主題 15 + 18
        (3, 4),    # 合併後主題 3 + 4
        (7, 8),    # 合併後主題 7 + 8
    ]
    
    merged_topic_pairs_to_merge = []
    
    for merged_id1, merged_id2 in manual_merge_merged_pairs:
        # 檢查是否在有效範圍內
        if merged_id1 >= K_merged_temp or merged_id2 >= K_merged_temp:
            print(f"[Warn] Invalid merged topic IDs: {merged_id1} or {merged_id2} (K_merged={K_merged_temp})")
            continue
        
        if merged_id1 == merged_id2:
            print(f"[Info] Merged topics {merged_id1} and {merged_id2} are the same, skipping")
            continue
        
        merged_topic_pairs_to_merge.append((merged_id1, merged_id2))
        print(f"[Merge] Will merge: Merged topic {merged_id1} + Merged topic {merged_id2}")
    
    # 執行合併
    if merged_topic_pairs_to_merge:
        print(f"[Merge] Applying {len(merged_topic_pairs_to_merge)} manual merges on merged topics: {merged_topic_pairs_to_merge}")
        groups = merge_groups_manually_by_merged_topic_ids(groups, merged_topic_pairs_to_merge, old2new)
    else:
        print(f"[Warn] No manual merges applied - no matching topics found")
    
    # 重新建立 old2new 映射
    old2new = {}
    for new_id, g in enumerate(groups):
        for old_id in g:
            old2new[int(old_id)] = int(new_id)
    
    merged_labels = remap_labels(labels, old2new)
    K_merged = int(merged_labels.max()) + 1

    with open(MERGE_INFO_PATH, "w", encoding="utf-8") as f:
        merge_info = {
            "K_original": int(K),
            "K_merged": int(K_merged),
            "merge_threshold": float(MERGE_THRESHOLD),
            "groups": groups,
            "note": "Each group is a set of original cluster ids merged into one topic_id."
        }
        json.dump(to_py(merge_info), f, ensure_ascii=False, indent=2)

    print(f"[OK] Merge info saved: {MERGE_INFO_PATH}")
    print(f"[INFO] Original K={K} -> Merged K={K_merged} (threshold={MERGE_THRESHOLD})")

    # 7) Topic keywords + sizes (merged)
    top_terms = get_top_terms_per_topic(X, merged_labels, feature_names, topn=12)
    sizes = cluster_sizes_from_labels(merged_labels)

    summary = {
        "K_original": int(K),
        "K_merged": int(K_merged),
        "merge_threshold": float(MERGE_THRESHOLD),
        "topic_sizes": sizes,
        "top_terms": {str(k): v for k, v in top_terms.items()},
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(to_py(summary), f, ensure_ascii=False, indent=2)

    eval_info = {
        "K_original": int(K),
        "K_merged": int(K_merged),
        "merge_threshold": float(MERGE_THRESHOLD),
        "inertia_rss_original": float(kmeans.inertia_),
        "silhouette_metric": "cosine",
        "silhouette_sampled_n": int(min(SIL_SAMPLE_N, X.shape[0])),
        "silhouette_original_sampled": float(sampled_silhouette_cosine(X, labels, sample_n=SIL_SAMPLE_N, random_state=RANDOM_STATE)),
        "silhouette_merged_sampled": float(sampled_silhouette_cosine(X, merged_labels, sample_n=SIL_SAMPLE_N, random_state=RANDOM_STATE)),
        "cluster_stats_original": cluster_stats(labels, K),
        "cluster_stats_merged": cluster_stats(merged_labels, K_merged),
        "sim_pairs_ge_threshold": int(n_pairs),
        "cluster_sim_pairs_path": os.path.basename(CLUSTER_SIM_PAIRS_PATH),
        "cluster_sim_topN_path": os.path.basename(CLUSTER_SIM_TOPN_PATH),
        "merge_info_path": os.path.basename(MERGE_INFO_PATH),
    }
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(to_py(eval_info), f, ensure_ascii=False, indent=2)

    with open(KEYWORDS_TSV_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("topic_id\tsize\ttop_keywords\n")
        for tid in range(K_merged):
            kw = ", ".join(top_terms[tid])
            f.write(f"{tid}\t{sizes[tid]}\t{kw}\n")

    print(f"[OK] Topic summary saved: {SUMMARY_PATH}")
    print(f"[OK] Topic eval saved   : {EVAL_PATH}")
    print(f"[OK] Topic keywords TSV : {KEYWORDS_TSV_PATH}")

    # 8) Save aligned topic vectors (merged one-hot)
    topic_vec = make_topic_vector_hard(merged_labels, K_merged)
    np.save(TOPIC_VEC_PATH, topic_vec)

    meta = {
        "type": "kmeans_onehot_merged",
        "K_original": int(K),
        "K_merged": int(K_merged),
        "merge_threshold": float(MERGE_THRESHOLD),
        "shape": [int(topic_vec.shape[0]), int(topic_vec.shape[1])],
        "aligned_to_song_ids_json": True,
        "song_ids_path": SONG_IDS_PATH,
        "note": "Row i corresponds to used_song_ids[i] (subset of song_ids.json if missing_policy=drop).",
        "align_stats": align_stats,
        "tfidf_path": TFIDF_PATH,
        "model_path": MODEL_PATH,
        "cluster_sim_pairs_path": CLUSTER_SIM_PAIRS_PATH,
        "cluster_sim_topN_path": CLUSTER_SIM_TOPN_PATH,
        "merge_info_path": MERGE_INFO_PATH,
    }
    with open(TOPIC_VEC_META_PATH, "w", encoding="utf-8") as f:
        json.dump(to_py(meta), f, ensure_ascii=False, indent=2)

    print(f"[OK] Topic vectors saved: {TOPIC_VEC_PATH}")
    print(f"[OK] Topic vector meta  : {TOPIC_VEC_META_PATH}")

    # 9) Assignments
    top_terms_orig = get_top_terms_per_cluster(X, labels, feature_names, topn=12)

    with open(ASSIGN_PATH, "w", encoding="utf-8") as f:
        for i in range(n):
            sid = used_song_ids[i]
            orig_cid = int(labels[i])
            tid = int(merged_labels[i])
            record = {
                "row_idx": int(i),
                "song_id": sid,
                "topic_id": int(tid),
                "topic_top_keywords": top_terms[tid][:12],
                "orig_cluster_id": int(orig_cid),
                "orig_cluster_top_keywords": top_terms_orig[orig_cid][:12],
            }
            f.write(json.dumps(to_py(record), ensure_ascii=False) + "\n")

    print(f"[OK] Assignments saved: {ASSIGN_PATH}")
    print("[DONE]")


if __name__ == "__main__":
    main()
