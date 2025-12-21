import os
import csv
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ✅ Lyrics-topic 專用 stopwords（只影響 topic TF-IDF）
LYRICS_TOPIC_STOPWORDS = {
    "yeah", "oh", "uh", "na", "la", "ya",
    "choru", "chorus", "verse", "hook",
    "da", "ah", "woah", "ooh",
    # 你這份結果很常見的 filler，也可以加（可選）
    "hey", "huh", "whoa", "mmm", "mm",
    # 歌唱結構
    "choru", "chorus", "verse", "vers", "bridg", "outro", "hook",

    # 功能動詞 / 口語
    "got", "get", "gotta", "got ta",
    "wan", "want", "gon", "gonna",
    "im", "em", "ta",
    "say", "said", "tell", "look", "come", "let", "make",

    # 填充語氣
    "yeah", "oh", "uh", "na", "la", "ah", "ooh", "woah",
    "hey", "huh", "mmm", "mm",

    # 髒話（topic 無資訊）
    "fuck", "shit", "bitch", "nigga", "ass", "damn", "hell", "yo",
}

def build_stopwords_for_lyrics_topic() -> List[str]:
    # english stopwords ∪ lyrics stopwords
    return sorted(set(ENGLISH_STOP_WORDS) | set(LYRICS_TOPIC_STOPWORDS))


# ===============================
# Config
# ===============================
LYRICS_TOKENS_PATH = r"data/processed/lyrics/lyrics_tokens.csv"
SONG_IDS_PATH = r"outputs/bm25_vectors/song_ids.json"   # ✅ C 的對齊順序

OUT_DIR = "outputs/topic_vectors_lyrics"
os.makedirs(OUT_DIR, exist_ok=True)

TFIDF_PATH = os.path.join(OUT_DIR, "lyrics_tfidf.npz")
TFIDF_META_PATH = os.path.join(OUT_DIR, "lyrics_tfidf_meta.json")

# Scan-K outputs
SCAN_TABLE_PATH = os.path.join(OUT_DIR, "lyrics_kmeans_scanK.tsv")
SCAN_JSON_PATH = os.path.join(OUT_DIR, "lyrics_kmeans_scanK.json")

# Final chosen-K outputs
MODEL_PATH = os.path.join(OUT_DIR, "lyrics_kmeans_model.joblib")
ASSIGN_PATH = os.path.join(OUT_DIR, "lyrics_topic_assignments.jsonl")
SUMMARY_PATH = os.path.join(OUT_DIR, "lyrics_topic_summary.json")
EVAL_PATH = os.path.join(OUT_DIR, "lyrics_topic_eval.json")
KEYWORDS_TSV_PATH = os.path.join(OUT_DIR, "lyrics_topic_keywords.tsv")

# ✅ 向量輸出（給 cosine similarity）
TOPIC_VEC_PATH = os.path.join(OUT_DIR, "TopicVec_songs_kmeans.npy")
TOPIC_VEC_META_PATH = os.path.join(OUT_DIR, "TopicVec_songs_kmeans_meta.json")

# K range to scan
K_MIN = 29
K_MAX = 29   # inclusive
K_STEP = 1

# Evaluation settings
SIL_SAMPLE_N = 5000
RANDOM_STATE = 42

# Optional: treat tiny clusters as "bad"
MIN_CLUSTER_SIZE = 30  # set 0 to disable


# ===============================
# Utilities
# ===============================
def clean_for_json(s: str) -> str:
    return str(s).replace("\t", " ").replace("\n", " ").strip()


def load_song_ids(song_ids_path: str) -> List[str]:
    with open(song_ids_path, "r", encoding="utf-8") as f:
        song_ids = json.load(f)
    if not isinstance(song_ids, list) or not song_ids:
        raise ValueError("song_ids.json is empty or not a list.")
    return [str(x) for x in song_ids]


def load_lyrics_tokens_csv(csv_path: str) -> Dict[str, str]:
    """
    Read lyrics_tokens.csv with columns: song_id,tokens
    Return: dict song_id -> tokens(str)
    """
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
    """
    Align lyrics documents to C's song_ids order.
    Returns:
      texts_aligned: list[str] in aligned order
      used_song_ids: list[str] matched to texts_aligned
      stats: dict about missing
    """
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
                # skip
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
    return {
        "size_min": mn,
        "size_max": mx,
        "n_small_clusters": n_small,
    }


def make_topic_vector_hard(labels: np.ndarray, K: int) -> np.ndarray:
    """
    Hard KMeans -> one-hot topic vector (n, K)
    """
    n = labels.shape[0]
    vec = np.zeros((n, K), dtype=np.float32)
    vec[np.arange(n), labels] = 1.0
    return vec


# ===============================
# Main
# ===============================
def main():
    # --------
    # 1) Load + align lyrics to C's song_ids order
    # --------
    song_ids = load_song_ids(SONG_IDS_PATH)
    songid_to_tokens = load_lyrics_tokens_csv(LYRICS_TOKENS_PATH)

    texts, used_song_ids, align_stats = align_lyrics_to_song_ids(
        song_ids=song_ids,
        songid_to_tokens=songid_to_tokens,
        missing_policy="drop",  # ✅ 建議 drop，因為空文本會污染 TF-IDF/聚類
    )

    n = len(texts)
    print(f"[INFO] Loaded song_ids: {len(song_ids)}")
    print(f"[INFO] Lyrics docs used: {n}")
    print(f"[INFO] Missing lyrics in song_ids: {align_stats['n_missing']} (policy={align_stats['missing_policy']})")

    if n < 10:
        raise RuntimeError("Too few lyrics documents after alignment; check your files/paths.")

    # --------
    # 2) TF-IDF (lyrics-friendly)
    # --------
    stopwords = build_stopwords_for_lyrics_topic()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        max_features=50000,
        min_df=5,
        max_df=0.5,
        stop_words=stopwords,  # ✅ 合併後的 stopwords
    )
    X = vectorizer.fit_transform(texts)  # sparse (n, V)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # ============================================================
    # DEBUG A: Global TF-IDF importance (sum over all documents)
    # ============================================================
    term_tfidf_sum = np.asarray(X.sum(axis=0)).ravel()
    order_tfidf = np.argsort(-term_tfidf_sum)

    TOP_N = 200  # 想看多少個，先 200 很剛好

    print("\n" + "="*60)
    print("[DEBUG A] Top terms by GLOBAL TF-IDF weight")
    print("="*60)
    print("rank\tterm\ttfidf_sum")

    for rank, idx in enumerate(order_tfidf[:TOP_N], start=1):
        print(f"{rank}\t{feature_names[idx]}\t{term_tfidf_sum[idx]:.4f}")


    # ============================================================
    # DEBUG B: Document Frequency (how many songs contain the term)
    # ============================================================
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    df_ratio = df / X.shape[0]
    order_df = np.argsort(-df)

    print("\n" + "="*60)
    print("[DEBUG B] Top terms by DOCUMENT FREQUENCY")
    print("="*60)
    print("rank\tterm\tdf\tdf_ratio")

    for rank, idx in enumerate(order_df[:TOP_N], start=1):
        print(
            f"{rank}\t{feature_names[idx]}"
            f"\t{df[idx]}"
            f"\t{df_ratio[idx]:.2%}"
        )


    sparse.save_npz(TFIDF_PATH, X)
    with open(TFIDF_META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "lyrics_tokens_path": LYRICS_TOKENS_PATH,
                "song_ids_path": SONG_IDS_PATH,
                "align_stats": align_stats,
                "n_songs_used": int(n),
                "vocab_size": int(X.shape[1]),
                "tfidf": {
                    "ngram_range": [1, 2],
                    "max_features": 50000,
                    "min_df": 5,
                    "max_df": 0.5,
                    "stop_words": {
                        "english": True,
                        "lyrics_topic_stopwords": sorted(list(LYRICS_TOPIC_STOPWORDS)),
                    }
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] TF-IDF saved: {TFIDF_PATH}")
    print(f"[INFO] TF-IDF shape: {X.shape}")

    # --------
    # 3) Scan K
    # --------
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

    # Save scan table (TSV)
    with open(SCAN_TABLE_PATH, "w", encoding="utf-8", newline="") as f:
        cols = [
            "K", "inertia_rss", "silhouette_cosine_sampled", "silhouette_sampled_n",
            "size_min", "size_max", "n_small_clusters"
        ]
        f.write("\t".join(cols) + "\n")
        for r in scan_rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    # Save scan json
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
        json.dump(scan_summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Scan table saved: {SCAN_TABLE_PATH}")
    print(f"[OK] Scan json saved : {SCAN_JSON_PATH}")
    print(f"[INFO] Best K by silhouette: {best_k} (sil={best_sil:.4f})")

    # --------
    # 4) Train final KMeans with chosen K
    # --------
    if best_k is None:
        raise RuntimeError("No valid silhouette score produced; cannot choose K automatically.")

    K = best_k
    kmeans = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(X)
    joblib.dump(kmeans, MODEL_PATH)

    print(f"[OK] Final KMeans model saved: {MODEL_PATH}")
    print(f"[INFO] Final KMeans inertia (RSS): {kmeans.inertia_:.2f}")

    # --------
    # 5) Topic keywords + sizes + eval
    # --------
    top_terms = get_top_terms_per_cluster(X, labels, feature_names, topn=12)
    sizes = {int(cid): int((labels == cid).sum()) for cid in range(K)}

    summary = {
        "K": int(K),
        "cluster_sizes": sizes,
        "top_terms": {str(k): v for k, v in top_terms.items()},
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    eval_info = {
        "K": int(K),
        "inertia_rss": float(kmeans.inertia_),
        "silhouette_metric": "cosine",
        "silhouette_sampled_n": int(min(SIL_SAMPLE_N, X.shape[0])),
        "silhouette_sampled": float(sampled_silhouette_cosine(X, labels, sample_n=SIL_SAMPLE_N, random_state=RANDOM_STATE)),
        "cluster_stats": cluster_stats(labels, K),
    }
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_info, f, ensure_ascii=False, indent=2)

    # TSV keywords for report
    with open(KEYWORDS_TSV_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("topic_id\tsize\ttop_keywords\n")
        for cid in range(K):
            kw = ", ".join(top_terms[cid])
            f.write(f"{cid}\t{sizes[cid]}\t{kw}\n")

    print(f"[OK] Topic summary saved: {SUMMARY_PATH}")
    print(f"[OK] Topic eval saved   : {EVAL_PATH}")
    print(f"[OK] Topic keywords TSV : {KEYWORDS_TSV_PATH}")

    # --------
    # 6) Save aligned topic vectors (for cosine reranking)
    # --------
    topic_vec = make_topic_vector_hard(labels, K)  # (n_used, K)
    np.save(TOPIC_VEC_PATH, topic_vec)

    meta = {
        "type": "kmeans_onehot",
        "K": int(K),
        "shape": [int(topic_vec.shape[0]), int(topic_vec.shape[1])],
        "aligned_to_song_ids_json": True,
        "song_ids_path": SONG_IDS_PATH,
        "note": "Row i corresponds to used_song_ids[i] (subset of song_ids.json if missing_policy=drop).",
        "align_stats": align_stats,
        "tfidf_path": TFIDF_PATH,
        "model_path": MODEL_PATH,
    }
    with open(TOPIC_VEC_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Topic vectors saved: {TOPIC_VEC_PATH}")
    print(f"[OK] Topic vector meta  : {TOPIC_VEC_META_PATH}")

    # --------
    # 7) Save per-song assignments (for inspection)
    # --------
    with open(ASSIGN_PATH, "w", encoding="utf-8") as f:
        for i in range(n):
            sid = used_song_ids[i]
            record = {
                "row_idx": i,
                "song_id": sid,
                "cluster_id": int(labels[i]),
                "cluster_top_keywords": top_terms[int(labels[i])][:12],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] Assignments saved: {ASSIGN_PATH}")
    print("[DONE]")


if __name__ == "__main__":
    main()
