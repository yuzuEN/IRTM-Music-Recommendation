import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


# ===============================
# Config
# ===============================
POSTS_PATH = r"C:\Users\tinti\Desktop\IRTM_music\data\processed\posts\posts_clean_expanded.jsonl"

OUT_DIR = "outputs/topic_vectors"
os.makedirs(OUT_DIR, exist_ok=True)

TFIDF_PATH = os.path.join(OUT_DIR, "posts_tfidf.npz")
TFIDF_META_PATH = os.path.join(OUT_DIR, "posts_tfidf_meta.json")

# Scan-K outputs
SCAN_TABLE_PATH = os.path.join(OUT_DIR, "posts_kmeans_scanK.tsv")
SCAN_JSON_PATH = os.path.join(OUT_DIR, "posts_kmeans_scanK.json")

# Final chosen-K outputs (same style as your original)
MODEL_PATH = os.path.join(OUT_DIR, "posts_kmeans_model.joblib")
ASSIGN_PATH = os.path.join(OUT_DIR, "posts_topic_assignments.jsonl")
SUMMARY_PATH = os.path.join(OUT_DIR, "posts_topic_summary.json")
EVAL_PATH = os.path.join(OUT_DIR, "posts_topic_eval.json")
KEYWORDS_TSV_PATH = os.path.join(OUT_DIR, "posts_topic_keywords.tsv")

# K range to scan
K_MIN = 3
K_MAX = 30   # inclusive
K_STEP = 1

# Evaluation settings
SIL_SAMPLE_N = 5000
RANDOM_STATE = 42

# Optional: treat tiny clusters as "bad"
MIN_CLUSTER_SIZE = 30  # you can set 0 to disable


# ===============================
# Utilities
# ===============================
def tokens_to_text(tokens: Any) -> str:
    if isinstance(tokens, list):
        return " ".join([str(t) for t in tokens if str(t).strip()])
    if isinstance(tokens, str):
        return tokens
    return ""


def load_posts(posts_jsonl_path: str, prefer: str = "expanded_tokens") -> Tuple[List[str], List[Dict[str, Any]]]:
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


def clean_for_tsv(s: str) -> str:
    return str(s).replace("\t", " ").replace("\n", " ").strip()


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


def sampled_silhouette_cosine(X: sparse.spmatrix, labels: np.ndarray, sample_n: int, random_state: int) -> Optional[float]:
    n = X.shape[0]
    if n <= 2:
        return None
    sample_n = min(sample_n, n)

    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    # silhouette supports sparse + cosine in sklearn
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


# ===============================
# Main
# ===============================
def main():
    # --------
    # 1) Load posts
    # --------
    texts, raws = load_posts(POSTS_PATH, prefer="expanded_tokens")
    n = len(texts)
    print(f"[INFO] Loaded posts: n={n}")

    # --------
    # 2) TF-IDF
    # --------
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    sparse.save_npz(TFIDF_PATH, X)
    with open(TFIDF_META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "posts_path": POSTS_PATH,
                "n_posts": int(n),
                "vocab_size": int(X.shape[1]),
                "tfidf": {
                    "ngram_range": [1, 2],
                    "max_features": 50000,
                    "min_df": 2,
                    "stop_words": "english",
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

        # choose best K by silhouette (simple + common)
        if sil is not None and sil > best_sil:
            best_sil = sil
            best_k = K

    # Save scan table (TSV)
    with open(SCAN_TABLE_PATH, "w", encoding="utf-8", newline="") as f:
        cols = ["K", "inertia_rss", "silhouette_cosine_sampled", "silhouette_sampled_n", "size_min", "size_max", "n_small_clusters"]
        f.write("\t".join(cols) + "\n")
        for r in scan_rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    # Save scan json
    scan_summary = {
        "posts_path": POSTS_PATH,
        "tfidf_path": TFIDF_PATH,
        "K_range": {"min": K_MIN, "max": K_MAX, "step": K_STEP},
        "silhouette": {"metric": "cosine", "sample_n": int(min(SIL_SAMPLE_N, X.shape[0]))},
        "min_cluster_size": int(MIN_CLUSTER_SIZE),
        "best_by_silhouette": {"K": int(best_k) if best_k is not None else None, "silhouette": float(best_sil) if best_k is not None else None},
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
    # 5) Topic keywords + sizes
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
    # 6) Save per-post assignments
    # --------
    with open(ASSIGN_PATH, "w", encoding="utf-8") as f:
        for i in range(n):
            obj = raws[i]
            raw_text = clean_for_tsv(obj.get("raw_text", ""))

            record = {
                "idx": i,
                "cluster_id": int(labels[i]),
                "cluster_top_keywords": top_terms[int(labels[i])][:12],
                "raw_text": raw_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] Assignments saved: {ASSIGN_PATH}")
    print("[DONE]")


if __name__ == "__main__":
    main()
