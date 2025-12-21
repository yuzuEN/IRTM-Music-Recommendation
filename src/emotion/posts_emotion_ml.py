import os
import json
import re
import joblib
import numpy as np

from typing import Tuple, List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ✅ NEW: HuggingFace datasets loader
from datasets import load_dataset


# ===============================
# Unified 8-class emotion space (MUST match project)
# ===============================
EMOTIONS = ["joy", "anger", "fear", "sadness", "surprise", "disgust", "excitement", "neutral"]
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}


# ===============================
# Helpers: robust dataset loading (kept for optional local files)
# ===============================
def _is_int_like(x: Any) -> bool:
    try:
        int(x)
        return True
    except Exception:
        return False


def _normalize_label(label: Any) -> int:
    """
    Accepts:
      - int-like 0..7
      - string label in EMOTIONS
    Returns:
      int in [0,7]
    """
    if label is None:
        raise ValueError("label is None")

    # numeric label
    if _is_int_like(label):
        y = int(label)
        if 0 <= y < len(EMOTIONS):
            return y
        raise ValueError(f"label int out of range: {y}")

    # string label
    lab = str(label).strip().lower()
    if lab in EMO2IDX:
        return EMO2IDX[lab]
    raise ValueError(f"unknown label string: {label}")


def _guess_text_key(obj: Dict[str, Any]) -> str:
    """
    Common text keys in HF / custom datasets.
    """
    candidates = ["text", "sentence", "content", "caption", "raw_text"]
    for k in candidates:
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            return k
    # fallback: find any str field with longest length
    best_k, best_len = None, 0
    for k, v in obj.items():
        if isinstance(v, str):
            ln = len(v.strip())
            if ln > best_len:
                best_k, best_len = k, ln
    if best_k is None:
        raise ValueError("Cannot guess text key from sample.")
    return best_k


def _guess_label_key(obj: Dict[str, Any]) -> str:
    """
    Common label keys.
    """
    candidates = ["label", "labels", "emotion", "target", "y"]
    for k in candidates:
        if k in obj:
            return k
    raise ValueError("Cannot guess label key from sample.")


def load_supervised_dataset(path: str) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """
    Supports:
      - .jsonl : each line is a JSON object
      - .json  : a list of JSON objects, or { "train": [...] } style
      - .csv   : requires pandas

    Returns:
      texts: List[str]
      y: np.ndarray (n,)
      info: dict with detected schema keys
    """
    path_lower = path.lower()
    texts: List[str] = []
    labels: List[int] = []

    detected = {"text_key": None, "label_key": None, "format": None}

    if path_lower.endswith(".jsonl"):
        detected["format"] = "jsonl"
        with open(path, "r", encoding="utf-8") as f:
            first_obj = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                first_obj = obj
                break
        if first_obj is None:
            raise ValueError("Empty jsonl dataset.")

        text_key = _guess_text_key(first_obj)
        label_key = _guess_label_key(first_obj)
        detected["text_key"], detected["label_key"] = text_key, label_key

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                try:
                    text = obj.get(text_key, "")
                    y = _normalize_label(obj.get(label_key))
                except Exception:
                    continue
                if isinstance(text, str) and text.strip():
                    texts.append(text)
                    labels.append(y)

    elif path_lower.endswith(".json"):
        detected["format"] = "json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "train" in data and isinstance(data["train"], list):
                items = data["train"]
            else:
                list_values = [v for v in data.values() if isinstance(v, list)]
                if not list_values:
                    raise ValueError("JSON dict contains no list data.")
                items = list_values[0]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError("Unsupported JSON structure.")

        if not items:
            raise ValueError("Empty json dataset list.")

        text_key = _guess_text_key(items[0])
        label_key = _guess_label_key(items[0])
        detected["text_key"], detected["label_key"] = text_key, label_key

        for obj in items:
            if not isinstance(obj, dict):
                continue
            try:
                text = obj.get(text_key, "")
                y = _normalize_label(obj.get(label_key))
            except Exception:
                continue
            if isinstance(text, str) and text.strip():
                texts.append(text)
                labels.append(y)

    elif path_lower.endswith(".csv"):
        detected["format"] = "csv"
        import pandas as pd
        df = pd.read_csv(path)

        text_col = None
        for c in ["text", "sentence", "content", "caption", "raw_text"]:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            obj_cols = [c for c in df.columns if df[c].dtype == "object"]
            if not obj_cols:
                raise ValueError("Cannot guess text column from CSV.")
            text_col = obj_cols[0]

        label_col = None
        for c in ["label", "emotion", "target", "y"]:
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            raise ValueError("Cannot guess label column from CSV.")

        detected["text_key"], detected["label_key"] = text_col, label_col

        for _, row in df.iterrows():
            try:
                text = row[text_col]
                y = _normalize_label(row[label_col])
            except Exception:
                continue
            if isinstance(text, str) and text.strip():
                texts.append(text)
                labels.append(y)
    else:
        raise ValueError("Unsupported dataset file type. Use .jsonl / .json / .csv")

    y_arr = np.array(labels, dtype=np.int64)
    if len(texts) != len(y_arr) or len(texts) == 0:
        raise ValueError("Loaded dataset is empty or mismatched.")

    detected["n"] = len(texts)
    return texts, y_arr, detected


# ===============================
# NEW: Load HuggingFace dataset
# ===============================
def load_hf_dataset_tanaos() -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """
    Load:
      tanaos/synthetic-emotion-detection-dataset-v1

    Expected fields (HF typical):
      - text: str
      - label: int 0..7 (aligned with EMOTIONS order in your project)

    Returns:
      texts, y, info
    """
    ds = load_dataset("tanaos/synthetic-emotion-detection-dataset-v1")

    # Usually only 'train' split exists (or train/test). We'll use train.
    split = "train" if "train" in ds else list(ds.keys())[0]
    items = ds[split]

    texts: List[str] = []
    labels: List[int] = []

    # Quick schema check from first item
    sample = items[0]
    text_key = "text" if "text" in sample else _guess_text_key(sample)
    label_key = "label" if "label" in sample else _guess_label_key(sample)

    for obj in items:
        try:
            text = obj.get(text_key, "")
            y = _normalize_label(obj.get(label_key))
        except Exception:
            continue
        if isinstance(text, str) and text.strip():
            texts.append(text)
            labels.append(y)

    y_arr = np.array(labels, dtype=np.int64)
    info = {
        "source": "huggingface",
        "name": "tanaos/synthetic-emotion-detection-dataset-v1",
        "split": split,
        "text_key": text_key,
        "label_key": label_key,
        "n": len(texts),
    }
    return texts, y_arr, info


# ===============================
# Inference input prep (posts)
# ===============================
def tokens_to_text(tokens: Any) -> str:
    if isinstance(tokens, list):
        return " ".join([str(t) for t in tokens if str(t).strip()])
    if isinstance(tokens, str):
        return tokens
    return ""


def load_posts_as_texts(posts_jsonl_path: str, prefer: str = "expanded_tokens") -> List[str]:
    texts: List[str] = []
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
                    tokens = obj.get("expanded_tokens") or obj.get("clean_tokens")
                text = tokens_to_text(tokens)

            texts.append(text if isinstance(text, str) else "")
    return texts


# ===============================
# Train + Evaluate + Save model
# ===============================
def train_lr_model(texts: List[str], y: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
        )),
        ("lr", LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1,
        ))
    ])
    model.fit(texts, y)
    return model




def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 2) -> float:
    """
    probs: (n,8), y_true: (n,)
    Return: fraction where true label is in top-k predicted classes.
    """
    topk = np.argsort(-probs, axis=1)[:, :k]
    hit = np.any(topk == y_true[:, None], axis=1)
    return float(hit.mean())


def threshold_hit_accuracy(probs: np.ndarray, y_true: np.ndarray, thr: float = 0.2) -> float:
    """
    Your rule:
      - if any prob >= thr: correct if true label is among those
      - else: fallback to argmax
    """
    mask = (probs >= thr)  # (n,8)
    any_mask = mask.any(axis=1)  # (n,)

    # Hit if true label included in threshold set
    hit_when_any = mask[np.arange(len(y_true)), y_true]  # (n,)

    # Fallback: argmax if none >= thr
    top1 = np.argmax(probs, axis=1)
    hit_when_none = (top1 == y_true)

    hit = np.where(any_mask, hit_when_any, hit_when_none)
    return float(hit.mean())


def probs_to_multihot(probs: np.ndarray, thr: float = 0.2) -> np.ndarray:
    """
    Convert probs (n,8) to multi-hot (n,8) using your threshold rule.
    Used for recommendation vectors if you want 0/1 emotions.
    """
    out = (probs >= thr).astype(np.int8)
    none_mask = (out.sum(axis=1) == 0)
    if np.any(none_mask):
        top = np.argmax(probs[none_mask], axis=1)
        out[none_mask] = 0
        out[none_mask, top] = 1
    return out



def main():
    # ============
    # Paths (edit these)
    # ============
    # ✅ No local dataset path needed now (HF will download automatically)
    POSTS_PATH = r"C:\Users\tinti\Desktop\IRTM_music\data\processed\posts\posts_clean_expanded.jsonl"

    MODEL_DIR = "outputs/models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, "post_emotion_lr.joblib")

    OUT_VEC_PATH = "outputs/emotion_vectors/EmotionVec_posts_model.npy"
    OUT_META_PATH = "outputs/emotion_vectors/post_emotion_ml_metadata.json"
    os.makedirs(os.path.dirname(OUT_VEC_PATH), exist_ok=True)

    # ============
    # 1) Load supervised dataset (HuggingFace)
    # ============
    X, y, info = load_hf_dataset_tanaos()
    print(f"[INFO] Loaded dataset: n={len(X)} | source={info.get('source')} | name={info.get('name')} | split={info.get('split')} | text_key={info.get('text_key')} | label_key={info.get('label_key')}")

    # ============
    # 2) Train/Test split & train
    # ============
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = train_lr_model(X_train, y_train)

    # ============
    # 3) Evaluate
    # ============
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"[RESULT] Top-1 Accuracy (argmax): {acc:.4f}")
    print("[RESULT] Confusion Matrix:")
    print(cm)
    print("[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))

    # ---- NEW: extra metrics for "close scores" + multi-emotion threshold ----
    probs_test = clf.predict_proba(X_test).astype(np.float32)  # (n_test, 8)
    thr = 0.2  # ✅ 你要的門檻，想改成 0.25 就改這裡

    acc_top2 = topk_accuracy(probs_test, y_test, k=2)
    acc_thr_hit = threshold_hit_accuracy(probs_test, y_test, thr=thr)

    print(f"[RESULT] Top-2 Accuracy: {acc_top2:.4f}")
    print(f"[RESULT] Threshold-hit Accuracy (thr={thr}): {acc_thr_hit:.4f}")


    joblib.dump(clf, MODEL_PATH)
    print(f"[OK] Saved model to {MODEL_PATH}")

    # ============
    # 4) Inference on your posts
    # ============
    post_texts = load_posts_as_texts(POSTS_PATH, prefer="expanded_tokens")
    probs = clf.predict_proba(post_texts).astype(np.float32)  # (num_posts, 8)

    s = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, np.maximum(s, 1e-8))

    np.save(OUT_VEC_PATH, probs)
    print(f"[OK] Saved Post Emotion (ML) vectors to {OUT_VEC_PATH}")
    print(f"Shape: {probs.shape}")
    print(f"Row[0] sum: {probs[0].sum():.4f}")

    # ============
    # 5) Save metadata for report + reproducibility
    # ============
    meta = {
        "model": "TFIDF + LogisticRegression(multinomial)",
        "emotions_order": EMOTIONS,
        "dataset": info,
        "train_test_split": {"test_size": 0.2, "random_state": 42, "stratify": True},
        "tfidf": {"ngram_range": [1, 2], "max_features": 50000, "min_df": 2},
        "logreg": {"solver": "lbfgs", "multi_class": "multinomial", "max_iter": 2000},
        "eval": {"accuracy": float(acc)},
        "posts_path": POSTS_PATH,
        "output_vector_path": OUT_VEC_PATH,
        "output_shape": list(probs.shape),
    }
    with open(OUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved metadata to {OUT_META_PATH}")


if __name__ == "__main__":
    main()
