"""
情緒融合權重調優實驗
=================================================

功能：
測試不同的情緒融合權重組合，找出最佳配置。

方法：
1. 載入帶標籤的貼文數據（posts_clean_expanded.jsonl 或 HuggingFace 數據集）
2. 對每篇貼文計算三種情緒向量（Lexicon, Emoji, Classifier）
3. 使用不同權重組合進行融合
4. 評估每種組合的準確率、F1-score 等指標
5. 找出最佳權重組合

使用方式：
  python pipeline/evaluate_emotion_fusion_weights.py --dataset posts
  python pipeline/evaluate_emotion_fusion_weights.py --dataset hf
  python pipeline/evaluate_emotion_fusion_weights.py --dataset posts --output results.json
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
import pathlib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# 添加項目根目錄到路徑
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from datasets import load_dataset

# 導入必要的模組
from src.emotion.posts_emotion_lex import compute_post_emotion, load_nrc_lexicon, EMOTIONS, EMOTION2IDX
import numpy as np
from src.emotion.posts_emotion_emoji import post_emoji_emotion, build_emoji_emotion_table, load_emoji_joined
from src.preprocessing.preprocess_post import PostPreprocessor

############################################################
# 0. PATH CONFIGURATION
############################################################

NRC_LEXICON_PATH = PROJECT_ROOT / "data" / "raw" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
EMOJI_JOINED_PATH = PROJECT_ROOT / "data" / "processed" / "posts" / "emoji_joined.txt"
EMOJI_TABLE_PATH = PROJECT_ROOT / "outputs" / "emotion_vectors" / "EmojiEmotionTable.npy"
EMOTION_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "post_emotion_lr.joblib"
POSTS_PATH = PROJECT_ROOT / "data" / "processed" / "posts" / "posts_clean_expanded.jsonl"

############################################################
# 1. 載入模型和數據
############################################################

def load_models_and_resources():
    """載入所有必要的模型和資源"""
    print("[Load] Loading models and resources...")
    
    # 1. NRC Lexicon
    nrc_lexicon = load_nrc_lexicon(str(NRC_LEXICON_PATH))
    
    # 2. Emoji Table
    if EMOJI_TABLE_PATH.exists():
        emoji_table = np.load(str(EMOJI_TABLE_PATH), allow_pickle=True).item()
    else:
        emo2phrases = load_emoji_joined(str(EMOJI_JOINED_PATH))
        emoji_table = build_emoji_emotion_table(emo2phrases, nrc_lexicon)
    
    # 3. Emotion Model
    emotion_model = joblib.load(str(EMOTION_MODEL_PATH))
    
    # 4. Preprocessor
    preprocessor = PostPreprocessor()
    
    print("[Load] All models loaded!")
    
    return {
        "nrc_lexicon": nrc_lexicon,
        "emoji_table": emoji_table,
        "emotion_model": emotion_model,
        "preprocessor": preprocessor,
    }

############################################################
# 2. 載入帶標籤的數據
############################################################

def load_posts_with_labels(posts_path: pathlib.Path) -> Tuple[List[str], np.ndarray]:
    """從 posts_clean_expanded.jsonl 載入貼文和標籤"""
    print(f"[Load] Loading posts from {posts_path}...")
    
    texts = []
    labels = []
    
    with open(posts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            
            raw_text = obj.get("raw_text", "")
            emotion = obj.get("emotion", "")
            
            if not raw_text or not emotion:
                continue
            
            # 將情緒名稱轉換為索引
            emotion_lower = emotion.lower()
            if emotion_lower in EMOTION2IDX:
                texts.append(raw_text)
                labels.append(EMOTION2IDX[emotion_lower])
    
    labels = np.array(labels, dtype=np.int64)
    print(f"[Load] Loaded {len(texts)} posts with labels")
    print(f"[Load] Label distribution: {np.bincount(labels)}")
    
    return texts, labels

def load_hf_dataset() -> Tuple[List[str], np.ndarray]:
    """從 HuggingFace 載入帶標籤的數據集"""
    print("[Load] Loading HuggingFace dataset...")
    
    ds = load_dataset("tanaos/synthetic-emotion-detection-dataset-v1")
    split = "train" if "train" in ds else list(ds.keys())[0]
    items = ds[split]
    
    texts = []
    labels = []
    
    # 檢查第一個樣本的結構
    sample = items[0]
    text_key = "text" if "text" in sample else "content"
    label_key = "label" if "label" in sample else "labels"
    
    for obj in items:
        text = obj.get(text_key, "")
        label = obj.get(label_key)
        
        if isinstance(text, str) and text.strip() and label is not None:
            # 確保 label 是整數
            try:
                label = int(label)
                if 0 <= label < 8:  # 8 種情緒
                    texts.append(text)
                    labels.append(label)
            except (ValueError, TypeError):
                continue
    
    labels = np.array(labels, dtype=np.int64)
    print(f"[Load] Loaded {len(texts)} posts from HuggingFace")
    print(f"[Load] Label distribution: {np.bincount(labels)}")
    
    return texts, labels

############################################################
# 3. 計算融合情緒向量
############################################################

def compute_fused_emotion(
    post_text: str,
    tokens: List[str],
    nrc_lexicon: dict,
    emoji_table: dict,
    emotion_model,
    weight_lexicon: float,
    weight_emoji: float,
    weight_classifier: float,
    conditional_fusion: bool = True
) -> np.ndarray:
    """
    計算融合後的情緒向量
    
    Args:
        conditional_fusion: 如果 True，當沒有 emoji 時會排除 emoji 信號並重新分配權重
    """
    # 1. Lexicon 情緒
    lex_emotion = compute_post_emotion(tokens, nrc_lexicon)
    
    # 2. Emoji 情緒
    emoji_emotion, emoji_used, emoji_seen = post_emoji_emotion(post_text, emoji_table)
    
    # 3. ML 模型情緒
    post_text_for_model = " ".join(tokens)
    model_probs = emotion_model.predict_proba([post_text_for_model])[0]
    model_emotion = model_probs.astype(np.float32)
    
    # 4. 檢查是否有實際信號（排除 neutral fallback）
    # Lexicon: 檢查是否為全 neutral（即 sum == 0 後的 fallback）
    # 如果只有 neutral 位置是 1.0，其他都是 0，表示是 fallback
    neutral_idx = 7  # neutral 是索引 7
    has_lexicon_signal = not (lex_emotion[neutral_idx] == 1.0 and lex_emotion.sum() == 1.0)
    
    # Emoji: 檢查是否有 emoji 且被 emoji_table 覆蓋
    # 如果 emoji_used == 0，會返回全 neutral，這是 fallback
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

############################################################
# 4. 評估輔助函數
############################################################

def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 2) -> float:
    """
    計算 Top-K 準確率：真實標籤是否在預測的前 K 個類別中。
    
    Args:
        probs: (n, 8) - 情緒概率向量
        y_true: (n,) - 真實標籤索引
        k: int - Top-K
    
    Returns:
        float - Top-K 準確率
    """
    topk = np.argsort(-probs, axis=1)[:, :k]  # 每行取前 k 個最大值的索引
    hit = np.any(topk == y_true[:, None], axis=1)  # 檢查真實標籤是否在前 k 個中
    return float(hit.mean())

def threshold_hit_accuracy(probs: np.ndarray, y_true: np.ndarray, thr: float = 0.2) -> float:
    """
    計算閾值命中準確率：
    - 如果任何概率 >= 閾值：正確當且僅當真實標籤在這些高概率類別中
    - 否則：回退到 argmax
    
    Args:
        probs: (n, 8) - 情緒概率向量
        y_true: (n,) - 真實標籤索引
        thr: float - 閾值
    
    Returns:
        float - 閾值命中準確率
    """
    mask = (probs >= thr)  # (n, 8) - 哪些類別 >= 閾值
    any_mask = mask.any(axis=1)  # (n,) - 是否有任何類別 >= 閾值
    
    # 如果有類別 >= 閾值：檢查真實標籤是否在其中
    hit_when_any = mask[np.arange(len(y_true)), y_true]  # (n,)
    
    # 如果沒有類別 >= 閾值：回退到 argmax
    top1 = np.argmax(probs, axis=1)
    hit_when_none = (top1 == y_true)
    
    hit = np.where(any_mask, hit_when_any, hit_when_none)
    return float(hit.mean())

############################################################
# 5. 評估函數
############################################################

def evaluate_weights(
    texts: List[str],
    true_labels: np.ndarray,
    models_and_resources: dict,
    weight_combinations: List[Tuple[float, float, float]]
) -> Dict[str, Dict]:
    """評估不同權重組合"""
    print(f"\n[Eval] Evaluating {len(weight_combinations)} weight combinations on {len(texts)} posts...")
    
    results = {}
    preprocessor = models_and_resources["preprocessor"]
    
    for idx, (w_lex, w_emoji, w_clf) in enumerate(weight_combinations):
        print(f"\n[Eval] Combination {idx+1}/{len(weight_combinations)}: "
              f"Lexicon={w_lex:.2f}, Emoji={w_emoji:.2f}, Classifier={w_clf:.2f}")
        
        # 計算所有貼文的融合情緒向量（保存完整向量）
        predictions = []
        emotion_vectors = []  # 保存完整的概率向量
        
        for i, text in enumerate(texts):
            if (i + 1) % 1000 == 0:
                print(f"  Processing {i+1}/{len(texts)}...")
            
            tokens = preprocessor.preprocess_text(text)
            fused_emotion = compute_fused_emotion(
                text,
                tokens,
                models_and_resources["nrc_lexicon"],
                models_and_resources["emoji_table"],
                models_and_resources["emotion_model"],
                w_lex,
                w_emoji,
                w_clf,
                conditional_fusion=True  # 啟用條件融合
            )
            
            # 保存完整向量
            emotion_vectors.append(fused_emotion)
            
            # 取 argmax 作為預測（用於傳統指標）
            pred = np.argmax(fused_emotion)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        emotion_vectors = np.array(emotion_vectors)  # (n, 8)
        
        # ========== 傳統指標（基於 argmax）==========
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        f1_per_class = f1_score(true_labels, predictions, average=None)
        
        # ========== 向量相似度指標（考慮完整向量）==========
        # 1. Top-K Accuracy (K=2, 3)
        top2_accuracy = topk_accuracy(emotion_vectors, true_labels, k=2)
        top3_accuracy = topk_accuracy(emotion_vectors, true_labels, k=3)
        
        # 2. Threshold Hit Accuracy
        threshold_accuracy_02 = threshold_hit_accuracy(emotion_vectors, true_labels, thr=0.2)
        threshold_accuracy_025 = threshold_hit_accuracy(emotion_vectors, true_labels, thr=0.25)
        threshold_accuracy_03 = threshold_hit_accuracy(emotion_vectors, true_labels, thr=0.3)
        
        # 3. Cosine Similarity（預測向量 vs 真實 one-hot 向量）
        true_onehot = np.eye(8)[true_labels]  # (n, 8)
        cosine_sims = np.sum(emotion_vectors * true_onehot, axis=1)  # 因為是 one-hot，直接取對應位置的值
        mean_cosine_sim = float(cosine_sims.mean())
        
        # 4. Cross-Entropy Loss（越小越好）
        # 避免 log(0)，加上小常數
        eps = 1e-15
        emotion_vectors_clipped = np.clip(emotion_vectors, eps, 1.0 - eps)
        cross_entropy = -np.sum(true_onehot * np.log(emotion_vectors_clipped), axis=1)
        mean_cross_entropy = float(cross_entropy.mean())
        
        # 5. Mean Squared Error (MSE)
        mse = np.mean((emotion_vectors - true_onehot) ** 2)
        mean_mse = float(mse)
        
        # 6. KL Divergence（預測分布 vs 真實分布）
        kl_div = np.sum(true_onehot * np.log((true_onehot + eps) / (emotion_vectors_clipped + eps)), axis=1)
        mean_kl_div = float(kl_div.mean())
        
        # 7. 預測向量的熵（衡量不確定性，越低越好）
        entropy = -np.sum(emotion_vectors * np.log(emotion_vectors + eps), axis=1)
        mean_entropy = float(entropy.mean())
        
        # 8. 預測分布分析（檢測 neutral 過度預測問題）
        pred_distribution = np.bincount(predictions, minlength=8) / len(predictions)  # 預測分布
        true_distribution = np.bincount(true_labels, minlength=8) / len(true_labels)  # 真實分布
        
        # Neutral 預測比例 vs 真實比例
        neutral_pred_ratio = float(pred_distribution[7])  # neutral 是索引 7
        neutral_true_ratio = float(true_distribution[7])
        neutral_overprediction = neutral_pred_ratio - neutral_true_ratio  # 過度預測量
        
        # 每個類別的預測比例 vs 真實比例差異
        class_distribution_diff = np.abs(pred_distribution - true_distribution)
        mean_distribution_diff = float(class_distribution_diff.mean())
        max_distribution_diff = float(class_distribution_diff.max())
        
        # Neutral 的 F1（單獨檢查）
        neutral_f1 = float(f1_per_class[7]) if len(f1_per_class) > 7 else 0.0
        
        weight_key = f"lex{w_lex:.2f}_emoji{w_emoji:.2f}_clf{w_clf:.2f}"
        results[weight_key] = {
            "weights": {
                "lexicon": float(w_lex),
                "emoji": float(w_emoji),
                "classifier": float(w_clf),
            },
            # 傳統指標（基於 argmax）
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "f1_per_class": [float(f) for f in f1_per_class],
            # 向量相似度指標
            "top2_accuracy": float(top2_accuracy),
            "top3_accuracy": float(top3_accuracy),
            "threshold_accuracy_0.2": float(threshold_accuracy_02),
            "threshold_accuracy_0.25": float(threshold_accuracy_025),
            "threshold_accuracy_0.3": float(threshold_accuracy_03),
            "mean_cosine_similarity": float(mean_cosine_sim),
            "mean_cross_entropy": float(mean_cross_entropy),
            "mean_mse": float(mean_mse),
            "mean_kl_divergence": float(mean_kl_div),
            "mean_entropy": float(mean_entropy),
            # 預測分布分析
            "prediction_distribution": [float(p) for p in pred_distribution],
            "true_distribution": [float(t) for t in true_distribution],
            "neutral_pred_ratio": float(neutral_pred_ratio),
            "neutral_true_ratio": float(neutral_true_ratio),
            "neutral_overprediction": float(neutral_overprediction),
            "neutral_f1": float(neutral_f1),
            "mean_distribution_diff": float(mean_distribution_diff),
            "max_distribution_diff": float(max_distribution_diff),
            # 預測結果（可選，如果數據量大可以不保存）
            # "predictions": predictions.tolist(),
            # "emotion_vectors": emotion_vectors.tolist(),  # 如果數據量大，建議不保存
        }
        
        print(f"  Accuracy (argmax): {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  Top-2 Accuracy: {top2_accuracy:.4f}")
        print(f"  Mean Cosine Similarity: {mean_cosine_sim:.4f}")
        print(f"  Mean Cross-Entropy: {mean_cross_entropy:.4f}")
        print(f"  Mean MSE: {mean_mse:.4f}")
        print(f"  Neutral: 預測={neutral_pred_ratio:.2%}, 真實={neutral_true_ratio:.2%}, "
              f"過度預測={neutral_overprediction:+.2%}, F1={neutral_f1:.4f}")
    
    return results

############################################################
# 6. 主函數
############################################################

def main():
    parser = argparse.ArgumentParser(description="Evaluate emotion fusion weight combinations")
    parser.add_argument("--dataset", type=str, default="posts", 
                       choices=["posts", "hf"],
                       help="Dataset to use: 'posts' (posts_clean_expanded.jsonl) or 'hf' (HuggingFace)")
    parser.add_argument("--output", type=str, default="outputs/emotion_vectors/weight_evaluation_results.json",
                       help="Output JSON file path")
    parser.add_argument("--max-posts", type=int, default=None,
                       help="Maximum number of posts to evaluate (for faster testing)")
    
    args = parser.parse_args()
    
    # 載入模型和資源
    models_and_resources = load_models_and_resources()
    
    # 載入數據
    if args.dataset == "posts":
        texts, true_labels = load_posts_with_labels(POSTS_PATH)
    else:
        texts, true_labels = load_hf_dataset()
    
    # 限制數據量（如果指定）
    if args.max_posts and args.max_posts < len(texts):
        print(f"[Info] Limiting to {args.max_posts} posts for faster evaluation")
        indices = np.random.choice(len(texts), args.max_posts, replace=False)
        texts = [texts[i] for i in indices]
        true_labels = true_labels[indices]
    
    # 定義要測試的權重組合
    weight_combinations = [
        # 當前配置
        (0.3, 0.2, 0.5),
        
        # ========== 精細測試區域：Classifier 0.85-1.0 ==========
        # Lexicon: 0.00-0.15, Emoji: 0.00-0.15, Classifier: 0.85-1.0
        
        # 純 ML 基準
        (0, 0, 1),
        
        # Emoji 0.02-0.15, Lexicon 0
        (0, 0.02, 0.98),
        (0, 0.03, 0.97),
        (0, 0.04, 0.96),
        (0, 0.05, 0.95),
        (0, 0.06, 0.94),
        (0, 0.07, 0.93),
        (0, 0.08, 0.92),
        (0, 0.09, 0.91),
        (0, 0.10, 0.90),
        (0, 0.12, 0.88),
        (0, 0.15, 0.85),
        
        # Lexicon 0.02-0.15, Emoji 0
        (0.02, 0, 0.98),
        (0.03, 0, 0.97),
        (0.04, 0, 0.96),
        (0.05, 0, 0.95),
        (0.06, 0, 0.94),
        (0.07, 0, 0.93),
        (0.08, 0, 0.92),
        (0.09, 0, 0.91),
        (0.10, 0, 0.90),
        (0.12, 0, 0.88),
        (0.15, 0, 0.85),
        
        # Lexicon + Emoji 組合（總和 0.05-0.15）
        (0.01, 0.04, 0.95),
        (0.02, 0.03, 0.95),
        (0.03, 0.02, 0.95),
        (0.04, 0.01, 0.95),
        (0.02, 0.05, 0.93),
        (0.03, 0.04, 0.93),
        (0.04, 0.03, 0.93),
        (0.05, 0.02, 0.93),
        (0.03, 0.05, 0.92),
        (0.04, 0.04, 0.92),
        (0.05, 0.03, 0.92),
        (0.02, 0.08, 0.90),
        (0.03, 0.07, 0.90),
        (0.04, 0.06, 0.90),
        (0.05, 0.05, 0.90),
        (0.06, 0.04, 0.90),
        (0.07, 0.03, 0.90),
        (0.08, 0.02, 0.90),
        (0.05, 0.08, 0.87),
        (0.06, 0.07, 0.87),
        (0.07, 0.06, 0.87),
        (0.08, 0.05, 0.87),
        (0.05, 0.10, 0.85),
        (0.07, 0.08, 0.85),
        (0.08, 0.07, 0.85),
        (0.10, 0.05, 0.85),
        
        # 模型優先（高權重）- 其他組合
        (0.2, 0.1, 0.7),
        (0.1, 0.1, 0.8),
        (0.15, 0.05, 0.8),
        
        
    ]
    
    print(f"\n[Info] Testing {len(weight_combinations)} weight combinations")
    print(f"[Info] Dataset: {args.dataset}, Posts: {len(texts)}")
    
    # 評估
    results = evaluate_weights(
        texts,
        true_labels,
        models_and_resources,
        weight_combinations
    )
    
    # ========== 按照推薦策略篩選 ==========
    # 策略：先用 Top-2 Accuracy 和 Mean Cosine Similarity 篩選
    #       再用 Cross-Entropy 和 MSE 細化
    #       最後看傳統 Accuracy 和 F1 作為參考
    
    # 第一步：篩選 Top-2 Accuracy 和 Cosine Similarity 都較高的組合
    # 取 Top-2 Accuracy 和 Cosine Similarity 的加權平均
    def combined_score_v1(result_data):
        """第一階段篩選：Top-2 Accuracy + Cosine Similarity"""
        top2 = result_data["top2_accuracy"]
        cosine = result_data["mean_cosine_similarity"]
        # 兩者等權重
        return 0.5 * top2 + 0.5 * cosine
    
    # 第二步：在第一步的候選中，用 Cross-Entropy 和 MSE 細化
    def combined_score_v2(result_data):
        """第二階段細化：Cross-Entropy + MSE（越小越好，所以取負值）"""
        ce = result_data["mean_cross_entropy"]
        mse = result_data["mean_mse"]
        # 兩者等權重，取負值（因為越小越好）
        return -(0.5 * ce + 0.5 * mse)
    
    # 綜合評分：結合兩階段
    def final_score(result_data):
        """最終評分：結合所有推薦指標（優先考慮 Top-2 Accuracy）"""
        # 主要看 Top-2 Accuracy（70% 權重），因為最終用途是計算相似度
        top2 = result_data["top2_accuracy"]
        cosine = result_data["mean_cosine_similarity"]
        v1 = combined_score_v1(result_data)  # Top-2 + Cosine
        v2 = combined_score_v2(result_data)  # Cross-Entropy + MSE（越小越好，取了負值）
        # Top-2 Accuracy 權重最高（0.7），其他指標綜合（0.3）
        return 0.7 * top2 + 0.15 * cosine + 0.15 * (v2 + 2.0)
    
    # 找出最佳組合
    best_by_accuracy = max(results.items(), key=lambda x: x[1]["accuracy"])
    best_by_f1_macro = max(results.items(), key=lambda x: x[1]["f1_macro"])
    best_by_f1_weighted = max(results.items(), key=lambda x: x[1]["f1_weighted"])
    best_by_top2 = max(results.items(), key=lambda x: x[1]["top2_accuracy"])
    best_by_cosine = max(results.items(), key=lambda x: x[1]["mean_cosine_similarity"])
    best_by_cross_entropy = min(results.items(), key=lambda x: x[1]["mean_cross_entropy"])
    best_by_combined_v1 = max(results.items(), key=lambda x: combined_score_v1(x[1]))
    best_by_combined_v2 = max(results.items(), key=lambda x: combined_score_v2(x[1]))
    best_by_final = max(results.items(), key=lambda x: final_score(x[1]))
    
    # 找出 neutral 過度預測最少的組合
    best_by_neutral_balance = min(results.items(), key=lambda x: abs(x[1]["neutral_overprediction"]))
    
    print("\n" + "=" * 80)
    print("評估結果總結")
    print("=" * 80)
    
    # 顯示真實標籤分布
    from src.emotion.posts_emotion_lex import EMOTIONS
    print(f"\n【數據集信息】")
    print(f"  總樣本數: {len(texts)}")
    true_dist = np.bincount(true_labels, minlength=8)
    print(f"  真實標籤分布:")
    for i, emotion in enumerate(EMOTIONS):
        print(f"    {emotion}: {true_dist[i]} ({true_dist[i]/len(true_labels):.2%})")
    neutral_true_count = np.sum(true_labels == 7)
    print(f"  Neutral 真實比例: {neutral_true_count/len(true_labels):.2%}")
    
    print(f"\n【推薦策略 - 最佳綜合組合】")
    print(f"  權重: {best_by_final[1]['weights']}")
    print(f"  Top-2 準確率: {best_by_final[1]['top2_accuracy']:.4f}")
    print(f"  平均餘弦相似度: {best_by_final[1]['mean_cosine_similarity']:.4f}")
    print(f"  平均交叉熵: {best_by_final[1]['mean_cross_entropy']:.4f}")
    print(f"  平均 MSE: {best_by_final[1]['mean_mse']:.4f}")
    print(f"  準確率 (參考): {best_by_final[1]['accuracy']:.4f}")
    print(f"  F1 (macro, 參考): {best_by_final[1]['f1_macro']:.4f}")
    print(f"  Neutral 過度預測: {best_by_final[1]['neutral_overprediction']:+.2%}")
    
    print(f"\n【第一階段篩選 - Top-2 + Cosine】")
    print(f"  權重: {best_by_combined_v1[1]['weights']}")
    print(f"  Top-2 準確率: {best_by_combined_v1[1]['top2_accuracy']:.4f}")
    print(f"  平均餘弦相似度: {best_by_combined_v1[1]['mean_cosine_similarity']:.4f}")
    
    print(f"\n【第二階段細化 - Cross-Entropy + MSE】")
    print(f"  權重: {best_by_combined_v2[1]['weights']}")
    print(f"  平均交叉熵: {best_by_combined_v2[1]['mean_cross_entropy']:.4f}")
    print(f"  平均 MSE: {best_by_combined_v2[1]['mean_mse']:.4f}")
    
    print(f"\n【Neutral 平衡最佳組合】")
    print(f"  權重: {best_by_neutral_balance[1]['weights']}")
    print(f"  Neutral 過度預測: {best_by_neutral_balance[1]['neutral_overprediction']:+.2%}")
    print(f"  Neutral 預測比例: {best_by_neutral_balance[1]['neutral_pred_ratio']:.2%}")
    print(f"  Neutral 真實比例: {best_by_neutral_balance[1]['neutral_true_ratio']:.2%}")
    print(f"  Neutral F1: {best_by_neutral_balance[1]['neutral_f1']:.4f}")
    print(f"  Top-2 準確率: {best_by_neutral_balance[1]['top2_accuracy']:.4f}")
    
    print(f"\n【傳統指標參考 - 最佳準確率】")
    print(f"  權重: {best_by_accuracy[1]['weights']}")
    print(f"  準確率: {best_by_accuracy[1]['accuracy']:.4f}")
    print(f"  F1 (macro): {best_by_accuracy[1]['f1_macro']:.4f}")
    print(f"  Neutral 過度預測: {best_by_accuracy[1]['neutral_overprediction']:+.2%}")
    
    # 保存結果
    output_data = {
        "dataset": args.dataset,
        "num_posts": len(texts),
        "label_distribution": np.bincount(true_labels).tolist(),
        "results": results,
        "best": {
            "by_accuracy": {
                "weights": best_by_accuracy[1]["weights"],
                "metrics": {
                    "accuracy": best_by_accuracy[1]["accuracy"],
                    "f1_macro": best_by_accuracy[1]["f1_macro"],
                    "f1_weighted": best_by_accuracy[1]["f1_weighted"],
                    "top2_accuracy": best_by_accuracy[1]["top2_accuracy"],
                }
            },
            "by_f1_macro": {
                "weights": best_by_f1_macro[1]["weights"],
                "metrics": {
                    "accuracy": best_by_f1_macro[1]["accuracy"],
                    "f1_macro": best_by_f1_macro[1]["f1_macro"],
                    "f1_weighted": best_by_f1_macro[1]["f1_weighted"],
                }
            },
            "by_top2_accuracy": {
                "weights": best_by_top2[1]["weights"],
                "metrics": {
                    "top2_accuracy": best_by_top2[1]["top2_accuracy"],
                    "mean_cosine_similarity": best_by_top2[1]["mean_cosine_similarity"],
                    "mean_cross_entropy": best_by_top2[1]["mean_cross_entropy"],
                }
            },
            "by_cosine_similarity": {
                "weights": best_by_cosine[1]["weights"],
                "metrics": {
                    "mean_cosine_similarity": best_by_cosine[1]["mean_cosine_similarity"],
                    "accuracy": best_by_cosine[1]["accuracy"],
                    "top2_accuracy": best_by_cosine[1]["top2_accuracy"],
                }
            },
            "by_cross_entropy": {
                "weights": best_by_cross_entropy[1]["weights"],
                "metrics": {
                    "mean_cross_entropy": best_by_cross_entropy[1]["mean_cross_entropy"],
                    "mean_mse": best_by_cross_entropy[1]["mean_mse"],
                    "accuracy": best_by_cross_entropy[1]["accuracy"],
                }
            },
            "by_combined_v1": {
                "weights": best_by_combined_v1[1]["weights"],
                "metrics": {
                    "top2_accuracy": best_by_combined_v1[1]["top2_accuracy"],
                    "mean_cosine_similarity": best_by_combined_v1[1]["mean_cosine_similarity"],
                }
            },
            "by_combined_v2": {
                "weights": best_by_combined_v2[1]["weights"],
                "metrics": {
                    "mean_cross_entropy": best_by_combined_v2[1]["mean_cross_entropy"],
                    "mean_mse": best_by_combined_v2[1]["mean_mse"],
                }
            },
            "by_final_recommended": {
                "weights": best_by_final[1]["weights"],
                "metrics": {
                    "top2_accuracy": best_by_final[1]["top2_accuracy"],
                    "mean_cosine_similarity": best_by_final[1]["mean_cosine_similarity"],
                    "mean_cross_entropy": best_by_final[1]["mean_cross_entropy"],
                    "mean_mse": best_by_final[1]["mean_mse"],
                    "accuracy": best_by_final[1]["accuracy"],
                    "f1_macro": best_by_final[1]["f1_macro"],
                    "neutral_overprediction": best_by_final[1]["neutral_overprediction"],
                }
            },
            "by_neutral_balance": {
                "weights": best_by_neutral_balance[1]["weights"],
                "metrics": {
                    "neutral_overprediction": best_by_neutral_balance[1]["neutral_overprediction"],
                    "neutral_pred_ratio": best_by_neutral_balance[1]["neutral_pred_ratio"],
                    "neutral_true_ratio": best_by_neutral_balance[1]["neutral_true_ratio"],
                    "neutral_f1": best_by_neutral_balance[1]["neutral_f1"],
                    "top2_accuracy": best_by_neutral_balance[1]["top2_accuracy"],
                }
            },
        }
    }
    
    # 確保輸出目錄存在
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Results saved to {args.output}")
    
    # 顯示前 5 名（按 Top-2 Accuracy 排序）
    print("\n" + "=" * 80)
    print("Top 5 組合（按 Top-2 Accuracy 排序）")
    print("=" * 80)
    sorted_results_top2 = sorted(results.items(), key=lambda x: x[1]["top2_accuracy"], reverse=True)
    for rank, (key, data) in enumerate(sorted_results_top2[:5], 1):
        print(f"\n{rank}. {key}")
        print(f"   權重: Lexicon={data['weights']['lexicon']:.2f}, "
              f"Emoji={data['weights']['emoji']:.2f}, "
              f"Classifier={data['weights']['classifier']:.2f}")
        print(f"   Top-2 Accuracy: {data['top2_accuracy']:.4f} | "
              f"Accuracy: {data['accuracy']:.4f} | "
              f"F1 (macro): {data['f1_macro']:.4f}")
        print(f"   餘弦相似度: {data['mean_cosine_similarity']:.4f} | "
              f"交叉熵: {data['mean_cross_entropy']:.4f} | "
              f"MSE: {data['mean_mse']:.4f}")
        print(f"   Neutral 過度預測: {data['neutral_overprediction']:+.2%}")
    
    # 也顯示按推薦策略綜合評分排序的 Top 5（作為參考）
    print("\n" + "=" * 80)
    print("Top 5 組合（按推薦策略綜合評分排序 - 參考）")
    print("=" * 80)
    sorted_results_final = sorted(results.items(), key=lambda x: final_score(x[1]), reverse=True)
    for rank, (key, data) in enumerate(sorted_results_final[:5], 1):
        print(f"\n{rank}. {key}")
        print(f"   權重: Lexicon={data['weights']['lexicon']:.2f}, "
              f"Emoji={data['weights']['emoji']:.2f}, "
              f"Classifier={data['weights']['classifier']:.2f}")
        print(f"   Top-2: {data['top2_accuracy']:.4f} | "
              f"餘弦相似度: {data['mean_cosine_similarity']:.4f} | "
              f"交叉熵: {data['mean_cross_entropy']:.4f} | "
              f"MSE: {data['mean_mse']:.4f}")
        print(f"   準確率 (參考): {data['accuracy']:.4f} | "
              f"F1 (參考): {data['f1_macro']:.4f}")
        print(f"   Neutral 過度預測: {data['neutral_overprediction']:+.2%}")
    
    print("\n" + "=" * 80)
    print("Top 5 組合（按 Neutral 平衡排序）")
    print("=" * 80)
    sorted_results_neutral = sorted(results.items(), key=lambda x: abs(x[1]["neutral_overprediction"]))
    for rank, (key, data) in enumerate(sorted_results_neutral[:5], 1):
        print(f"\n{rank}. {key}")
        print(f"   權重: Lexicon={data['weights']['lexicon']:.2f}, "
              f"Emoji={data['weights']['emoji']:.2f}, "
              f"Classifier={data['weights']['classifier']:.2f}")
        print(f"   Neutral 過度預測: {data['neutral_overprediction']:+.2%} "
              f"(預測={data['neutral_pred_ratio']:.2%}, 真實={data['neutral_true_ratio']:.2%})")
        print(f"   Top-2: {data['top2_accuracy']:.4f} | "
              f"餘弦相似度: {data['mean_cosine_similarity']:.4f}")

if __name__ == "__main__":
    main()

