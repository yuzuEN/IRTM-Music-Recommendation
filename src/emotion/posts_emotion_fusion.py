"""
組員 B — 貼文情緒向量融合模組
=================================================

功能：
組合多種情緒訊號，生成融合後的貼文情緒向量。

方法：
E_post = α·E_lexicon + β·E_emoji + γ·E_classifier

其中：
- E_lexicon: 基於 NRC Lexicon 的情緒向量
- E_emoji: 基於 Emoji 的情緒向量
- E_classifier: 基於 ML 模型的情緒向量
- α + β + γ = 1.0（權重和為 1）

輸入：
  outputs/emotion_vectors/EmotionVec_posts_lex.npy (Lexicon 情緒向量)
  outputs/emotion_vectors/EmotionVec_posts_emoji.npy (Emoji 情緒向量)
  outputs/emotion_vectors/EmotionVec_posts_model.npy (ML 模型情緒向量)

輸出：
  outputs/emotion_vectors/EmotionVec_posts_fused.npy (融合後的情緒向量)
  outputs/emotion_vectors/post_emotion_fusion_meta.json (融合元資料)
"""

import os
import json
import numpy as np
from typing import Tuple

############################################################
# 0. PATH CONFIGURATION
############################################################

OUT_DIR = "outputs/emotion_vectors"
os.makedirs(OUT_DIR, exist_ok=True)

# 輸入檔案
LEX_EMOTION_PATH = os.path.join(OUT_DIR, "EmotionVec_posts_lex.npy")
EMOJI_EMOTION_PATH = os.path.join(OUT_DIR, "EmotionVec_posts_emoji.npy")
MODEL_EMOTION_PATH = os.path.join(OUT_DIR, "EmotionVec_posts_model.npy")

# 輸出檔案
FUSED_EMOTION_PATH = os.path.join(OUT_DIR, "EmotionVec_posts_fused.npy")
FUSION_META_PATH = os.path.join(OUT_DIR, "post_emotion_fusion_meta.json")

# 融合權重（可調整）
WEIGHT_LEXICON = 0.3   # α: Lexicon 權重
WEIGHT_EMOJI = 0.2     # β: Emoji 權重
WEIGHT_CLASSIFIER = 0.5  # γ: ML 模型權重

# 驗證權重和為 1.0
assert abs(WEIGHT_LEXICON + WEIGHT_EMOJI + WEIGHT_CLASSIFIER - 1.0) < 1e-6, \
    f"Weights must sum to 1.0, got {WEIGHT_LEXICON + WEIGHT_EMOJI + WEIGHT_CLASSIFIER}"

############################################################
# 1. 載入情緒向量
############################################################

def load_emotion_vectors() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    載入三種情緒向量。
    
    返回：
        lex_emotion: (M, 8) - Lexicon 情緒向量
        emoji_emotion: (M, 8) - Emoji 情緒向量
        model_emotion: (M, 8) - ML 模型情緒向量
    """
    print("[Load] Loading emotion vectors...")
    
    # 載入 Lexicon 情緒向量
    lex_emotion = np.load(LEX_EMOTION_PATH, allow_pickle=True)
    print(f"[Load] Lexicon emotion vectors: {lex_emotion.shape}")
    
    # 載入 Emoji 情緒向量
    emoji_emotion = np.load(EMOJI_EMOTION_PATH, allow_pickle=True)
    print(f"[Load] Emoji emotion vectors: {emoji_emotion.shape}")
    
    # 載入 ML 模型情緒向量
    model_emotion = np.load(MODEL_EMOTION_PATH, allow_pickle=True)
    print(f"[Load] ML model emotion vectors: {model_emotion.shape}")
    
    # 檢查維度一致性
    num_posts = lex_emotion.shape[0]
    assert emoji_emotion.shape[0] == num_posts, \
        f"Emoji emotion count mismatch: {emoji_emotion.shape[0]} != {num_posts}"
    assert model_emotion.shape[0] == num_posts, \
        f"Model emotion count mismatch: {model_emotion.shape[0]} != {num_posts}"
    assert lex_emotion.shape[1] == 8, f"Lexicon emotion should be 8D, got {lex_emotion.shape[1]}D"
    assert emoji_emotion.shape[1] == 8, f"Emoji emotion should be 8D, got {emoji_emotion.shape[1]}D"
    assert model_emotion.shape[1] == 8, f"Model emotion should be 8D, got {model_emotion.shape[1]}D"
    
    return lex_emotion, emoji_emotion, model_emotion

############################################################
# 2. 融合情緒向量
############################################################

def fuse_emotion_vectors(
    lex_emotion: np.ndarray,
    emoji_emotion: np.ndarray,
    model_emotion: np.ndarray,
    weight_lex: float,
    weight_emoji: float,
    weight_model: float
) -> np.ndarray:
    """
    融合三種情緒向量。
    
    公式：
        E_fused = α·E_lexicon + β·E_emoji + γ·E_classifier
    
    參數：
        lex_emotion: (M, 8) - Lexicon 情緒向量
        emoji_emotion: (M, 8) - Emoji 情緒向量
        model_emotion: (M, 8) - ML 模型情緒向量
        weight_lex: float - Lexicon 權重 (α)
        weight_emoji: float - Emoji 權重 (β)
        weight_model: float - ML 模型權重 (γ)
    
    返回：
        fused_emotion: (M, 8) - 融合後的情緒向量（已正規化）
    """
    print(f"[Fuse] Fusing emotion vectors with weights: lex={weight_lex:.2f}, emoji={weight_emoji:.2f}, model={weight_model:.2f}")
    
    # 加權融合
    fused = (
        weight_lex * lex_emotion +
        weight_emoji * emoji_emotion +
        weight_model * model_emotion
    )
    
    # 正規化為機率分布（每行和為 1）
    row_sums = fused.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # 避免除以 0
    fused = fused / row_sums
    
    print(f"[Fuse] Fused emotion vectors shape: {fused.shape}")
    print(f"[Fuse] Sample row sum: {fused[0].sum():.4f} (should be ~1.0)")
    
    return fused

############################################################
# 3. Main
############################################################

def main():
    """
    主流程：融合貼文情緒向量。
    """
    print("=" * 60)
    print("Post Emotion Fusion")
    print("=" * 60)
    
    # 檢查輸入檔案
    required_files = [LEX_EMOTION_PATH, EMOJI_EMOTION_PATH, MODEL_EMOTION_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # 1. 載入情緒向量
    lex_emotion, emoji_emotion, model_emotion = load_emotion_vectors()
    
    # 2. 融合情緒向量
    fused_emotion = fuse_emotion_vectors(
        lex_emotion,
        emoji_emotion,
        model_emotion,
        WEIGHT_LEXICON,
        WEIGHT_EMOJI,
        WEIGHT_CLASSIFIER
    )
    
    # 3. 儲存融合後的情緒向量
    print(f"\n[Save] Saving fused emotion vectors...")
    np.save(FUSED_EMOTION_PATH, fused_emotion)
    print(f"[Save] Saved to {FUSED_EMOTION_PATH}")
    print(f"[Save] Shape: {fused_emotion.shape}")
    
    # 4. 儲存元資料
    meta = {
        "type": "fused_emotion",
        "method": "weighted_combination",
        "weights": {
            "lexicon": float(WEIGHT_LEXICON),
            "emoji": float(WEIGHT_EMOJI),
            "classifier": float(WEIGHT_CLASSIFIER),
        },
        "formula": "E_fused = α·E_lexicon + β·E_emoji + γ·E_classifier",
        "input_files": {
            "lexicon": os.path.basename(LEX_EMOTION_PATH),
            "emoji": os.path.basename(EMOJI_EMOTION_PATH),
            "classifier": os.path.basename(MODEL_EMOTION_PATH),
        },
        "output_file": os.path.basename(FUSED_EMOTION_PATH),
        "shape": list(fused_emotion.shape),
        "emotions": ["joy", "anger", "fear", "sadness", "surprise", "disgust", "excitement", "neutral"],
    }
    with open(FUSION_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Save] Saved metadata to {FUSION_META_PATH}")
    
    print("\n" + "=" * 60)
    print("✅ Post Emotion Fusion Complete!")
    print("=" * 60)
    print(f"\nOutput file:")
    print(f"  - {FUSED_EMOTION_PATH}")
    print(f"\nFusion weights:")
    print(f"  - Lexicon: {WEIGHT_LEXICON:.2f}")
    print(f"  - Emoji: {WEIGHT_EMOJI:.2f}")
    print(f"  - Classifier: {WEIGHT_CLASSIFIER:.2f}")

if __name__ == "__main__":
    main()

