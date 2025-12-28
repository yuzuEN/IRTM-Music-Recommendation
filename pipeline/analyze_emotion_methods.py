"""
情緒方法分析腳本
=================================================

分析為什麼純 ML 效果最好，以及 Emoji 和 Lexicon 的貢獻
"""

import json
import numpy as np
import pathlib
import sys
from collections import Counter
from typing import Dict, List, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def analyze_coverage():
    """分析 Emoji 和 Lexicon 的覆蓋率"""
    print("=" * 80)
    print("1. 覆蓋率分析")
    print("=" * 80)
    
    posts_path = PROJECT_ROOT / "data" / "processed" / "posts" / "posts_clean_expanded.jsonl"
    
    # 統計有 emoji 的貼文比例
    total = 0
    has_emoji = 0
    has_emoji_covered = 0
    
    # 統計有 lexicon 匹配的貼文比例
    from src.emotion.posts_emotion_lex import load_nrc_lexicon, compute_post_emotion
    from src.emotion.posts_emotion_emoji import post_emoji_emotion, build_emoji_emotion_table, load_emoji_joined
    from src.preprocessing.preprocess_post import PostPreprocessor
    
    nrc_lexicon = load_nrc_lexicon(str(PROJECT_ROOT / "data" / "raw" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"))
    emo2phrases = load_emoji_joined(str(PROJECT_ROOT / "data" / "processed" / "posts" / "emoji_joined.txt"))
    emoji_table = build_emoji_emotion_table(emo2phrases, nrc_lexicon)
    preprocessor = PostPreprocessor()
    
    lexicon_coverage = 0
    emoji_coverage = 0
    both_coverage = 0
    neither_coverage = 0
    
    with open(posts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_text = obj.get("raw_text", "")
            tokens = preprocessor.preprocess_text(raw_text)
            
            total += 1
            
            # 檢查 Lexicon
            lex_emotion = compute_post_emotion(tokens, nrc_lexicon)
            has_lexicon = lex_emotion.sum() > 0
            
            # 檢查 Emoji
            emoji_emotion, used, seen = post_emoji_emotion(raw_text, emoji_table)
            has_emoji_signal = used > 0
            
            if has_lexicon:
                lexicon_coverage += 1
            if has_emoji_signal:
                emoji_coverage += 1
            if has_lexicon and has_emoji_signal:
                both_coverage += 1
            if not has_lexicon and not has_emoji_signal:
                neither_coverage += 1
    
    print(f"\n總貼文數: {total}")
    print(f"\nLexicon 覆蓋率:")
    print(f"  有匹配的貼文: {lexicon_coverage} ({lexicon_coverage/total:.2%})")
    print(f"  無匹配的貼文: {total - lexicon_coverage} ({(total-lexicon_coverage)/total:.2%})")
    
    print(f"\nEmoji 覆蓋率:")
    print(f"  有 emoji 且被覆蓋的貼文: {emoji_coverage} ({emoji_coverage/total:.2%})")
    print(f"  無 emoji 或未被覆蓋的貼文: {total - emoji_coverage} ({(total-emoji_coverage)/total:.2%})")
    
    print(f"\n覆蓋組合:")
    print(f"  兩者都有: {both_coverage} ({both_coverage/total:.2%})")
    print(f"  只有 Lexicon: {lexicon_coverage - both_coverage} ({(lexicon_coverage-both_coverage)/total:.2%})")
    print(f"  只有 Emoji: {emoji_coverage - both_coverage} ({(emoji_coverage-both_coverage)/total:.2%})")
    print(f"  兩者都沒有: {neither_coverage} ({neither_coverage/total:.2%})")
    
    return {
        "total": total,
        "lexicon_coverage": lexicon_coverage,
        "emoji_coverage": emoji_coverage,
        "both_coverage": both_coverage,
        "neither_coverage": neither_coverage,
    }

def analyze_quality():
    """分析各方法的預測質量"""
    print("\n" + "=" * 80)
    print("2. 預測質量分析")
    print("=" * 80)
    
    results_path = PROJECT_ROOT / "outputs" / "emotion_vectors" / "weight_evaluation_results.json"
    
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data["results"]
    
    # 提取純方法結果
    pure_ml = results.get("lex0.00_emoji0.00_clf1.00", {})
    pure_emoji = results.get("lex0.00_emoji1.00_clf0.00", {})
    pure_lexicon = results.get("lex1.00_emoji0.00_clf0.00", {})
    current = results.get("lex0.30_emoji0.20_clf0.50", {})
    
    print(f"\n【純 ML (Classifier)】")
    if pure_ml:
        print(f"  準確率: {pure_ml.get('accuracy', 0):.4f}")
        print(f"  Top-2 準確率: {pure_ml.get('top2_accuracy', 0):.4f}")
        print(f"  F1 (macro): {pure_ml.get('f1_macro', 0):.4f}")
        print(f"  平均餘弦相似度: {pure_ml.get('mean_cosine_similarity', 0):.4f}")
        print(f"  平均交叉熵: {pure_ml.get('mean_cross_entropy', 0):.4f}")
        print(f"  Neutral 過度預測: {pure_ml.get('neutral_overprediction', 0):+.2%}")
    
    print(f"\n【純 Lexicon】")
    if pure_lexicon:
        print(f"  準確率: {pure_lexicon.get('accuracy', 0):.4f}")
        print(f"  Top-2 準確率: {pure_lexicon.get('top2_accuracy', 0):.4f}")
        print(f"  F1 (macro): {pure_lexicon.get('f1_macro', 0):.4f}")
        print(f"  平均餘弦相似度: {pure_lexicon.get('mean_cosine_similarity', 0):.4f}")
        print(f"  平均交叉熵: {pure_lexicon.get('mean_cross_entropy', 0):.4f}")
        print(f"  Neutral 過度預測: {pure_lexicon.get('neutral_overprediction', 0):+.2%}")
    
    print(f"\n【純 Emoji】")
    if pure_emoji:
        print(f"  準確率: {pure_emoji.get('accuracy', 0):.4f}")
        print(f"  Top-2 準確率: {pure_emoji.get('top2_accuracy', 0):.4f}")
        print(f"  F1 (macro): {pure_emoji.get('f1_macro', 0):.4f}")
        print(f"  平均餘弦相似度: {pure_emoji.get('mean_cosine_similarity', 0):.4f}")
        print(f"  平均交叉熵: {pure_emoji.get('mean_cross_entropy', 0):.4f}")
        print(f"  Neutral 過度預測: {pure_emoji.get('neutral_overprediction', 0):+.2%}")
        print(f"  [警告] 注意：交叉熵異常高 ({pure_emoji.get('mean_cross_entropy', 0):.2f})，表示預測分布很差")
    
    print(f"\n【當前組合 (0.3, 0.2, 0.5)】")
    if current:
        print(f"  準確率: {current.get('accuracy', 0):.4f}")
        print(f"  Top-2 準確率: {current.get('top2_accuracy', 0):.4f}")
        print(f"  F1 (macro): {current.get('f1_macro', 0):.4f}")
        print(f"  平均餘弦相似度: {current.get('mean_cosine_similarity', 0):.4f}")
        print(f"  平均交叉熵: {current.get('mean_cross_entropy', 0):.4f}")
        print(f"  Neutral 過度預測: {current.get('neutral_overprediction', 0):+.2%}")

def analyze_why_ml_better():
    """分析為什麼 ML 效果最好"""
    print("\n" + "=" * 80)
    print("3. 為什麼純 ML 效果最好？")
    print("=" * 80)
    
    print("""
【可能原因分析】

1. **ML 模型經過訓練，學習了複雜模式**
   - ML 模型在 11,384 條標註數據上訓練
   - 準確率達到 84.28%（訓練集測試）
   - 能夠捕捉上下文、語義、語法等多種特徵
   - 使用 TF-IDF + Logistic Regression，能學習詞彙組合模式

2. **Lexicon 的限制**
   - 只能匹配詞典中的詞，無法處理：
     * 新詞、網絡用語
     * 上下文語義（同一個詞在不同語境下可能有不同情緒）
     * 否定句、反諷等複雜表達
   - 覆蓋率可能不夠高（需要統計）
   - 無法學習詞彙之間的組合關係

3. **Emoji 的限制**
   - 覆蓋率低：不是所有貼文都有 emoji
   - 從結果看，純 Emoji 的準確率只有 29.16%
   - 交叉熵異常高 (15.51)，表示預測分布很差
   - 可能原因：
     * emoji_joined.txt 覆蓋的 emoji 不夠全面
     * emoji -> phrase -> NRC 的轉換鏈路太長，誤差累積
     * 很多 emoji 沒有對應的 phrase，導致回退到 neutral

4. **融合可能引入噪音**
   - 當 Lexicon 和 Emoji 的準確率較低時，融合可能反而降低 ML 的表現
   - 特別是當它們的預測與 ML 不一致時，會"拖累" ML 的預測
   - 當前組合 (0.3, 0.2, 0.5) 的準確率 (58.78%) 低於純 ML (74.39%)
   """)

def analyze_emoji_contribution():
    """分析 Emoji 是否有增益效果"""
    print("\n" + "=" * 80)
    print("4. Emoji 是否有增益效果？")
    print("=" * 80)
    
    results_path = PROJECT_ROOT / "outputs" / "emotion_vectors" / "weight_evaluation_results.json"
    
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data["results"]
    
    # 比較有 emoji 和沒有 emoji 的組合
    print("\n【比較分析】")
    
    # 純 ML vs ML + Emoji
    pure_ml = results.get("lex0.00_emoji0.00_clf1.00", {})
    ml_emoji = results.get("lex0.00_emoji0.20_clf0.80", {})
    if not ml_emoji:
        # 找一個接近的組合
        for key, value in results.items():
            if "lex0.00_emoji" in key and "clf0.8" in key:
                ml_emoji = value
                break
    
    if pure_ml and ml_emoji:
        print(f"\n純 ML (0, 0, 1):")
        print(f"  準確率: {pure_ml.get('accuracy', 0):.4f}")
        print(f"  Top-2: {pure_ml.get('top2_accuracy', 0):.4f}")
        
        print(f"\nML + Emoji (0, 0.2, 0.8):")
        print(f"  準確率: {ml_emoji.get('accuracy', 0):.4f}")
        print(f"  Top-2: {ml_emoji.get('top2_accuracy', 0):.4f}")
        
        acc_diff = ml_emoji.get('accuracy', 0) - pure_ml.get('accuracy', 0)
        top2_diff = ml_emoji.get('top2_accuracy', 0) - pure_ml.get('top2_accuracy', 0)
        
        print(f"\n差異:")
        print(f"  準確率變化: {acc_diff:+.4f} ({acc_diff/pure_ml.get('accuracy', 1)*100:+.2f}%)")
        print(f"  Top-2 變化: {top2_diff:+.4f} ({top2_diff/pure_ml.get('top2_accuracy', 1)*100:+.2f}%)")
        
        if acc_diff < 0:
            print(f"  [負面] Emoji 降低了準確率！")
        elif acc_diff > 0:
            print(f"  [正面] Emoji 提升了準確率！")
        else:
            print(f"  [中性] Emoji 沒有明顯影響")
    
    # 分析 Emoji 在哪些情況下可能有幫助
    print("""
【Emoji 可能的增益場景】

1. **有 Emoji 的貼文**
   - 如果貼文包含 emoji，Emoji 方法可能提供額外信號
   - 但需要檢查：有 emoji 的貼文比例是多少？

2. **Emoji 表達的情緒更明確**
   - 某些情況下，emoji 比文字更能表達情緒
   - 例如：😊 比 "happy" 更直接

3. **但從結果看，Emoji 整體表現很差**
   - 純 Emoji 準確率只有 29.16%
   - 交叉熵異常高 (15.51)
   - 可能原因：
     * emoji_joined.txt 覆蓋不全
     * emoji -> phrase -> NRC 轉換鏈路誤差大
     * 很多 emoji 沒有對應 phrase，回退到 neutral

【建議】

1. **檢查 Emoji 覆蓋率**
   - 統計有多少貼文有 emoji
   - 統計有多少 emoji 能在 emoji_joined.txt 中找到對應

2. **改進 Emoji 方法**
   - 直接使用 emoji 情緒詞典（不經過 phrase）
   - 使用更大的 emoji 情緒數據集
   - 考慮 emoji 的組合（多個 emoji 的組合情緒）

3. **條件融合**
   - 只在有 emoji 的貼文上使用 emoji 信號
   - 或者只在 emoji 置信度高時使用
    """)

def main():
    print("=" * 80)
    print("情緒方法深度分析")
    print("=" * 80)
    
    # 1. 覆蓋率分析
    coverage = analyze_coverage()
    
    # 2. 質量分析
    analyze_quality()
    
    # 3. 為什麼 ML 最好
    analyze_why_ml_better()
    
    # 4. Emoji 貢獻分析
    analyze_emoji_contribution()
    
    print("\n" + "=" * 80)
    print("總結")
    print("=" * 80)
    print(f"""
1. **覆蓋率問題**
   - Lexicon 覆蓋: {coverage['lexicon_coverage']/coverage['total']:.2%}
   - Emoji 覆蓋: {coverage['emoji_coverage']/coverage['total']:.2%}
   - 兩者都沒有: {coverage['neither_coverage']/coverage['total']:.2%}

2. **純 ML 效果最好的原因**
   - ML 模型經過訓練，學習了複雜模式
   - Lexicon 和 Emoji 的準確率較低，融合可能引入噪音
   - 特別是 Emoji 表現很差（29.16% 準確率）

3. **Emoji 增益效果**
   - 從結果看，Emoji 目前沒有明顯增益
   - 可能因為覆蓋率低、轉換鏈路誤差大
   - 建議改進 Emoji 方法或使用條件融合
    """)

if __name__ == "__main__":
    main()

