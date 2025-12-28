"""
分析 Top-5 組合，給出選擇建議
"""

import json
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def analyze_top5():
    """分析 Top-5 組合"""
    
    results_path = PROJECT_ROOT / "outputs" / "emotion_vectors" / "weight_evaluation_results.json"
    
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data["results"]
    
    # Top-5 組合
    top5_keys = [
        "lex0.03_emoji0.07_clf0.90",
        "lex0.03_emoji0.04_clf0.93",
        "lex0.03_emoji0.05_clf0.92",
        "lex0.04_emoji0.06_clf0.90",
        "lex0.01_emoji0.04_clf0.95",
    ]
    
    print("=" * 80)
    print("Top-5 組合詳細對比分析")
    print("=" * 80)
    
    for i, key in enumerate(top5_keys, 1):
        if key not in results:
            continue
        data = results[key]
        print(f"\n【組合 {i}】{key}")
        print(f"  權重: Lexicon={data['weights']['lexicon']:.2f}, "
              f"Emoji={data['weights']['emoji']:.2f}, "
              f"Classifier={data['weights']['classifier']:.2f}")
        print(f"  Top-2 Accuracy: {data['top2_accuracy']:.4f}")
        print(f"  Accuracy: {data['accuracy']:.4f}")
        print(f"  F1 (macro): {data['f1_macro']:.4f}")
        print(f"  餘弦相似度: {data['mean_cosine_similarity']:.4f}")
        print(f"  交叉熵: {data['mean_cross_entropy']:.4f}")
        print(f"  MSE: {data['mean_mse']:.4f}")
        print(f"  Neutral 過度預測: {data['neutral_overprediction']:+.2%}")
    
    print("\n" + "=" * 80)
    print("綜合分析與建議")
    print("=" * 80)
    
    print("""
【指標重要性（針對相似度計算場景）】

1. Top-2 Accuracy ⭐⭐⭐⭐⭐ (最重要)
   - 如果真實情緒在預測的前 2 名中，表示向量方向大致正確
   - 對於相似度計算：即使不是最高分，只要在 top-2，相似度計算時仍可能匹配到

2. Accuracy ⭐⭐⭐⭐ (重要)
   - 主要情緒方向正確，相似度計算時更容易匹配

3. Neutral 過度預測 ⭐⭐⭐ (需要考慮)
   - 過度預測會影響其他情緒的識別
   - 建議控制在 +3% 以內

4. 餘弦相似度 ⭐⭐ (參考)
   - 當前計算的是與 one-hot 的相似度，可能不夠準確反映實際使用場景

5. Cross-Entropy / MSE ⭐⭐ (參考)
   - 差異很小時可以忽略

【各組合特點】

組合 1: (0.03, 0.07, 0.90)
  ✅ Top-2 Accuracy 最高 (0.8650)
  ✅ Accuracy 較高 (0.7559)
  ✅ Neutral 過度預測較低 (+3.09%)
  ⚠️ 餘弦相似度稍低 (0.3654)
  
組合 2: (0.03, 0.04, 0.93)
  ✅ Top-2 Accuracy 第二高 (0.8649)
  ✅ 餘弦相似度較高 (0.3674)
  ✅ Cross-Entropy 較低 (1.1495)
  ⚠️ Accuracy 稍低 (0.7532)
  ⚠️ Neutral 過度預測稍高 (+3.19%)
  
組合 4: (0.04, 0.06, 0.90)
  ✅ Accuracy 最高 (0.7565)
  ✅ F1 (macro) 最高 (0.7403)
  ✅ Neutral 過度預測最低 (+2.99%)
  ✅ Top-2 Accuracy 也很高 (0.8644)
  ⚠️ 餘弦相似度稍低 (0.3648)
  ⚠️ Cross-Entropy 稍高 (1.1536)

【推薦】

基於以下考慮：
1. 最終用途是計算與歌曲情緒向量的相似度
2. Top-2 Accuracy 最重要（容錯性好）
3. Accuracy 也很重要（主要情緒方向）
4. Neutral 過度預測需要控制

推薦選擇：組合 1 或 組合 4

【組合 1: (0.03, 0.07, 0.90)】- 如果優先考慮 Top-2 Accuracy
  - Top-2 Accuracy: 0.8650 (最高)
  - Accuracy: 0.7559 (很好)
  - Neutral: +3.09% (可接受)
  - 特點：Emoji 權重稍高 (0.07)，可能對有 emoji 的貼文有幫助

【組合 4: (0.04, 0.06, 0.90)】- 如果優先考慮整體平衡
  - Top-2 Accuracy: 0.8644 (很高，僅差 0.0006)
  - Accuracy: 0.7565 (最高)
  - F1 (macro): 0.7403 (最高)
  - Neutral: +2.99% (最低，最好)
  - 特點：Lexicon 和 Emoji 權重更平衡，整體表現最均衡

【最終建議】

如果只能選一個，建議選擇：組合 4 (0.04, 0.06, 0.90)

理由：
1. Top-2 Accuracy 僅比最高低 0.0006 (0.06%)，差異極小
2. Accuracy 和 F1 都是最高，表示主要情緒更準確
3. Neutral 過度預測最低 (+2.99%)，最平衡
4. 整體表現最均衡，沒有明顯短板

如果更看重 Top-2 Accuracy 的極致表現，可以選組合 1 (0.03, 0.07, 0.90)
    """)

if __name__ == "__main__":
    analyze_top5()

