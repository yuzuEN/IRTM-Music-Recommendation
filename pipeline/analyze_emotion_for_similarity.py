"""
分析情緒向量在相似度計算場景下的表現
=================================================

如果最終用途是計算與歌曲情緒向量的相似度，應該關注哪些指標？
"""

import json
import numpy as np
import pathlib
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def analyze_for_similarity_use():
    """分析哪個指標最適合相似度計算場景"""
    print("=" * 80)
    print("情緒向量相似度計算場景分析")
    print("=" * 80)
    
    results_path = PROJECT_ROOT / "outputs" / "emotion_vectors" / "weight_evaluation_results.json"
    
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data["results"]
    
    # 提取兩個候選組合
    combo1 = results.get("lex0.00_emoji0.10_clf0.90", {})
    combo2 = results.get("lex0.00_emoji0.00_clf1.00", {})
    
    print("\n【候選組合對比】")
    print(f"\n組合 1: (0, 0.1, 0.9)")
    print(f"  準確率: {combo1.get('accuracy', 0):.4f}")
    print(f"  Top-2 準確率: {combo1.get('top2_accuracy', 0):.4f}")
    print(f"  平均餘弦相似度: {combo1.get('mean_cosine_similarity', 0):.4f}")
    print(f"  平均交叉熵: {combo1.get('mean_cross_entropy', 0):.4f}")
    print(f"  平均 MSE: {combo1.get('mean_mse', 0):.4f}")
    
    print(f"\n組合 2: (0, 0, 1.0) - 純 ML")
    print(f"  準確率: {combo2.get('accuracy', 0):.4f}")
    print(f"  Top-2 準確率: {combo2.get('top2_accuracy', 0):.4f}")
    print(f"  平均餘弦相似度: {combo2.get('mean_cosine_similarity', 0):.4f}")
    print(f"  平均交叉熵: {combo2.get('mean_cross_entropy', 0):.4f}")
    print(f"  平均 MSE: {combo2.get('mean_mse', 0):.4f}")
    
    print("\n" + "=" * 80)
    print("指標解讀（針對相似度計算場景）")
    print("=" * 80)
    
    print("""
【當前 Mean Cosine Similarity 的問題】

當前的 mean_cosine_similarity 是計算：
  預測向量 vs 真實 one-hot 向量

但實際使用時是：
  貼文情緒向量 vs 歌曲情緒向量（都是概率分布）

這兩者不同！所以這個指標可能不夠準確。

【應該關注的指標】

1. **Top-2 Accuracy** ⭐⭐⭐⭐⭐
   - 為什麼重要：如果真實情緒在預測的前 2 名中，表示向量方向大致正確
   - 對於相似度計算：即使不是最高分，只要在 top-2，相似度計算時仍可能匹配到
   - 組合 1: 0.8626 > 組合 2: 0.8604 ✅

2. **Mean Cross-Entropy** ⭐⭐⭐⭐
   - 為什麼重要：衡量概率分布的質量
   - 對於相似度計算：如果概率分布更準確，與歌曲向量的相似度也會更準確
   - 組合 1: 1.1483 > 組合 2: 1.1405（越小越好，組合 2 稍好）
   - 但差異很小（0.0078），可以忽略

3. **Mean MSE** ⭐⭐⭐⭐
   - 為什麼重要：衡量向量整體差異
   - 對於相似度計算：MSE 小表示向量更接近真實分布
   - 組合 1: 0.0645 > 組合 2: 0.0639（越小越好，組合 2 稍好）
   - 但差異很小（0.0006），可以忽略

4. **Accuracy** ⭐⭐⭐
   - 為什麼重要：argmax 正確表示主要情緒方向正確
   - 對於相似度計算：如果主要情緒正確，相似度計算時更容易匹配
   - 組合 1: 0.7542 > 組合 2: 0.7439 ✅

5. **Mean Cosine Similarity（當前）** ⭐⭐
   - 為什麼不夠準確：計算的是與 one-hot 的相似度，不是與概率分布的相似度
   - 實際使用時，歌曲情緒向量也是概率分布，不是 one-hot
   - 組合 1: 0.3672 < 組合 2: 0.3737
   - 但這個差異可能不反映實際使用場景

【建議】

如果最終用途是計算與歌曲情緒向量的相似度：

1. **主要看 Top-2 Accuracy**
   - 組合 1 (0, 0.1, 0.9) 的 Top-2 更高（0.8626 vs 0.8604）
   - 這表示向量方向更準確

2. **次要看 Accuracy 和 Cross-Entropy**
   - 組合 1 的準確率更高（0.7542 vs 0.7439）
   - Cross-Entropy 差異很小，可以忽略

3. **Mean Cosine Similarity 可以參考，但不是決定性指標**
   - 因為它計算的是與 one-hot 的相似度
   - 實際使用時是與概率分布比較

【結論】

建議選擇組合 1: (0, 0.1, 0.9)
- Top-2 Accuracy 更高（容錯性更好）
- Accuracy 更高（主要情緒更準確）
- Cross-Entropy 和 MSE 差異很小，可以忽略
- Mean Cosine Similarity 稍低，但這個指標可能不夠準確反映實際使用場景

【進一步驗證建議】

如果要更準確地評估，可以：
1. 載入實際的歌曲情緒向量
2. 計算貼文向量與歌曲向量的相似度
3. 檢查哪些組合能找到更相關的歌曲

這樣才能最準確地評估哪個組合在實際使用中表現更好。
    """)

def suggest_better_evaluation():
    """建議更好的評估方法"""
    print("\n" + "=" * 80)
    print("更好的評估方法建議")
    print("=" * 80)
    
    print("""
【問題】

當前的評估指標（Mean Cosine Similarity）計算的是：
  預測向量 vs 真實 one-hot 向量

但實際使用時是：
  貼文情緒向量 vs 歌曲情緒向量（都是概率分布）

【解決方案】

可以創建一個新的評估指標：

1. **載入歌曲情緒向量**
   - 從 outputs/emotion_vectors/EmotionVec_lyrics.npy 載入

2. **模擬實際使用場景**
   - 對每篇貼文，計算其情緒向量
   - 計算與所有歌曲情緒向量的 cosine similarity
   - 找出相似度最高的歌曲
   - 檢查這些歌曲的情緒是否與貼文真實情緒匹配

3. **評估指標**
   - Hit Rate: 相似度最高的歌曲中，有多少首的情緒與貼文匹配
   - Mean Similarity: 與匹配歌曲的平均相似度
   - Mean Similarity (non-matching): 與不匹配歌曲的平均相似度
   - 差異越大越好（匹配歌曲相似度高，不匹配歌曲相似度低）

這樣才能最準確地評估哪個組合在實際推薦場景中表現更好。
    """)

if __name__ == "__main__":
    analyze_for_similarity_use()
    suggest_better_evaluation()

