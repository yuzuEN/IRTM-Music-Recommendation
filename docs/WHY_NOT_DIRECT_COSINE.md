# 為什麼不用 28 維向量直接做 Cosine Similarity？

## 🤔 你的問題

**為什麼不這樣做？**
```
歌曲：Emotion (8維) + Topic (20維) = 28維向量
貼文：Emotion (8維) + Topic (20維) = 28維向量
    ↓
直接計算 cosine similarity
    ↓
排序
```

---

## 💡 兩種方法對比

### 方法 1：直接用 28 維向量（你的想法）

```
貼文 28 維向量
    ↓
與所有 30,246 首歌的 28 維向量計算 cosine similarity
    ↓
排序，取 top-N
```

**優點**：
- ✅ 簡單直接
- ✅ 不需要 BM25
- ✅ 只需要一個步驟

**缺點**：
- ⚠️ 可能丟失「文字相關性」資訊
- ⚠️ Emotion + Topic 可能無法完全捕捉「詞彙重疊」

---

### 方法 2：兩階段檢索（目前的方法）

```
Stage 1: BM25 找出 top-100（文字相關）
    ↓
Stage 2: 在 top-100 內用 Emotion + Topic 重新排序
```

**優點**：
- ✅ BM25 擅長找「文字相關」的歌曲
- ✅ 兩階段可以互補（文字 + 語境）
- ✅ 更符合 IR 標準做法

**缺點**：
- ⚠️ 需要兩個步驟
- ⚠️ 如果 BM25 漏掉好歌，就找不到了

---

## 🎯 關鍵差異

### 問題：BM25 和 28 維向量捕捉的資訊不同

#### BM25 捕捉的：
- **詞彙重疊**：貼文有 "sad"，歌詞也有 "sad"
- **詞頻**：出現次數多的詞更重要
- **稀有詞**：少見的詞更重要（IDF）

#### 28 維向量捕捉的：
- **情緒分布**：貼文的情緒是 [0.8, 0.0, 0.0, 0.2, ...]（sadness 很高）
- **主題分布**：貼文的主題是 [0.0, 0.9, 0.1, ...]（壓力主題）

**例子**：
```
貼文："I'm tired after work"

BM25 會找到：
  - 包含 "tired", "work", "exhausted" 等詞的歌
  - 即使情緒/主題不完全符合，但文字相關

28 維向量會找到：
  - 情緒和主題相似的歌
  - 即使沒有 "tired" 這個詞，但情境相似
```

---

## 📊 實際例子

### 場景：貼文 "I'm tired"

#### 方法 1：直接用 28 維向量
```
貼文 28 維向量：
  Emotion: [0.1, 0.0, 0.0, 0.8, ...] (sadness 很高)
  Topic: [0.0, 0.9, 0.1, ...] (壓力主題)

可能找到：
  - 歌曲 A: 情緒/主題相似，但歌詞沒有 "tired" 這個詞
  - 歌曲 B: 情緒/主題相似，但歌詞完全不同
```

#### 方法 2：兩階段檢索
```
Stage 1 (BM25):
  找到包含 "tired", "exhausted" 等詞的歌
  → 歌曲 C: 有 "tired" 這個詞，BM25 分數高

Stage 2 (Reranking):
  在 top-100 內，用 Emotion + Topic 重新排序
  → 歌曲 C: BM25=0.9, 情緒相似=0.95, 主題相似=0.90
  → 總分很高，排名第一
```

---

## 🤷 哪個方法更好？

### 實際上，兩種方法都可以！

**你的方法（28 維向量直接排序）**：
- 更簡單
- 如果 Emotion + Topic 能充分捕捉語境，效果可能很好
- **適合**：如果貼文的情緒和主題比文字更重要

**兩階段方法（BM25 + Reranking）**：
- 更複雜，但更全面
- BM25 確保文字相關，Emotion + Topic 確保情境符合
- **適合**：如果文字相關性也很重要

---

## 💭 建議

### 可以兩種方法都試試！

1. **方法 A：直接用 28 維向量**
   ```python
   # 計算貼文和所有歌曲的 cosine similarity
   similarities = cosine_similarity(post_28d, songs_28d)
   # 排序，取 top-N
   ```

2. **方法 B：兩階段檢索**
   ```python
   # Stage 1: BM25 top-100
   # Stage 2: Reranking (BM25 + Emotion + Topic)
   ```

3. **方法 C：混合方法**
   ```python
   # 先用 BM25 找出 top-1000（更寬鬆）
   # 然後在 top-1000 內用 28 維向量排序
   ```

---

## 🎯 為什麼目前用兩階段方法？

1. **符合 IR 標準做法**：先檢索（retrieval）再重排（reranking）
2. **報告好寫**：可以清楚說明每個階段的作用
3. **可解釋性高**：每個分數的意義很清楚
4. **互補性**：BM25（文字）+ Emotion/Topic（語境）

---

## 📝 總結

### 你的想法（28 維向量直接排序）：
- ✅ 完全可行
- ✅ 更簡單
- ⚠️ 可能丟失文字相關性

### 目前的方法（兩階段）：
- ✅ 更全面（文字 + 語境）
- ✅ 符合 IR 標準
- ⚠️ 更複雜

### 建議：
- 可以兩種方法都實作
- 比較效果，選擇更好的
- 或者混合使用

---

## 🔜 實作建議

如果你想試試 28 維向量直接排序，可以這樣做：

```python
# 1. 載入貼文和歌曲的 28 維向量
post_emotion = load_post_emotion()  # (8,)
post_topic = load_post_topic()       # (20,)
post_28d = np.hstack([post_emotion, post_topic])  # (28,)

songs_emotion = load_songs_emotion()  # (30246, 8)
songs_topic = load_songs_topic()      # (30246, 20)
songs_28d = np.hstack([songs_emotion, songs_topic])  # (30246, 28)

# 2. 計算 cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(post_28d.reshape(1, -1), songs_28d)[0]

# 3. 排序，取 top-N
top_indices = np.argsort(similarities)[::-1][:N]
```

這樣就可以直接排序了！

