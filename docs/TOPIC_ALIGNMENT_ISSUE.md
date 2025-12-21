# 主題對齊問題分析

## ⚠️ 發現的問題

### 當前狀態：

1. **歌曲主題向量**：
   - 維度：**20 維**（`K_merged = 20`）
   - 檔案：`outputs/topic_vectors/TopicVec_lyrics_kmeans.npy`
   - 模型：`outputs/topic_vectors/lyrics_kmeans_model.joblib`

2. **貼文主題向量**：
   - 維度：**29 維**（`K = 29`）
   - 檔案：`outputs/topic_vectors/posts_topic_assignments.jsonl`
   - 模型：`outputs/topic_vectors/posts_kmeans_model.joblib`

### 問題：
- ❌ **維度不一致**：歌曲是 20 維，貼文是 29 維
- ❌ **無法直接計算 cosine similarity**
- ❌ **無法用於 PPR 的 teleport 權重計算**

---

## 🔧 解決方案

### 方案 1：用歌曲的 KMeans 模型預測貼文主題（推薦）⭐

**優點**：
- ✅ 貼文和歌曲使用**同一個主題空間**（20 維）
- ✅ 可以直接計算 cosine similarity
- ✅ 不需要重新訓練

**方法**：
```python
# 1. 載入歌曲的 KMeans 模型和 TF-IDF vectorizer
lyrics_kmeans = joblib.load('outputs/topic_vectors/lyrics_kmeans_model.joblib')
lyrics_tfidf = ...  # 需要從 lyrics_topic 模組載入

# 2. 對貼文做相同的 TF-IDF 轉換
post_tfidf = lyrics_tfidf.transform(post_texts)

# 3. 用歌曲的 KMeans 模型預測貼文主題
post_topic_labels = lyrics_kmeans.predict(post_tfidf)

# 4. 轉換成 20 維 one-hot 向量
post_topic_vectors = make_topic_vector_hard(post_topic_labels, K=20)
```

### 方案 2：重新訓練，讓貼文和歌曲用同一個主題空間

**方法**：
- 將貼文和歌詞合併，一起訓練 KMeans
- 確保兩者使用相同的 20 個主題

**缺點**：
- ⚠️ 需要重新訓練
- ⚠️ 可能影響現有結果

---

## 📝 關於 Encoder

### Emotion Encoder（已有）✅

**函數**：
- `compute_post_emotion(tokens, nrc_lexicon)` - 從 tokens 計算 8 維情緒向量
- `post_emoji_emotion(raw_text, emoji_table)` - 從 emoji 計算情緒向量
- ML 模型：`clf.predict_proba(post_texts)` - 從文本預測情緒

**可以對新貼文使用**：
```python
# 方法 1: Lexicon
tokens = ["sad", "lonely", "tired"]
emotion_vec = compute_post_emotion(tokens, nrc_lexicon)  # (8,)

# 方法 2: ML 模型
post_text = "I'm feeling sad and lonely"
emotion_vec = clf.predict_proba([post_text])[0]  # (8,)
```

### Topic Encoder（需要對齊）⚠️

**當前問題**：
- 貼文主題是 29 維（自己的 KMeans）
- 歌曲主題是 20 維（自己的 KMeans）
- **沒有對齊**

**解決方法**：
- 用歌曲的 KMeans 模型來預測貼文主題（對齊到 20 維）

---

## 🎯 對於 PPR Teleport 權重的建議

### 你的想法（用 cosine similarity 作為 teleport 權重）：

```
對每首歌計算：
  similarity = cosine(貼文_28d, 歌曲_28d)
  
Teleport 向量 v：
  v[song_i] = similarity[song_i] / sum(similarities)
```

**這是一個很好的想法！** 但需要確保：
1. ✅ 貼文和歌曲的 Emotion 向量都是 8 維（已對齊）
2. ⚠️ 貼文和歌曲的 Topic 向量都是 20 維（**需要對齊**）

---

## 🔜 需要做的事情

### 1. 建立貼文主題對齊模組

**檔案**：`src/topic/align_post_topics.py`

**功能**：
- 載入歌曲的 KMeans 模型和 TF-IDF vectorizer
- 對貼文做相同的 TF-IDF 轉換
- 用歌曲的 KMeans 模型預測貼文主題
- 輸出 20 維主題向量（與歌曲對齊）

### 2. 建立 Encoder 模組

**檔案**：`src/graph_ppr/encode_post.py`

**功能**：
- 輸入：貼文文本
- 輸出：28 維向量（8 維 Emotion + 20 維 Topic）
- 用於：PPR 的 teleport 權重計算

### 3. 實作 PPR with Cosine Similarity Teleport

**檔案**：`src/graph_ppr/pagerank.py`

**功能**：
- 計算貼文與所有歌曲的 28 維向量 cosine similarity
- 用 similarity 作為 teleport 權重
- 執行 Personalized PageRank

---

## 📊 對齊後的流程

```
新貼文 "I'm tired"
    ↓
[Encoder] 轉換成 28 維向量
  - Emotion: 8 維（用 lexicon 或 ML 模型）
  - Topic: 20 維（用歌曲的 KMeans 模型預測）
    ↓
計算與所有歌曲的 cosine similarity
    ↓
用 similarity 作為 PPR 的 teleport 權重
    ↓
執行 Personalized PageRank
    ↓
最終推薦
```

---

## ✅ 總結

### 當前狀態：
- ✅ Emotion encoder：已有（lexicon 或 ML）
- ⚠️ Topic encoder：需要對齊（貼文 29 維 → 歌曲 20 維）

### 需要做的事情：
1. 建立貼文主題對齊模組（用歌曲的 KMeans 模型）
2. 建立統一的 encoder（輸入貼文 → 輸出 28 維向量）
3. 實作 PPR with cosine similarity teleport

