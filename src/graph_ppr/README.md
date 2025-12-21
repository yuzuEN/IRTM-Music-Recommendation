# Graph + Personalized PageRank 模組

組員 D 負責：**Graph Construction + Personalized PageRank 推薦引擎**

## 功能概述

本模組負責：
1. **建立相似度圖譜**：從歌曲的 emotion 和 topic 向量計算相似度，建立稀疏圖
2. **Personalized PageRank**：使用查詢（post）與歌曲的相似度作為 teleportation vector，在圖譜上執行 PPR
3. **產生最終推薦**：輸出 top-K 推薦歌曲

## 檔案說明

### `build_graph.py`
- **功能**：建立歌曲相似度圖譜
- **輸入**：
  - `outputs/emotion_vectors/EmotionVec_lyrics.npy` - 歌曲情緒向量
  - `outputs/topic_vectors/TopicVec_lyrics_kmeans.npy` - 歌曲主題向量
  - `outputs/bm25_vectors/song_ids.json` - 歌曲 ID 列表
- **輸出**：
  - `outputs/graph/song_graph.npz` - 稀疏相似度圖譜
  - `outputs/graph/graph_metadata.json` - 圖譜元資料

### `personalized_pagerank.py`
- **功能**：執行 Personalized PageRank 推薦
- **輸入**：
  - `outputs/graph/song_graph.npz` - 相似度圖譜
  - `outputs/emotion_vectors/EmotionVec_lyrics.npy` - 歌曲情緒向量
  - `outputs/emotion_vectors/EmotionVec_posts_model.npy` - 貼文情緒向量
  - `outputs/topic_vectors/TopicVec_lyrics_kmeans.npy` - 歌曲主題向量
  - `outputs/topic_vectors/posts_topic_assignments.jsonl` - 貼文主題分配
  - `outputs/bm25_vectors/song_ids.json` - 歌曲 ID 列表
  - `outputs/retrieval/bm25_topk.jsonl` - BM25 候選（用於查詢對齊）
- **輸出**：
  - `outputs/recommendations/ppr_recommendations.jsonl` - PPR 推薦結果

## 使用方法

### 1. 建立相似度圖譜（離線，只需執行一次）

```bash
python src/graph_ppr/build_graph.py
```

這會產生 `outputs/graph/song_graph.npz` 和相關元資料。

### 2. 執行 Personalized PageRank（為所有查詢產生推薦）

```bash
python src/graph_ppr/personalized_pagerank.py
```

這會讀取所有查詢，為每個查詢產生 top-10 推薦歌曲。

## 方法說明

### Teleportation Vector 設計

使用 **查詢-歌曲相似度**（Option 1: Full Similarity）作為 teleportation vector：

1. **計算查詢與所有歌曲的相似度**：
   - 情緒相似度：`cosine_similarity(query_emotion, song_emotion)`
   - 主題相似度：`cosine_similarity(query_topic, song_topic)`
   - 組合相似度：`0.6 * emotion_sim + 0.4 * topic_sim`

2. **正規化為機率分布**：
   - 所有歌曲都獲得與相似度成比例的 teleportation 機率
   - 相似度越高，teleportation 機率越大

3. **優點**：
   - 直接測量查詢與歌曲的 emotion/topic 匹配度
   - 不依賴 BM25（可以發現文字不相關但情緒/主題相關的歌曲）
   - 與圖譜結構互補（圖譜 = 歌曲間相似度，teleportation = 查詢-歌曲相似度）

### Personalized PageRank 演算法

```
r = (1 - α) · M · r + α · v

其中：
- r: PPR 分數向量
- M: 正規化的圖譜（row-stochastic）
- v: teleportation vector（查詢-歌曲相似度）
- α: damping factor（預設 0.85）
```

**迭代過程**：
1. 隨機遊走：跟隨圖譜邊緣（歌曲間相似度）
2. Teleportation：跳回查詢相關的歌曲（teleportation vector）
3. 收斂：重複直到分數穩定

**結果**：
- 查詢相關的歌曲（高 teleportation 機率）獲得高分
- 與查詢相關歌曲相似的歌曲（透過圖譜邊緣）也獲得高分
- 最終推薦結合了直接相關性和間接相似性

## 輸出格式

### `ppr_recommendations.jsonl`

每行一個 JSON，格式如下：

```json
{
  "query_id": "post_0",
  "query_idx": 0,
  "recommendations": [
    {
      "rank": 1,
      "song_id": "hf_13986",
      "ppr_score": 0.0234
    },
    {
      "rank": 2,
      "song_id": "hf_25522",
      "ppr_score": 0.0198
    },
    ...
  ],
  "num_songs": 30246,
  "alpha": 0.85
}
```

## 參數說明

### Graph Construction (`build_graph.py`)
- `top_m`: 每首歌保留前 M 個最相似的鄰居（預設 20）
- `normalize`: 是否正規化 emotion 和 topic 向量（預設 True）

### Personalized PageRank (`personalized_pagerank.py`)
- `alpha`: Damping factor（預設 0.85）
  - 較高（接近 1）：更多 teleportation，更聚焦於查詢相關歌曲
  - 較低（接近 0）：更多隨機遊走，更多探索圖譜結構
- `max_iter`: 最大迭代次數（預設 100）
- `tol`: 收斂容忍度（預設 1e-6）
- `top_k`: 返回前 K 個推薦（預設 10）
- `emotion_weight`: 情緒相似度權重（預設 0.6）
- `topic_weight`: 主題相似度權重（預設 0.4）

## 與其他模組的介面

### 輸入（來自其他組員）

**來自 B（Emotion & Topic）**：
- `EmotionVec_lyrics.npy` - 歌曲情緒向量
- `EmotionVec_posts_model.npy` - 貼文情緒向量
- `TopicVec_lyrics_kmeans.npy` - 歌曲主題向量
- `posts_topic_assignments.jsonl` - 貼文主題分配

**來自 C（BM25）**：
- `song_ids.json` - 歌曲 ID 列表（用於對齊）
- `bm25_topk.jsonl` - BM25 候選（用於查詢對齊，但 PPR 不直接使用 BM25 分數）

### 輸出（給最終推薦系統）

- `ppr_recommendations.jsonl` - PPR 推薦結果
  - 可與 BM25 結果結合或直接使用

## 系統架構

```
查詢（Post）
    ↓
[計算查詢-歌曲相似度] ← Teleportation Vector
    ↓
[Personalized PageRank]
    - 圖譜邊緣 = 歌曲間相似度（emotion + topic）
    - Teleportation = 查詢-歌曲相似度
    ↓
[PPR 分數]
    ↓
[Top-K 推薦]
```

## 注意事項

1. **圖譜必須先建立**：執行 PPR 前需先執行 `build_graph.py`
2. **向量對齊**：所有向量必須與 `song_ids.json` 的順序對齊
3. **查詢對齊**：查詢索引從 `query_id` 提取（例如 "post_0" -> 0），必須與 post vectors 對齊
4. **主題向量轉換**：貼文的主題是 cluster_id，需轉換為 one-hot 向量以匹配歌曲主題向量維度

## 範例使用

### 單一查詢推薦（程式碼）

```python
from src.graph_ppr.personalized_pagerank import (
    load_graph_and_vectors,
    load_post_vectors,
    generate_recommendations_for_query,
    cluster_id_to_topic_vector
)

# 載入資料
graph, song_emotion, song_topic, song_ids = load_graph_and_vectors()
post_emotion, post_topic_assignments = load_post_vectors()

# 查詢索引
query_idx = 0
query_emotion = post_emotion[query_idx]
cluster_id = post_topic_assignments[query_idx]["cluster_id"]
query_topic = cluster_id_to_topic_vector(cluster_id, num_clusters=20)

# 產生推薦
result = generate_recommendations_for_query(
    query_id="post_0",
    query_idx=0,
    query_emotion=query_emotion,
    query_topic=query_topic,
    graph=graph,
    song_emotion=song_emotion,
    song_topic=song_topic,
    song_ids=song_ids,
    top_k=10,
    alpha=0.85
)

print(result["recommendations"])
```

## 未來擴展

- [ ] 加入 popularity reweighting（Spotify 熱門度）
- [ ] 支援自訂查詢（不從檔案讀取）
- [ ] 調整 emotion/topic 權重
- [ ] 實驗不同的 teleportation 策略

