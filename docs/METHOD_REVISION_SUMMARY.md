# 方法改進總結（Method Revision Summary）

## 核心問題

**舊方法的問題**：
```
FinalVec = β·LyricsVec_BM25 + (1-β)·Emotion_song + λ·TopicVec_song
```

**問題**：
- BM25 是 53,477 維（詞彙空間）
- Emotion 是 8 維（情緒空間）
- Topic 是 4 維（主題空間）
- **不同維度、不同意義的向量無法直接相加**

## 新方法：兩階段檢索 + 分數融合

### Stage 1: BM25 候選生成（語義檢索）
- 用 BM25 找出 top-K 候選歌曲（例如 top-100）
- 輸出：`outputs/retrieval/bm25_topk.jsonl`

### Stage 2: 分數融合 Reranking（在 top-K 內）
- 對每首候選歌曲計算融合分數：
```
Score = w_sem · BM25分數
      + w_emo · cosine(Emotion_post, Emotion_song)
      + w_topic · cosine(Topic_post, Topic_song)
      + w_pop · popularity(song)
```
- 排序後取 top-N

### Stage 3: Personalized PageRank（可選）
- 在歌曲相似度圖上，以高分候選為 seed 做 PPR
- 擴散到相似歌曲，得到最終推薦

## 需要實現的模組

### ✅ 已完成
- [x] BM25 檢索模組（`src/bm25/compute_bm25.py`）- 已實現 top-K 候選生成
- [x] Emotion 模組（`src/emotion/`）- 已完成
- [x] Topic 模組（`src/topic/`）- 已完成

### ⏳ 待實現（優先級排序）

#### 1. **Reranking 模組**（B/C 協作）
**檔案**：`src/bm25/rerank_fusion.py` 或 `src/graph_ppr/rerank.py`

**功能**：
```python
def rerank_candidates(
    candidates: List[Tuple[str, float]],  # BM25 top-K 候選
    query_emotion: np.ndarray,  # 8 維
    query_topic: np.ndarray,  # T 維
    song_emotions: np.ndarray,  # (num_songs, 8)
    song_topics: np.ndarray,  # (num_songs, T)
    song_ids: List[str],
    weights: dict = {"sem": 0.5, "emo": 0.3, "topic": 0.2}
) -> List[Tuple[str, float]]:
    """
    在 BM25 候選內做 reranking。
    
    輸出：
        reranked: List[Tuple[song_id, final_score]]
    """
    # 實現分數融合邏輯
```

#### 2. **Graph Construction**（D 負責）
**檔案**：`src/graph_ppr/build_graph.py`

**功能**：
- 建立歌曲相似度圖（cosine similarity）
- 稀疏化：每首歌保留 top-M 相似鄰居
- 輸出：`outputs/graph/song_graph.npz`

#### 3. **Personalized PageRank**（D 負責）
**檔案**：`src/graph_ppr/pagerank.py`

**功能**：
- 在歌曲圖上做 PPR
- 以 rerank 的高分候選作為 teleport seed
- 輸出最終推薦列表

## 資料流

```
貼文 (post)
    ↓
[Stage 1] BM25 → top-100 候選
    ↓
[Stage 2] Reranking（情緒+主題融合）
    ↓
[Stage 3] PPR（在圖上擴散）
    ↓
最終推薦 Top-N
```

## 輸出檔案規範

### BM25 模組（C）
- `outputs/retrieval/bm25_topk.jsonl` - 每篇貼文的 top-K 候選
- `outputs/retrieval/idf.json` - 供 B 模組使用（可選）

### Emotion/Topic 模組（B）
- `outputs/emotion/song_emotion.npy` - (num_songs, 8)
- `outputs/emotion/post_emotion.npy` - (num_posts, 8)
- `outputs/topic/song_topic.npy` - (num_songs, T)
- `outputs/topic/post_topic.npy` - (num_posts, T)

### Graph/PPR 模組（D）
- `outputs/graph/song_graph.npz` - 稀疏相似度圖
- `outputs/recommendations/final_recommendations.jsonl` - 最終推薦結果

## 為什麼新方法更好？

1. **符合 IR 標準做法**：先檢索（retrieval）再重排（reranking）
2. **可解釋性高**：每個分數的意義很清楚
3. **效率更好**：PPR 只在 top-K 內跑，不用處理 3 萬首歌
4. **權重容易調整**：w_sem, w_emo, w_topic 的意義明確
5. **報告好寫**：可以清楚說明每個階段的作用

