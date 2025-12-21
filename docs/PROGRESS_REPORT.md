# 專案進度報告（Project Progress Report）

## 📊 整體進度總覽

根據兩階段檢索架構，以下是各模組的完成狀態：

---

## ✅ 已完成（Completed）

### 1. **BM25 檢索模組**（C 負責）✅

#### 程式碼檔案
- [x] `src/bm25/compute_bm25.py` - 已實現 BM25 矩陣建立與 top-K 候選生成
- [x] `src/bm25/demo_bm25_query.py` - 示範程式
- [x] `src/bm25/README.md` - 文檔說明
- [x] `src/bm25/METHOD_REVISION_GUIDE.md` - 方法修正指南

#### 輸出檔案（已生成）
- [x] `outputs/bm25_vectors/bm25_matrix.npz` - BM25 矩陣
- [x] `outputs/bm25_vectors/vocabulary.json` - 詞彙表
- [x] `outputs/bm25_vectors/song_ids.json` - 歌曲 ID 列表
- [x] `outputs/bm25_vectors/idf.json` - IDF 值
- [x] `outputs/bm25_vectors/metadata.json` - 元資料
- [x] `outputs/retrieval/bm25_topk.jsonl` - **主要輸出：每篇貼文的 top-K 候選** ⭐
- [x] `outputs/retrieval/idf.json` - 供 B 模組使用

#### 功能確認
- [x] Stage 1: BM25 候選生成已完成
- [x] 對所有貼文計算 top-K 候選
- [x] 輸出格式符合規範

---

### 2. **Emotion 模組**（B 負責）✅

#### 程式碼檔案
- [x] `src/emotion/lyrics_emotion.py` - 歌詞情緒向量（NRC lexicon）
- [x] `src/emotion/melody_emotion.py` - 旋律情緒向量（Spotify features）
- [x] `src/emotion/posts_emotion_lex.py` - 貼文情緒向量（Lexicon）
- [x] `src/emotion/posts_emotion_emoji.py` - 貼文情緒向量（Emoji）
- [x] `src/emotion/posts_emotion_ml.py` - 貼文情緒向量（ML 分類器）

#### 輸出檔案（已生成）
- [x] `outputs/emotion_vectors/EmotionVec_lyrics.npy` - 歌詞情緒向量 (num_songs, 8)
- [x] `outputs/emotion_vectors/EmotionVec_posts_emoji.npy` - 貼文情緒向量（Emoji）
- [x] `outputs/emotion_vectors/EmotionVec_posts_lex.npy` - 貼文情緒向量（Lexicon）
- [x] `outputs/emotion_vectors/EmotionVec_posts_model.npy` - 貼文情緒向量（ML）⭐
- [x] `outputs/emotion_vectors/post_emotion_ml_metadata.json` - ML 模型元資料
- [x] `outputs/models/post_emotion_lr.joblib` - 訓練好的 ML 模型

#### 功能確認
- [x] 歌詞情緒向量已生成
- [x] 貼文情緒向量已生成（多種方法）
- [x] 8 維情緒空間統一

#### ⚠️ 注意事項
- 輸出檔案命名與規範略有不同：
  - 規範：`outputs/emotion/song_emotion.npy`, `outputs/emotion/post_emotion.npy`
  - 實際：`outputs/emotion_vectors/EmotionVec_*.npy`
  - **建議**：統一檔案命名，或確認最終使用哪個檔案作為標準輸出

---

### 3. **Topic 模組**（B 負責）✅

#### 程式碼檔案
- [x] `src/topic/lyrics_topic_kmeans_scanK.py` - 歌詞主題分類（K-means）
- [x] `src/topic/lyrics_topic_kmeans_scanK_merge.py` - 歌詞主題分類（含合併）
- [x] `src/topic/posts_topic_kmeans_scanK.py` - 貼文主題分類

#### 輸出檔案（已生成）
- [x] `outputs/topic_vectors/TopicVec_lyrics_kmeans.npy` - 歌詞主題向量 ⭐
- [x] `outputs/topic_vectors/lyrics_kmeans_model.joblib` - K-means 模型
- [x] `outputs/topic_vectors/lyrics_topic_assignments.jsonl` - 主題分配
- [x] `outputs/topic_vectors/lyrics_topic_keywords.tsv` - 主題關鍵詞
- [x] `outputs/topic_vectors/lyrics_topic_summary.json` - 主題摘要
- [x] `outputs/topic_vectors/posts_topic_assignments.jsonl` - 貼文主題分配
- [x] `outputs/topic_vectors/posts_topic_keywords.tsv` - 貼文主題關鍵詞

#### 功能確認
- [x] 歌詞主題向量已生成
- [x] 貼文主題分類已完成
- [x] K-means 模型已訓練

#### ⚠️ 注意事項
- 輸出檔案命名與規範略有不同：
  - 規範：`outputs/topic/song_topic.npy`, `outputs/topic/post_topic.npy`
  - 實際：`outputs/topic_vectors/TopicVec_*.npy` 和 `*_assignments.jsonl`
  - **需要確認**：貼文主題向量的標準格式（是 npy 還是 jsonl？）

---

## ⏳ 待實現（To Be Implemented）

### 4. **Reranking 模組**（分數融合）❌ **高優先級**

#### 需要建立的檔案
- [ ] `src/bm25/rerank_fusion.py` 或 `src/graph_ppr/rerank.py`

#### 需要實現的功能
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
    Stage 2: 在 BM25 候選內做 reranking。
    
    計算融合分數：
    Score = w_sem * BM25分數
          + w_emo * cosine(Emotion_post, Emotion_song)
          + w_topic * cosine(Topic_post, Topic_song)
    
    輸出：reranked: List[Tuple[song_id, final_score]]
    """
```

#### 輸出檔案
- [ ] `outputs/retrieval/reranked_topk.jsonl` - Rerank 後的候選列表

#### 依賴關係
- ✅ 需要：BM25 top-K 候選（已完成）
- ✅ 需要：Emotion 向量（已完成）
- ✅ 需要：Topic 向量（已完成）
- **可以開始實作！**

---

### 5. **Graph Construction 模組**（D 負責）❌

#### 需要建立的檔案
- [ ] `src/graph_ppr/build_graph.py`

#### 需要實現的功能
- [ ] 讀取歌曲向量（Emotion + Topic 或 BM25）
- [ ] 計算歌曲間 cosine similarity
- [ ] 建立稀疏圖（每首歌保留 top-M 相似鄰居）
- [ ] 儲存為稀疏矩陣格式

#### 輸出檔案
- [ ] `outputs/graph/song_graph.npz` - 稀疏相似度圖

#### 依賴關係
- ✅ 需要：Emotion 向量（已完成）
- ✅ 需要：Topic 向量（已完成）
- **可以開始實作！**

---

### 6. **Personalized PageRank 模組**（D 負責）❌

#### 需要建立的檔案
- [ ] `src/graph_ppr/pagerank.py`

#### 需要實現的功能
- [ ] 讀取歌曲相似度圖
- [ ] 讀取 reranked 候選（作為 teleport seed）
- [ ] 實作 Personalized PageRank 演算法
- [ ] 輸出最終推薦列表

#### 輸出檔案
- [ ] `outputs/recommendations/final_recommendations.jsonl` - 最終推薦結果

#### 依賴關係
- ⏳ 需要：Graph（待實現）
- ⏳ 需要：Reranked 候選（待實現）
- **需等待 Graph 和 Reranking 完成**

---

### 7. **Pipeline 整合腳本**❌

#### 需要建立的檔案
- [ ] `pipeline/run_all.py` - 一鍵重建完整推薦系統
- [ ] `pipeline/query_once.py` - 輸入貼文 → 輸出推薦結果

#### 功能需求
- [ ] `run_all.py`：依序執行 A → B → C → D 模組
- [ ] `query_once.py`：單次查詢介面

---

## 📋 檔案命名規範檢查

### 規範 vs 實際對照表

| 規範命名 | 實際命名 | 狀態 | 備註 |
|---------|---------|------|------|
| `outputs/emotion/song_emotion.npy` | `outputs/emotion_vectors/EmotionVec_lyrics.npy` | ⚠️ | 需確認統一 |
| `outputs/emotion/post_emotion.npy` | `outputs/emotion_vectors/EmotionVec_posts_*.npy` | ⚠️ | 多個版本，需選擇 |
| `outputs/topic/song_topic.npy` | `outputs/topic_vectors/TopicVec_lyrics_kmeans.npy` | ⚠️ | 需確認統一 |
| `outputs/topic/post_topic.npy` | `outputs/topic_vectors/posts_topic_assignments.jsonl` | ⚠️ | 格式不同 |
| `outputs/retrieval/bm25_topk.jsonl` | `outputs/retrieval/bm25_topk.jsonl` | ✅ | 符合規範 |
| `outputs/graph/song_graph.npz` | - | ❌ | 未生成 |
| `outputs/recommendations/final_recommendations.jsonl` | - | ❌ | 未生成 |

---

## 🎯 下一步行動建議（優先級排序）

### 立即可以做的（不依賴其他模組）

1. **建立 Reranking 模組** ⭐⭐⭐
   - 所有輸入資料都已就緒
   - 實作分數融合邏輯
   - 輸出 reranked 候選列表

2. **建立 Graph Construction 模組** ⭐⭐
   - 可以開始建立相似度圖
   - 使用現有的 Emotion/Topic 向量

### 需要等待的

3. **建立 Personalized PageRank 模組** ⭐
   - 需等待 Graph 和 Reranking 完成

4. **建立 Pipeline 整合腳本**
   - 需等待所有模組完成

---

## 📝 總結

### 完成度
- ✅ **Stage 1 (BM25 檢索)**：100% 完成
- ⏳ **Stage 2 (Reranking)**：0% 完成（待實作）
- ⏳ **Stage 3 (PPR)**：0% 完成（待實作）

### 關鍵缺口
1. **Reranking 模組** - 這是連接 Stage 1 和 Stage 3 的關鍵
2. **Graph + PPR 模組** - 最終推薦系統的核心
3. **檔案命名統一** - 確保各模組能正確讀取彼此的輸出

### 建議
1. 優先實作 **Reranking 模組**（所有依賴已完成）
2. 同時開始 **Graph Construction**（可並行開發）
3. 統一檔案命名規範，或建立檔案映射文件

