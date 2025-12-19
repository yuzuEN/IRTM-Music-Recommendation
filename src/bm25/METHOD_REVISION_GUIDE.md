# Method 修正指南：為什麼改、改什麼、怎麼做

## 🤔 為什麼要改？

### 問題 1：向量維度不匹配，無法直接相加

**原本的設計**：
```
FinalVec = β·LyricsVec_BM25 + (1-β)·Emotion_song + λ·TopicVec_song
```

**問題**：
- `LyricsVec_BM25` = 53,477 維（詞彙空間）
- `Emotion_song` = 8 維（情緒空間）
- `TopicVec_song` = 4 維（主題空間）

**這就像**：
- 把「身高 180 公分」+「體重 70 公斤」+「血型 A 型」直接相加
- 單位不同、意義不同，硬加在一起沒有意義

### 問題 2：BM25 的本質是「分數」，不是「向量」

**BM25 的設計目的**：
- 計算 query 和 document 的「相關性分數」（一個數字）
- 不是用來跟其他向量做 cosine similarity 的

**正確用法**：
- 用 BM25 找出「哪些歌最相關」（top-K 候選）
- 然後在候選內用其他方法（情緒、主題）做 reranking

---

## ✅ 改成什麼？

### 新設計：兩階段檢索 + 分數融合

```
┌─────────────────────────────────────────────────┐
│ Stage 1: BM25 候選生成（語意檢索）              │
│                                                  │
│ Query (貼文) → BM25 分數 → Top-K 候選歌曲      │
│ 例如：Top-100 首最相關的歌                      │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Stage 2: 分數融合 Reranking（在 Top-K 內）     │
│                                                  │
│ 對每首候選歌曲計算：                            │
│ Score = w1·BM25分數                             │
│      + w2·情緒相似度(cosine)                    │
│      + w3·主題相似度(cosine)                    │
│      + w4·熱門度                                │
│                                                  │
│ 然後排序，取 Top-N                              │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Stage 3: Personalized PageRank（可選）         │
│                                                  │
│ 在歌曲圖譜上，以高分候選為 seed 做 PPR         │
│ 擴散到相似歌曲，得到最終推薦                    │
└─────────────────────────────────────────────────┘
```

---

## 📊 具體例子：為什麼新方法更好

### 例子：貼文 "I'm tired"

**舊方法（FinalVec 混加）的問題**：
```
貼文 "I'm tired" 的向量：
- LyricsVec: [0.001, 0.002, ..., 0.0, ...] (53,477 維，大部分是 0)
- Emotion: [0.1, 0.0, 0.0, 0.8, ...] (8 維，sadness 很高)
- Topic: [0.0, 0.0, 0.9, 0.1] (4 維，壓力主題)

硬加在一起 → 意義不明確，權重很難解釋
```

**新方法（分數融合）的優勢**：
```
Step 1: BM25 找出 top-100 候選
  - 包含 "tired", "exhausted", "burnout" 等詞的歌

Step 2: 在 top-100 內做 reranking
  - 歌曲 A: BM25=0.8, 情緒相似=0.9, 主題相似=0.95 → 總分高
  - 歌曲 B: BM25=0.9, 情緒相似=0.3, 主題相似=0.2 → 總分低
  → 歌曲 A 排名更高（因為情境更符合）

Step 3: PPR 擴散
  - 從高分候選開始，在圖譜上找到相似歌曲
```

---

## 🛠️ 你（C 組員）該怎麼做？

### 階段 1：修改 BM25 模組（你現在就可以做）

#### 1.1 修改 `compute_bm25.py`：輸出 top-K 候選

**新增功能**：對每篇貼文計算 BM25 分數，輸出 top-K 候選

**新增函式**：
```python
def compute_bm25_topk(
    query_tokens: List[str],
    bm25_matrix: csr_matrix,
    song_ids: List[str],
    artifacts: dict,
    top_k: int = 100
) -> List[Tuple[str, float]]:
    """
    對 query 計算 BM25 分數，回傳 top-K 候選歌曲。
    
    輸出：
        results: List[Tuple[song_id, bm25_score]]
    """
    # 1. 編碼 query 成 BM25 向量
    query_vec = encode_query_tokens(...)
    
    # 2. 計算與所有歌曲的 BM25 分數（用 dot product）
    scores = bm25_matrix.dot(query_vec)
    
    # 3. 找出 top-K
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # 4. 回傳結果
    return [(song_ids[i], float(scores[i])) for i in top_indices]
```

**新增輸出檔案**：
- `outputs/retrieval/bm25_topk.jsonl`：每行是一個 query 的 top-K 候選
  ```json
  {"query_id": "post_0", "candidates": [{"song_id": "...", "bm25_score": 0.8}, ...]}
  ```

#### 1.2 修改 `demo_bm25_query.py`：展示 top-K 候選

**不改動核心邏輯**，只是讓它輸出「這是 Stage 1 的結果」

---

### 階段 2：準備分數融合介面（等 B 完成後）

#### 2.1 建立 `src/bm25/rerank_fusion.py`

**主要函式**：
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
    results = []
    
    for song_id, bm25_score in candidates:
        # 找到這首歌的 index
        song_idx = song_ids.index(song_id)
        
        # 計算情緒相似度
        emotion_sim = cosine_similarity(
            query_emotion.reshape(1, -1),
            song_emotions[song_idx].reshape(1, -1)
        )[0, 0]
        
        # 計算主題相似度
        topic_sim = cosine_similarity(
            query_topic.reshape(1, -1),
            song_topics[song_idx].reshape(1, -1)
        )[0, 0]
        
        # 分數融合
        final_score = (
            weights["sem"] * bm25_score +
            weights["emo"] * emotion_sim +
            weights["topic"] * topic_sim
        )
        
        results.append((song_id, final_score))
    
    # 排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results
```

---

### 階段 3：與 D 組員整合（等 D 完成後）

#### 3.1 提供介面給 D

**輸出檔案**：
- `outputs/retrieval/bm25_topk.jsonl`：給 D 做 reranking 和 PPR

**函式**：
- `compute_bm25_topk()`：給 D 呼叫，取得候選

---

## 📝 修改清單（Checklist）

### 立即可以做的（不依賴 B/D）

- [ ] **修改 `compute_bm25.py`**
  - [ ] 新增 `compute_bm25_topk()` 函式
  - [ ] 修改 `main()`，對所有貼文計算 top-K 候選
  - [ ] 輸出 `outputs/retrieval/bm25_topk.jsonl`

- [ ] **更新 `demo_bm25_query.py`**
  - [ ] 說明這是「Stage 1: BM25 候選生成」
  - [ ] 輸出格式改為「候選列表 + 分數」

- [ ] **更新 README.md**
  - [ ] 說明新的兩階段架構
  - [ ] 更新輸出檔案說明

### 等 B 完成後做的

- [ ] **建立 `rerank_fusion.py`**
  - [ ] 實作 `rerank_candidates()` 函式
  - [ ] 實作分數融合邏輯

- [ ] **更新 TODO.md**
  - [ ] 移除「FinalVec 融合」相關任務
  - [ ] 新增「分數融合 reranking」任務

### 等 D 完成後做的

- [ ] **與 D 整合測試**
  - [ ] 確認 `bm25_topk.jsonl` 格式正確
  - [ ] 協助 D 整合 reranking 和 PPR

---

## 🎯 核心改變總結

| 項目 | 舊方法 | 新方法 |
|------|--------|--------|
| **BM25 用途** | 轉成向量，跟情緒/主題相加 | 計算分數，找出 top-K 候選 |
| **情緒/主題用途** | 跟 BM25 向量相加 | 在候選內做 reranking |
| **向量融合** | FinalVec = 硬加在一起 | 分數融合 = 加權和 |
| **優點** | 概念簡單 | 可解釋、符合 IR 標準做法 |
| **缺點** | 維度不匹配、意義不明 | 需要多一個階段 |

---

## 💡 為什麼新方法更好？

1. **符合 IR 標準做法**：先檢索（retrieval）再重排（reranking）
2. **可解釋性高**：每個分數的意義很清楚
3. **效率更好**：PPR 只在 top-K 內跑，不用處理 3 萬首歌
4. **權重容易調整**：w1, w2, w3 的意義明確
5. **報告好寫**：可以清楚說明每個階段的作用

---

## 📚 參考資料

- **BM25 原始論文**：Robertson & Zaragoza (2009)
- **Reranking 方法**：Learning to Rank (LTR) 的基礎概念
- **Score Fusion**：多模態檢索的標準做法

