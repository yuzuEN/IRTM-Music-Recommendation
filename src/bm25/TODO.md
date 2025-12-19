# C 組員待辦事項與 B 組員介面規範

## 📋 需要從 B 組員取得的資料

### 1. 歌曲情緒向量（Emotion_song）

**檔案路徑**：`outputs/emotion_vectors/song_emotion.npy`

**格式**：
- NumPy Array，形狀：`(num_songs, 8)`
- 每一列代表一首歌的 8 維情緒向量
- 順序必須與 `outputs/bm25_vectors/song_ids.json` 對齊

**情緒類別（8 類）**：
```
[joy, anger, fear, sadness, surprise, disgust, excitement, neutral]
```

**範例**：
```python
import numpy as np
song_emotion = np.load("outputs/emotion_vectors/song_emotion.npy")
# shape: (30246, 8)
# song_emotion[0] = [0.1, 0.2, 0.05, 0.6, 0.0, 0.0, 0.05, 0.0]  # 第 0 首歌的情緒向量
```

**說明**：
- 這是融合後的情緒向量：`Emotion_song = α * EmotionVec_lyrics + (1-α) * EmotionVec_melody`
- α 參數由 B 組員決定（建議在 metadata 中記錄）

---

### 2. 歌曲主題向量（TopicVec_song）

**檔案路徑**：`outputs/topic_vectors/song_topic.npy`

**格式**：
- NumPy Array，形狀：`(num_songs, T)`
- T = 主題維度（建議 4-6 維，例如：校園、旅行、壓力、感情等）
- 每一列代表一首歌的主題向量
- 順序必須與 `outputs/bm25_vectors/song_ids.json` 對齊

**範例**：
```python
import numpy as np
song_topic = np.load("outputs/topic_vectors/song_topic.npy")
# shape: (30246, 4)  # 假設 T=4
# song_topic[0] = [0.0, 0.3, 0.7, 0.0]  # 第 0 首歌的主題向量
```

**說明**：
- 主題維度 T 由 B 組員決定（建議在 metadata 中記錄）
- 可以是 keyword-based 或 topic modeling 的結果

---

### 3. 貼文情緒向量（Emotion_query）

**檔案路徑**：`outputs/emotion_vectors/query_emotion.npy`（可選）

**格式**：
- NumPy Array，形狀：`(num_queries, 8)`
- 每一列代表一篇貼文的 8 維情緒向量
- 順序必須與 `data/processed/posts/posts_clean_expanded.jsonl` 對齊

**替代方案**：如果 B 組員提供的是函式，則：
```python
def encode_query_emotion(query_tokens: List[str], raw_text: str) -> np.ndarray:
    """
    輸入：query tokens 或原始貼文
    輸出：8 維情緒向量
    """
    # B 組員實作
    pass
```

**說明**：
- 這是融合後的情緒向量：`Emotion_query = η1 * EmotionVec_lex + η2 * EmotionVec_emoji + η3 * EmotionVec_model`
- η1, η2, η3 參數由 B 組員決定

---

### 4. 貼文主題向量（TopicVec_post）

**檔案路徑**：`outputs/topic_vectors/query_topic.npy`（可選）

**格式**：
- NumPy Array，形狀：`(num_queries, T)`
- T 必須與 `song_topic.npy` 的 T 相同
- 每一列代表一篇貼文的主題向量
- 順序必須與 `data/processed/posts/posts_clean_expanded.jsonl` 對齊

**替代方案**：如果 B 組員提供的是函式，則：
```python
def encode_query_topic(query_tokens: List[str], raw_text: str) -> np.ndarray:
    """
    輸入：query tokens 或原始貼文
    輸出：T 維主題向量
    """
    # B 組員實作
    pass
```

---

### 5. Metadata（可選但建議）

**檔案路徑**：`outputs/emotion_vectors/metadata.json`、`outputs/topic_vectors/metadata.json`

**內容**：
```json
{
  "emotion_dim": 8,
  "topic_dim": 4,
  "alpha": 0.7,
  "eta1": 0.4,
  "eta2": 0.3,
  "eta3": 0.3,
  "emotion_labels": ["joy", "anger", "fear", "sadness", "surprise", "disgust", "excitement", "neutral"],
  "topic_labels": ["校園", "旅行", "壓力", "感情"]
}
```

---

## ✅ C 組員待辦事項

### Phase 1：BM25 Retrieval（已完成 ✅）

- [x] **實作 `compute_bm25.py`**
  - [x] 建立 BM25 矩陣
  - [x] 對所有貼文計算 BM25 top-K 候選
  - [x] 輸出 `outputs/retrieval/bm25_topk.jsonl`
  - [x] 輸出 `outputs/retrieval/idf.json` 給 B 組員

- [x] **建立 Demo**
  - [x] `demo_bm25_query.py`：BM25-only baseline

---

### Phase 2：等待 B/D 組員（目前無需實作）

**重要**：根據新的方法設計，C 組員**不需要**實作 FinalVec 融合或 reranking。

**原因**：
- BM25 負責 Stage 1（候選生成）← **已完成**
- Reranking 和 PPR 由 D 組員負責（在 top-K 候選內做分數融合）

**C 組員的任務已完成**，接下來：
- 等待 B 組員完成情緒/主題向量
- 等待 D 組員實作 reranking 和 PPR
- 協助 D 組員整合（如有需要）

---

### Phase 3：協助 D 組員整合（可選）

- [ ] **確認輸出格式**
  - 確認 `outputs/retrieval/bm25_topk.jsonl` 格式符合 D 組員需求
  - 如有問題，協助調整格式

- [ ] **提供技術支援**
  - 協助 D 組員理解 BM25 分數的意義
  - 協助 D 組員整合 reranking 邏輯

---

### Phase 4：報告撰寫

- [ ] **Method 2（BM25 Retrieval）章節**
  - BM25 公式說明
  - 為什麼選擇 BM25（相對於 TF-IDF 的優勢）
  - BM25 參數選擇理由（k1, b）
  - Stage 1 候選生成的設計

- [ ] **Experiments 章節**
  - BM25-only baseline 結果（`demo_bm25_query.py`）
  - 可選：TF-IDF vs BM25 比較（如果有做實驗）
  - 說明 Stage 1 的候選品質

---

## 🔗 與其他模組的依賴關係

### 依賴 A 組員（已完成 ✅）
- `data/processed/lyrics/lyrics_tokens.csv`
- `data/processed/lyrics/vocabulary.json`
- `data/processed/posts/posts_clean_expanded.jsonl`

### 依賴 B 組員（等待中 ⏳）
- `outputs/emotion_vectors/song_emotion.npy`
- `outputs/topic_vectors/song_topic.npy`
- `outputs/emotion_vectors/query_emotion.npy`（或函式）
- `outputs/topic_vectors/query_topic.npy`（或函式）

### 提供給 D 組員（已完成 ✅）
- `outputs/retrieval/bm25_topk.jsonl`：每篇貼文的 top-K 候選歌曲（Stage 1 輸出）
- `outputs/bm25_vectors/bm25_matrix.npz`：BM25 矩陣（可選，用於進階查詢）
- `outputs/bm25_vectors/vocabulary.json`、`idf.json`、`metadata.json`：檢索所需 artifacts

---

## 📝 注意事項

1. **BM25 top-K 候選是 Stage 1 的輸出**：後續 reranking 和 PPR 由 D 組員負責
2. **候選數量 K**：目前設為 100，可根據實驗結果調整
3. **可重現性**：所有參數（k1, b, top_k）都記錄在程式碼和輸出中
4. **與 D 組員的介面**：`bm25_topk.jsonl` 格式需與 D 組員確認

## 🎯 核心改變總結

**舊方法（已廢棄）**：
- FinalVec = 硬加 BM25 向量 + 情緒向量 + 主題向量
- 問題：維度不匹配，意義不明確

**新方法（目前採用）**：
- Stage 1（C 組員）：BM25 檢索 → top-K 候選
- Stage 2（D 組員）：在候選內做 reranking（分數融合）+ PPR
- 優點：符合 IR 標準做法，可解釋性高，效率更好

