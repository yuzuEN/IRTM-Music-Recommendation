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

### Phase 1：準備 FinalVec 融合（等待 B 組員完成）

- [ ] **確認 B 組員的輸出格式**
  - 與 B 組員確認上述檔案的格式與路徑
  - 確認向量維度（emotion: 8, topic: T）
  - 確認順序對齊方式

- [ ] **設計 `finalvec_fusion.py` 的介面**
  - 規劃函式清單與 I/O
  - 設計權重參數（β, λ, γ）的調整機制

---

### Phase 2：實作 FinalVec 融合（Method 3.6）

- [ ] **實作 `src/bm25/finalvec_fusion.py`**

  **主要函式**：
  
  1. `load_emotion_topic_vectors()`：載入 B 組員的情緒與主題向量
  2. `build_song_finalvec(beta, lambda_)`：
     - 載入 `LyricsVec_BM25`（從 `bm25_matrix.npz`）
     - 載入 `Emotion_song`（從 B）
     - 載入 `TopicVec_song`（從 B）
     - 套用公式：`FinalVec = β * LyricsVec + (1-β) * Emotion + λ * Topic`
     - 輸出：`outputs/bm25_vectors/finalvec_song.npy`
  
  3. `encode_query_vector(beta, gamma)`：
     - 接收 query tokens
     - 計算 `QueryLyricsVec`（使用 `encode_query_tokens`）
     - 載入或計算 `Emotion_query`（從 B）
     - 載入或計算 `TopicVec_post`（從 B）
     - 套用公式：`QueryVec = β * QueryLyricsVec + (1-β) * Emotion_query + γ * TopicVec_post`
     - 輸出：QueryVec（numpy array）

  4. `normalize_vectors()`：對向量做 L2 normalization（可選）

- [ ] **處理向量維度不一致問題**
  - LyricsVec_BM25：`(num_songs, vocab_size)` = `(30246, 53477)`
  - Emotion_song：`(num_songs, 8)`
  - TopicVec_song：`(num_songs, T)`
  - **解決方案**：需要將情緒與主題向量擴展到與 LyricsVec 相同的維度，或使用 concatenation + projection

- [ ] **實作向量融合策略**
  - 方案 A：Concatenation（拼接）
    - `FinalVec = [LyricsVec, Emotion_song, TopicVec_song]`
    - 形狀：`(num_songs, vocab_size + 8 + T)`
  - 方案 B：Weighted Sum（加權和，需要先做 dimension matching）
    - 需要將 Emotion/Topic 向量擴展到 vocab_size 維度（例如用 embedding）
  - **建議**：先實作方案 A（較簡單），之後可嘗試方案 B

---

### Phase 3：實驗與參數調整

- [ ] **實作 `experiments_finalvec_ablation.py`**
  
  **實驗設計**：
  - Baseline 1：只用 `LyricsVec_BM25`
  - Baseline 2：`LyricsVec_BM25 + Emotion_song`
  - Full Model：`LyricsVec_BM25 + Emotion_song + TopicVec_song`
  
  **比較指標**：
  - 推薦結果的差異（Top-K overlap）
  - 人工評估（選幾筆 query，比較哪個版本更合理）
  - 參數 grid search（β, λ, γ 的組合）

- [ ] **參數 Grid Search**
  - β ∈ {0.3, 0.5, 0.7, 0.9}
  - λ ∈ {0.1, 0.2, 0.3}
  - γ ∈ {0.1, 0.2, 0.3}
  - 在 gold set 上評估最佳組合

---

### Phase 4：與 D 組員整合

- [ ] **提供 FinalVec 給 D 組員**
  - 確保 `finalvec_song.npy` 格式正確
  - 確保順序與 `song_ids.json` 對齊
  - 提供 `encode_query_vector()` 函式給 D 組員使用

- [ ] **撰寫介面文件**
  - 說明如何載入 FinalVec
  - 說明如何編碼 query
  - 提供範例程式碼

---

### Phase 5：報告撰寫

- [ ] **Method 2（BM25）章節**
  - TF-IDF vs BM25 比較實驗結果
  - BM25 參數選擇理由

- [ ] **Method 3.6（FinalVec）章節**
  - 向量融合公式
  - 維度處理策略
  - 參數選擇（β, λ, γ）

- [ ] **Experiments 章節**
  - TF-IDF vs BM25 實驗結果
  - FinalVec Ablation Study 結果
  - 參數 Grid Search 結果

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

### 提供給 D 組員（待完成 📝）
- `outputs/bm25_vectors/finalvec_song.npy`
- `src/bm25/finalvec_fusion.py` 中的 `encode_query_vector()` 函式

---

## 📝 注意事項

1. **向量順序對齊**：所有向量必須與 `song_ids.json` 的順序一致
2. **向量維度**：需要處理不同維度的向量融合問題
3. **Normalization**：考慮是否需要對向量做 L2 normalization
4. **記憶體**：FinalVec 可能很大，考慮使用稀疏矩陣或分塊處理
5. **可重現性**：所有參數（β, λ, γ）都要記錄在 metadata 中

