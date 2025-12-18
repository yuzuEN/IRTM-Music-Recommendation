# BM25 語意向量模組

組員 C 負責：**Method 2（BM25 Feature Representation）**

## 功能概述

本模組負責將 A 組員前處理好的歌詞 tokens 轉換成語意向量（TF-IDF / BM25），供後續 FinalVec 融合與推薦系統使用。

## 檔案說明

- `compute_bm25.py`：主要實作檔案
  - 讀取 `data/processed/lyrics/lyrics_tokens.csv`
  - 建立 vocabulary、計算 TF/DF/IDF
  - 產生 TF-IDF 與 BM25 向量矩陣
  - 輸出所有 artifacts 到 `outputs/bm25_vectors/`

## 使用方法

### 1. 執行完整 pipeline（離線建模）

```bash
python src/bm25/compute_bm25.py
```

### 2. 輸出檔案

執行後會在 `outputs/bm25_vectors/` 產生：

- `song_ids.json`：歌曲 ID 列表（順序與矩陣對齊）
- `vocabulary.json`：詞彙表（詞 → index）
- `tfidf_matrix.npz`：TF-IDF 稀疏矩陣（scipy.sparse.csr_matrix）
- `bm25_matrix.npz`：BM25 稀疏矩陣（scipy.sparse.csr_matrix）
- `idf.json`：IDF 值（詞 → idf）
- `metadata.json`：統計量與參數（N, avgdl, k1, b, min_df, max_df_ratio 等）

### 3. 在其他模組中使用

```python
from src.bm25.compute_bm25 import encode_query_tokens
import json
import numpy as np
from scipy.sparse import load_npz

# 載入 vocabulary 與 IDF
with open("outputs/bm25_vectors/vocabulary.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
with open("outputs/bm25_vectors/idf.json", "r", encoding="utf-8") as f:
    idf = json.load(f)
with open("outputs/bm25_vectors/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# 載入 BM25 矩陣
bm25_matrix = load_npz("outputs/bm25_vectors/bm25_matrix.npz")

# 編碼 query tokens（例如一篇貼文的 expanded_tokens）
query_tokens = ["love", "heartbreak", "night"]
query_vec = encode_query_tokens(
    query_tokens,
    vocab,
    idf,
    metadata["N"],
    metadata["avgdl"],
    k1=metadata["k1"],
    b=metadata["b"],
    mode="bm25"
)

# 計算 cosine similarity（範例）
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(query_vec.reshape(1, -1), bm25_matrix)
```

## 參數說明

### Vocabulary Pruning

- `min_df=5`：document frequency < 5 的詞丟棄（過罕見）
- `max_df_ratio=0.5`：document frequency > 50% 的詞丟棄（過常見）

### BM25 參數

- `k1=1.5`：控制 term frequency 飽和度
- `b=0.75`：控制文件長度正規化強度

### IDF 公式

- **BM25 版本**：\\( idf(t) = \\log\\left(\\frac{N - df_t + 0.5}{df_t + 0.5} + 1\\right) \\)
- **TF-IDF 版本**：\\( idf(t) = \\log\\frac{N}{df_t + 1} \\)

## 與其他模組的介面

### 輸入（來自 A）

- `data/processed/lyrics/lyrics_tokens.csv`：歌詞 tokens（已清理、stemming）

### 輸出（給 C 的 FinalVec / D 的 Graph）

- `outputs/bm25_vectors/bm25_matrix.npz`：所有歌曲的 BM25 向量
- `outputs/bm25_vectors/vocabulary.json`：vocabulary（給 query encoding 用）
- `outputs/bm25_vectors/idf.json`：IDF 值（給 query encoding 用）
- `outputs/bm25_vectors/metadata.json`：統計量（N, avgdl, k1, b 等）

### Query Encoding 函式

`encode_query_tokens()` 函式可將任意 tokens（例如貼文的 `expanded_tokens`）轉成與歌詞向量同維度的向量，供後續相似度計算或 FinalVec 融合使用。

## 注意事項

1. 執行前需確保 A 組員已執行 `preprocess_lyrics_full.py` 產生 `lyrics_tokens.csv`
2. 矩陣使用 scipy.sparse.csr_matrix 格式，節省記憶體
3. vocabulary 會做 pruning，實際向量維度可能小於原始詞彙數
4. 所有輸出檔案都使用 UTF-8 編碼

## 實驗用途

本模組產生的 TF-IDF 與 BM25 矩陣可用於：

- **TF-IDF vs BM25 比較實驗**：比較兩種語意表示法的推薦效果
- **FinalVec Ablation Study**：只使用語意向量 vs 語意+情緒+主題的差異

