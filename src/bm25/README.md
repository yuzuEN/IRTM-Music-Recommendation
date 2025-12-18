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

### 2.1. 輸出檔案格式詳解

#### `song_ids.json`
- **格式**：JSON Array
- **內容**：`["song_id_1", "song_id_2", ...]`
- **說明**：歌曲 ID 列表，順序與矩陣的 row index 對齊（第 i 首歌對應矩陣的第 i 列）
- **範例**：
  ```json
  [
    "scraped_eminem_houdini",
    "scraped_ariana_grande_imperfect_for_you",
    "hf_00001",
    ...
  ]
  ```

#### `vocabulary.json`
- **格式**：JSON Object
- **內容**：`{"詞": column_index}`
- **說明**：詞彙表，將詞映射到向量矩陣的 column index（0 ~ |V|-1）
- **範例**：
  ```json
  {
    "a": 0,
    "aa": 1,
    "hey": 5428,
    "stork": 44530,
    ...
  }
  ```
- **注意**：詞彙按照字母順序排序，index 從 0 開始連續編號

#### `tfidf_matrix.npz`
- **格式**：NumPy Sparse Matrix（scipy.sparse.csr_matrix）
- **內容**：TF-IDF 稀疏矩陣
- **形狀**：`(num_songs, vocab_size)`，例如 `(30246, 53477)`
- **說明**：
  - 每一列代表一首歌的 TF-IDF 向量
  - 每一欄代表一個詞的 TF-IDF 分數
  - 使用稀疏格式節省記憶體
- **載入方式**：
  ```python
  from scipy.sparse import load_npz
  tfidf_matrix = load_npz("outputs/bm25_vectors/tfidf_matrix.npz")
  ```

#### `bm25_matrix.npz`
- **格式**：NumPy Sparse Matrix（scipy.sparse.csr_matrix）
- **內容**：BM25 稀疏矩陣
- **形狀**：`(num_songs, vocab_size)`，例如 `(30246, 53477)`
- **說明**：
  - 每一列代表一首歌的 BM25 向量
  - 每一欄代表一個詞的 BM25 分數
  - 使用稀疏格式節省記憶體
- **載入方式**：
  ```python
  from scipy.sparse import load_npz
  bm25_matrix = load_npz("outputs/bm25_vectors/bm25_matrix.npz")
  ```

#### `idf.json`
- **格式**：JSON Object
- **內容**：`{"詞": idf_value}`
- **說明**：BM25 版本的 IDF 值（用於 query encoding）
- **公式**：\\( idf(t) = \\log\\left(\\frac{N - df_t + 0.5}{df_t + 0.5} + 1\\right) \\)
- **範例**：
  ```json
  {
    "hey": 7.399381552902967,
    "em": 7.643003635560718,
    "stork": 8.445350108085655,
    ...
  }
  ```

#### `metadata.json`
- **格式**：JSON Object
- **內容**：統計量與參數
- **欄位說明**：
  - `N`：總文件數（歌曲數）
  - `avgdl`：平均文件長度（token 數）
  - `doc_lengths`：每首歌的長度列表（順序與 `song_ids.json` 對齊）
  - `vocab_size`：vocabulary 大小
  - `k1`：BM25 參數 k1（預設 1.5）
  - `b`：BM25 參數 b（預設 0.75）
  - `min_df`：vocabulary pruning 參數（預設 5）
  - `max_df_ratio`：vocabulary pruning 參數（預設 0.5）
  - `max_df`：最大 document frequency（計算值）
- **範例**：
  ```json
  {
    "N": 30246,
    "avgdl": 145.80083316802222,
    "doc_lengths": [385, 128, 132, ...],
    "vocab_size": 53477,
    "k1": 1.5,
    "b": 0.75,
    "min_df": 5,
    "max_df_ratio": 0.5,
    "max_df": 15123
  }
  ```

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

## Vocabulary 檔案比較

本模組使用 A 組員建立的 `data/processed/lyrics/vocabulary.json` 作為詞彙來源，並產生 `outputs/bm25_vectors/vocabulary.json` 供向量化使用。兩個檔案的差異如下：

| 特性 | `data/processed/lyrics/vocabulary.json` | `outputs/bm25_vectors/vocabulary.json` |
|------|----------------------------------------|----------------------------------------|
| **建立者** | A 組員（`preprocess_lyrics_full.py`） | C 組員（`compute_bm25.py`） |
| **格式** | `{"詞": 詞頻}` | `{"詞": column_index}` |
| **值的意義** | 該詞在整個語料中出現的總次數（term frequency count） | 該詞在向量矩陣中的欄位索引（0 ~ \|V\|-1） |
| **用途** | 詞頻統計、分析用 | 向量化索引、查詢用 |
| **Pruning** | 無（包含所有出現過的詞） | 目前無（直接使用 A 的完整 vocabulary） |
| **總詞數** | 53,477 個詞 | 53,477 個詞（與 A 的相同） |
| **範例** | `{"hey": 5428, "stork": 2}` | `{"hey": 5428, "stork": 44530}` |

**說明**：
- A 組員的 vocabulary 存的是**詞頻**（該詞出現幾次），用於統計分析
- C 組員的 vocabulary 存的是**索引**（該詞在向量中的位置），用於向量化
- 目前兩個檔案包含相同的詞彙（因為直接使用 A 的 vocabulary，沒有做額外的 pruning）
- 詞彙按照字母順序排序，確保一致性

## 注意事項

1. 執行前需確保 A 組員已執行 `preprocess_lyrics_full.py` 產生 `lyrics_tokens.csv` 和 `vocabulary.json`
2. 矩陣使用 scipy.sparse.csr_matrix 格式，節省記憶體
3. 目前 vocabulary 直接使用 A 組員的完整詞彙表，沒有做額外的 pruning（向量維度 = 53,477）
4. 所有輸出檔案都使用 UTF-8 編碼

## Demo：BM25-only Query

執行 `demo_bm25_query.py` 可以快速驗證 BM25 語意向量是否有效：

```bash
python src/bm25/demo_bm25_query.py
```

這個 demo 會：
1. 載入所有 BM25 artifacts
2. 從 `posts_clean_expanded.jsonl` 讀取 5 筆貼文當 query
3. 對每筆 query 做 BM25 檢索，找出 Top-10 推薦歌曲
4. 顯示結果（包含原始貼文、推薦歌曲 ID、相似度分數）

**用途**：
- 驗證 BM25 語意向量是否有效
- 提供 D 組員（Graph + PPR）的 baseline 對照
- 可用於報告中的案例展示


## 實驗用途

- **Baseline 對照**：D 組員可以用 `demo_bm25_query.py` 的結果作為 Graph + PPR 的 baseline

