# BM25 語意檢索模組

組員 C 負責：**Method 2（BM25 Retrieval - 語意候選擷取）**

## 功能概述

本模組負責：
1. **建立 BM25 檢索系統**：將 A 組員前處理好的歌詞 tokens 轉換成 BM25 向量矩陣
2. **Stage 1 候選生成**：對每篇貼文計算 BM25 分數，輸出 top-K 候選歌曲集合
3. **提供檢索骨幹**：供後續 D 組員進行 reranking 與 Personalized PageRank

**核心設計**：採用「兩階段檢索」架構
- **Stage 1（本模組）**：BM25 語意檢索 → top-K 候選
- **Stage 2（D 組員）**：在候選內做 reranking（情緒/主題相似度）+ PPR

## 檔案說明

- `compute_bm25.py`：主要實作檔案
  - 讀取 `data/processed/lyrics/lyrics_tokens.csv`
  - 建立 vocabulary、計算 TF/DF/IDF
  - 產生 TF-IDF 與 BM25 向量矩陣
  - 對所有貼文計算 BM25 top-K 候選
  - 輸出 artifacts 到 `outputs/bm25_vectors/` 和 `outputs/retrieval/`

- `demo_bm25_query.py`：BM25-only 查詢 demo（baseline 展示）

- `METHOD_REVISION_GUIDE.md`：方法修正說明（為什麼改、改什麼、怎麼做）

- `TODO.md`：待辦事項與 B 組員介面規範

## 使用方法

### 1. 執行完整 pipeline（離線建模）

```bash
python src/bm25/compute_bm25.py
```

### 2. 輸出檔案

執行後會產生兩類輸出：

#### 2.1 BM25 向量（`outputs/bm25_vectors/`）

**必要檔案**：
- `song_ids.json`：歌曲 ID 列表（順序與矩陣對齊）
- `vocabulary.json`：詞彙表（詞 → index）
- `idf.json`：IDF 值（詞 → idf）
- `metadata.json`：統計量與參數（N, avgdl, k1, b, min_df, max_df_ratio 等）

**可選檔案**：
- `bm25_matrix.npz`：BM25 稀疏矩陣（scipy.sparse.csr_matrix）
  - **用途**：用於對新 query 做檢索（進階用途）
  - **注意**：如果只需要使用已產生的 `bm25_topk.jsonl`，則不需要此檔案

#### 2.2 檢索結果（`outputs/retrieval/`）

- `bm25_topk.jsonl`：**每篇貼文的 top-K 候選歌曲**（Stage 1 輸出）
  - 格式：JSONL，每行一個 query 的候選列表
  - 內容：`query_id`, `raw_text`, `emotion`, `top_k`, `candidates`（含 `song_id` 和 `bm25_score`）
  - **用途**：給 D 組員做 reranking 和 PPR 的輸入

- `idf.json`：IDF 值（供 B 組員使用，若需要 tf-idf 權重）

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

#### `bm25_matrix.npz`（可選）
- **格式**：NumPy Sparse Matrix（scipy.sparse.csr_matrix）
- **內容**：BM25 稀疏矩陣
- **形狀**：`(num_songs, vocab_size)`，例如 `(30246, 53477)`
- **說明**：
  - 每一列代表一首歌的 BM25 向量
  - 每一欄代表一個詞的 BM25 分數
  - 使用稀疏格式節省記憶體
- **用途**：
  - **主要用途**：對新 query 做檢索（進階用途）
  - **注意**：如果只需要使用已產生的 `bm25_topk.jsonl`，則**不需要**此檔案
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

#### `outputs/retrieval/bm25_topk.jsonl`（Stage 1 輸出）
- **格式**：JSONL（每行一個 JSON）
- **內容**：每篇貼文的 top-K 候選歌曲與 BM25 分數
- **範例**：
  ```json
  {
    "query_id": "post_0",
    "raw_text": "Why is it always customer service...",
    "emotion": "anger",
    "top_k": 100,
    "candidates": [
      {"song_id": "hf_09875", "bm25_score": 0.098776},
      {"song_id": "hf_25390", "bm25_score": 0.095280},
      ...
    ]
  }
  ```
- **用途**：
  - 給 D 組員做 reranking（在 top-K 內用情緒/主題相似度重新排序）
  - 給 D 組員做 PPR（以高分候選為 seed 在圖譜上擴散）

#### `outputs/retrieval/idf.json`
- **格式**：JSON Object（與 `outputs/bm25_vectors/idf.json` 相同）
- **內容**：`{"詞": idf_value}`
- **用途**：供 B 組員使用（若需要 tf-idf 權重）

### 3. 在其他模組中使用

#### 3.1 載入 BM25 top-K 候選（給 D 組員）

```python
import json

# 載入 BM25 top-K 候選
with open("outputs/retrieval/bm25_topk.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        query_result = json.loads(line)
        query_id = query_result["query_id"]
        candidates = query_result["candidates"]  # List[{"song_id": str, "bm25_score": float}]
        
        # 在 top-K 候選內做 reranking（D 組員的任務）
        # 例如：計算情緒/主題相似度，重新排序
```

#### 3.2 直接計算 BM25 分數（進階用法）

```python
from src.bm25.compute_bm25 import encode_query_tokens, compute_bm25_score_for_query
import json
from scipy.sparse import load_npz

# 載入 artifacts
with open("outputs/bm25_vectors/vocabulary.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
with open("outputs/bm25_vectors/idf.json", "r", encoding="utf-8") as f:
    idf = json.load(f)
with open("outputs/bm25_vectors/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
with open("outputs/bm25_vectors/song_ids.json", "r", encoding="utf-8") as f:
    song_ids = json.load(f)

bm25_matrix = load_npz("outputs/bm25_vectors/bm25_matrix.npz")

# 對 query 計算 BM25 分數
query_tokens = ["love", "heartbreak", "night"]
scores = compute_bm25_score_for_query(
    query_tokens,
    bm25_matrix,
    vocab,
    idf,
    metadata,
    song_ids
)

# 找出 top-K
top_k = 100
top_indices = np.argsort(scores)[::-1][:top_k]
top_songs = [(song_ids[i], float(scores[i])) for i in top_indices]
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

### 輸出（給 D 組員的 Reranking / PPR）

**必要檔案**：
- `outputs/retrieval/bm25_topk.jsonl`：**每篇貼文的 top-K 候選歌曲**（Stage 1 輸出）
  - 這是 D 組員做 reranking 和 PPR 的**主要輸入**
  - 每行包含一個 query 的候選列表與 BM25 分數

**可選檔案**（僅用於新 query 檢索）：
- `outputs/bm25_vectors/bm25_matrix.npz`：BM25 矩陣（用於對新 query 做檢索）
- `outputs/bm25_vectors/vocabulary.json`：vocabulary（用於 query encoding）
- `outputs/bm25_vectors/idf.json`：IDF 值（用於 query encoding）
- `outputs/bm25_vectors/metadata.json`：統計量（N, avgdl, k1, b 等）

### 輸出（給 B 組員）

- `outputs/retrieval/idf.json`：IDF 值（若 B 組員需要 tf-idf 權重）

### 系統架構

```
貼文 (Query)
    ↓
[Stage 1: BM25 Retrieval] ← 本模組負責
    ↓
Top-K 候選歌曲 (C_K)
    ↓
[Stage 2: Reranking] ← D 組員負責
    - 情緒相似度 (cosine)
    - 主題相似度 (cosine)
    - BM25 分數
    - 熱門度
    ↓
[Stage 3: Personalized PageRank] ← D 組員負責
    - 在歌曲圖譜上擴散
    ↓
最終推薦結果
```

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

## 系統架構說明

### 兩階段檢索設計

本系統採用「兩階段檢索 + 分數融合」架構，而非「向量硬加」：

**為什麼不直接融合向量？**
- BM25 向量是 53,477 維（詞彙空間）
- 情緒向量是 8 維（情緒空間）
- 主題向量是 T 維（主題空間）
- **維度不匹配，無法直接相加**

**新設計的優勢**：
1. **符合 IR 標準做法**：先檢索（retrieval）再重排（reranking）
2. **可解釋性高**：每個分數的意義清楚
3. **效率更好**：PPR 只在 top-K 內跑，不用處理 3 萬首歌
4. **權重容易調整**：w1, w2, w3 的意義明確

詳細說明請參考 `METHOD_REVISION_GUIDE.md`。

## 注意事項

1. 執行前需確保 A 組員已執行 `preprocess_lyrics_full.py` 產生 `lyrics_tokens.csv` 和 `vocabulary.json`
2. 矩陣使用 scipy.sparse.csr_matrix 格式，節省記憶體
3. 目前 vocabulary 直接使用 A 組員的完整詞彙表，沒有做額外的 pruning（向量維度 = 53,477）
4. 所有輸出檔案都使用 UTF-8 編碼
5. **BM25 top-K 候選是 Stage 1 的輸出**，後續 reranking 和 PPR 由 D 組員負責

## Demo：BM25-only Query（Baseline）

執行 `demo_bm25_query.py` 可以快速驗證 BM25 語意檢索是否有效：

```bash
python src/bm25/demo_bm25_query.py
```

這個 demo 會：
1. 載入所有 BM25 artifacts
2. 從 `posts_clean_expanded.jsonl` 讀取 5 筆貼文當 query
3. 對每筆 query 做 BM25 檢索，找出 Top-10 推薦歌曲
4. 顯示結果（包含原始貼文、推薦歌曲 title/artist、BM25 分數）

**用途**：
- 驗證 BM25 語意檢索是否有效（Stage 1 的 baseline）
- 提供 D 組員（Reranking + PPR）的 baseline 對照
- 可用於報告中的案例展示

**注意**：這是「純 BM25」的結果，後續 D 組員會在此基礎上加入情緒/主題 reranking 和 PPR。

