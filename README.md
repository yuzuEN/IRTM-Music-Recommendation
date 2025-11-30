# IRTM Final Project  
# **Semantic + Emotion + Topic Music Recommendation System**

This project implements a hybrid recommendation system that takes a **social media post** as input and outputs a ranked list of **songs that best match the post’s semantic content, emotions, and topic**.

The system integrates:

- BM25-based semantic vectors  
- 8-class emotion modeling (lyrics + melody + post)  
- Topic keyword detection  
- Similarity Graph construction  
- Personalized PageRank (PPR)  
- Popularity reweighting (Spotify)

The entire system supports:  
✔ One-click full pipeline reconstruction   
✔ One-command query mode (input post → output recommendations)

---

# **Project Structure Overview**

```
IRTM-Music-Recommendation/
│
├── README.md                     ← 主專案說明（助教會讀）
├── CONTRIBUTING.md               ← 組內協作規範與分工
├── requirements.txt              ← 套件列表
│
├── data/
│   ├── raw/
│   │   ├── lyrics/               ← 未清理的歌詞資料（原始資料）
│   │   └── posts/                ← 未清理的貼文資料（原始資料）
│   │
│   └── processed/
│       ├── lyrics/               ← A 清理後歌詞（clean）
│       └── posts/                ← A 清理後貼文（clean）
│
├── outputs/
│   ├── tokens/                   ← A 產生 tokens JSON / CSV / PKL
│   ├── emotion_vectors/          ← B 產生 歌詞/旋律/貼文 的 8 維情緒
│   ├── topic_vectors/            ← B 產生 T 維主題向量
│   ├── bm25_vectors/             ← C 產生 BM25 與 FinalVec
│   └── recommendations/          ← D 產生推薦結果（Top-K）
│
├── src/
│   ├── preprocessing/            ← A：Preprocessing + Query Expansion
│   │   ├── lyrics_preprocess.py
│   │   ├── posts_preprocess.py
│   │   ├── query_expansion.py
│   │   └── README.md
│   │
│   ├── emotion/                  ← B：Emotion & Topic Modeling
│   │   ├── lyrics_emotion.py
│   │   ├── melody_emotion.py
│   │   ├── post_emotion.py
│   │   ├── topic_detection.py
│   │   └── README.md
│   │
│   ├── bm25/                     ← C：BM25 + FinalVec 融合
│   │   ├── compute_bm25.py
│   │   ├── finalvec_fusion.py
│   │   └── README.md
│   │
│   └── graph_ppr/                ← D：Graph + Personalized PageRank
│       ├── build_graph.py
│       ├── personalized_pagerank.py
│       ├── reweight_popularity.py
│       └── README.md
│
├── pipeline/
│   ├── run_all.py                ← 一鍵重建完整推薦系統
│   └── query_once.py             ← 輸入貼文 → 輸出推薦結果
│
└── docs/
    ├── method_diagrams/
    ├── pipeline_flowchart/
    └── demo_images/
```

---

# **Installation**

```
pip install -r requirements.txt
```

---

# **How to Run the Entire System**

## **1. Full Pipeline (Offline Model Building)**  
從 raw data → preprocessing → emotion → BM25 → FinalVec → Graph → PR：

```
python pipeline/run_all.py
```

輸出的結果會放在：

```
outputs/tokens/
outputs/emotion_vectors/
outputs/topic_vectors/
outputs/bm25_vectors/
outputs/recommendations/
```

---

## **2. Online Query Mode（輸入貼文 → 得到歌曲推薦）**

```
python pipeline/query_once.py --post "I feel lonely today"
```

範例輸出：

```
Top 10 Recommended Songs:
1. Fix You – Coldplay       Score: 0.892
2. Someone Like You – Adele Score: 0.875
3. Let Her Go – Passenger   Score: 0.861
...
```

---

# **Team Responsibility (Module-Based Division of Labor)**

| Member | Module | Folder | Description |
|--------|--------|---------|-------------|
| **A** | Preprocessing + Query Expansion | `src/preprocessing/` | 歌詞/貼文清理、token、slang、hashtag、QE |
| **B** | Emotion & Topic Modeling | `src/emotion/` | 歌詞/旋律/貼文情緒、主題偵測 |
| **C** | BM25 + FinalVec Fusion | `src/bm25/` | 語意向量、情緒向量融合 |
| **D** | Graph + PPR | `src/graph_ppr/` | Similarity Graph、PageRank 推薦引擎、Popularity reweighting |
---

# **Data Flow**

```
Raw Data
  ↓

A. Preprocessing → tokens
  ↓
B. Emotion / Topic Modeling → emotion vectors / topic vectors
  ↓
C. BM25 + FinalVec (semantic + emotion + topic)
  ↓
D. Similarity Graph → Personalized PageRank → song ranking
```

---

# **Reproducibility**

本專案可保證完全可重現：

- 所有模組皆有清楚 input/output
- 所有向量化結果都存放於 outputs/
- 完整 pipeline 可自動執行並產生所有中間結果
- query_once.py 支援助教任意輸入貼文測試

---

# **Notes**

- 本專案不需 UI（非必要項目），但 query_once.py 提供功能性測試。
- 所有模型/向量在 offline pipeline 中計算。
- 查詢階段僅使用已建立的向量與 graph。

---

**End of Main README**

