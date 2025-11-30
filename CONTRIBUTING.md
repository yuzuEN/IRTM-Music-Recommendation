# 🤝 Team Collaboration & Development Guide  
(IRTM Music Recommendation Project)

此文件為團隊內部的協作規範、分工說明、使用方式與開發流程。

---

# 📌 Branch Rules

每位組員擁有自己的開發 branch，不直接在 main 上修改。

| Member | Branch name |
|--------|--------------|
| A | `preprocess-A` |
| B | `emotion-B` |
| C | `bm25-C` |
| D | `recommend-D` |

開分支方式：

```
git checkout -b preprocess-A
```

更新 main 後再進入自己的分支：

```
git checkout main
git pull
git checkout preprocess-A
git merge main
```

---

# 📌 Folder Ownership（誰負責什麼資料夾）

| Folder | Owner | Description |
|--------|--------|--------------|
| `src/preprocessing/` | A | Preprocessing, Query Expansion |
| `src/emotion/` | B | Lyrics/Melody/Post Emotion, Topic Detection |
| `src/bm25/` | C | BM25, FinalVec Fusion |
| `src/graph_ppr/` | D | Graph Construction, PPR, Re-ranking |
| `pipeline/` | C or D | run_all.py + query_once.py |

每位組員 **不得修改他人模組的程式碼**，避免衝突。

---

# 📌 Input / Output Specification

## A. Preprocessing (A)
輸入：
```
data/raw/lyrics/*
data/raw/posts/*
```

輸出：
```
data/processed/lyrics/*
data/processed/posts/*
outputs/tokens/*.json / .csv / .pkl
```

---

## B. Emotion & Topic (B)
輸入：
```
outputs/tokens/*
data/processed/posts/*
```

輸出：
```
outputs/emotion_vectors/*
outputs/topic_vectors/*
```

---

## C. BM25 + FinalVec (C)
輸入：
```
outputs/tokens/*
outputs/emotion_vectors/*
outputs/topic_vectors/*
```

輸出：
```
outputs/bm25_vectors/*
```

---

## D. Graph + PPR (D)
輸入：
```
outputs/bm25_vectors/*
```

輸出：
```
outputs/recommendations/*
```

---

# 📌 How to Add Code to Repo

1️⃣ 切換到自己的專屬 branch  
2️⃣ 把 code 放進自己的資料夾（如 `src/preprocessing/`）  
3️⃣ 推到 GitHub

```
git add .
git commit -m "Add preprocessing module"
git push origin preprocess-A
```

4️⃣ 開 Pull Request  
5️⃣ 全組確認 → merge 進 main  

---

# 📌 Pipeline Workflow

## Offline 模式（重建所有資料）
```
python pipeline/run_all.py
```

會依序執行：

- A：Preprocessing  
- B：Emotion & Topic  
- C：BM25 + FinalVec  
- D：Graph + PageRank  

---

## Online Query Mode（助教輸入貼文）
```
python pipeline/query_once.py --post "I feel sad today"
```

輸出：Top-K 推薦歌曲

---

# 📌 Final Deliverables（每位組員最後要交的成果）

| Member | Final Deliverables |
|--------|---------------------|
| **A** | preprocessing 程式碼、clean 資料、QE、tokens（json/csv/pkl） |
| **B** | 歌詞/旋律/貼文情緒向量、topic vectors |
| **C** | BM25、FinalVec、融合實驗 |
| **D** | graph、PageRank、re-ranking、query_once.py |
| **Team** | 能跑通的 pipeline、報告、GitHub repo |

---

# 📌 Commit Message Rules

```
[A] Add preprocessing for lyrics
[B] Implement melody emotion mapping
[C] Add BM25 vector generation
[D] Implement PageRank ranking
```

---

# 📌 Code Style

- Python 3.10+
- 每個模組內需提供 `README.md`
- functions 必須有 docstring
- 所有檔案需可單獨執行或被 import

---

**End of CONTRIBUTING.md**
