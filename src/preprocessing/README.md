# **README — Data Preprocessing Pipeline (Lyrics + Posts)**

# # **IRTM Final Project – Data Preprocessing Pipeline (Lyrics + Posts)**

本專案的目標是實作「以文本找歌曲」(Text-to-Song Recommendation) 的檢索系統。
其中這個部份是實做 **歌詞資料建立與清理（Lyrics Preprocessing）**

---

# # Project Structure (Preprocessing 部分)

```
src/
  preprocessing/
    preprocess_lyrics_full.py -> 對收集到的 lyrics 原始資料進行 preprocessing
    preprocess_post.py -> 對 post 原始資料進行 preprocessing
    crawl_latest_lyrics.py -> 爬取 2024-2025 熱門歌曲及其歌詞
    download_lyrics_dataset.py -> 下載 lyrics dataset
    duplicate_checker.py -> 檢查 lyrics 原始資料中有沒有重複的歌
    merge_lyrics_corpus.py -> 整合 lyrics dataset 和爬到的 lyrics，去除雜訊

data/
  raw/
    lyrics/
      latest_lyrics_raw.json
      lyrics_raw.json
      all_lyrics_raw.json
    posts/
      generated_social_posts_10k.json -> 包含情緒標籤
      test_text(1).txt -> tweeteval dataset emotion part
      train_text(1).txt -> tweeteval dataset emotion part
      val_text(1).txt -> tweeteval dataset emotion part
      test_text.txt -> tweeteval dataset emoji part
      train_text(1).txt -> tweeteval dataset emoji part
      val_text(1).txt -> tweeteval dataset emoji part
      all_captions.txt -> Instagram posts dataset post text part
    notused/ -> not used data crawl from old reddit and Tumblr
      TumblrReddit_posts_v1.csv 
      TumblrReddit_posts_v1.jsonl
      TumblrReddit_posts_v2.csv
      TumblrReddit_posts_v2.jsonl
  processed/
    lyrics/
      clean_lyrics.json
      lyrics_tokens.csv
      vocabulary.json
    posts/
      posts_clean_expanded.jsonl
```

---

# # **Dataset Resource**

Lyrics:
* Genius: https://genius.com/
* Hugging face lyrics dataset: https://huggingface.co/datasets/mrYou/lyrics-dataset

實作方式：
* 使用爬蟲爬取 billboard 排行榜得知 2024-2025 熱門英文歌曲，再使用該列表去 Genius 爬取對應歌曲的歌詞，
並將爬取到有其他翻譯資訊或其他說明的資料進行清理。最後結合 hugging face dataset 中的歌詞。

Posts:
* Instagram Posts Dataset: https://www.kaggle.com/datasets/thecoderenroute/instagram-posts-dataset
* Tweeteval: https://github.com/cardiffnlp/tweeteval/tree/main

實作方式：
* Instagram Posts Dataset 包含 post 文字本身、圖片和很多其他資訊，因此先使用 python 將所有 post 文字部分，
整理為單一 txt 檔案，名為 all_captions.txt。此外，取用 tweeteval dataset 中和我們主題叫相關的 emoji 和 emotion 資料，
作為社群文章敘述風格參考，再結合 Instagram Posts Dataset 的貼文內容作為 instagram 貼文風格的範例，讓
大型語言模型 chatGPT 生成 10000 筆模仿 Instagram 的現代社群貼文的資料。

找到但沒有用：
* Goemotion: https://github.com/google-research/google-research/tree/master/goemotions

---

# # **Why Build Our Own Dataset?**

老師要求：

* 學習「如何收集、清理、融合原始資料」
* 不能直接用現成資料庫作為唯一來源
* 必須能處理真實世界的 noisy text（歌詞、社群貼文）

因此這部分實作了：

✔ 2024–2025 最新熱門歌曲（Billboard Hot 100 每週抓取）
✔ + Genius API 歌詞
✔ + HuggingFace 歌詞資料庫（2011–2023, 30k songs）
✔ 社群貼文（Tumblr, Reddit, YouTube）
✔ Instagram Post Dataset
✔ TwitterEval（提供情緒標籤平衡資料）
✔ ChatGPT 生成 10k IG-style caption（平衡情緒 + 補齊 IG 語氣）

透過 **混合資料（Hybrid Corpus）** 解決：

* IG caption 不足
* Twitter 的 style 與 IG 不完全相同
* 貼文太短，情緒不明 → 用 LLM 模仿風格生成
* 歌詞資料需併合多來源才能完整覆蓋 2024–2025 熱門曲

---

# ## 🔍 Why Use a Hybrid Dataset (Raw Social Posts + TwitterEval + LLM-Generated Captions)?

本專案採用 **混合資料（Hybrid Corpus）** 作為模型訓練與貼文風格學習的核心方式，結合：

* **真實 IG/Tumblr/YT 貼文**（具備自然語言風格與真實噪音）
* **TwitterEval（高品質情緒標註資料）**
* **LLM 生成的 IG-style captions（擴大資料量並平衡情緒分布）**

這樣的混合策略不只是為了「補資料不夠」，而是有理論支持、實證證明更有效。以下是整理自五篇近年 NLP/LLM 文獻的結論摘要。

---

# ## 📘 Evidence from Recent Research

### ### **1. LLM-driven augmentation consistently improves text classification performance**

**(Neshaei et al., 2025, EDM)**
🔗 [https://arxiv.org/abs/2025.EDM.long-papers.54](https://arxiv.org/abs/2025.EDM.long-papers.54) (EMNLP/EDM proceedings)

* 研究對 78 篇 LLM augmentation 文獻做系統性回顧，並提出五階段 Data-Aug pipeline。
* 在三個教育場景的分類任務中，所有任務都因為加入 LLM 生成資料而 **balanced accuracy 提升 4–10%**。
* 特別是：

  * **少量真實資料 + LLM 合成資料 > 單用真實資料 / 單用生成資料**。
* 他們也指出 **資料不平衡與不足是問題主因**，而 LLM 能有效補齊 minority labels。

📌 **我們的任務（多情緒、多主題）同樣高度 imbalanced，因此 LLM-generated IG captions 能有效補強資料分布。**

---

### ### **2. Synthetic text helps most when real data is limited**

**(Li et al., 2024, ICML)**
🔗 [https://arxiv.org/abs/2407.12813](https://arxiv.org/abs/2407.12813)

這篇工作系統測試 6 個 NLP 任務，發現：

* **使用純 synthetic data 訓練模型效果很差**（語言模型的風格與真實資料不同）。

* 但如果使用：

  ```
  少量 raw data（100） + 大量 synthetic data（1000）
  ```

  → 所有任務表現都有提升。

* 結論非常重要：

> **Mixing raw + synthetic is necessary.**（原文 §5.1）

* 同時發現：

  * synthetic data 的 **diversity** 能提升分類能力
  * topic-conditioned generation（例如生成 IG 風格）效果最好

📌 **對我們的 IG/Twitter 風格貼文任務，同樣需要讓模型學習語氣，因此加入 LLM 生成資料（高多樣性）非常必要。**

---

### ### **3. LLM-generated data可補足 domain style gap**

**(EMNLP Findings 2020 – Style transfer / augmentation)**

這篇研究顯示：

* LLM（當時較小模型）能捕捉 domain-specific style
* synthetic utterances 用於分類時能 **改善 domain mismatch 問題**

📌 **由於 IG caption 與 Twitter 不同，在我們的資料流程中：
LLM 用 IG 語氣生成 captions → 補足 TwitterEval 的 domain gap。**

---

### ### **4. Synthetic data balances emotional labels better than natural corpora**

**(arXiv 2005.00547 – Data augmentation survey)**

這篇 survey 指出：

* 自然語料中，**情緒資料分布通常高度不均**
* augmentation 能「增強稀有類別的 representation」
* 結果更能提升 **macro-F1 / minority label performance**

📌 TwitterEval 雖然有情緒標籤，但 **IG-style caption 的情緒呈現方式完全不同**，需要 synthetic IG captions 來填補分布。

---

### ### **5. LLM generation helps classification even if LLM本身在任務上表現不佳**

**(Li et al., 2024)**

這篇的另一個驚人結果：

> 即使 LLM 在某個 task 上 zero-shot 表現不佳，
> **它生成的資料仍然可幫助 RoBERTa/BERT classifier 提升效果。**

原因：

* LLM 很擅長「製造符合 label 的例子」
* 但不一定擅長做推論
  → 所以 synthetic data 作為 supervised examples 仍然有用。

📌 對我們的情緒分類也是一樣：
LLM 的情緒判定不完美，但 **生成的 IG-style 貼文卻非常適合作為訓練資料**。

---

# ## 📌 Final Summary — Why Hybrid Data?

以下是可以直接放進 README 的總結：

### ### **Why Hybrid?（為什麼使用混合資料？）**

為了建立一個能理解 **IG-style captions**、情緒、多主題的檢索模型，我們需要同時滿足：

| 來源                    | 長處                      | 缺點                              |
| --------------------- | ----------------------- | ------------------------------- |
| **IG/Tumblr/YT 真實貼文** | 真實風格、emoji 用法、自然語言      | 標籤不足、不平衡、資料稀少                   |
| **TwitterEval**       | 高品質情緒標註                 | 語氣與 IG 不同，內容偏向 Twitter          |
| **LLM 生成貼文（10k）**     | 可模仿 IG 語氣、多樣性高、可控制情緒與主題 | 需與真實資料混合才能避免 distribution shift |

基於上述五篇論文的實證結果：

* **少量真實資料 + synthetic data > 單獨使用任一種**
* synthetic data 能改善 **資料稀疏 & 標籤不平衡**
* topic-conditioned generation（例如「生成 IG caption」）能最佳化 domain matching
* synthetic data diversity ↑ → classifier performance ↑
* 即使 LLM 本身分類不準，它生成的例子仍能提高模型效能
* real data 是不可缺少的 grounding，避免模型學到錯誤 distribution

---

# ## ⭐ 簡潔版本

## ### Why We Use Hybrid Data (Real Posts + TwitterEval + LLM-Generated Captions)

Our task—emotion/topic-aware caption-to-song retrieval—requires a dataset that matches the **style, emotional density, and linguistic noise** of IG captions. Real IG/Tumblr/YT posts alone are insufficient (too few, highly imbalanced), while TwitterEval provides strong emotion labels but a different linguistic domain. Recent research shows that:

* **Synthetic + real data > either one alone**
  – LLM-driven augmentation significantly improves text classification, especially in low-resource settings (Neshaei et al., 2025; Li et al., 2024).
* **Synthetic data fixes label imbalance**
  – Minority emotion classes benefit the most from augmentation (Feng et al., 2020).
* **Domain-specific synthetic text reduces style mismatch**
  – LLM-generated captions help bridge the gap between Twitter-style labeled data and IG-style user text.
* **LLM-generated examples remain helpful even when LLM itself performs poorly on the task**
  – The classifier can still learn from high-quality synthetic examples (Li et al., 2024).

Therefore, we use a **hybrid corpus** combining:

1. **Real social posts** → authentic style, emoji usage, noise
2. **TwitterEval** → high-quality emotion labels
3. **LLM-generated IG-style posts (10k)** → balanced emotion distribution + realistic caption style

This hybrid design follows best practices in current NLP augmentation research and provides a robust foundation for downstream IR-based music recommendation.

---

# # **Lyrics Preprocessing Pipeline**

主要程式：**preprocess_lyrics_full.py**

來源：

1. Billboard + Genius（用 crawl_latest_lyrics.py）
2. HuggingFace lyrics-dataset（2011–2023）
3. 兩者合併後用 merge_lyrics_corpus.py 清理去重、語言判斷

流程：

---

## ✔ Step 1: 收集最新 Hot 100 歌曲（2024–2025）

由 Billboard 每週榜單自動抓取歌曲清單。

## ✔ Step 2: 用 Genius API 抓取歌詞

並且：

* 去除翻譯版本
* 去除包含 “Translation”，"Traducción"，俄文/德文等外文頁面
* 保留英文歌曲

---

## ✔ Step 3: 清理歌詞

使用 merge_lyrics_corpus.py【】：

* 移除 metadata（Read More、Lyrics Provided、Producers）
* 移除 (Intro)、(Chorus)、(2x) 等非歌詞行
* 移除過長敘述行
* 刪除非英文歌詞（langdetect）
* 刪除過短 (<50 字) 或過少行數 (<3 行) 的歌詞

---

## ✔ Step 4: 合併 HuggingFace 30k 歌詞

與 Billboard/Genius 資料合併後去重。
duplicate_checker.py【】使用 fuzzy matching 判重：

* 同歌手 + 高相似度 (>0.85) → 視為 duplicate
* 去除 feat. / remix / live 版本

---

## ✔ Step 5: Tokenization + Stopwords + Stemming

preprocess_lyrics_full.py【】：

* lowercase
* 移除標點、數字、網址
* Treebank tokenizer
* Stopwords removal
* Porter stemming
* 建立 vocabulary 字典
* 產生 lyrics_tokens.csv → BM25 / embedding 使用

**為什麼歌詞需要 stemming？**
因為歌詞是長文本（document）：

* stemming 可以泛化詞彙（love/loving → love）
* 增加查詢覆蓋率
* 減少詞彙稀疏

貼文則不做 stemming（因為語氣強度重要，後述）。

---

# # **Post Preprocessing Pipeline**

主要程式：**preprocess_post.py**【】

來源：

1. Instagram Posts Dataset
2. TwitterEval（emojis, emotions 部分資料夾）
3. ChatGPT 生成 IG-style captions（10k 筆，用於平衡資料與 IG 語氣）

---

## ✔ Step 1: URL / @username 移除

移除無語意的 noise。

## ✔ Step 2: Hashtag 分詞（#NotOkay → not okay）

利用 CamelCase 判斷切割：

```
[A-Z][a-z]+ | [a-z]+ | \d+
```

避免整串 hashtag 變成一個 token。

---

## ✔ Step 3: Emoji 分離（embedding 由組員 B 處理）

Emoji 是社群貼文中的情緒來源（尤其 IG）：

* 😭 → sadness
* ✨ → excitement
* ❤️ → affection
* 🔥 → hype


**把 emoji 從字串中拆成獨立 token**（不要刪除）

---

## ✔ Step 4: Slang 正規化

例如：

| Slang | Normalized   |
| ----- | ------------ |
| idk   | i don't know |
| omg   | oh my god    |
| u     | you          |
| ur    | your         |
| lmao  | laughing     |

提升貼文與歌詞詞彙重疊率。

---

## ✔ Step 5: 重複字壓縮（soooo → soo）

保留語氣但避免詞彙爆炸。

---

## ✔ Step 6: Tokenization

使用簡易 regex：

```
\w+ | [^\w\s]
```

可以同時抓到：

* 英文詞彙
* 標點符號
* emoji

---

# # **Query Expansion (QE)**

### 實作的 QE 在 preprocess_post.py 裡：

✔ WordNet 擴展
✔ Emotion Lexicon 擴展（可選）
✔ PRF（Pseudo Relevance Feedback，可選）

---

## ✔ WordNet 同義詞擴展

例如：

```
tired → exhausted, weary
sad → unhappy, sorrowful
```

貼文非常短 → QE 是必要的。

---

## ✔ Emotion Lexicon 擴展

如貼文帶 sadness：

```
sad → depressed, heartbroken, lonely
```

更容易找到相同情緒的歌詞。

---

## ✔ PRF（Pseudo Relevance Feedback）

利用「第一次搜尋到的 top documents」反向強化 query。
<!-- （可選功能） -->

---

# # **Why We Do NOT Stem Posts（但要 Stem Lyrics）？**

✔ 貼文是 Query：

* 貼文很短（通常 < 20 tokens）
* 情緒強度（crying vs cry）需要被保留
* slang 不適合 stemming
* emoji 要保留語氣

❌ stemming 會破壞「語氣 / 情緒資訊」。

✔ 歌詞是 Document：

* 文本長
* stemming 可以泛化詞彙，提高查詢覆蓋率
* 不會失去語氣（歌詞本身已完整表達情緒）

---

# # Output Files

## Lyrics

```
data/processed/clean_lyrics.json
data/processed/lyrics_tokens.csv
data/processed/vocabulary.json
```

## Posts

```
data/processed/posts/posts_clean_expanded.jsonl
```

---

# # Summary: preprocessing part

### ✔ 收集資料

* Billboard/Genius 歌詞（2024–2025）
* HuggingFace lyrics（2011–2023）
* Tumblr / Reddit / YouTube posts -> but not used (和主題不太符合)
* Instagram Posts Dataset
* TwitterEval
* ChatGPT 生成 10k IG captions

### ✔ 歌詞 preprocessing

* 中文/非英文移除
* metadata 清理
* fuzzy duplicate removal
* 英文判斷
* token + stopwords + stemming

### ✔ 貼文 preprocessing

* hashtag 分詞
* emoji 分離
* URL / username 過濾
* slang 修正
* 重複字壓縮
* tokenization

### ✔ Query Expansion

* WordNet
* Emotion Lexicon
* PRF（optional）

### ✔ 產生乾淨語料供 BM25 / embedding 模型使用
