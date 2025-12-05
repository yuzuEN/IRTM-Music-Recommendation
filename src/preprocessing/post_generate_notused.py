import os
import json
import random
import time
import pathlib
import numpy as np

from dotenv import load_dotenv
load_dotenv()  # ← 從 .env 自動載入 OPENAI_API_KEY

from openai import OpenAI
client = OpenAI()   # API key 會自動從 OPENAI_API_KEY 環境變數讀取

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



############################################################
# 0. PATH CONFIGURATION (Windows / Mac / Linux safe)
############################################################

# 此 Python 檔案所在位置：src/preprocessing
CURRENT_DIR = pathlib.Path(__file__).resolve().parent

# 專案根目錄：IRTM-Music-Recommendation/
PROJECT_ROOT = CURRENT_DIR.parent.parent

# raw 資料
BASE_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "posts"

# synthetic output 資料夾
BASE_OUT_DIR = PROJECT_ROOT / "data" / "processed"
os.makedirs(BASE_OUT_DIR, exist_ok=True)

print("CURRENT_DIR:", CURRENT_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("BASE_RAW_DIR:", BASE_RAW_DIR)
print("BASE_OUT_DIR:", BASE_OUT_DIR, "\n")

RAW_FILES = [
    "all_captions.txt",
    "train_text.txt",
    "train_text (1).txt",
    "val_text.txt",
    "val_text (1).txt",
    "test_text.txt",
    "test_text (1).txt"
]



############################################################
# 1. LOAD RAW TEXT FILES
############################################################

def load_text_file(path):
    """讀取 txt 檔案，每行一筆資料。"""
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return []

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(line)
    return items


all_posts = []

print("===== LOADING DATASETS =====")
for filename in RAW_FILES:
    full_path = BASE_RAW_DIR / filename
    data = load_text_file(full_path)
    print(f"Loaded {len(data):5d} lines from {filename}")
    all_posts.extend(data)

print(f"\nTOTAL posts loaded: {len(all_posts)}")
print("====================================\n")



############################################################
# 2. SELECT STYLE EXAMPLES FOR TONE LEARNING
############################################################

def extract_style_samples(data, n=300):
    return random.sample(data, min(n, len(data)))

style_examples = extract_style_samples(all_posts, n=300)
STYLE_TEXT = "\n".join(style_examples)



############################################################
# 3. SYSTEM PROMPT
############################################################

SYSTEM_PROMPT = f"""
You are a model generating ORIGINAL Instagram/Twitter-style posts.

You MUST NOT copy or paraphrase ANY of the provided examples.
Only learn tone, emoji frequency, spacing, sentence style, and mood.

===== STYLE EXAMPLES (DO NOT COPY) =====
{STYLE_TEXT}
===== END EXAMPLES =====

Rules:
- 1–3 sentences max
- natural human tone
- casual emotional writing
- occasional emojis (not too many)
- avoid artificial or repetitive patterns
- avoid hashtags unless natural
- avoid real people's names
"""



############################################################
# 4. USER PROMPT (batch generation)
############################################################

USER_PROMPT = """
Generate 50 ORIGINAL social media posts.

Each output line must be a JSON object:

{
  "text": "...",
  "emotion_label": "...",
  "strong_weak": "..."
}

Rules:
- emotion_label ∈ {joy, anger, fear, sadness, surprise, disgust, excitement, neutral}
- strong_weak ∈ {"strong", "weak"}
- DO NOT resemble or paraphrase the style examples
- Output EXACTLY 50 lines in JSONL format.
"""



############################################################
# 5. CALL OPENAI API  (new SDK)
############################################################

def call_model(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=1.0,
        max_tokens=2000,
    )
    return response.choices[0].message.content



############################################################
# 6. SEMANTIC SIMILARITY FILTER
############################################################

def build_vectorizer():
    vec = CountVectorizer(max_features=3000)
    vec.fit(style_examples)
    return vec

vectorizer = build_vectorizer()
style_vectors = vectorizer.transform(style_examples)

def too_similar(text, threshold=0.65):
    """避免生成內容與 IG/Twitter 原文太相似。"""
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, style_vectors).flatten()
    return max(sims) > threshold



############################################################
# 7. GENERATE SYNTHETIC POSTS
############################################################

TARGET = 10000
generated = []

print("===== START GENERATION =====\n")

while len(generated) < TARGET:
    print(f"[Batch] Current: {len(generated)} / {TARGET}")

    try:
        raw_output = call_model(SYSTEM_PROMPT, USER_PROMPT)
    except Exception as e:
        print("OpenAI API error, retry in 3 sec:", e)
        time.sleep(3)
        continue

    lines = raw_output.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # JSON parse
        try:
            obj = json.loads(line)
        except:
            continue

        text = obj.get("text", "")
        if len(text) < 5:
            continue

        # 過濾太相似的（避免抄 IG/Twitter）
        if too_similar(text):
            continue

        generated.append(obj)

        if len(generated) >= TARGET:
            break

    time.sleep(1)

print(f"\n===== FINISHED! Generated {len(generated)} posts. =====\n")



############################################################
# 8. SAVE FINAL JSONL
############################################################

OUTPUT_PATH = BASE_OUT_DIR / "synthetic_posts.jsonl"

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for obj in generated:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Saved synthetic dataset to:\n{OUTPUT_PATH}")
