# ============================================================
# FINAL VERSION — FIXED Tumblr + YouTube issues
# (with missing import os added)
# ============================================================

import os
import json
import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ------------------------------------------------------------
# 強化 headers（Tumblr & YouTube 反爬蟲繞過）
# ------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.tumblr.com/",
}

# ------------------------------------------------------------
# 自動讀 cookie.txt（Tumblr & YouTube用）
# ------------------------------------------------------------
COOKIE_FILE = "cookies.txt"
COOKIES = {}

if COOKIE_FILE and os.path.exists(COOKIE_FILE):
    with open(COOKIE_FILE, "r", encoding="utf-8") as f:
        cookie_str = f.read().strip()
        for pair in cookie_str.split(";"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                COOKIES[k.strip()] = v.strip()

# ------------------------------------------------------------
# 工具
# ------------------------------------------------------------
ASCII_RE = re.compile(r"^[\x00-\x7F]+$")

def is_english(text):
    return bool(ASCII_RE.match(text))

def clean(text):
    text = re.sub(r"\s+", " ", text.replace("\n", " ").replace("\r", " ")).strip()
    return text


# ============================================================
# Tumblr Crawler（已修：加 cookie + retry）
# ============================================================
def crawl_tumblr(blog, pages=5):
    print(f"\n🔎 Tumblr: {blog}")
    base = f"https://{blog}.tumblr.com"
    all_posts = []

    for p in range(1, pages + 1):
        url = f"{base}/page/{p}"
        print("  →", url)

        # retry 3 次
        for _ in range(3):
            try:
                r = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=10)
                break
            except:
                time.sleep(1)

        if r.status_code != 200:
            print("    ❌ Status:", r.status_code)
            break

        soup = BeautifulSoup(r.text, "html.parser")

        posts = soup.find_all("article")
        if not posts:
            posts = soup.find_all("div", {"class": lambda x: x and "post" in x.lower()})

        for p in posts:
            text = clean(p.get_text())
            if 3 <= len(text.split()) <= 60 and is_english(text):
                all_posts.append({
                    "source": "tumblr",
                    "topic": blog,
                    "text": text,
                })

        time.sleep(1)

    print("  ✔", len(all_posts), "posts")
    return all_posts


# ============================================================
# Reddit Crawler（穩定）
# ============================================================
def crawl_reddit(sub, pages=4):
    print(f"\n🔎 Reddit: r/{sub}")
    url = f"https://old.reddit.com/r/{sub}/"
    all_posts = []

    for _ in range(pages):
        print("  →", url)
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            break

        soup = BeautifulSoup(r.text, "html.parser")
        posts = soup.find_all("div", class_="thing")

        for post in posts:
            title = post.find("a", class_="title")
            body = post.find("div", class_="expando")

            text = clean(
                (title.get_text(" ", strip=True) if title else "") + " " +
                (body.get_text(" ", strip=True) if body else "")
            )

            if is_english(text) and 3 <= len(text.split()) <= 60:
                all_posts.append({
                    "source": "reddit",
                    "topic": sub,
                    "text": text,
                })

        nxt = soup.find("span", class_="next-button")
        if not nxt:
            break

        url = nxt.find("a")["href"]
        time.sleep(1.5)

    print("  ✔", len(all_posts), "posts")
    return all_posts


# ============================================================
# YouTube Crawler（修正版）
# ============================================================
def crawl_youtube(url, max_comments=300):
    try:
        from yt_dlp import YoutubeDL
    except:
        print("⚠ yt_dlp not installed — skipping")
        return []

    print(f"\n🔎 YouTube: {url}")
    comments = []

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "extract_flat": False,
        "getcomments": True,
        "cookiesfrombrowser": ("chrome",),
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        print("  ❌ error:", e)
        return []

    for c in (info.get("comments") or []):
        text = clean(c.get("text", ""))
        if is_english(text) and 3 <= len(text.split()) <= 40:
            comments.append({
                "source": "youtube",
                "topic": "yt",
                "text": text,
            })
        if len(comments) >= max_comments:
            break

    print("  ✔", len(comments), "comments")
    return comments


# ============================================================
# Save
# ============================================================
def save(records):
    df = pd.DataFrame(records)
    df.to_csv("final_corpus.csv", index=False, encoding="utf-8-sig")
    print("\n💾 Saved final_corpus.csv:", len(df))

    with open("final_corpus.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print("💾 Saved final_corpus.jsonl")


# ============================================================
# MAIN
# ============================================================
def main():

    print("\n=====================================")
    print("     START FINAL SCRAPING (FIXED)")
    print("=====================================\n")

    data = []

    # --------------------------------------------------------
    # Tumblr（修正版，可抓）
    # --------------------------------------------------------
    tumblr_list = [
        "lovequotesrus",
        "quotesilove",
        "deepfeelingsclub",
        "sadpoems-quotes",
        "sadbutcute",
    ]

    for blog in tqdm(tumblr_list, desc="Tumblr"):
        data.extend(crawl_tumblr(blog))

    # --------------------------------------------------------
    # Reddit（正常）
    # --------------------------------------------------------
    reddit_list = [
        "offmychest", "lonely", "depression", "anxiety", "confession",
        "casualconversation", "college", "Adulting", "selfimprovement",
        "relationships", "relationship_advice", "breakups",
    ]

    for sub in tqdm(reddit_list, desc="Reddit"):
        data.extend(crawl_reddit(sub))

    # --------------------------------------------------------
    # YouTube（region-free MV）
    # --------------------------------------------------------
    youtube_list = [
        "https://www.youtube.com/watch?v=lY2yjAdbvdQ",
        "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
        "https://www.youtube.com/watch?v=fKopy74weus",
    ]

    for u in tqdm(youtube_list, desc="YouTube"):
        data.extend(crawl_youtube(u))

    save(data)


if __name__ == "__main__":
    main()
