import os
import json
import time
import re
from typing import Dict, Any, Optional

import numpy as np
import requests

EMOTIONS = [
    "joy", "anger", "fear", "sadness",
    "surprise", "disgust", "excitement", "neutral",
]
EMOTION2IDX = {e: i for i, e in enumerate(EMOTIONS)}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s[:120]

# -----------------------------
# Spotify API Client
# -----------------------------
class SpotifyClient:
    def __init__(self, client_id: str, client_secret: str, cache_dir: str = "outputs/spotify_cache"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.cache_dir = cache_dir
        ensure_dir(self.cache_dir)

        self._token: Optional[str] = None
        self._token_expire_at: float = 0.0

    def _token_cache_path(self) -> str:
        return os.path.join(self.cache_dir, "token.json")

    def _load_token_cache(self) -> bool:
        p = self._token_cache_path()
        if not os.path.exists(p):
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            token = obj.get("access_token")
            expire_at = float(obj.get("expire_at", 0.0))
            if token and time.time() < expire_at - 30:
                self._token = token
                self._token_expire_at = expire_at
                return True
        except Exception:
            return False
        return False

    def _save_token_cache(self, access_token: str, expires_in: int):
        obj = {"access_token": access_token, "expire_at": time.time() + int(expires_in)}
        with open(self._token_cache_path(), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def get_access_token(self) -> str:
        if self._token and time.time() < self._token_expire_at - 30:
            return self._token
        if self._load_token_cache():
            return self._token  # type: ignore

        url = "https://accounts.spotify.com/api/token"
        data = {"grant_type": "client_credentials"}
        resp = requests.post(url, data=data, auth=(self.client_id, self.client_secret), timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"Spotify token error: {resp.status_code} {resp.text}")

        obj = resp.json()
        access_token = obj["access_token"]
        expires_in = int(obj.get("expires_in", 3600))
        self._token = access_token
        self._token_expire_at = time.time() + expires_in
        self._save_token_cache(access_token, expires_in)
        return access_token

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.get_access_token()}"}

    def search_track(self, title: str, artist: str) -> Optional[Dict[str, Any]]:
        key = slugify(f"search__{title}__{artist}")
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        q = f'track:"{title}" artist:"{artist}"'
        url = "https://api.spotify.com/v1/search"
        params = {"q": q, "type": "track", "limit": 5}

        resp = requests.get(url, headers=self._headers(), params=params, timeout=20)
        if resp.status_code != 200:
            # cache failure (store error for debugging)
            err_obj = {"_error": True, "status": resp.status_code, "text": resp.text}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(err_obj, f, ensure_ascii=False, indent=2)
            return None

        items = resp.json().get("tracks", {}).get("items", [])
        best = items[0] if items else None

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
        return best

    def get_audio_features(self, track_id: str, debug: bool = False, max_retry: int = 2) -> Optional[Dict[str, Any]]:
        key = slugify(f"features__{track_id}")
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # 若之前 cache 的是錯誤資訊，也直接當作 None
            if isinstance(obj, dict) and obj.get("_error"):
                return None
            return obj

        url = f"https://api.spotify.com/v1/audio-features/{track_id}"

        for attempt in range(max_retry + 1):
            resp = requests.get(url, headers=self._headers(), timeout=20)

            # 429 rate limit: obey Retry-After
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "1"))
                if debug:
                    print(f"[429] rate limited, sleep {retry_after}s (attempt {attempt+1}/{max_retry+1})")
                time.sleep(retry_after)
                continue

            if resp.status_code != 200:
                # cache error detail for debugging (instead of null)
                err_obj = {"_error": True, "status": resp.status_code, "text": resp.text}
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(err_obj, f, ensure_ascii=False, indent=2)

                if debug:
                    print(f"[features fail] track_id={track_id} status={resp.status_code} text={resp.text[:200]}")
                return None

            obj = resp.json()
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            return obj

        return None

# -----------------------------
# Melody -> 8D Emotion Mapping
# -----------------------------
def melody_features_to_emotion_vec(feat: Dict[str, Any]) -> np.ndarray:
    v = float(feat.get("valence", 0.0))
    e = float(feat.get("energy", 0.0))
    d = float(feat.get("danceability", 0.0))
    a = float(feat.get("acousticness", 0.0))
    tempo = float(feat.get("tempo", 0.0))

    tempo_n = (tempo - 60.0) / (180.0 - 60.0)
    tempo_n = max(0.0, min(1.0, tempo_n))

    raw = np.zeros(len(EMOTIONS), dtype=np.float32)
    raw[EMOTION2IDX["joy"]] = 0.60 * v + 0.15 * (1.0 - a) + 0.10 * (1.0 - abs(e - 0.5))
    raw[EMOTION2IDX["excitement"]] = 0.35 * v + 0.30 * e + 0.20 * d + 0.15 * tempo_n
    raw[EMOTION2IDX["sadness"]] = 0.45 * (1.0 - v) + 0.30 * a + 0.25 * (1.0 - e)
    raw[EMOTION2IDX["anger"]] = 0.55 * e + 0.25 * (1.0 - v) + 0.20 * tempo_n
    raw[EMOTION2IDX["fear"]] = 0.40 * (1.0 - v) + 0.35 * (1.0 - e) + 0.25 * a
    raw[EMOTION2IDX["disgust"]] = 0.45 * (1.0 - v) + 0.35 * e + 0.20 * (1.0 - d)
    raw[EMOTION2IDX["surprise"]] = 0.50 * tempo_n + 0.30 * e + 0.20 * (1.0 - d)
    raw[EMOTION2IDX["neutral"]] = (
        0.35 * (1.0 - abs(v - 0.5) * 2) +
        0.35 * (1.0 - abs(e - 0.5) * 2) +
        0.30 * a
    )
    raw = np.clip(raw, 0.0, None)

    s = float(raw.sum())
    if s > 0:
        raw /= s
    else:
        raw[:] = 0.0
        raw[EMOTION2IDX["neutral"]] = 1.0
    return raw

def load_songs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    lyrics_path = "data/processed/lyrics/clean_lyrics.json"
    output_path = "outputs/emotion_vectors/EmotionVec_melody.npy"
    cache_dir = "outputs/spotify_cache"

    ensure_dir(os.path.dirname(output_path))
    ensure_dir(cache_dir)

    client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("Missing Spotify credentials. Set SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET")

    # ✅ 只跑前 N 首（預設不限制；要 10 就設 SPOTIFY_LIMIT=10）
    limit_env = os.getenv("SPOTIFY_LIMIT", "").strip()
    limit = 10

    # ✅ debug 印出 features 失敗原因（要開就設 SPOTIFY_DEBUG=1）
    debug = os.getenv("SPOTIFY_DEBUG", "0").strip() == "1"

    sp = SpotifyClient(client_id, client_secret, cache_dir=cache_dir)
    songs = load_songs(lyrics_path)

    emotion_vectors: Dict[str, np.ndarray] = {}
    n_total = n_ok = n_fail = 0

    for song in songs:
        if limit is not None and n_total >= limit:
            break

        song_id = song.get("song_id")
        title = (song.get("title") or "").strip()
        artist = (song.get("artist") or "").strip()
        if not song_id or not title or not artist:
            continue

        n_total += 1

        track = sp.search_track(title=title, artist=artist)
        if not track or not track.get("id"):
            vec = np.zeros(len(EMOTIONS), dtype=np.float32)
            vec[EMOTION2IDX["neutral"]] = 1.0
            emotion_vectors[song_id] = vec
            n_fail += 1
            if debug:
                print(f"[search miss] {title} - {artist}")
            continue

        feat = sp.get_audio_features(track["id"], debug=debug)
        if not feat:
            vec = np.zeros(len(EMOTIONS), dtype=np.float32)
            vec[EMOTION2IDX["neutral"]] = 1.0
            emotion_vectors[song_id] = vec
            n_fail += 1
            if debug:
                print(f"[features miss] {title} - {artist} track_id={track['id']}")
            continue

        emotion_vectors[song_id] = melody_features_to_emotion_vec(feat)
        n_ok += 1

        # gentle
        if n_total % 20 == 0:
            time.sleep(0.2)

    np.save(output_path, emotion_vectors, allow_pickle=True)
    print(f"[DONE] Saved to {output_path}")
    print(f"Processed: {n_total}, OK: {n_ok}, Fallback(neutral): {n_fail}")
    print("Cache dir:", cache_dir)

if __name__ == "__main__":
    main()
