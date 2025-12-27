"""
Post preprocessing + query expansion (v4.2)
===========================================

This is a tuned version of v4.1 with the goal:
- Keep QE helpful for BM25 recall
- Reduce semantic drift and "generic" expansion noise

v4.2 adds:
1) Candidate-level new-term blocklist (DO_NOT_ADD): prevents adding very generic / unhelpful terms.
2) Automatic "new-term DF gate": if a newly-added QE term appears in too many posts, we auto-block it.
3) POS restriction: by default, expand only noun/adj/adv synsets; allow a small whitelist of emotion/expressive verbs.
4) More informative blocked report: records *all* filtered candidates (not only anti-drift list).
5) Optional skipped-sources report: explains why some source tokens produced no expansions.
"""

import os
import re
import json
import pathlib
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from typing import List, Set, Tuple, Optional, Dict

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Optional dependency: pip install contractions
try:
    import contractions  # type: ignore
except Exception:
    contractions = None


############################################################
# 0. PATH CONFIGURATION
############################################################

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

RAW_POST_DIR = PROJECT_ROOT / "data" / "raw" / "posts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "posts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output filenames
OUTPUT_JSONL_NAME = "posts_clean_expanded.jsonl"
QE_REPORT_NAME = "qe_report_top50.tsv"
BLOCKED_REPORT_NAME = "qe_blocked_report_top50.tsv"
HF_TOKENS_NAME = "qe_high_freq_tokens.txt"
NEWTERM_GATE_NAME = "qe_newterm_gate.txt"
SKIPPED_SOURCES_NAME = "qe_skipped_sources.tsv"

# OUTPUT_JSONL_NAME = "posts_clean_expanded_new4_2.jsonl"
# QE_REPORT_NAME = "qe_report_top50_v4_2.tsv"
# BLOCKED_REPORT_NAME = "qe_blocked_report_top50_v4_2.tsv"
# HF_TOKENS_NAME = "qe_high_freq_tokens_v4_2.txt"
# NEWTERM_GATE_NAME = "qe_newterm_gate_v4_2.txt"
# SKIPPED_SOURCES_NAME = "qe_skipped_sources_v4_2.tsv"


############################################################
# 0.2 QE PARAMETERS (tune here)
############################################################

# High-frequency SOURCE-token gate (skip QE for overly common source tokens)
HF_DF_RATIO = 0.05   # 5% of posts
HF_MIN_DF = 200      # also require at least this many posts (avoid tiny datasets)
HF_MAX_TOKENS = 500  # safety cap

# Allow-list: even if high frequency, still allow as SOURCE token for QE
# (emotion words often worth expanding, even if frequent)
HF_EXEMPT: Set[str] = {
    "sad", "happy", "angry", "mad", "fear", "scared", "anxious", "anxiety",
    "excited", "exciting", "surprised", "surprise", "love", "hate", "lonely",
    "tired", "calm", "peace", "peaceful",
}

# Automatic new-term DF gate (block NEW terms that show up too often as QE additions)
NEWTERM_DF_RATIO = 0.03  # 3% of posts
NEWTERM_MIN_DF = 150
NEWTERM_MAX_TOKENS = 400  # safety cap

# Allowed POS for synset expansion (default: no verbs)
ALLOWED_POS: Set[str] = {"n", "a", "r"}  # noun, adj, adv

# Allow a small set of expressive verbs to expand (keeps some useful emotion actions)
VERB_EXPAND_ALLOWLIST: Set[str] = {
    "cry", "cries", "cried", "crying",
    "laugh", "laughs", "laughed", "laughing",
    "smile", "smiles", "smiled", "smiling",
    "scream", "screams", "screamed", "screaming",
    "dance", "dances", "danced", "dancing",
    "sing", "sings", "sang", "sung", "singing",
    "miss", "misses", "missed", "missing",
    "hurt", "hurts", "hurted", "hurting",
    "heal", "heals", "healed", "healing",
    "panic", "panics", "panicked", "panicking",
    "worry", "worries", "worried", "worrying",
}

# Candidate-level "do not add" list: blocks very generic expansions that often dilute queries.
# This list should be small and focused. You can extend it based on qe_report_top50.
DO_NOT_ADD: Set[str] = {
    # generic / low-information expansions frequently seen in WordNet QE
    "make", "manner", "style", "sort", "idea", "think", "entirely", "wholly", "all",
    # known drift/noise from earlier experiments
    "mass",
}


############################################################
# Anti-drift candidate filters (from v4.1)
############################################################

ANTI_DRIFT_BLOCKLIST: Set[str] = {
    # classic drift from WordNet multi-sense / unrelated domains
    "ace", "climate", "temper", "humor", "approve", "sanction",
    # money / finance drift (allow only if money context exists)
    "earn", "profit", "wage", "salary", "income", "revenue",
    # housing / rent drift (allow only if housing context exists)
    "rent", "lease", "mortgage",
}

MONEY_CONTEXT: Set[str] = {
    "money", "cash", "paid", "pay", "paying", "price", "cost", "bucks", "dollar", "dollars",
    "job", "work", "salary", "wage", "income", "profit", "revenue", "tip", "tips", "bonus",
}

HOUSING_CONTEXT: Set[str] = {
    "rent", "rented", "renting", "lease", "tenant", "landlord", "apartment", "room", "house",
    "home", "mortgage", "move", "moved", "moving", "deposit", "utility", "utilities",
}

ALLOW_CONTEXT_BY_TERM: Dict[str, Set[str]] = {
    "earn": MONEY_CONTEXT,
    "profit": MONEY_CONTEXT,
    "wage": MONEY_CONTEXT,
    "salary": MONEY_CONTEXT,
    "income": MONEY_CONTEXT,
    "revenue": MONEY_CONTEXT,
    "rent": HOUSING_CONTEXT,
    "lease": HOUSING_CONTEXT,
    "mortgage": HOUSING_CONTEXT,
}


############################################################
# 0.1 TOKEN UTILITIES
############################################################

POST_STOPWORDS_MINIMAL: Set[str] = {
    "i", "me", "my", "mine", "we", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs",
    "a", "an", "the",
    "and", "or", "but",
    "to", "of", "in", "on", "at", "for", "with", "as", "by", "from", "into",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those",
    "now", "then", "just", "very", "so",
}


# Align stopwords with lyrics pipeline (NLTK english stopwords)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

LYRICS_STOPWORDS: Set[str] = set(stopwords.words("english"))

# Use lyrics stopwords as the canonical stopword set for QE eligibility + BM25 normalization
STOPWORDS: Set[str] = LYRICS_STOPWORDS


DROP_SINGLE_SYMBOLS: Set[str] = {"@", "#", "&"}

# Tokens that should remain in clean_tokens but should NOT be used for WordNet QE.
DO_NOT_EXPAND: Set[str] = {
    "always", "never", "ever",
    "why", "where", "when", "how",
    "this", "that", "these", "those", "it",
    "now", "then",

    # drift-prone in social context (keep them, but don't expand)
    "mood",
    "okay",
    "really",
    "someone",
    "after",

    # avoids "about -> approximately/roughly" noise
    "about",

    # avoid "one -> ace" noise
    "one",

    # multi-sense triggers that often drift in social posts
    "let", "lets", "letting",
    "realize", "realizes", "realized", "realizing",

    # highly generic sources that often lead to generic expansions
    "lot",
}


# fallback contraction map
FALLBACK_CONTRACTIONS = {
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "i'm": "i am",
    "im": "i am",
    "you're": "you are",
    "youre": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would",
    "y'all": "you all",
    "yall": "you all",
}
_FALLBACK_CONTRACTIONS_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(FALLBACK_CONTRACTIONS.keys(), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)


# Porter stemming (to align with lyrics pipeline)
STEMMER = PorterStemmer()

def stem_token(tok: str) -> str:
    """
    Stem ONLY plain alphabetic tokens (a-z). Keep emojis/punct/digits as-is.
    This is applied after QE so WordNet expansion still operates on normal words.
    """
    if tok and re.fullmatch(r"[a-z]+", tok):
        return STEMMER.stem(tok)
    return tok

def stem_tokens(tokens: List[str]) -> List[str]:
    return [stem_token(t) for t in tokens]



def normalize_for_bm25(tokens: List[str]) -> List[str]:
    """
    Produce BM25-consistent tokens aligned with lyrics pipeline:
    - keep only alphabetic tokens [a-z]+
    - drop NLTK english stopwords (same as lyrics)
    - drop very short tokens (len<=1) before and after stemming
    - apply Porter stemming
    """
    out: List[str] = []
    for tok in tokens:
        if not tok:
            continue
        if not re.fullmatch(r"[a-z]+", tok):
            continue
        if len(tok) <= 1:
            continue
        if tok in STOPWORDS:
            continue
        stem = STEMMER.stem(tok)
        if not stem or len(stem) <= 1:
            continue
        out.append(stem)
    return out




def unique_preserve_order(tokens: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def is_emoji(tok: str) -> bool:
    return any(unicodedata.category(ch) == "So" for ch in tok)


def is_punct_only(tok: str) -> bool:
    tok = tok.strip()
    if not tok:
        return True
    return all(unicodedata.category(ch).startswith("P") for ch in tok)


def is_valid_content_token(tok: str) -> bool:
    """Eligibility for WordNet QE as a SOURCE token."""
    if not tok or tok in DROP_SINGLE_SYMBOLS:
        return False
    if tok in STOPWORDS:
        return False
    if tok in DO_NOT_EXPAND:
        return False
    if is_emoji(tok):
        return False
    if is_punct_only(tok):
        return False
    if tok.isdigit():
        return False
    if len(tok) < 3:
        return False
    if not re.fullmatch(r"[a-z]+", tok):
        return False
    return True


def expand_contractions(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    if contractions is not None:
        text = contractions.fix(text)
        text = re.sub(r"\bcan not\b", "cannot", text, flags=re.IGNORECASE)
        return text

    def _repl(m: re.Match) -> str:
        key = m.group(1).lower()
        return FALLBACK_CONTRACTIONS.get(key, m.group(0))

    return _FALLBACK_CONTRACTIONS_RE.sub(_repl, text)


############################################################
# 1. POST PREPROCESSOR
############################################################

class PostPreprocessor:
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    USER_PATTERN = re.compile(r"@\w+")
    REPEAT_PATTERN = re.compile(r"(.)\1{2,}")  # 3+ -> 2
    CAMEL_SPLIT_RE = re.compile(r"[A-Z][a-z]+|[a-z]+|\d+")

    # slang expansions
    SLANG: Dict[str, str] = {
        "rn": "right now",
        "dm": "direct message",
        "dms": "direct messages",
        "idk": "i don't know",
        "imo": "in my opinion",
        "imho": "in my humble opinion",
        "omg": "oh my god",
        "lmao": "laughing",
        "lol": "laughing",
        "u": "you",
        "ur": "your",
        "thx": "thanks",
        "pls": "please",
        "plz": "please",
        "tbh": "to be honest",
        "smh": "shaking my head",
        "btw": "by the way",
        "ngl": "not gonna lie",
        "fr": "for real",
        "ikr": "i know right",
        "brb": "be right back",
        "lmk": "let me know",
        "omw": "on my way",
        "hmu": "hit me up",
        "wyd": "what are you doing",
        "idc": "i do not care",
        "ily": "i love you",
        "jk": "just kidding",
        "nvm": "never mind",
        "ttyl": "talk to you later",
    }

    SLANG_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\bw/\b", flags=re.IGNORECASE), "with"),
        (re.compile(r"\bw/o\b", flags=re.IGNORECASE), "without"),
        (re.compile(r"\bya\b", flags=re.IGNORECASE), "you"),
    ]

    COMMON_SUFFIXES = [
        "vibes", "mood", "life", "time", "day", "night", "love", "goals",
        "energy", "feels", "feeling", "moments", "moment", "happy", "sad",
        "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "weekend",
    ]

    CUSTOM_HASHTAG_MAP = {
        "notokay": ["not", "okay"],
        "mentalhealth": ["mental", "health"],
        "selfcare": ["self", "care"],
        "sundayfunday": ["sunday", "fun", "day"],
        "weekendvibes": ["weekend", "vibes"],
        "smalljoys": ["small", "joys"],
        # fixes
        "plotwist": ["plot", "twist"],
        "instamood": ["insta", "mood"],
    }

    # allow some 3-letter words; otherwise 3-letter fragments are often garbage from DP segmentation
    SHORT_OK = {"not", "fun", "day", "sad", "bad", "big", "new", "old", "wow", "yay"}

    def __init__(self):
        self._lemma_dict = self._build_wordnet_lemma_dict()

    def _build_wordnet_lemma_dict(self) -> Set[str]:
        lemma_set: Set[str] = set()
        for name in wn.all_lemma_names():
            w = name.lower()
            if "_" in w:
                continue
            if not w.isalpha():
                continue
            if len(w) < 2:
                continue
            lemma_set.add(w)

        lemma_set.update({"not", "okay", "ok"})
        lemma_set.update(self.COMMON_SUFFIXES)
        lemma_set.update({"funday", "selfcare", "small", "joys", "cozy", "cosy", "insta", "plot", "twist"})
        return lemma_set

    def _split_hashtag_camel(self, tag: str) -> List[str]:
        parts = self.CAMEL_SPLIT_RE.findall(tag)
        parts = [p.lower() for p in parts if p]
        return parts if parts else [tag.lower()]

    @lru_cache(maxsize=20000)
    def _split_hashtag_lowercase(self, tag: str) -> List[str]:
        tag = tag.lower()

        if tag in self.CUSTOM_HASHTAG_MAP:
            return self.CUSTOM_HASHTAG_MAP[tag]

        if not tag.isalnum():
            return [tag]

        if tag.startswith("not") and len(tag) > 5:
            rest = tag[3:]
            if rest in self._lemma_dict:
                return ["not", rest]

        for suf in sorted(self.COMMON_SUFFIXES, key=len, reverse=True):
            if tag.endswith(suf) and len(tag) > len(suf) + 2:
                pre = tag[: -len(suf)]
                left = self._dp_word_break(pre)
                if left:
                    return left + [suf]
                if pre in self._lemma_dict:
                    return [pre, suf]

        res = self._dp_word_break(tag)
        return res if res else [tag]

    def _dp_word_break(self, s: str) -> Optional[List[str]]:
        n = len(s)
        dp: List[Optional[Tuple[int, int, List[str]]]] = [None] * (n + 1)
        dp[0] = (0, 0, [])
        max_word_len = 20

        for i in range(n):
            if dp[i] is None:
                continue
            base_cnt, base_score, base_tokens = dp[i]

            for j in range(i + 2, min(n, i + max_word_len) + 1):
                w = s[i:j]
                if w not in self._lemma_dict:
                    continue

                cnt = base_cnt + 1
                score = base_score - (len(w) * len(w))
                tokens = base_tokens + [w]
                cand = (cnt, score, tokens)
                cur = dp[j]
                if cur is None or cand < cur:
                    dp[j] = cand

        best = dp[n]
        if best is None:
            return None

        tokens = best[2]
        if len(tokens) >= 6:
            return None

        # v4+ safeguard: avoid garbage 3-letter fragments
        if any((len(w) == 3 and w not in self.SHORT_OK) for w in tokens):
            return None

        return tokens

    def split_hashtag(self, tag: str) -> List[str]:
        if re.search(r"[A-Z]", tag):
            return self._split_hashtag_camel(tag)
        return self._split_hashtag_lowercase(tag)

    def preprocess_text(self, text: str) -> List[str]:
        # 0) remove emoji variation selectors (VS16/VS15)
        text = text.replace("\uFE0F", "").replace("\uFE0E", "")

        # 1) remove URL
        text = self.URL_PATTERN.sub(" ", text)

        # 2) remove @username
        text = self.USER_PATTERN.sub(" ", text)

        # 2.5) apply small slang patterns before contractions/tokenization
        for pat, repl in self.SLANG_PATTERNS:
            text = pat.sub(repl, text)

        # 3) expand contractions BEFORE tokenization
        text = expand_contractions(text)

        # 4) expand hashtags
        def _hashtag_repl(m: re.Match) -> str:
            raw_tag = m.group(1)
            raw_lower = raw_tag.lower()
            split = self.split_hashtag(raw_tag)
            expanded = unique_preserve_order([raw_lower] + split)
            return " " + " ".join(expanded) + " "

        text = re.sub(r"#(\w+)", _hashtag_repl, text)

        # 5) separate emoji-ish high codepoints
        text = "".join(f" {ch} " if ord(ch) > 10000 else ch for ch in text)

        # 6) lowercase
        text = text.lower()

        # tokenizer that keeps apostrophes inside words
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", text)

        processed: List[str] = []
        for t in tokens:
            t = self.REPEAT_PATTERN.sub(r"\1\1", t)

            if t in self.SLANG:
                t = self.SLANG[t]

            for sub in t.split():
                sub = sub.strip()
                if not sub:
                    continue
                if sub in DROP_SINGLE_SYMBOLS:
                    continue
                if is_punct_only(sub):
                    continue
                processed.append(sub)

        return processed


############################################################
# 2. QE: Reduced-drift WordNet Expander + richer diagnostics
############################################################

class ReducedDriftQueryExpander:
    """
    WordNet QE with lightweight context sense selection (Lesk-style), plus:
    - High-frequency SOURCE-token gate
    - Candidate-level blocklists (DO_NOT_ADD, NEWTERM DF gate, anti-drift domain gates)
    - POS restriction (skip verb synsets unless source token is in VERB_EXPAND_ALLOWLIST)
    - Richer blocked/skip diagnostics
    """

    def __init__(
        self,
        max_synsets_per_token: int = 15,
        use_only_top_synsets: int = 8,
        min_signature_overlap: int = 1,
        max_expansion_per_token: int = 3,
        max_total_new_terms: int = 30,
        min_lemma_count: int = 2,
        fallback_min_lemma_count: int = 10,
        allow_adj_similar_tos: bool = True,
        high_freq_tokens: Optional[Set[str]] = None,
        newterm_blocklist: Optional[Set[str]] = None,
        auto_newterm_gate: Optional[Set[str]] = None,
    ):
        self.max_synsets_per_token = max_synsets_per_token
        self.use_only_top_synsets = use_only_top_synsets
        self.min_signature_overlap = min_signature_overlap
        self.max_expansion_per_token = max_expansion_per_token
        self.max_total_new_terms = max_total_new_terms
        self.min_lemma_count = min_lemma_count
        self.fallback_min_lemma_count = fallback_min_lemma_count
        self.allow_adj_similar_tos = allow_adj_similar_tos

        self.high_freq_tokens: Set[str] = set(high_freq_tokens or set())
        self.newterm_blocklist: Set[str] = set(newterm_blocklist or set())
        self.auto_newterm_gate: Set[str] = set(auto_newterm_gate or set())

        self.reset_inspection_stats()

    def set_high_freq_tokens(self, s: Set[str]) -> None:
        self.high_freq_tokens = set(s)

    def set_auto_newterm_gate(self, s: Set[str]) -> None:
        self.auto_newterm_gate = set(s)

    def set_newterm_blocklist(self, s: Set[str]) -> None:
        self.newterm_blocklist = set(s)

    # ----------------------------
    # Diagnostics / inspection stats
    # ----------------------------
    def reset_inspection_stats(self) -> None:
        # blocked candidate term stats
        self.blocked_term_counts: Counter = Counter()
        self.blocked_sources: Dict[str, Counter] = defaultdict(Counter)
        self.blocked_reasons: Dict[str, Counter] = defaultdict(Counter)

        # skipped source token stats (why a source token produced no expansions)
        self.skipped_sources: Dict[str, Counter] = defaultdict(Counter)

    def _record_blocked(self, term: str, source_token: str, reason: str) -> None:
        self.blocked_term_counts[term] += 1
        if source_token:
            self.blocked_sources[term][source_token] += 1
        self.blocked_reasons[term][reason] += 1

    def _record_skipped_source(self, source_token: str, reason: str) -> None:
        if source_token:
            self.skipped_sources[source_token][reason] += 1

    def _is_candidate_blocked(self, cand: str, context_set: Set[str]) -> Tuple[bool, str]:
        # candidate-level gates
        if cand in self.auto_newterm_gate:
            return True, "newterm_df_gate"
        if cand in self.newterm_blocklist:
            return True, "newterm_blocklist"
        if cand in DO_NOT_EXPAND:
            return True, "do_not_expand_term"
        if cand in STOPWORDS:
            return True, "stopword"

        # anti-drift domain gates
        if cand in ANTI_DRIFT_BLOCKLIST:
            allow_ctx = ALLOW_CONTEXT_BY_TERM.get(cand)
            if allow_ctx:
                if context_set & allow_ctx:
                    return False, ""
                return True, "anti_drift_missing_context"
            return True, "anti_drift_always"

        return False, ""

    @staticmethod
    def _simple_word_tokens(text: str) -> List[str]:
        return re.findall(r"[a-z]+", text.lower())

    @lru_cache(maxsize=200000)
    def _synset_signature(self, syn_name: str) -> Set[str]:
        syn = wn.synset(syn_name)
        sig_text = syn.definition() + " " + " ".join(syn.examples())
        hypers = syn.hypernyms()[:2]
        for h in hypers:
            sig_text += " " + h.definition()
        toks = [t for t in self._simple_word_tokens(sig_text) if is_valid_content_token(t)]
        return set(toks)

    def _pos_hint(self, tokens: List[str], idx: int) -> Optional[str]:
        prev = tokens[idx - 1] if idx > 0 else None
        if not prev:
            return None
        if prev == "to":
            return "v"
        if prev in {"very", "so", "too", "really"}:
            return "a"
        if prev in {"more", "most", "less"}:
            return "a"
        if prev in {"a", "an", "the", "my", "your", "his", "her", "their"}:
            return "n"
        return None

    def _choose_synset(self, token: str, context_set: Set[str], synsets: List["wn.synset"], pos_hint: Optional[str]) -> Optional["wn.synset"]:
        scored: List[Tuple[int, int, "wn.synset"]] = []
        cand = synsets[: self.use_only_top_synsets]
        for rank, syn in enumerate(cand):
            if pos_hint is not None and syn.pos() != pos_hint:
                continue
            sig = self._synset_signature(syn.name())
            overlap = len(sig & context_set)
            scored.append((overlap, -rank, syn))

        if not scored:
            return None

        scored.sort(reverse=True)
        best_overlap, _, best_syn = scored[0]

        if best_overlap < self.min_signature_overlap:
            # no confident overlap -> allow fallback later
            return None

        return best_syn

    def _lemma_candidates_from_synset(
        self,
        chosen: "wn.synset",
        source_token: str,
        context_set: Set[str],
        base_set: Set[str],
        lemma_count_threshold: int,
    ) -> List[str]:
        """
        Collect lemma candidates, and record why candidates are rejected.
        """
        out: List[Tuple[int, str]] = []
        for lemma in chosen.lemmas():
            name = lemma.name().lower()

            # format filters
            if "_" in name:
                self._record_blocked(name, source_token, "underscore_multiword")
                continue
            if not name.isalpha():
                self._record_blocked(name, source_token, "non_alpha")
                continue
            if len(name) < 3:
                self._record_blocked(name, source_token, "too_short")
                continue
            if name == source_token:
                self._record_blocked(name, source_token, "same_as_source")
                continue
            if name in base_set:
                # already present -> not "blocked", just not needed
                continue

            # usage frequency filter
            cnt = lemma.count()
            if cnt < lemma_count_threshold:
                self._record_blocked(name, source_token, f"lemma_count<{lemma_count_threshold}")
                continue

            # candidate-level block checks
            blocked, reason = self._is_candidate_blocked(name, context_set)
            if blocked:
                self._record_blocked(name, source_token, reason)
                continue

            out.append((cnt, name))

        out.sort(reverse=True)
        return [w for _, w in out]

    def expand_wordnet_unique_with_trace(self, tokens: List[str]) -> Tuple[List[str], Dict[str, str]]:
        # Keep the ORIGINAL token sequence (including duplicates) for output.
        # We still compute a unique view for QE decisions to avoid repeated work.
        base_tokens = list(tokens)
        base_unique = unique_preserve_order(base_tokens)
        base_set = set(base_tokens)
        new_terms: List[str] = []
        trace: Dict[str, str] = {}

        # build context set once (content tokens only)
        content_tokens = [t for t in base_unique if is_valid_content_token(t)]
        content_set_all = set(content_tokens)

        for idx, t in enumerate(base_unique):
            if not is_valid_content_token(t):
                continue

            if t in self.high_freq_tokens and t not in HF_EXEMPT:
                self._record_skipped_source(t, "source_high_freq_gate")
                continue

            synsets = wn.synsets(t)
            if not synsets:
                self._record_skipped_source(t, "no_synsets")
                continue
            if len(synsets) > self.max_synsets_per_token:
                self._record_skipped_source(t, "too_many_synsets")
                continue

            context_set = set(content_set_all)
            context_set.discard(t)

            pos_hint = self._pos_hint(base_unique, idx)

            chosen = self._choose_synset(t, context_set, synsets, pos_hint)

            # POS restriction:
            # - if we got a chosen synset and its POS is verb, only allow if source token is whitelisted
            # - if no chosen synset (low overlap), we will do fallback; still apply the same POS check on fallback synsets
            def _pos_allowed(syn: "wn.synset") -> bool:
                p = syn.pos()
                if p in ALLOWED_POS:
                    return True
                if p == "v" and t in VERB_EXPAND_ALLOWLIST:
                    return True
                return False

            if chosen is not None and not _pos_allowed(chosen):
                self._record_skipped_source(t, f"pos_blocked:{chosen.pos()}")
                continue

            # primary candidates
            candidates: List[str] = []
            if chosen is not None:
                candidates = self._lemma_candidates_from_synset(
                    chosen=chosen,
                    source_token=t,
                    context_set=context_set,
                    base_set=base_set.union(set(new_terms)),
                    lemma_count_threshold=self.min_lemma_count,
                )

                # (optional) similar_tos for adjectives
                if self.allow_adj_similar_tos and chosen.pos() == "a":
                    for syn2 in chosen.similar_tos():
                        # still only allow allowed POS (similar_tos returns adj synsets)
                        candidates.extend(self._lemma_candidates_from_synset(
                            chosen=syn2,
                            source_token=t,
                            context_set=context_set,
                            base_set=base_set.union(set(new_terms)),
                            lemma_count_threshold=self.min_lemma_count,
                        ))

            # fallback: if no chosen synset or we got no candidates, try top synsets with stricter lemma_count
            if (chosen is None or not candidates) and synsets:
                fallback_synsets = synsets[: min(3, len(synsets))]
                for fs in fallback_synsets:
                    if not _pos_allowed(fs):
                        continue
                    candidates.extend(self._lemma_candidates_from_synset(
                        chosen=fs,
                        source_token=t,
                        context_set=context_set,
                        base_set=base_set.union(set(new_terms)),
                        lemma_count_threshold=self.fallback_min_lemma_count,
                    ))

            # dedup candidates while preserving order
            cand_unique: List[str] = []
            seen_c = set()
            for w in candidates:
                if w not in seen_c:
                    seen_c.add(w)
                    cand_unique.append(w)

            added = 0
            for w in cand_unique:
                # final safety: do-not-add checks (in case blocklists updated externally)
                blocked, reason = self._is_candidate_blocked(w, context_set)
                if blocked:
                    self._record_blocked(w, t, reason)
                    continue

                if w not in base_set and w not in new_terms:
                    new_terms.append(w)
                    trace.setdefault(w, t)
                    added += 1
                if added >= self.max_expansion_per_token:
                    break
                if len(new_terms) >= self.max_total_new_terms:
                    break

            if added == 0:
                self._record_skipped_source(t, "no_valid_candidates")

            if len(new_terms) >= self.max_total_new_terms:
                break

        return base_tokens + new_terms, trace

    def expand_wordnet_unique(self, tokens: List[str]) -> List[str]:
        expanded, _ = self.expand_wordnet_unique_with_trace(tokens)
        return expanded


############################################################
# 3. REPORT HELPERS
############################################################

def write_high_freq_tokens(path: pathlib.Path, hf: Set[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in sorted(hf):
            f.write(t + "\n")


def write_newterm_gate(path: pathlib.Path, gate: Set[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in sorted(gate):
            f.write(t + "\n")


def write_qe_report_tsv(path: pathlib.Path, added_term_counts: Counter, term_sources: Dict[str, Counter], top_k: int = 50) -> None:
    """
    TSV columns:
      new_term, count, top_sources (source:count,...)
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("new_term\tcount\ttop_sources\n")
        for term, cnt in added_term_counts.most_common(top_k):
            s_counter = term_sources.get(term, Counter())
            top_sources = ", ".join([f"{s}:{c}" for s, c in s_counter.most_common(5)])
            f.write(f"{term}\t{cnt}\t{top_sources}\n")


def write_blocked_report_tsv(
    path: pathlib.Path,
    blocked_term_counts: Counter,
    blocked_sources: Dict[str, Counter],
    blocked_reasons: Dict[str, Counter],
    top_k: int = 50
) -> None:
    """
    TSV columns:
      blocked_term, count, top_reasons (reason:count,...), top_sources (source:count,...)
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("blocked_term\tcount\ttop_reasons\ttop_sources\n")
        for term, cnt in blocked_term_counts.most_common(top_k):
            r_counter = blocked_reasons.get(term, Counter())
            s_counter = blocked_sources.get(term, Counter())
            top_reasons = ", ".join([f"{r}:{c}" for r, c in r_counter.most_common(3)])
            top_sources = ", ".join([f"{s}:{c}" for s, c in s_counter.most_common(5)])
            f.write(f"{term}\t{cnt}\t{top_reasons}\t{top_sources}\n")


def write_skipped_sources_tsv(path: pathlib.Path, skipped_sources: Dict[str, Counter], top_k: int = 200) -> None:
    """
    TSV columns:
      source_token, total_skips, top_reasons
    """
    totals = []
    for s, c in skipped_sources.items():
        totals.append((sum(c.values()), s))
    totals.sort(reverse=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source_token\ttotal_skips\ttop_reasons\n")
        for total, s in totals[:top_k]:
            c = skipped_sources[s]
            top_reasons = ", ".join([f"{r}:{n}" for r, n in c.most_common(4)])
            f.write(f"{s}\t{total}\t{top_reasons}\n")


############################################################
# 4. CORPUS-LEVEL GATES
############################################################

def compute_high_freq_tokens(clean_tokens_list: List[List[str]]) -> Set[str]:
    """
    DF over eligible content tokens. Returns a set of SOURCE tokens to skip QE for.
    HF_EXEMPT will be removed from the gate.
    """
    n_posts = len(clean_tokens_list)
    df = Counter()

    for toks in clean_tokens_list:
        uniq = set(t for t in toks if is_valid_content_token(t))
        df.update(uniq)

    hf = {t for t, c in df.items() if c >= HF_MIN_DF and (c / max(1, n_posts)) >= HF_DF_RATIO}
    hf = set(t for t in hf if t not in HF_EXEMPT)

    if len(hf) > HF_MAX_TOKENS:
        hf = set([t for t, _ in df.most_common(HF_MAX_TOKENS) if t in hf])

    return hf


def compute_newterm_gate(added_term_counts: Counter, n_posts: int) -> Set[str]:
    """
    Block NEW terms that are added too frequently (generic/unhelpful expansions).
    """
    gate = {t for t, c in added_term_counts.items() if c >= NEWTERM_MIN_DF and (c / max(1, n_posts)) >= NEWTERM_DF_RATIO}
    if len(gate) > NEWTERM_MAX_TOKENS:
        gate = set([t for t, _ in added_term_counts.most_common(NEWTERM_MAX_TOKENS) if t in gate])
    return gate


############################################################
# 5. MAIN PIPELINE
############################################################

def process_posts():
    pre = PostPreprocessor()

    all_files = list(RAW_POST_DIR.glob("generated_social_posts_10k.json"))
    print(f"找到 {len(all_files)} 個貼文檔案")
    if not all_files:
        print(f"[WARN] 在 {RAW_POST_DIR} 找不到 input 檔案：generated_social_posts_10k.json")
        return

    # Load all data (10k is fine)
    data_all: List[Dict] = []
    for fp in all_files:
        print(f"讀取：{fp.name}")
        data_all.extend(json.load(open(fp, "r", encoding="utf-8")))

    n_posts = len(data_all)
    print(f"總貼文數：{n_posts}")

    # Pass 1: preprocess to clean_tokens and compute high-freq SOURCE tokens
    clean_tokens_list: List[List[str]] = []
    for item in data_all:
        raw_text = item.get("text", "")
        clean_tokens_list.append(pre.preprocess_text(raw_text))

    hf = compute_high_freq_tokens(clean_tokens_list)
    print(f"High-frequency source-gate tokens: {len(hf)} (HF_DF_RATIO={HF_DF_RATIO}, HF_MIN_DF={HF_MIN_DF})")

    hf_path = OUTPUT_DIR / HF_TOKENS_NAME
    write_high_freq_tokens(hf_path, hf)
    print(f"已輸出 high-freq token 清單：{hf_path}")

    # Pass 2: provisional QE (no auto new-term gate yet) to learn new-term DF gate
    qe_prov = ReducedDriftQueryExpander(
        max_synsets_per_token=15,
        use_only_top_synsets=8,
        min_signature_overlap=1,
        max_expansion_per_token=3,
        max_total_new_terms=30,
        min_lemma_count=2,
        fallback_min_lemma_count=10,
        allow_adj_similar_tos=True,
        high_freq_tokens=hf,
        newterm_blocklist=DO_NOT_ADD,
        auto_newterm_gate=set(),
    )

    prov_added_counts = Counter()
    prov_sources: Dict[str, Counter] = defaultdict(Counter)

    for toks in clean_tokens_list:
        expanded, trace = qe_prov.expand_wordnet_unique_with_trace(toks)
        new_set = set(expanded) - set(toks)
        for w in new_set:
            prov_added_counts[w] += 1
            src = trace.get(w)
            if src:
                prov_sources[w][src] += 1

    auto_gate = compute_newterm_gate(prov_added_counts, n_posts)
    print(f"Auto new-term DF gate size: {len(auto_gate)} (NEWTERM_DF_RATIO={NEWTERM_DF_RATIO}, NEWTERM_MIN_DF={NEWTERM_MIN_DF})")

    gate_path = OUTPUT_DIR / NEWTERM_GATE_NAME
    write_newterm_gate(gate_path, auto_gate)
    print(f"已輸出 new-term gate 清單：{gate_path}")

    # Pass 3: final QE with both HF source gate + auto new-term gate
    qe = ReducedDriftQueryExpander(
        max_synsets_per_token=15,
        use_only_top_synsets=8,
        min_signature_overlap=1,
        max_expansion_per_token=3,
        max_total_new_terms=30,
        min_lemma_count=2,
        fallback_min_lemma_count=10,
        allow_adj_similar_tos=True,
        high_freq_tokens=hf,
        newterm_blocklist=DO_NOT_ADD,
        auto_newterm_gate=auto_gate,
    )

    # report counters for FINAL output
    added_term_counts = Counter()
    term_sources: Dict[str, Counter] = defaultdict(Counter)

    output_path = OUTPUT_DIR / OUTPUT_JSONL_NAME
    with open(output_path, "w", encoding="utf-8") as fout:
        for item, clean_tokens in zip(data_all, clean_tokens_list):
            raw_text = item.get("text", "")
            expanded_tokens, trace = qe.expand_wordnet_unique_with_trace(clean_tokens)

            new_set = set(expanded_tokens) - set(clean_tokens)
            for w in new_set:
                added_term_counts[w] += 1
                src = trace.get(w)
                if src:
                    term_sources[w][src] += 1

            obj = {
                "raw_text": raw_text,

                # Raw tokens (for debugging / emotion-topic modeling)
                "clean_tokens_raw": clean_tokens,
                "expanded_tokens_raw": expanded_tokens,

                # BM25-aligned tokens (same stopwords + stemming rules as lyrics)
                "clean_tokens": normalize_for_bm25(clean_tokens),
                "expanded_tokens": normalize_for_bm25(expanded_tokens),

                "emotion": item.get("emotion"),
                "strength": item.get("strength"),
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\n已輸出：{output_path}")

    # Reports
    # NOTE: Reports are in raw (unstemmed) token space.
    # Output JSONL tokens are stemmed for BM25 consistency with lyrics.
    report_path = OUTPUT_DIR / QE_REPORT_NAME
    write_qe_report_tsv(report_path, added_term_counts, term_sources, top_k=50)
    print(f"已輸出 QE 報表：{report_path}")

    blocked_path = OUTPUT_DIR / BLOCKED_REPORT_NAME
    write_blocked_report_tsv(
        blocked_path,
        qe.blocked_term_counts,
        qe.blocked_sources,
        qe.blocked_reasons,
        top_k=50,
    )
    print(f"已輸出 QE blocked 報表：{blocked_path}")

    skipped_path = OUTPUT_DIR / SKIPPED_SOURCES_NAME
    write_skipped_sources_tsv(skipped_path, qe.skipped_sources, top_k=200)
    print(f"已輸出 QE skipped-sources 報表：{skipped_path}")


if __name__ == "__main__":
    process_posts()
