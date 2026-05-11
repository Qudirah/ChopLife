"""
profile_extractor.py
=======================
ChopLife — User Profile Builder

Output:
  data/user_profiles.json       full profiles
  data/profile_index.json       lightweight lookup
  data/sampling_report.json     bucket distribution for solution paper

Usage:
  python profile_extractor.py --data_dir "./Yelp JSON/yelp_dataset" --output_dir ./data
  python profile_extractor.py --data_dir "./Yelp JSON/yelp_dataset" --output_dir ./data --no_llm --limit 200000
  python profile_extractor.py --data_dir "./Yelp JSON/yelp_dataset" --output_dir ./data --user_id <id>

"""

import json
import os
import argparse
import math
import random
import statistics
from collections import defaultdict, Counter
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from groq import Groq

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

GROQ_MODEL           = "llama-3.3-70b-versatile"
DRIFT_WINDOW_MONTHS  = 18
DRIFT_THRESHOLD      = 0.6
VOCABULARY_TOP_N     = 20
MAX_REVIEWS_FOR_LLM  = 15
MAX_TIPS_FOR_LLM     = 5
MAX_CHARS_PER_TEXT   = 300
RANDOM_SEED          = 42

# ── Category filter ───────────────────────────────────────────────────────────
# Primary anchors — most food businesses carry at least one of these
PRIMARY_FOOD_CATS = {"restaurants", "food"}

# Secondary — food venues that sometimes appear without the primary anchors
SECONDARY_FOOD_CATS = {
    "bakeries", "cafes", "coffee & tea", "food trucks", "bubble tea",
    "juice bars & smoothies", "desserts", "ice cream & frozen yogurt",
    "delis", "diners", "buffets", "street vendors", "grocery",
    "specialty food", "beer bar", "wine bars",
}

# ── Stratified sampling targets ───────────────────────────────────────────────
# (tendency, tone) -> target count   tone is: casual | formal only
#
# Pidgin removed as a Yelp-derived tone bucket. Yelp is US-dominated —
# pidgin detection would produce near-zero or false-positive results.
# Nigerian register lives at the agent output layer via the Nairaland
# few-shot corpus, not derived from Yelp user history.
WARM_BUCKETS = {
    ("generous", "casual"): 150,
    ("generous", "formal"):  60,
    ("balanced", "casual"): 150,
    ("balanced", "formal"):  60,
    ("harsh",    "casual"): 100,
    ("harsh",    "formal"):  40,
}
# Cold-start users: deliberately sampled to demonstrate fallback handling
# These showcase the cold-start criterion (25 pts Task B)
COLD_BUCKETS = {
    "thin":      {"min": 5,  "max": 14, "target": 80},   # thin history
    "very_thin": {"min": 1,  "max": 4,  "target": 50},   # near cold-start
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_jsonl(filepath: str, limit: int = None) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def is_food_business(business: dict) -> bool:
    """
    Return True if this business is food-related.
    Checks primary anchors first (Restaurants, Food), then secondary.
    Case-insensitive substring matching handles compound categories.
    """
    cats_raw = business.get("categories") or ""
    if not cats_raw:
        return False
    cats_lower = cats_raw.lower()
    # primary check first — fast path
    for cat in PRIMARY_FOOD_CATS:
        if cat in cats_lower:
            return True
    # secondary check
    for cat in SECONDARY_FOOD_CATS:
        if cat in cats_lower:
            return True
    return False


def months_between(date_str: str, reference: datetime) -> float:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return (reference - d).days / 30.44
    except Exception:
        return 999


def recency_weight(months_ago: float) -> float:
    """Exponential decay, half-life ~18 months."""
    return math.exp(-0.0385 * months_ago)


# ─── RATING PROFILE ───────────────────────────────────────────────────────────

def extract_rating_profile(reviews: list, now: datetime) -> dict:
    """
    Weighted mean is the primary metric — bounded 1–5 scale means
    outliers are genuine behaviour signals, not noise to discard.
    Median stored as a secondary cross-check only, not used in prompts.
    """
    if not reviews:
        return {}

    stars = [r["stars"] for r in reviews]
    dist  = Counter(int(s) for s in stars)

    raw_mean = statistics.mean(stars)
    median   = float(statistics.median(stars))
    std_dev  = statistics.stdev(stars) if len(stars) > 1 else 0.0

    weighted_sum = weighted_count = 0.0
    for r in reviews:
        age = months_between(r.get("date", ""), now)
        w   = recency_weight(age)
        weighted_sum   += r["stars"] * w
        weighted_count += w
    weighted_mean = weighted_sum / weighted_count if weighted_count else raw_mean

    if raw_mean >= 4.2:
        tendency = "generous"
    elif raw_mean <= 2.8:
        tendency = "harsh"
    else:
        tendency = "balanced"

    return {
        "raw_mean":      round(raw_mean, 2),
        "weighted_mean": round(weighted_mean, 2),
        "median":        median,           # stored but not injected into prompts
        "std_dev":       round(std_dev, 2),
        "tendency":      tendency,
        "distribution":  {str(k): dist.get(k, 0) for k in range(1, 6)},
        "total_reviews": len(reviews),
    }


# ─── DRIFT DETECTION ──────────────────────────────────────────────────────────

def detect_drift(reviews: list, now: datetime) -> dict:
    if len(reviews) < 15:
        return {
            "detected":     False,
            "reason":       "insufficient_history",
            "recent_count": 0,
        }

    sorted_r   = sorted(reviews, key=lambda r: r.get("date", "1900-01-01"))
    mid        = len(sorted_r) // 2
    older_mean = statistics.mean(r["stars"] for r in sorted_r[:mid])
    newer_mean = statistics.mean(r["stars"] for r in sorted_r[mid:])
    delta      = abs(newer_mean - older_mean)

    recent = [
        r for r in reviews
        if months_between(r.get("date", ""), now) <= DRIFT_WINDOW_MONTHS
    ]
    recent_mean = statistics.mean(r["stars"] for r in recent) if recent else newer_mean

    return {
        "detected":     delta >= DRIFT_THRESHOLD,
        "older_mean":   round(older_mean, 2),
        "newer_mean":   round(newer_mean, 2),
        "recent_mean":  round(recent_mean, 2),
        "delta":        round(delta, 2),
        "direction":    "increasing" if newer_mean > older_mean else "decreasing",
        "recent_count": len(recent),
        "reason":       "rating_shift" if delta >= DRIFT_THRESHOLD else "stable",
    }


# ─── WRITING STYLE ────────────────────────────────────────────────────────────

def classify_tone(texts: list) -> dict:
    """
    Heuristic tone classification from combined review + tip texts.
    Tips are included because they represent the user's most natural,
    unprompted register — shorter, more casual, vocabulary-rich.
    """
    FORMAL_MARKERS = [
        "however", "nevertheless", "furthermore", "exceptional", "ambience",
        "sophisticated", "impeccable", "whilst", "establishment",
        "commendable", "satisfactory", "delightful", "exquisite",
    ]
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "was", "it", "this", "that", "i",
        "my", "we", "they", "he", "she", "are", "were", "be", "been",
        "have", "has", "had", "not", "so", "very", "just", "food",
        "place", "good", "great", "nice", "get", "got", "went", "go",
        "would", "will", "also", "like", "one", "time", "back", "really",
        "more", "here", "there", "if", "you", "your", "their", "our",
        "from", "do", "did", "no", "yes", "all", "me", "its", "us",
        "them", "into", "out", "up", "about", "when", "what", "re",
    }

    total_words = 0
    formal_hits = 0
    word_counts = Counter()

    for text in texts:
        lower = text.lower()
        words = lower.split()
        total_words += len(words)
        word_counts.update(w for w in words if w.isalpha())
        formal_hits += sum(1 for m in FORMAL_MARKERS if m in lower)

    avg_length = total_words / len(texts) if texts else 0

    # Tone is simplified to casual | formal only.
    # Pidgin detection is not applied — Yelp data is US-dominated and
    # Nigerian register is handled at the agent output layer instead.
    if formal_hits > len(texts) * 0.3:
        tone = "formal"
    else:
        tone = "casual"

    vocab = [
        w for w, _ in word_counts.most_common(200)
        if w not in STOPWORDS and len(w) > 2
    ][:VOCABULARY_TOP_N]

    return {
        "avg_length_words":       round(avg_length, 1),
        "tone":                   tone,   # casual | formal
        "vocabulary_fingerprint": vocab,
    }


# ─── LLM TASTE EXTRACTION (GROQ) ──────────────────────────────────────────────

def extract_taste_signals_llm(
    reviews: list,
    tips:    list,
    user_id: str,
    client:  Groq,
) -> dict:
    """
    Use Groq (Llama 3.3 70B) to extract structured taste signals from
    a sample of the user's reviews and tips.

    Replaces brittle keyword matching from v1. Handles:
    - Nigerian pidgin and colloquial expressions naturally
    - Implicit signals ("e no make sense" = service complaint)
    - Contextual nuance keyword lists cannot capture

    Called once per user at profile build time only — not at inference.
    """
    review_sample = sorted(
        [r for r in reviews if r.get("text", "").strip()],
        key=lambda r: r.get("date", ""),
        reverse=True
    )[:MAX_REVIEWS_FOR_LLM]

    tip_sample = sorted(
        [t for t in tips if t.get("text", "").strip()],
        key=lambda t: t.get("date", ""),
        reverse=True
    )[:MAX_TIPS_FOR_LLM]

    all_texts = []
    for r in review_sample:
        all_texts.append(
            f"[REVIEW — {r.get('stars')}★ — {r.get('date','')[:10]}]\n"
            f"{r['text'][:MAX_CHARS_PER_TEXT]}"
        )
    for t in tip_sample:
        all_texts.append(
            f"[TIP — {t.get('date','')[:10]}]\n"
            f"{t['text'][:MAX_CHARS_PER_TEXT]}"
        )

    if not all_texts:
        return _fallback_taste_signals()

    prompt = f"""You are analysing the food review history of a single user to build a behavioural taste profile.

Here are their reviews and tips (most recent first):

{chr(10).join(all_texts)}

Extract the following and return ONLY valid JSON — no preamble, no markdown fences:

{{
  "top_cuisines": ["up to 5 cuisine types this user reviews most"],
  "flavour_preferences": ["up to 5 flavour qualities they respond positively to"],
  "praise_triggers": ["up to 5 specific things that consistently earn them high ratings"],
  "complaint_triggers": ["up to 5 specific things that consistently earn low ratings"],
  "preferred_price_sensitivity": "budget | mid-range | upscale | unclear",
  "occasion_patterns": ["patterns in when or why they eat out — e.g. quick lunch, date nights"],
  "notable_behaviours": ["any other distinctive patterns useful for simulating their reviews"]
}}

Rules:
- Be specific. Use the user's own language where it captures something precisely.
- If the user writes in Nigerian Pidgin or Nigerian English, preserve those signals.
- Do not invent signals not present in the text.
- If there is insufficient data for a field, return an empty list."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        # strip markdown fences if model adds them despite instructions
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"  [WARN] LLM extraction failed for {user_id}: {e}")
        return _fallback_taste_signals()


def _fallback_taste_signals() -> dict:
    return {
        "top_cuisines":                [],
        "flavour_preferences":         [],
        "praise_triggers":             [],
        "complaint_triggers":          [],
        "preferred_price_sensitivity": "unclear",
        "occasion_patterns":           [],
        "notable_behaviours":          [],
    }


# ─── USER METADATA ────────────────────────────────────────────────────────────

def parse_user_metadata(user_record: dict) -> dict:
    """
    Extract writing personality signals from Yelp user metadata.

    Retained signals:
    - funny/useful/cool votes — crowd-sourced writing personality annotation
    - compliment breakdown — fine-grained personality signal
    - elite status — power reviewer flag, expect detailed opinionated writing

    Removed:
    - yelping_since — platform-specific tenure, not transferable to Chowdeck context
    """
    if not user_record:
        return {}

    funny  = user_record.get("funny",  0) or 0
    useful = user_record.get("useful", 0) or 0
    cool   = user_record.get("cool",   0) or 0

    compliment_writer = user_record.get("compliment_writer", 0) or 0
    compliment_funny  = user_record.get("compliment_funny",  0) or 0
    compliment_hot    = user_record.get("compliment_hot",    0) or 0
    compliment_cool   = user_record.get("compliment_cool",   0) or 0

    elite_raw  = user_record.get("elite", "") or ""
    elite_years = [
        y.strip() for y in str(elite_raw).split(",")
        if y.strip().isdigit()
    ]

    # derive plain-English personality signals for persona injection
    personality_signals = []
    if funny > 50 or compliment_funny > 10:
        personality_signals.append("writes with humour — reflect this in tone")
    if useful > 100 or compliment_writer > 10:
        personality_signals.append("writes detailed, informative reviews")
    if cool > 50 or compliment_cool > 10:
        personality_signals.append("engaging and relatable writing style")
    if compliment_hot > 10:
        personality_signals.append("writes passionately and expressively about food")
    if elite_years:
        personality_signals.append(
            f"Yelp Elite reviewer ({len(elite_years)} years) — high standards, opinionated"
        )

    return {
        "name":         user_record.get("name", ""),
        "is_elite":     len(elite_years) > 0,
        "elite_years":  elite_years,
        "social_votes": {
            "useful": useful,
            "funny":  funny,
            "cool":   cool,
        },
        "compliments": {
            "writer": compliment_writer,
            "funny":  compliment_funny,
            "hot":    compliment_hot,
            "cool":   compliment_cool,
            "plain":  user_record.get("compliment_plain", 0) or 0,
        },
        "yelp_avg_stars":      user_record.get("average_stars"),
        "personality_signals": personality_signals,
    }


# ─── CONFIDENCE SCORE ─────────────────────────────────────────────────────────

def compute_confidence_score(
    total_reviews: int,
    drift:         dict,
    has_text:      bool,
    has_tips:      bool,
    is_elite:      bool,
) -> dict:
    score   = 0
    reasons = []

    # history depth — 40 pts
    if total_reviews >= 50:
        score += 40; reasons.append("rich review history (50+ reviews)")
    elif total_reviews >= 30:
        score += 32; reasons.append("good review history (30–49 reviews)")
    elif total_reviews >= 15:
        score += 22; reasons.append("moderate review history (15–29 reviews)")
    elif total_reviews >= 5:
        score += 10; reasons.append("thin history (5–14 reviews) — cold start fallback active")
    else:
        score += 4;  reasons.append("very thin history (under 5 reviews) — cold start")

    # text availability — 15 pts
    if has_text:
        score += 15; reasons.append("review text available for tone modelling")
    else:
        score += 3;  reasons.append("ratings only — tone modelling limited")

    # tips — 10 pts
    if has_tips:
        score += 10; reasons.append("tip data available — richer vocabulary signal")

    # elite bonus — 5 pts
    if is_elite:
        score += 5; reasons.append("Yelp Elite — detailed writing expected")

    # drift penalty
    if drift.get("detected"):
        score -= 15; reasons.append("preference drift detected — older history downweighted")
    else:
        score += 10; reasons.append("stable preferences over time")

    # recent activity — 10 pts
    recent_count = drift.get("recent_count", 0)
    if recent_count >= 10:
        score += 10; reasons.append("recently active")
    elif recent_count >= 5:
        score += 5;  reasons.append("some recent activity")

    score = max(0, min(100, score))
    label = "high" if score >= 75 else "medium" if score >= 50 else "low"
    return {"score": score, "label": label, "reasons": reasons}


# ─── FULL PROFILE BUILDER ─────────────────────────────────────────────────────

def build_user_profile(
    user_id:        str,
    reviews:        list,
    tips:           list,
    user_record:    dict,
    now:            datetime,
    groq_client:    Groq,
    use_llm:        bool = True,
) -> dict:

    if not reviews and not tips:
        return None

    reviews = sorted(reviews, key=lambda r: r.get("date", "1900-01-01"))

    all_texts = (
        [r.get("text", "") for r in reviews if r.get("text", "").strip()] +
        [t.get("text", "") for t in tips    if t.get("text", "").strip()]
    )

    rating_profile = extract_rating_profile(reviews, now)
    writing_style  = classify_tone(all_texts)
    drift          = detect_drift(reviews, now)
    user_metadata  = parse_user_metadata(user_record)

    if use_llm and all_texts:
        taste_signals = extract_taste_signals_llm(
            reviews, tips, user_id, groq_client
        )
    else:
        taste_signals = _fallback_taste_signals()

    confidence = compute_confidence_score(
        total_reviews = len(reviews),
        drift         = drift,
        has_text      = bool(all_texts),
        has_tips      = bool(tips),
        is_elite      = user_metadata.get("is_elite", False),
    )

    # sample texts for few-shot prompting in Task A
    sample_items = []
    for r in sorted(reviews, key=lambda r: r.get("date", ""), reverse=True)[:4]:
        if r.get("text", "").strip():
            sample_items.append({
                "source": "review",
                "stars":  r["stars"],
                "text":   r["text"][:300],
                "date":   r.get("date", "")[:10],
            })
    for t in sorted(tips, key=lambda t: t.get("date", ""), reverse=True)[:2]:
        if t.get("text", "").strip():
            sample_items.append({
                "source": "tip",
                "stars":  None,
                "text":   t["text"][:200],
                "date":   t.get("date", "")[:10],
            })

    return {
        "user_id":        user_id,
        "profile_built":  now.isoformat(),
        "rating_profile": rating_profile,
        "writing_style":  writing_style,
        "taste_signals":  taste_signals,
        "drift":          drift,
        "confidence":     confidence,
        "user_metadata":  user_metadata,
        "sample_reviews": sample_items,
        "meta": {
            "total_reviews": len(reviews),
            "total_tips":    len(tips),
            "first_review":  reviews[0].get("date", "")[:10] if reviews else None,
            "last_review":   reviews[-1].get("date", "")[:10] if reviews else None,
        },
    }


# ─── PERSONA STRING ───────────────────────────────────────────────────────────

def build_prompt_persona(profile: dict) -> str:
    """
    Render profile as plain-English persona string for LLM prompt injection.
    Weighted mean is the primary rating signal — median is not injected.
    """
    r = profile.get("rating_profile", {})
    w = profile.get("writing_style", {})
    t = profile.get("taste_signals", {})
    d = profile.get("drift", {})
    c = profile.get("confidence", {})
    m = profile.get("user_metadata", {})

    lines = [
        f"USER PERSONA — confidence {c.get('score', 0)}% ({c.get('label', 'unknown')})",
        "",
    ]

    if m.get("name"):
        lines.append(f"Name: {m['name']}")
    if m.get("is_elite"):
        lines.append(f"Status: Yelp Elite ({len(m.get('elite_years', []))} years) — detailed, opinionated writer")
    if m.get("personality_signals"):
        lines.append(f"Personality: {' · '.join(m['personality_signals'])}")

    lines += ["", "RATING BEHAVIOUR:"]
    lines.append(
        f"  Weighted mean: {r.get('weighted_mean', 3.0):.1f}★  "
        f"Std dev: {r.get('std_dev', 0):.2f}  "
        f"Tendency: {r.get('tendency', 'balanced')}"
    )
    lines.append(f"  Total reviews: {r.get('total_reviews', 0)}")

    if d.get("detected"):
        lines.append(
            f"  ⚠ Drift ({d['direction']}): "
            f"recent avg {d['recent_mean']:.1f}★ vs older {d['older_mean']:.1f}★ — "
            f"weight recent behaviour more heavily"
        )

    lines += ["", "WRITING STYLE:"]
    lines.append(f"  Tone: {w.get('tone', 'casual_english')}")
    lines.append(f"  Avg length: {w.get('avg_length_words', 0):.0f} words")
    if w.get("vocabulary_fingerprint"):
        lines.append(f"  Signature vocabulary: {', '.join(w['vocabulary_fingerprint'][:12])}")

    if t and any(t.get(k) for k in ["top_cuisines", "praise_triggers", "complaint_triggers"]):
        lines += ["", "TASTE PROFILE:"]
        if t.get("top_cuisines"):
            lines.append(f"  Cuisines: {', '.join(t['top_cuisines'])}")
        if t.get("flavour_preferences"):
            lines.append(f"  Flavour preferences: {', '.join(t['flavour_preferences'])}")
        if t.get("praise_triggers"):
            lines.append(f"  High ratings when: {', '.join(t['praise_triggers'])}")
        if t.get("complaint_triggers"):
            lines.append(f"  Low ratings when: {', '.join(t['complaint_triggers'])}")
        if t.get("preferred_price_sensitivity") not in (None, "unclear"):
            lines.append(f"  Price sensitivity: {t['preferred_price_sensitivity']}")
        if t.get("occasion_patterns"):
            lines.append(f"  Occasion patterns: {', '.join(t['occasion_patterns'])}")
        if t.get("notable_behaviours"):
            lines.append(f"  Notable behaviours: {', '.join(t['notable_behaviours'])}")

    if profile.get("sample_reviews"):
        lines += ["", "SAMPLE TEXTS — calibrate voice from these:"]
        for s in profile["sample_reviews"][:4]:
            stars  = f"{s['stars']}★ — " if s.get("stars") else ""
            lines.append(
                f"  [{s['source'].upper()} — {stars}{s['date']}] "
                f"\"{s['text'][:200]}\""
            )

    lines += [
        "",
        "SIMULATION INSTRUCTION:",
        "Match this user's exact tone, length, vocabulary, and rating tendency.",
        "Do NOT impose a Nigerian register if their history does not reflect one.",
        "Simulate this specific person.",
    ]

    return "\n".join(lines)


# ─── STRATIFIED SAMPLING ──────────────────────────────────────────────────────

def stratified_sample(
    user_reviews: dict,
    user_tips:    dict,
    user_records: dict,
    now:          datetime,
) -> list:
    """
    Build a stratified sample of user IDs that covers all behavioural
    buckets needed for evaluation and demonstration.

    Warm buckets (15+ reviews): cover all tendency × tone combinations
    Cold buckets: explicitly sampled to demonstrate fallback handling
                  and score the cold-start criterion (25 pts Task B)
    """
    random.seed(RANDOM_SEED)

    # ── quick-classify each user without full LLM processing ─────────────────
    print("  Quick-classifying users for stratified sampling...")

    classified = {}   # user_id -> {tendency, tone, review_count}

    for uid, reviews in tqdm(user_reviews.items(), desc="  Classifying"):
        if not reviews:
            continue
        count  = len(reviews)
        stars  = [r["stars"] for r in reviews]
        mean   = statistics.mean(stars)

        tendency = (
            "generous" if mean >= 4.2 else
            "harsh"    if mean <= 2.8 else
            "balanced"
        )

        texts = (
            [r.get("text", "") for r in reviews if r.get("text", "").strip()] +
            [t.get("text", "") for t in user_tips.get(uid, []) if t.get("text", "").strip()]
        )
        tone_data = classify_tone(texts) if texts else {"tone": "casual"}
        tone      = tone_data["tone"]  # casual | formal

        classified[uid] = {
            "tendency":     tendency,
            "tone":         tone,
            "review_count": count,
        }

    # ── warm buckets — 15+ reviews ────────────────────────────────────────────
    sampled_ids = set()

    for (tendency, tone), target in WARM_BUCKETS.items():
        candidates = [
            uid for uid, info in classified.items()
            if info["tendency"]     == tendency
            and info["tone"]        == tone
            and info["review_count"] >= 15
        ]
        # prefer users with more reviews within each bucket
        candidates.sort(key=lambda uid: classified[uid]["review_count"], reverse=True)
        chosen = candidates[:target]
        sampled_ids.update(chosen)
        print(f"  Bucket ({tendency}, {tone}): {len(chosen)}/{target} sampled "
              f"({len(candidates)} available)")

    # ── cold-start buckets ────────────────────────────────────────────────────
    for bucket_name, cfg in COLD_BUCKETS.items():
        candidates = [
            uid for uid, info in classified.items()
            if cfg["min"] <= info["review_count"] <= cfg["max"]
            and uid not in sampled_ids
        ]
        random.shuffle(candidates)
        chosen = candidates[:cfg["target"]]
        sampled_ids.update(chosen)
        print(f"  Cold-start bucket '{bucket_name}' "
              f"({cfg['min']}–{cfg['max']} reviews): "
              f"{len(chosen)}/{cfg['target']} sampled")

    print(f"\n  Total sampled: {len(sampled_ids):,} users")
    return list(sampled_ids)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--limit",       type=int,  default=None,
                        help="Limit reviews loaded (for quick testing)")
    parser.add_argument("--user_id",     default=None,
                        help="Process a single user ID (for testing)")
    parser.add_argument("--no_llm",      action="store_true",
                        help="Skip Groq LLM extraction (faster, for testing)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    now = datetime.now()

    # ── Init Groq client ─────────────────────────────────────────────────────
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key and not args.no_llm:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Add it to your .env file or pass --no_llm."
        )
    groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

    # ── Load businesses — food filter ────────────────────────────────────────
    print("\n[1/5] Loading and filtering businesses...")
    biz_path  = os.path.join(args.data_dir, "yelp_academic_dataset_business.json")
    all_biz   = load_jsonl(biz_path)
    food_biz  = {b["business_id"]: b for b in all_biz if is_food_business(b)}
    print(f"  {len(all_biz):,} total → {len(food_biz):,} food businesses retained")

    # ── Load user metadata ───────────────────────────────────────────────────
    print("\n[2/5] Loading user metadata...")
    user_path    = os.path.join(args.data_dir, "yelp_academic_dataset_user.json")
    user_records = {u["user_id"]: u for u in load_jsonl(user_path)}
    print(f"  {len(user_records):,} user records loaded")

    # ── Load reviews (food only) ─────────────────────────────────────────────
    print("\n[3/5] Loading reviews (food businesses only)...")
    review_path  = os.path.join(args.data_dir, "yelp_academic_dataset_review.json")
    reviews_raw  = load_jsonl(review_path, limit=args.limit)
    user_reviews = defaultdict(list)
    for r in reviews_raw:
        if r.get("business_id") in food_biz and r.get("user_id"):
            user_reviews[r["user_id"]].append(r)

    total_food_reviews = sum(len(v) for v in user_reviews.values())
    print(f"  {len(reviews_raw):,} total → {total_food_reviews:,} food reviews "
          f"across {len(user_reviews):,} users")

    # ── Load tips (food only) ────────────────────────────────────────────────
    print("\n[4/5] Loading tips (food businesses only)...")
    tip_path  = os.path.join(args.data_dir, "yelp_academic_dataset_tip.json")
    tips_raw  = load_jsonl(tip_path)
    user_tips = defaultdict(list)
    for t in tips_raw:
        if t.get("business_id") in food_biz and t.get("user_id"):
            user_tips[t["user_id"]].append(t)
    print(f"  {len(tips_raw):,} total → "
          f"{sum(len(v) for v in user_tips.values()):,} food tips")

    # ── Determine target users ───────────────────────────────────────────────
    print("\n[5/5] Selecting users...")

    if args.user_id:
        target_ids = [args.user_id]
        print(f"  Single user mode: {args.user_id}")
    else:
        target_ids = stratified_sample(
            user_reviews, user_tips, user_records, now
        )

    # ── Build profiles ───────────────────────────────────────────────────────
    print(f"\nBuilding {len(target_ids):,} profiles "
          f"({'with' if not args.no_llm else 'without'} LLM extraction)...")

    profiles      = {}
    profile_index = []
    bucket_counts = Counter()

    for uid in tqdm(target_ids, desc="Building profiles"):
        profile = build_user_profile(
            user_id      = uid,
            reviews      = user_reviews.get(uid, []),
            tips         = user_tips.get(uid, []),
            user_record  = user_records.get(uid, {}),
            now          = now,
            groq_client  = groq_client,
            use_llm      = not args.no_llm,
        )
        if not profile:
            continue

        profiles[uid] = profile

        tendency = profile["rating_profile"].get("tendency", "unknown")
        tone     = profile["writing_style"].get("tone", "unknown")  # casual | formal
        count    = profile["meta"]["total_reviews"]
        bucket   = (
            "very_thin" if count < 5  else
            "thin"      if count < 15 else
            f"{tendency}_{tone}"
        )
        bucket_counts[bucket] += 1

        profile_index.append({
            "user_id":       uid,
            "name":          profile["user_metadata"].get("name", ""),
            "total_reviews": count,
            "total_tips":    profile["meta"]["total_tips"],
            "avg_rating":    profile["rating_profile"]["weighted_mean"],
            "median_rating": profile["rating_profile"]["median"],
            "tendency":      tendency,
            "tone":          tone,
            "confidence":    profile["confidence"]["score"],
            "drift":         profile["drift"]["detected"],
            "is_elite":      profile["user_metadata"].get("is_elite", False),
            "bucket":        bucket,
            "last_review":   profile["meta"]["last_review"],
        })

    # ── Write outputs ────────────────────────────────────────────────────────
    profiles_path = os.path.join(args.output_dir, "user_profiles.json")
    index_path    = os.path.join(args.output_dir, "profile_index.json")
    report_path   = os.path.join(args.output_dir, "sampling_report.json")

    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(profile_index, f, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_profiles":  len(profiles),
            "bucket_counts":   dict(bucket_counts),
            "built_at":        now.isoformat(),
        }, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    scores = [p["confidence"]["score"] for p in profiles.values()]
    drifts = sum(1 for p in profiles.values() if p["drift"]["detected"])
    elites = sum(1 for p in profiles.values() if p["user_metadata"].get("is_elite"))

    print("\n── COMPLETE ────────────────────────────────────────────────────")
    print(f"  Profiles built:    {len(profiles):,}")
    print(f"  Avg confidence:    {sum(scores)/len(scores):.1f}")
    print(f"  Drift detected:    {drifts:,} ({drifts/len(profiles)*100:.1f}%)")
    print(f"  Elite reviewers:   {elites:,}")
    print(f"\n  Bucket distribution:")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"    {bucket:<30} {count:>4}")
    print(f"\n  Outputs: {args.output_dir}/")
    print("────────────────────────────────────────────────────────────────\n")

    # ── Sample persona ───────────────────────────────────────────────────────
    if profile_index:
        best = max(profile_index, key=lambda x: x["confidence"])
        print("── SAMPLE PERSONA ──────────────────────────────────────────────")
        print(build_prompt_persona(profiles[best["user_id"]]))
        print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()