"""
profile_extractor.py
=======================
ChopLife — User Profile Builder

Output:
  data/user_profiles.json        full profiles
  data/profile_index.json        lightweight lookup
  data/sampling_report.json      bucket distribution (paste into paper)
  data/llm_checkpoint.json       tracks which users have LLM extraction

Usage:
  # First run — full (with LLM)
  python profile_extractor.py --data_dir "./Yelp JSON/yelp_dataset" --output_dir ./data
 
  # Resume after rate limit hit
  python profile_extractor.py --data_dir "./Yelp JSON/yelp_dataset" --output_dir ./data --resume
 
  # Pipeline test — no LLM calls
  python profile_extractor.py --data_dir "./Yelp JSON/yelp_dataset" --output_dir ./data --no_llm
 
  # Single user test
  python profile_extractor.py --data_dir "./Yelp JSON/yelp_dataset" --output_dir ./data --user_id <id>

"""

import json
import os
import re
import time
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
# Change GROQ_MODEL here to switch models for ablation study comparisons.
 
GROQ_MODEL           = "meta-llama/llama-4-scout-17b-16e-instruct"
# GROQ_MODEL         = "llama-3.3-70b-versatile"       # 1K RPD, 100K TPD
# GROQ_MODEL         = "llama-3.1-8b-instant"           # 14.4K RPD, 500K TPD
 
DRIFT_WINDOW_MONTHS  = 18
DRIFT_THRESHOLD      = 0.6
VOCABULARY_TOP_N     = 20
MAX_REVIEWS_FOR_LLM  = 8     # reduced from 15 — halves token usage
MAX_TIPS_FOR_LLM     = 3     # reduced from 5
MAX_CHARS_PER_TEXT   = 200   # reduced from 300
SAMPLE_HIGH_N        = 3     # top N highest-rated reviews for sample
SAMPLE_LOW_N         = 2     # bottom N lowest-rated reviews for sample
RATE_LIMIT_WAIT_SECS = 660   # 11 minutes default wait on 429
RANDOM_SEED          = 42
 
# ── Category filter ───────────────────────────────────────────────────────────
PRIMARY_FOOD_CATS = {"restaurants", "food"}
SECONDARY_FOOD_CATS = {
    "bakeries", "cafes", "coffee & tea", "food trucks", "bubble tea",
    "juice bars & smoothies", "desserts", "ice cream & frozen yogurt",
    "delis", "diners", "buffets", "street vendors", "grocery",
    "specialty food", "beer bar", "wine bars",
}
 
# ── Stratified sampling targets ───────────────────────────────────────────────
# Option 2 — targets adjusted to match actually available users.
# Aspirational targets replaced with realistic ones based on first run.
# tone: casual | formal only — no pidgin from Yelp (US-dominated dataset)
# Nigerian register applied at agent output layer via Nairaland corpus
#
# Cold-start fallback hierarchy (applies to BOTH tasks):
#   Level 1: use thin history as-is, weight low
#   Level 2: similar user borrowing (tendency + tone + cuisine match)
#   Level 3: global Yelp aggregate baseline
#   Level 4: onboarding questionnaire — Task B front end ONLY
#   Nigerian baseline is NOT a cold-start fallback. It is a Task B
#   voice calibration only. Using it as a prior degrades Task A fidelity.
WARM_BUCKETS = {
    ("generous", "casual"): 150,  # 195 available — take 150
    ("generous", "formal"):   4,  # 4 available   — take all
    ("balanced", "casual"): 150,  # 883 available — take 150
    ("balanced", "formal"):  30,  # 30 available  — take all
    ("harsh",    "casual"):  17,  # 17 available  — take all
    ("harsh",    "formal"):   1,  # 1 available   — take all
}
COLD_BUCKETS = {
    "thin":      {"min": 5,  "max": 14, "target": 80},
    "very_thin": {"min": 1,  "max": 4,  "target": 50},
}
 
 
# ─── HELPERS ──────────────────────────────────────────────────────────────────
 
def load_jsonl(filepath: str) -> list:
    """Load entire JSONL file. No limit — required for deterministic sampling."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records
 
 
def is_food_business(business: dict) -> bool:
    cats_raw = business.get("categories") or ""
    if not cats_raw:
        return False
    cats_lower = cats_raw.lower()
    for cat in PRIMARY_FOOD_CATS:
        if cat in cats_lower:
            return True
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
 
 
def truncate_at_sentence(text: str, max_chars: int) -> str:
    """
    Truncate text at the last sentence boundary before max_chars.
    Avoids mid-sentence cuts that give the LLM incomplete thoughts.
    Falls back to word boundary if no sentence boundary found.
    """
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    # try sentence boundary first
    last_sentence = max(
        chunk.rfind(". "),
        chunk.rfind("! "),
        chunk.rfind("? "),
    )
    if last_sentence > max_chars // 2:
        return chunk[:last_sentence + 1].strip()
    # fall back to word boundary
    last_space = chunk.rfind(" ")
    if last_space > 0:
        return chunk[:last_space].strip() + "..."
    return chunk.strip() + "..."
 
 
def load_checkpoint(checkpoint_path: str) -> set:
    """Load set of user IDs that already have LLM extraction complete."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        return set(data.get("completed", []))
    return set()
 
 
def save_checkpoint(checkpoint_path: str, completed: set) -> None:
    with open(checkpoint_path, "w") as f:
        json.dump({"completed": list(completed), "count": len(completed)}, f)
 
 
def load_existing_profiles(profiles_path: str) -> dict:
    """Load profiles already written to disk (for resume mode)."""
    if os.path.exists(profiles_path):
        with open(profiles_path, "r") as f:
            return json.load(f)
    return {}
 
 
# ─── RATING PROFILE ───────────────────────────────────────────────────────────
 
def extract_rating_profile(reviews: list, now: datetime) -> dict:
    """
    Weighted mean is primary — bounded 1–5 scale has no true outliers.
    Median stored as secondary cross-check, not injected into prompts.
    """
    if not reviews:
        return {}
 
    stars    = [r["stars"] for r in reviews]
    dist     = Counter(int(s) for s in stars)
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
 
    tendency = (
        "generous" if raw_mean >= 4.2 else
        "harsh"    if raw_mean <= 2.8 else
        "balanced"
    )
 
    return {
        "raw_mean":      round(raw_mean, 2),
        "weighted_mean": round(weighted_mean, 2),
        "median":        median,
        "std_dev":       round(std_dev, 2),
        "tendency":      tendency,
        "distribution":  {str(k): dist.get(k, 0) for k in range(1, 6)},
        "total_reviews": len(reviews),
    }
 
 
# ─── DRIFT DETECTION ──────────────────────────────────────────────────────────
 
def detect_drift(reviews: list, now: datetime) -> dict:
    if len(reviews) < 15:
        return {"detected": False, "reason": "insufficient_history", "recent_count": 0}
 
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
    Tone: casual | formal only.
    Pidgin not detected — Yelp is US-dominated. Nigerian register applied
    at agent output layer via Nairaland corpus, not from Yelp profiles.
    Tips are included — they reflect the user's most natural unprompted register.
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
 
    total_words = formal_hits = 0
    word_counts = Counter()
 
    for text in texts:
        lower = text.lower()
        words = lower.split()
        total_words += len(words)
        word_counts.update(w for w in words if w.isalpha())
        formal_hits += sum(1 for m in FORMAL_MARKERS if m in lower)
 
    avg_length = total_words / len(texts) if texts else 0
    tone = "formal" if formal_hits > len(texts) * 0.3 else "casual"
 
    vocab = [
        w for w, _ in word_counts.most_common(200)
        if w not in STOPWORDS and len(w) > 2
    ][:VOCABULARY_TOP_N]
 
    return {
        "avg_length_words":       round(avg_length, 1),
        "tone":                   tone,
        "vocabulary_fingerprint": vocab,
    }
 
 
# ─── SAMPLE REVIEW SELECTION ──────────────────────────────────────────────────
 
def select_sample_reviews(reviews: list, tips: list) -> list:
    """
    Select representative sample texts for few-shot prompting.
 
    Strategy: top SAMPLE_HIGH_N highest-rated + bottom SAMPLE_LOW_N lowest-rated
    The agent needs to know both what delight AND disappointment sound like
    for this specific user — not just their happy reviews.
    Tips included separately — they show natural unprompted voice.
    """
    text_reviews = [r for r in reviews if r.get("text", "").strip()]
    if not text_reviews:
        return []
 
    sorted_by_stars = sorted(text_reviews, key=lambda r: (r["stars"], r.get("date", "")))
 
    # bottom N (lowest rated — disappointment voice)
    low  = sorted_by_stars[:SAMPLE_LOW_N]
    # top N (highest rated — delight voice)
    high = sorted_by_stars[-SAMPLE_HIGH_N:]
 
    # deduplicate (in case user has very few reviews)
    seen = set()
    sample = []
    for r in high + low:
        rid = r.get("review_id") or r.get("text", "")[:50]
        if rid not in seen:
            seen.add(rid)
            sample.append({
                "source": "review",
                "stars":  r["stars"],
                "text":   truncate_at_sentence(r.get("text", ""), MAX_CHARS_PER_TEXT * 2),
                "date":   r.get("date", "")[:10],
            })
 
    # add most recent tips
    for t in sorted(tips, key=lambda t: t.get("date", ""), reverse=True)[:2]:
        if t.get("text", "").strip():
            sample.append({
                "source": "tip",
                "stars":  None,
                "text":   truncate_at_sentence(t.get("text", ""), MAX_CHARS_PER_TEXT),
                "date":   t.get("date", "")[:10],
            })
 
    return sample
 
 
# ─── LLM TASTE EXTRACTION ─────────────────────────────────────────────────────
 
def extract_taste_signals_llm(
    reviews:    list,
    tips:       list,
    user_id:    str,
    groq_client: Groq,
) -> tuple:
    """
    Extract structured taste signals via Groq LLM.
    Returns (signals_dict, status) where status is "complete" or "fallback".
 
    Rate limit handling:
    - Catches 429 error
    - Parses retry-after time from error message
    - Waits and retries once
    - If retry also fails, returns fallback with status "fallback"
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
            f"{truncate_at_sentence(r.get('text',''), MAX_CHARS_PER_TEXT)}"
        )
    for t in tip_sample:
        all_texts.append(
            f"[TIP — {t.get('date','')[:10]}]\n"
            f"{truncate_at_sentence(t.get('text',''), MAX_CHARS_PER_TEXT)}"
        )
 
    if not all_texts:
        return _fallback_taste_signals(), "fallback"
 
    prompt = f"""Analyse this user's food review history and extract a behavioural taste profile.
 
{chr(10).join(all_texts)}
 
Return ONLY valid JSON — no preamble, no markdown:
 
{{
  "top_cuisines": ["up to 5 cuisine types reviewed most"],
  "flavour_preferences": ["up to 5 flavour qualities they respond positively to"],
  "praise_triggers": ["up to 5 things that consistently earn high ratings"],
  "complaint_triggers": ["up to 5 things that consistently earn low ratings"],
  "preferred_price_sensitivity": "budget | mid-range | upscale | unclear",
  "occasion_patterns": ["patterns in when or why they eat out"],
  "notable_behaviours": ["distinctive patterns useful for simulating their reviews"],
  "dietary_context": {{
    "halal_only": true or false or null,
    "inferred_from": "text_mentions | reviewed_businesses | absence_signal | unclear",
    "confidence": "high | medium | low | unknown",
    "notes": "brief explanation of what signal led to this inference"
  }}
}}
 
Rules:
- Be specific. Use the user's own language where it captures something precisely.
- Do not invent signals not present in the text.
- For dietary_context:
    high confidence   = user explicitly mentions halal, haram, or avoids pork/alcohol by name
    medium confidence = user consistently reviews halal-certified restaurants
    low confidence    = user never mentions pork or alcohol across many reviews (absence signal)
    unknown           = insufficient signal to infer
- If you cannot determine dietary preference, set halal_only to null and confidence to unknown."""
 
    def call_api():
        return groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
 
    def parse_response(response):
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
 
    def attempt(retries_left: int, backoff: int) -> tuple:
        """
        Recursive retry with exponential backoff.
        Handles both 429 (rate limit) and 503 (capacity) separately.
        """
        try:
            response = call_api()
            return parse_response(response), "complete"
 
        except json.JSONDecodeError as e:
            # malformed JSON — attempt simple repair before giving up
            raw = getattr(e, 'doc', '') or ''
            # try to extract JSON object substring
            start = raw.find('{')
            end   = raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end+1]), "complete"
                except Exception:
                    pass
            print(f"  [WARN] JSON repair failed for {user_id}")
            return _fallback_taste_signals(), "fallback"
 
        except Exception as e:
            error_str = str(e)
 
            # ── 429 Rate limit ────────────────────────────────────────────
            if "429" in error_str or "rate limit" in error_str.lower():
                wait_secs = RATE_LIMIT_WAIT_SECS
                match = re.search(r'try again in (\d+)m(\d+)', error_str)
                if match:
                    wait_secs = int(match.group(1)) * 60 + int(match.group(2)) + 10
                else:
                    match = re.search(r'try again in ([\d.]+)s', error_str)
                    if match:
                        wait_secs = int(float(match.group(1))) + 5
 
                print(f"\n  [RATE LIMIT 429] Waiting {wait_secs}s "
                      f"(~{wait_secs//60}m). Ctrl+C then --resume if needed.")
                time.sleep(wait_secs)
                if retries_left > 0:
                    return attempt(retries_left - 1, backoff * 2)
                return _fallback_taste_signals(), "fallback"
 
            # ── 503 Capacity ──────────────────────────────────────────────
            if "503" in error_str or "over capacity" in error_str.lower():
                if retries_left > 0:
                    print(f"  [503 CAPACITY] Backing off {backoff}s, "
                          f"{retries_left} retries left...")
                    time.sleep(backoff)
                    return attempt(retries_left - 1, backoff * 2)
                print(f"  [WARN] 503 retries exhausted for {user_id}")
                return _fallback_taste_signals(), "fallback"
 
            # ── Other errors ──────────────────────────────────────────────
            print(f"  [WARN] LLM extraction failed for {user_id}: {e}")
            return _fallback_taste_signals(), "fallback"
 
    return attempt(retries_left=3, backoff=10)
 
 
def _fallback_taste_signals() -> dict:
    return {
        "top_cuisines":                [],
        "flavour_preferences":         [],
        "praise_triggers":             [],
        "complaint_triggers":          [],
        "preferred_price_sensitivity": "unclear",
        "occasion_patterns":           [],
        "notable_behaviours":          [],
        "dietary_context": {
            "halal_only":      None,
            "inferred_from":   "unclear",
            "confidence":      "unknown",
            "notes":           "LLM extraction failed — dietary preference unknown",
        },
    }
 
 
# ─── USER METADATA ────────────────────────────────────────────────────────────
 
def parse_user_metadata(user_record: dict) -> dict:
    """
    Writing personality signals from Yelp user metadata.
    Retained: funny/useful/cool votes, compliments, elite status.
    Removed: yelping_since — platform-specific, not transferable.
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
 
    elite_raw   = user_record.get("elite", "") or ""
    elite_years = [y.strip() for y in str(elite_raw).split(",") if y.strip().isdigit()]
 
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
        "name":               user_record.get("name", ""),
        "is_elite":           len(elite_years) > 0,
        "elite_years":        elite_years,
        "social_votes":       {"useful": useful, "funny": funny, "cool": cool},
        "compliments":        {
            "writer": compliment_writer, "funny": compliment_funny,
            "hot": compliment_hot, "cool": compliment_cool,
            "plain": user_record.get("compliment_plain", 0) or 0,
        },
        "yelp_avg_stars":     user_record.get("average_stars"),
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
    score = 0
    reasons = []
 
    if total_reviews >= 50:
        score += 40; reasons.append("rich review history (50+ reviews)")
    elif total_reviews >= 30:
        score += 32; reasons.append("good review history (30–49 reviews)")
    elif total_reviews >= 15:
        score += 22; reasons.append("moderate review history (15–29 reviews)")
    elif total_reviews >= 5:
        score += 10; reasons.append("thin history (5–14 reviews) — cold start fallback active")
    else:
        score += 4;  reasons.append("very thin history (<5 reviews) — cold start")
 
    if has_text:
        score += 15; reasons.append("review text available for tone modelling")
    else:
        score += 3;  reasons.append("ratings only — tone modelling limited")
 
    if has_tips:
        score += 10; reasons.append("tip data available — richer vocabulary signal")
 
    if is_elite:
        score += 5; reasons.append("Yelp Elite — detailed writing expected")
 
    if drift.get("detected"):
        score -= 15; reasons.append("preference drift detected — older history downweighted")
    else:
        score += 10; reasons.append("stable preferences over time")
 
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
    user_id:     str,
    reviews:     list,
    tips:        list,
    user_record: dict,
    now:         datetime,
    groq_client: Groq,
    use_llm:     bool = True,
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
    sample_reviews = select_sample_reviews(reviews, tips)
 
    llm_status = "skipped"
    if use_llm and all_texts and groq_client:
        taste_signals, llm_status = extract_taste_signals_llm(
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
 
    return {
        "user_id":        user_id,
        "profile_built":  now.isoformat(),
        "llm_status":     llm_status,       # complete | fallback | skipped
        "llm_model":      GROQ_MODEL if llm_status == "complete" else None,
        "rating_profile": rating_profile,
        "writing_style":  writing_style,
        "taste_signals":  taste_signals,
        "drift":          drift,
        "confidence":     confidence,
        "user_metadata":  user_metadata,
        "sample_reviews": sample_reviews,
        "meta": {
            "total_reviews": len(reviews),
            "total_tips":    len(tips),
            "first_review":  reviews[0].get("date", "")[:10] if reviews else None,
            "last_review":   reviews[-1].get("date", "")[:10] if reviews else None,
        },
    }
 
 
# ─── PERSONA STRING ───────────────────────────────────────────────────────────
 
def build_prompt_persona(profile: dict) -> str:
    """Render profile as plain-English persona string for LLM injection."""
    r = profile.get("rating_profile", {})
    w = profile.get("writing_style", {})
    t = profile.get("taste_signals", {})
    d = profile.get("drift", {})
    c = profile.get("confidence", {})
    m = profile.get("user_metadata", {})
 
    lines = [f"USER PERSONA — confidence {c.get('score',0)}% ({c.get('label','unknown')})", ""]
 
    if m.get("name"):
        lines.append(f"Name: {m['name']}")
    if m.get("is_elite"):
        lines.append(f"Status: Yelp Elite ({len(m.get('elite_years',[]))} years) — detailed, opinionated")
    if m.get("personality_signals"):
        lines.append(f"Personality: {' · '.join(m['personality_signals'])}")
 
    lines += ["", "RATING BEHAVIOUR:"]
    lines.append(
        f"  Weighted mean: {r.get('weighted_mean',3.0):.1f}★  "
        f"Std dev: {r.get('std_dev',0):.2f}  "
        f"Tendency: {r.get('tendency','balanced')}"
    )
    lines.append(f"  Total reviews: {r.get('total_reviews',0)}")
    if d.get("detected"):
        lines.append(
            f"  ⚠ Drift ({d['direction']}): recent avg {d['recent_mean']:.1f}★ "
            f"vs older {d['older_mean']:.1f}★ — weight recent behaviour more heavily"
        )
 
    lines += ["", "WRITING STYLE:"]
    lines.append(f"  Tone: {w.get('tone','casual')}")
    lines.append(f"  Avg length: {w.get('avg_length_words',0):.0f} words")
    if w.get("vocabulary_fingerprint"):
        lines.append(f"  Signature vocabulary: {', '.join(w['vocabulary_fingerprint'][:12])}")
 
    if t and any(t.get(k) for k in ["top_cuisines","praise_triggers","complaint_triggers"]):
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
        high = [s for s in profile["sample_reviews"] if s.get("stars") and s["stars"] >= 4]
        low  = [s for s in profile["sample_reviews"] if s.get("stars") and s["stars"] <= 2]
        tips = [s for s in profile["sample_reviews"] if s.get("source") == "tip"]
        for s in high:
            lines.append(f"  [HIGH — {s['stars']}★ — {s['date']}] \"{s['text']}\"")
        for s in low:
            lines.append(f"  [LOW — {s['stars']}★ — {s['date']}] \"{s['text']}\"")
        for s in tips:
            lines.append(f"  [TIP — {s['date']}] \"{s['text']}\"")
 
    lines += [
        "", "SIMULATION INSTRUCTION:",
        "Match this user's exact tone, length, vocabulary, and rating tendency.",
        "Do NOT impose a Nigerian register if their history does not reflect one.",
        "Simulate this specific person.",
    ]
    return "\n".join(lines)
 
 
# ─── STRATIFIED SAMPLING ──────────────────────────────────────────────────────
 
def stratified_sample(user_reviews: dict, user_tips: dict) -> list:
    """
    Deterministic stratified sample.
    Warm buckets: sort by review count descending, take top N — stable.
    Cold buckets: random.shuffle with fixed RANDOM_SEED — stable.
    Do NOT pass --limit when loading reviews — different slices cause
    different users to appear, making results non-deterministic.
    """
    random.seed(RANDOM_SEED)
 
    print("  Quick-classifying users for stratified sampling...")
    classified = {}
 
    for uid, reviews in tqdm(user_reviews.items(), desc="  Classifying"):
        if not reviews:
            continue
        count = len(reviews)
        mean  = statistics.mean(r["stars"] for r in reviews)
        tendency = (
            "generous" if mean >= 4.2 else
            "harsh"    if mean <= 2.8 else
            "balanced"
        )
        texts = (
            [r.get("text","") for r in reviews if r.get("text","").strip()] +
            [t.get("text","") for t in user_tips.get(uid,[]) if t.get("text","").strip()]
        )
        tone = classify_tone(texts)["tone"] if texts else "casual"
        classified[uid] = {"tendency": tendency, "tone": tone, "review_count": count}
 
    sampled_ids = set()
 
    for (tendency, tone), target in WARM_BUCKETS.items():
        candidates = [
            uid for uid, info in classified.items()
            if info["tendency"]     == tendency
            and info["tone"]        == tone
            and info["review_count"] >= 15
        ]
        candidates.sort(key=lambda uid: classified[uid]["review_count"], reverse=True)
        chosen = candidates[:target]
        sampled_ids.update(chosen)
        print(f"  Bucket ({tendency}, {tone}): {len(chosen)}/{target} "
              f"({len(candidates)} available)")
 
    for bucket_name, cfg in COLD_BUCKETS.items():
        candidates = [
            uid for uid, info in classified.items()
            if cfg["min"] <= info["review_count"] <= cfg["max"]
            and uid not in sampled_ids
        ]
        random.shuffle(candidates)
        chosen = candidates[:cfg["target"]]
        sampled_ids.update(chosen)
        print(f"  Cold '{bucket_name}' ({cfg['min']}–{cfg['max']} reviews): "
              f"{len(chosen)}/{cfg['target']}")
 
    print(f"\n  Total sampled: {len(sampled_ids):,} users")
    return list(sampled_ids)
 
 
# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--user_id",    default=None)
    parser.add_argument("--no_llm",     action="store_true")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip users already in checkpoint, merge with existing profiles")
    args = parser.parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)
    now = datetime.now()
 
    profiles_path    = os.path.join(args.output_dir, "user_profiles.json")
    index_path       = os.path.join(args.output_dir, "profile_index.json")
    report_path      = os.path.join(args.output_dir, "sampling_report.json")
    checkpoint_path  = os.path.join(args.output_dir, "llm_checkpoint.json")
 
    # ── Init Groq ────────────────────────────────────────────────────────────
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key and not args.no_llm:
        raise EnvironmentError("GROQ_API_KEY not in .env. Use --no_llm to skip LLM.")
    groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
 
    # ── Resume: load existing state ──────────────────────────────────────────
    completed_ids = set()
    existing_profiles = {}
    if args.resume:
        completed_ids    = load_checkpoint(checkpoint_path)
        existing_profiles = load_existing_profiles(profiles_path)
        print(f"\n[RESUME] {len(completed_ids):,} users already processed, "
              f"{len(existing_profiles):,} profiles loaded from disk")
 
    # ── [1/5] Load businesses ────────────────────────────────────────────────
    print("\n[1/5] Loading and filtering businesses...")
    biz_path = os.path.join(args.data_dir, "yelp_academic_dataset_business.json")
    all_biz  = load_jsonl(biz_path)
    food_biz_raw = [b for b in all_biz if is_food_business(b)]
    # flag halal businesses — used in dietary context inference and Task B filter
    for b in food_biz_raw:
        cats_lower = (b.get("categories") or "").lower()
        b["is_halal"] = "halal" in cats_lower
    food_biz = {b["business_id"]: b for b in food_biz_raw}
    halal_count = sum(1 for b in food_biz_raw if b["is_halal"])
    print(f"  {len(all_biz):,} total → {len(food_biz):,} food businesses "
          f"({halal_count:,} halal-flagged)")
 
    # ── [2/5] Load user metadata ─────────────────────────────────────────────
    print("\n[2/5] Loading user metadata...")
    user_path    = os.path.join(args.data_dir, "yelp_academic_dataset_user.json")
    user_records = {u["user_id"]: u for u in load_jsonl(user_path)}
    print(f"  {len(user_records):,} users loaded")
 
    # ── [3/5] Load reviews ───────────────────────────────────────────────────
    print("\n[3/5] Loading reviews (food only)...")
    review_path  = os.path.join(args.data_dir, "yelp_academic_dataset_review.json")
    reviews_raw  = load_jsonl(review_path)
    user_reviews = defaultdict(list)
    for r in reviews_raw:
        if r.get("business_id") in food_biz and r.get("user_id"):
            user_reviews[r["user_id"]].append(r)
    print(f"  {len(reviews_raw):,} total → "
          f"{sum(len(v) for v in user_reviews.values()):,} food reviews "
          f"across {len(user_reviews):,} users")
 
    # ── [4/5] Load tips ──────────────────────────────────────────────────────
    print("\n[4/5] Loading tips (food only)...")
    tip_path  = os.path.join(args.data_dir, "yelp_academic_dataset_tip.json")
    tips_raw  = load_jsonl(tip_path)
    user_tips = defaultdict(list)
    for t in tips_raw:
        if t.get("business_id") in food_biz and t.get("user_id"):
            user_tips[t["user_id"]].append(t)
    print(f"  {len(tips_raw):,} total → "
          f"{sum(len(v) for v in user_tips.values()):,} food tips")
 
    # ── [5/5] Select target users ────────────────────────────────────────────
    print("\n[5/5] Selecting users...")
    if args.user_id:
        target_ids = [args.user_id]
        print(f"  Single user: {args.user_id}")
    else:
        target_ids = stratified_sample(user_reviews, user_tips)
 
    # filter out already completed on resume
    if args.resume:
        pending = [uid for uid in target_ids if uid not in completed_ids]
        print(f"  Pending after resume filter: {len(pending):,} users")
    else:
        pending = target_ids
 
    # ── Build profiles ───────────────────────────────────────────────────────
    use_llm = not args.no_llm
    print(f"\nBuilding {len(pending):,} profiles "
          f"({'with' if use_llm else 'without'} LLM) "
          f"using model: {GROQ_MODEL if use_llm else 'none'}...")
 
    profiles      = dict(existing_profiles)
    bucket_counts = Counter()
    llm_stats     = Counter()
 
    for uid in tqdm(pending, desc="Building profiles"):
        profile = build_user_profile(
            user_id     = uid,
            reviews     = user_reviews.get(uid, []),
            tips        = user_tips.get(uid, []),
            user_record = user_records.get(uid, {}),
            now         = now,
            groq_client = groq_client,
            use_llm     = use_llm,
        )
        if not profile:
            continue
 
        profiles[uid] = profile
        llm_stats[profile["llm_status"]] += 1
 
        # update checkpoint immediately after each profile
        if use_llm and profile["llm_status"] == "complete":
            completed_ids.add(uid)
            save_checkpoint(checkpoint_path, completed_ids)
 
        tendency = profile["rating_profile"].get("tendency", "unknown")
        tone     = profile["writing_style"].get("tone", "unknown")
        count    = profile["meta"]["total_reviews"]
        bucket   = (
            "very_thin" if count < 5  else
            "thin"      if count < 15 else
            f"{tendency}_{tone}"
        )
        bucket_counts[bucket] += 1
 
    # ── Build profile index ──────────────────────────────────────────────────
    profile_index = []
    for uid, profile in profiles.items():
        tendency = profile["rating_profile"].get("tendency","unknown")
        tone     = profile["writing_style"].get("tone","unknown")
        count    = profile["meta"]["total_reviews"]
        bucket   = (
            "very_thin" if count < 5  else
            "thin"      if count < 15 else
            f"{tendency}_{tone}"
        )
        dietary = profile.get("taste_signals",{}).get("dietary_context",{})
        profile_index.append({
            "user_id":          uid,
            "name":             profile["user_metadata"].get("name",""),
            "total_reviews":    count,
            "total_tips":       profile["meta"]["total_tips"],
            "avg_rating":       profile["rating_profile"]["weighted_mean"],
            "median_rating":    profile["rating_profile"]["median"],
            "tendency":         tendency,
            "tone":             tone,
            "confidence":       profile["confidence"]["score"],
            "drift":            profile["drift"]["detected"],
            "is_elite":         profile["user_metadata"].get("is_elite",False),
            "llm_status":       profile.get("llm_status","unknown"),
            "bucket":           bucket,
            "last_review":      profile["meta"]["last_review"],
            # halal flag surfaced for fast Task B candidate filtering
            "halal_only":       dietary.get("halal_only"),
            "halal_confidence": dietary.get("confidence","unknown"),
        })
 
    # ── Write outputs ────────────────────────────────────────────────────────
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(profile_index, f, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_profiles":  len(profiles),
            "bucket_counts":   dict(bucket_counts),
            "llm_stats":       dict(llm_stats),
            "model_used":      GROQ_MODEL,
            "built_at":        now.isoformat(),
        }, f, indent=2)
 
    # ── Summary ──────────────────────────────────────────────────────────────
    scores = [p["confidence"]["score"] for p in profiles.values()]
    drifts = sum(1 for p in profiles.values() if p["drift"]["detected"])
    elites = sum(1 for p in profiles.values() if p["user_metadata"].get("is_elite"))
 
    print("\n── COMPLETE ────────────────────────────────────────────────────")
    print(f"  Total profiles:    {len(profiles):,}")
    print(f"  Avg confidence:    {sum(scores)/len(scores):.1f}")
    print(f"  Drift detected:    {drifts:,} ({drifts/len(profiles)*100:.1f}%)")
    print(f"  Elite reviewers:   {elites:,}")
    print(f"  LLM extraction:    {dict(llm_stats)}")
    print(f"\n  Bucket distribution:")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"    {bucket:<30} {count:>4}")
    print(f"\n  Model used: {GROQ_MODEL}")
    print(f"  Outputs: {args.output_dir}/")
    print("────────────────────────────────────────────────────────────────\n")
 
    if profile_index:
        best = max(profile_index, key=lambda x: x["confidence"])
        print("── SAMPLE PERSONA ──────────────────────────────────────────────")
        print(build_prompt_persona(profiles[best["user_id"]]))
        print("────────────────────────────────────────────────────────────────")
 
 
if __name__ == "__main__":
    main()