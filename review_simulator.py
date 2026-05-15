"""
review_simulator.py
===================
ChopLife — Task A: Review Simulation Agent

Takes a user profile + item profile and produces:
  - Predicted star rating (1-5)
  - Simulated review in the user's specific voice
  - Chain-of-thought reasoning trace
  - Authorship verification score
  - Confidence score with plain-English explanation

Pipeline:
  1. Load user profile (from user_profiles.json)
  2. Load item profile (from item_profiles.json)
  3. Chain-of-thought reasoning pass
  4. Review generation in user's voice
  5. Authorship verification (rewrite if score < 70%)
  6. Confidence scoring

Usage:
  from review_simulator import ReviewSimulator

  sim = ReviewSimulator(
      profiles_path="./data/user_profiles.json",
      items_path="./data/item_profiles.json"
  )

  result = sim.simulate(
      user_id="abc123",
      business_id="xyz789",
      context={"time": "Friday evening", "occasion": "date night"}
  )

  results = sim.evaluate(
      held_out_path="./data/held_out_pairs.json",
      output_path="./data/evaluation_results.json"
  )
"""

import json
import os
import re
import time
import statistics
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, util
from stylometry_extractor import build_stylometry_prompt_section

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

GROQ_MODEL                   = "meta-llama/llama-4-scout-17b-16e-instruct"
AUTHORSHIP_REWRITE_THRESHOLD = 70
MAX_REWRITE_ATTEMPTS         = 2
RATE_LIMIT_WAIT_SECS         = 660
TEMPERATURE_SIMULATE         = 0.7
TEMPERATURE_VERIFY           = 0.1
TEMPERATURE_REASON           = 0.2

# Semantic similarity threshold for trigger matching.
# 0.45 catches paraphrase-level matches ("generous portions" ↔ "large servings")
# without false positives. Lower = more hits but noisier; higher = stricter.
SIMILARITY_THRESHOLD = 0.45
EMBEDDER_MODEL = "all-MiniLM-L6-v2"

DAYS_MAP = {
    "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday",
    "thursday": "Thursday", "friday": "Friday", "saturday": "Saturday",
    "sunday": "Sunday", "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
    "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
}


# ─── DATA CLASSES ─────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    user_id:                  str
    business_id:              str
    predicted_rating:         int
    review_text:              str
    reasoning_trace:          str
    authorship_score:         int
    authorship_feedback:      str
    profile_strength:         int
    profile_strength_label:   str
    profile_strength_reasons: list
    hours_warning:            Optional[str]
    rewrite_count:            int
    trigger_match:            dict
    llm_model:                str
    context:                  dict = field(default_factory=dict)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_groq(client, prompt, temperature=0.3, max_tokens=600) -> str:
    def attempt(retries_left, backoff):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                wait = RATE_LIMIT_WAIT_SECS
                m = re.search(r"try again in (\d+)m(\d+)", error_str)
                if m:
                    wait = int(m.group(1)) * 60 + int(m.group(2)) + 10
                print(f"[RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
                if retries_left > 0:
                    return attempt(retries_left - 1, backoff * 2)
                raise
            if "503" in error_str or "over capacity" in error_str.lower():
                if retries_left > 0:
                    print(f"[503] Backing off {backoff}s...")
                    time.sleep(backoff)
                    return attempt(retries_left - 1, backoff * 2)
                raise
            raise
    return attempt(3, 10)


# ─── HOURS CHECK ──────────────────────────────────────────────────────────────

def check_hours(item: dict, context: dict) -> Optional[str]:
    hours_summary = item.get("hours_summary") or ""
    if not hours_summary or not context:
        return None
    time_context = (context.get("time") or "").lower()
    if not time_context:
        return None
    stated_day = None
    for key, canonical in DAYS_MAP.items():
        if key in time_context:
            stated_day = canonical
            break
    if not stated_day:
        return None
    if stated_day not in hours_summary and stated_day[:3] not in hours_summary:
        return (
            f"Warning: {item.get('name','Restaurant')} may not be open on "
            f"{stated_day} — hours: {hours_summary}. "
            f"Adjust visit context or note this in the simulation."
        )
    return None


# ─── SEMANTIC TRIGGER MATCHING ────────────────────────────────────────────────

def semantic_match(
    user_signals: list,
    item_signals: list,
    embedder:     SentenceTransformer,
    threshold:    float = SIMILARITY_THRESHOLD,
) -> list:
    """
    Match user taste signals against item signals using cosine similarity
    on sentence embeddings.

    Returns the subset of user_signals that semantically match at least
    one item signal above the threshold.

    Why sentence embeddings over exact/fuzzy word matching:
    - User profiles say "tender meat", item profiles say "falls off the bone"
    - User profiles say "generous portions", item profiles say "large servings"
    - Exact matching misses all of these; cosine similarity catches them.
    """
    if not user_signals or not item_signals:
        return []
    try:
        user_emb = embedder.encode(user_signals, convert_to_tensor=True)
        item_emb = embedder.encode(item_signals, convert_to_tensor=True)
        scores   = util.cos_sim(user_emb, item_emb)
        return [
            signal for i, signal in enumerate(user_signals)
            if scores[i].max().item() >= threshold
        ]
    except Exception as e:
        print(f"  [WARN] Semantic matching failed: {e}")
        return []


def compute_trigger_match(
    profile:  dict,
    item:     dict,
    embedder: SentenceTransformer,
) -> dict:
    """
    Match user's praise/complaint/flavour triggers against restaurant signals
    using semantic similarity.

    Rating = base (weighted mean) + praise bonus + flavour bonus - complaint penalty.
    Only complaint triggers relevant to THIS USER are surfaced.
    """
    taste           = profile.get("taste_signals", {})
    user_praises    = [p.lower() for p in taste.get("praise_triggers", [])]
    user_complaints = [c.lower() for c in taste.get("complaint_triggers", [])]
    user_flavours   = [f.lower() for f in taste.get("flavour_preferences", [])]

    item_praised  = [p.lower() for p in item.get("top_praised", [])]
    item_faults   = [f.lower() for f in item.get("top_complaints", [])]
    item_flavours = [f.lower() for f in item.get("flavour_tags", [])]
    item_menu     = [m.lower() for m in item.get("menu_highlights", [])]

    item_positive = item_praised + item_flavours + item_menu
    item_negative = item_faults

    praise_hits    = semantic_match(user_praises,    item_positive, embedder)
    complaint_hits = semantic_match(user_complaints, item_negative, embedder)
    flavour_hits   = semantic_match(user_flavours,   item_positive, embedder)

    base_rating = profile.get("rating_profile", {}).get("weighted_mean", 3.0)

    item_bias = item.get("aggregate_rating", 3.5) - 3.5

    praise_bonus = min(0.6, 0.2 * len(praise_hits))
    flavour_bonus = min(0.4, 0.15 * len(flavour_hits))
    complaint_penalty = min(0.8, 0.25 * len(complaint_hits))

    adjusted = base_rating + item_bias + praise_bonus + flavour_bonus - complaint_penalty
    predicted_raw  = max(1.0, min(5.0, adjusted))
    predicted_star = predicted_raw

    return {
        "base_rating":       round(base_rating, 2),
        "praise_hits":       praise_hits,
        "complaint_hits":    complaint_hits,
        "flavour_hits":      flavour_hits,
        "praise_bonus":      round(praise_bonus, 2),
        "complaint_penalty": round(complaint_penalty, 2),
        "flavour_bonus":     round(flavour_bonus, 2),
        "adjusted_rating":   round(predicted_raw, 2),
        "predicted_star":    predicted_star,
        "explanation": (
            f"Base: {base_rating:.1f}★"
            + (f" +{praise_bonus:.1f} praise match ({', '.join(praise_hits)})"
               if praise_hits else "")
            + (f" +{flavour_bonus:.1f} flavour match ({', '.join(flavour_hits)})"
               if flavour_hits else "")
            + (f" -{complaint_penalty:.1f} complaint match ({', '.join(complaint_hits)})"
               if complaint_hits else "")
            + f" = {predicted_raw:.1f}★ → {predicted_star}★"
        ),
    }


# ─── PROFILE STRENGTH ─────────────────────────────────────────────────────────

def compute_profile_strength(profile: dict, item: dict, trigger_match: dict) -> dict:
    """
    How well do we know this person? INPUT quality metric, not output quality.
    Computed before simulation — represents data richness, not result accuracy.
    """
    score   = 0
    reasons = []

    total = profile.get("rating_profile", {}).get("total_reviews", 0)
    if total >= 50:
        score += 45; reasons.append(f"Rich history — {total} reviews to learn from")
    elif total >= 30:
        score += 35; reasons.append(f"Good history — {total} reviews")
    elif total >= 15:
        score += 22; reasons.append(f"Moderate history — {total} reviews")
    elif total >= 5:
        score += 10; reasons.append(f"Thin history — {total} reviews (cold-start mode)")
    else:
        score += 4;  reasons.append(f"Very thin history — {total} reviews")

    taste = profile.get("taste_signals", {})
    taste_signals_count = sum(
        1 for k in ["praise_triggers", "complaint_triggers", "flavour_preferences"]
        if taste.get(k)
    )
    if taste_signals_count == 3:
        score += 20; reasons.append("Complete taste profile extracted")
    elif taste_signals_count == 2:
        score += 12; reasons.append("Partial taste profile")
    elif taste_signals_count >= 1:
        score += 6;  reasons.append("Minimal taste signals")
    else:
        reasons.append("No taste signals — cold start")

    total_hits = (
        len(trigger_match.get("praise_hits", [])) +
        len(trigger_match.get("flavour_hits", []))
    )
    if total_hits >= 3:
        score += 20; reasons.append(
            f"Strong match — {total_hits} of user's preferences found in this restaurant"
        )
    elif total_hits >= 1:
        score += 10; reasons.append(
            f"Partial match — {total_hits} preference overlap with this restaurant"
        )
    else:
        reasons.append("No direct preference overlap — extrapolating from general taste")

    if profile.get("stylometry", {}).get("available"):
        score += 10; reasons.append("Writing DNA captured — structural simulation possible")
    else:
        reasons.append("No stylometry — vocabulary-only simulation")

    drift = profile.get("drift", {})
    if drift.get("detected"):
        score -= 10
        reasons.append(
            f"Taste drift detected ({drift.get('direction', '')}) — "
            f"older history less reliable"
        )

    score = max(0, min(100, score))
    if score >= 75:
        label = "strong";   summary = "We know this person well"
    elif score >= 50:
        label = "moderate"; summary = "We have a reasonable picture of this person"
    else:
        label = "limited";  summary = "Limited data — simulation is our best estimate"

    return {"score": score, "label": label, "summary": summary, "reasons": reasons}


# ─── PERSONA BUILDER ──────────────────────────────────────────────────────────

def build_persona_string(profile: dict) -> str:
    r  = profile.get("rating_profile", {})
    w  = profile.get("writing_style", {})
    t  = profile.get("taste_signals", {})
    d  = profile.get("drift", {})
    m  = profile.get("user_metadata", {})
    st = profile.get("stylometry", {})

    lines = []

    if m.get("name"):
        lines.append(f"REVIEWER: {m['name']}")
    if m.get("is_elite"):
        lines.append(
            f"Status: Yelp Elite ({len(m.get('elite_years',[]))} years) — "
            f"detailed, opinionated"
        )
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
            f"  ⚠ Recent taste drift ({d['direction']}): "
            f"recent avg {d['recent_mean']:.1f}★ — weight this more heavily"
        )

    lines += ["", "WRITING STYLE:"]
    # sentence length from stylometry (median — robust to outliers)
    if st.get("available") and st.get("sentence_structure", {}).get("median_sentence_length"):
        med = st["sentence_structure"]["median_sentence_length"]
        lines.append(f"  Typical sentence length: {med:.0f} words (median)")
    else:
        lines.append(f"  Typical sentence length: unknown")
    # review length — median preferred over mean (robust to one-off long reviews)
    review_length = w.get("median_length_words", w.get("avg_length_words", 0))
    lines.append(f"  Typical review length: {review_length:.0f} words total (median)")
    lines.append(f"  Tone: {w.get('tone','casual')}")
    if w.get("vocabulary_fingerprint"):
        lines.append(
            f"  Signature vocabulary: "
            f"{', '.join(w['vocabulary_fingerprint'][:12])}"
        )

    stylometry_section = build_stylometry_prompt_section(st)
    if stylometry_section:
        lines += ["", stylometry_section]

    if t and any(t.get(k) for k in ["praise_triggers", "complaint_triggers"]):
        lines += ["", "TASTE PROFILE:"]
        if t.get("top_cuisines"):
            lines.append(f"  Cuisines: {', '.join(t['top_cuisines'])}")
        if t.get("flavour_preferences"):
            lines.append(f"  Flavour preferences: {', '.join(t['flavour_preferences'])}")
        if t.get("praise_triggers"):
            lines.append(f"  Gives high ratings when: {', '.join(t['praise_triggers'])}")
        if t.get("complaint_triggers"):
            lines.append(f"  Drops rating when: {', '.join(t['complaint_triggers'])}")
        dietary = t.get("dietary_context", {})
        if dietary.get("halal_only") and dietary.get("confidence") in ("high", "medium"):
            lines.append("  Dietary: halal only")

    if profile.get("sample_reviews"):
        lines += ["", "REAL EXAMPLES — study the STRUCTURE, not just the words:"]
        high = [s for s in profile["sample_reviews"] if s.get("stars") and s["stars"] >= 4]
        low  = [s for s in profile["sample_reviews"] if s.get("stars") and s["stars"] <= 2]
        tips = [s for s in profile["sample_reviews"] if s.get("source") == "tip"]
        for s in high[:2]:
            lines.append(f'  [HIGH — {s["stars"]}★] "{s["text"]}"')
        for s in low[:1]:
            lines.append(f'  [LOW — {s["stars"]}★] "{s["text"]}"')
        for s in tips[:1]:
            lines.append(f'  [TIP] "{s["text"]}"')

    return "\n".join(lines)


def build_item_context_string(item: dict) -> str:
    lines = [
        f"RESTAURANT: {item['name']}",
        f"Location: {item.get('city','')}, {item.get('state','')}",
    ]
    if item.get("categories"):
        lines.append(f"Type: {', '.join(item['categories'][:4])}")
    if item.get("price_range"):
        label = {1:"budget",2:"mid-range",3:"upscale",4:"luxury"}.get(item["price_range"],"")
        lines.append(f"Price: {'$'*item['price_range']} ({label})")
    lines.append(
        f"Rating: {item.get('aggregate_rating','?')}★ "
        f"({item.get('review_count',0)} reviews)"
    )
    if item.get("is_halal"):
        lines.append("Halal certified: Yes")
    if item.get("is_haram"):
        lines.append(f"Note — haram signals: {', '.join(item.get('haram_signals',[]))}")
    if item.get("menu_highlights"):
        lines.append(f"Known for: {', '.join(item['menu_highlights'][:6])}")
    if item.get("flavour_tags"):
        lines.append(f"Flavour profile: {', '.join(item['flavour_tags'])}")
    if item.get("ambience_tags"):
        lines.append(f"Ambience: {', '.join(item['ambience_tags'])}")
    if item.get("top_praised"):
        lines.append(f"Reviewers love: {', '.join(item['top_praised'])}")
    if item.get("top_complaints"):
        lines.append(f"Common complaints: {', '.join(item['top_complaints'])}")
    if item.get("hours_summary"):
        lines.append(f"Hours: {item['hours_summary']}")
    return "\n".join(lines)


# ─── REASONING PASS ───────────────────────────────────────────────────────────

def run_reasoning_pass(client, persona, item_context, trigger_match, context) -> str:
    context_str = "\n".join([
        f"{k.title()}: {v}" for k, v in context.items() if v
    ]) if context else ""

    prompt = f"""You are a behavioural analyst preparing to simulate a restaurant review.

{persona}

{item_context}

{f"Visit context:{chr(10)}{context_str}" if context_str else ""}

TRIGGER ANALYSIS (already computed — use this):
{trigger_match['explanation']}
Praise triggers matched: {trigger_match['praise_hits'] or 'none'}
Complaint triggers matched (relevant to this user only): {trigger_match['complaint_hits'] or 'none'}
Flavour overlap: {trigger_match['flavour_hits'] or 'none'}
Behavioural prediction: {trigger_match['predicted_star']}★

Based on this, reason through the following:

1. RATING: Confirm or adjust the behavioural prediction. The trigger match gives
   {trigger_match['predicted_star']}★ — does the full context support this?
   If the user would not have experienced the matched triggers, adjust.

2. TONE: How will this person write this review?
   Reference their structural DNA and sample reviews above.
   Story-driven or evaluation-driven? Punchy or winding sentences?
   What do they mention first — atmosphere, food, service, value?

3. KEY TENSIONS: What specific aspects create uncertainty in this prediction?
   Only mention tensions relevant to THIS user's known triggers.

4. HOURS NOTE: {f"⚠ {context.get('_hours_warning', '')}" if context.get('_hours_warning') else "Restaurant hours are compatible with visit context."}

Respond in this format:
RATING: [1-5]
RATING_REASONING: [2-3 sentences grounded in trigger matching]
TONE_PREDICTION: [2 sentences on structural style]
KEY_TENSIONS: [2-3 bullet points — only user-relevant tensions]
CONFIDENCE: [high/medium/low]
CONFIDENCE_REASONING: [1-2 sentences]"""

    return call_groq(client, prompt, temperature=TEMPERATURE_REASON, max_tokens=500)


def parse_reasoning(text: str) -> dict:
    """
    Robust parser for reasoning pass.
    Handles:
    - RATING: 5
    - Rating - 5 stars
    - Predicted rating: 4★
    - Markdown bold variants
    """
    result = {
        "predicted_rating":     3,
        "rating_reasoning":     "",
        "tone_prediction":      "",
        "key_tensions":         "",
        "confidence":           "medium",
        "confidence_reasoning": "",
        "raw":                  text,
    }

    current_key = None
    multiline_fields = {
        "rating_reasoning": [],
        "tone_prediction": [],
        "key_tensions": [],
        "confidence_reasoning": [],
    }

    for raw_line in text.split("\n"):
        line = raw_line.strip().lstrip("*").strip()
        if not line:
            continue

        # ─── FLEXIBLE RATING PARSE ─────────────────────────────
        if "rating" in line.lower():
            rating_match = re.search(
                r'([1-5])\s*(?:★|\*|stars?|star)?',
                line.lower()
            )
            if rating_match:
                try:
                    result["predicted_rating"] = max(
                        1, min(5, int(rating_match.group(1)))
                    )
                except Exception:
                    pass
            current_key = None
            continue

        # ─── SECTION HEADERS ───────────────────────────────────
        elif line.upper().startswith("RATING_REASONING:"):
            current_key = "rating_reasoning"
            content = line.split(":", 1)[1].strip()
            if content:
                multiline_fields[current_key].append(content)

        elif line.upper().startswith("TONE_PREDICTION:"):
            current_key = "tone_prediction"
            content = line.split(":", 1)[1].strip()
            if content:
                multiline_fields[current_key].append(content)

        elif line.upper().startswith("KEY_TENSIONS:"):
            current_key = "key_tensions"
            content = line.split(":", 1)[1].strip()
            if content:
                multiline_fields[current_key].append(content)

        elif line.upper().startswith("CONFIDENCE:"):
            c = line.split(":", 1)[1].strip().lower()
            if c in ("high", "medium", "low"):
                result["confidence"] = c
            current_key = None

        elif line.upper().startswith("CONFIDENCE_REASONING:"):
            current_key = "confidence_reasoning"
            content = line.split(":", 1)[1].strip()
            if content:
                multiline_fields[current_key].append(content)

        # ─── MULTILINE SUPPORT ─────────────────────────────────
        elif current_key:
            multiline_fields[current_key].append(
                line.lstrip("-• ").strip()
            )

    # flatten multiline fields
    for key, values in multiline_fields.items():
        if values:
            result[key] = " ".join(values).strip()

    return result

# ─── REVIEW GENERATION ────────────────────────────────────────────────────────

def generate_review(
    client, persona, item_context, reasoning,
    context, attempt_num=1, previous_feedback=""
) -> str:
    feedback_str = (
        f"\nREWRITE INSTRUCTION (attempt {attempt_num}):\n"
        f"Previous attempt rejected. Fix this specifically: {previous_feedback}"
        if previous_feedback else ""
    )

    prompt = f"""You are simulating a restaurant review written by a specific person.
    Produce a review indistinguishable from what this person actually writes.

    {persona}

    {item_context}

    REASONING:
    Predicted rating: {reasoning['predicted_rating']}★
    Why: {reasoning['rating_reasoning']}
    How they'll write it: {reasoning['tone_prediction']}
    Tensions to reflect: {reasoning['key_tensions']}
    {feedback_str}

    STRICT RULES:
    1. Copy their STRUCTURAL DNA — narrative mode, sentence rhythm, opening/closing style.
    Check their sample reviews and writing DNA section above.
    2. Predicted rating is {reasoning['predicted_rating']}★ — the entire tone must reflect this.
    A 3-star review sounds different from a 5-star review even in the same voice.
    3. Only reference restaurant aspects this user would actually notice based on their
    known triggers. Do not mention complaints they have never shown sensitivity to.
    4. Match their median sentence length AND their typical review length from the profile.
    If they use fragments, use fragments. If they open with action ("I stopped by..."),
    open with action.
    5. Do NOT impose Nigerian language if their history is in standard English.
    Simulate THIS person.

    Respond:
    RATING: {reasoning['predicted_rating']}
    REVIEW: [review text only — no explanation]"""

    return call_groq(client, prompt, temperature=TEMPERATURE_SIMULATE, max_tokens=500)


def parse_review_response(response: str, fallback_rating: int = 3) -> tuple:
    """
    Robust review parser.
    Handles:
    - RATING: 5
    - Rating - 5 stars
    - 5★
    - REVIEW:
    - Missing REVIEW header fallback

    If rating parse fails, falls back to reasoning prediction.
    """
    rating = fallback_rating
    review_lines = []
    found_review = False

    for raw_line in response.strip().split("\n"):
        line = raw_line.strip().lstrip("*").strip()

        if not line:
            continue

        # ─── FLEXIBLE RATING PARSE ─────────────────────────────
        if "rating" in line.lower():
            rating_match = re.search(
                r'([1-5])\s*(?:★|\*|stars?|star)?',
                line.lower()
            )
            if rating_match:
                try:
                    rating = max(1, min(5, int(rating_match.group(1))))
                except Exception:
                    pass
            continue

        # Backup parse for lines like "5 stars"
        elif re.match(r'^[1-5]\s*(?:★|\*|stars?|star)', line.lower()):
            try:
                rating = max(
                    1,
                    min(5, int(re.search(r'[1-5]', line).group()))
                )
            except Exception:
                pass
            continue

        # ─── REVIEW HEADER ─────────────────────────────────────
        elif line.upper().startswith("REVIEW:"):
            found_review = True
            text = line.split(":", 1)[1].strip()
            if text:
                review_lines.append(text)

        # ─── REVIEW BODY ───────────────────────────────────────
        elif found_review:
            review_lines.append(line)

    # If REVIEW header missing, strip likely rating lines and use body
    if not review_lines:
        filtered_lines = []
        for line in response.strip().split("\n"):
            clean = line.strip()
            if not clean:
                continue
            if "rating" in clean.lower():
                continue
            if re.match(r'^[1-5]\s*(?:★|\*|stars?|star)', clean.lower()):
                continue
            filtered_lines.append(clean)

        review_text = " ".join(filtered_lines).strip()
    else:
        review_text = " ".join(review_lines).strip()

    return rating, review_text or response.strip()

# ─── AUTHORSHIP VERIFICATION ──────────────────────────────────────────────────

def verify_authorship(client, persona, review_text) -> dict:
    """
    Checks: does this review sound like this specific person?
    OUTPUT quality metric — separate from Profile Strength (input quality).
    Parser handles markdown-bold responses from llama-4-scout.
    """
    sample_section = ""
    if "REAL EXAMPLES" in persona:
        sample_section = persona.split("REAL EXAMPLES")[1][:800]

    style_section = "\n".join(
        line for line in persona.split("\n")
        if any(k in line for k in [
            "Tone:", "Typical sentence", "Typical review", "Signature vocab",
            "Personality:", "Writing DNA", "Narrative mode", "Structure pattern",
            "Hedging:", "Opens with:", "Closes with:"
        ])
    )

    prompt = f"""Evaluate how well this simulated review matches a specific person's writing style.

    THEIR STYLE PROFILE:
    {style_section}

    THEIR ACTUAL WRITING SAMPLES:
    {sample_section}

    SIMULATED REVIEW TO EVALUATE:
    "{review_text}"

    Score the match on writing style — structure, rhythm, vocabulary, tone, narrative mode.
    NOT on content accuracy.

    SCORE: [0-100]
    VERDICT: [pass/fail — pass if score >= 70]
    FEEDBACK: [specific actionable feedback — what structure/pattern to fix if low]
    STRONGEST_MATCH: [what was done right]
    WEAKEST_MATCH: [what was done wrong]"""

    response = call_groq(client, prompt, temperature=TEMPERATURE_VERIFY, max_tokens=300)

    result = {
        "score": 50, "verdict": "fail",
        "feedback": "", "strongest_match": "", "weakest_match": "", "raw": response,
    }

    current_key   = None
    content_lines = []

    for line in response.split("\n"):
        # strip markdown bold markers that llama-4-scout adds
        line_clean = line.strip().lstrip("*").strip()

        if line_clean.startswith("SCORE:"):
            try:
                result["score"] = max(0, min(100,
                    int(re.search(r"\d+", line_clean).group())
                ))
            except Exception:
                pass
            current_key = None
        elif line_clean.startswith("VERDICT:"):
            result["verdict"] = "pass" if "pass" in line_clean.lower() else "fail"
            current_key = None
        elif line_clean.startswith("FEEDBACK:"):
            current_key   = "feedback"
            content_lines = []
        elif line_clean.startswith("STRONGEST_MATCH:"):
            current_key   = "strongest_match"
            content_lines = []
        elif line_clean.startswith("WEAKEST_MATCH:"):
            current_key   = "weakest_match"
            content_lines = []
        elif current_key and line_clean:
            content_lines.append(line_clean.lstrip("*• ").strip())
            result[current_key] = " ".join(content_lines)

    result["verdict"] = "pass" if result["score"] >= AUTHORSHIP_REWRITE_THRESHOLD else "fail"
    return result


# ─── MAIN SIMULATOR CLASS ─────────────────────────────────────────────────────

class ReviewSimulator:

    def __init__(self, profiles_path: str, items_path: str, api_key: str = None):
        self.profiles = load_json(profiles_path)
        self.items    = load_json(items_path)
        self.client   = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

        # load once — shared across all simulate() calls, no per-call cost
        print(f"Loading semantic embedder ({EMBEDDER_MODEL})...")
        self.embedder = SentenceTransformer(
            EMBEDDER_MODEL,
            token=os.getenv("HF_TOKEN"))

        print(f"ReviewSimulator loaded: "
              f"{len(self.profiles):,} profiles, {len(self.items):,} items")

    def simulate(
        self,
        user_id:     str,
        business_id: str,
        context:     dict = None,
    ) -> SimulationResult:

        context = context or {}

        if user_id not in self.profiles:
            raise ValueError(f"User {user_id} not found")
        if business_id not in self.items:
            raise ValueError(f"Business {business_id} not found")

        profile = self.profiles[user_id]
        item    = self.items[business_id]

        hours_warning = check_hours(item, context)
        if hours_warning:
            print(f"  ⚠ {hours_warning}")
            context["_hours_warning"] = hours_warning

        # semantic trigger matching — embedder passed in, loaded once
        trigger_match    = compute_trigger_match(profile, item, self.embedder)
        profile_strength = compute_profile_strength(profile, item, trigger_match)
        persona          = build_persona_string(profile)
        item_context     = build_item_context_string(item)

        name = profile.get("user_metadata", {}).get("name", "user")
        print(f"  [1/4] Reasoning: {name} visiting {item.get('name','')}...")
        print(f"        Trigger match: {trigger_match['explanation']}")
        reasoning_raw = run_reasoning_pass(
            self.client, persona, item_context, trigger_match, context
        )
        reasoning = parse_reasoning(reasoning_raw)
        raw_pred = trigger_match["predicted_star"]

        # blend reasoning + heuristic
        confidence = reasoning.get("confidence", "medium")

        if confidence == "high":
            alpha = 0.85
        elif confidence == "medium":
            alpha = 0.7
        else:
            alpha = 0.5

        final_pred = (
            alpha * reasoning["predicted_rating"] +
            (1 - alpha) * raw_pred
        )

        # convert to valid star rating
        final_pred = round(final_pred)
        final_pred = max(1, min(5, final_pred))
        print(f"        Predicted: {final_pred}★ (blended) | "
            f"Reasoning confidence: {reasoning['confidence']}")

        print(f"  [2/4] Generating review...")
        best_review     = ""
        best_rating     = final_pred
        best_auth_score = 0
        best_auth       = {}
        rewrite_count   = 0
        prev_feedback   = ""

        for attempt in range(MAX_REWRITE_ATTEMPTS + 1):
            raw = generate_review(
                self.client, persona, item_context, reasoning,
                context, attempt_num=attempt+1,
                previous_feedback=prev_feedback
            )

            # ✅ FIX: extract review text only (ignore parsed rating)
            _, review_text = parse_review_response(raw)

            print(f"  [3/4] Verifying authorship (attempt {attempt+1})...")

            auth = verify_authorship(self.client, persona, review_text)
            print(f"        Authorship score: {auth['score']}% ({auth['verdict']})")

            if auth["score"] > best_auth_score:
                best_review     = review_text
                best_auth_score = auth["score"]
                best_auth       = auth

            if auth["verdict"] == "pass":
                break

            if attempt < MAX_REWRITE_ATTEMPTS:
                rewrite_count += 1
                prev_feedback  = auth["feedback"]
                print(f"        Rewriting — {auth['feedback'][:80]}...")

        print(f"  [4/4] Profile strength: {profile_strength['score']}% "
              f"({profile_strength['label']}) | Authorship: {best_auth_score}%")

        return SimulationResult(
            user_id                  = user_id,
            business_id              = business_id,
            predicted_rating         = best_rating,
            review_text              = best_review,
            reasoning_trace          = reasoning_raw,
            authorship_score         = best_auth_score,
            authorship_feedback      = best_auth.get("strongest_match", ""),
            profile_strength         = profile_strength["score"],
            profile_strength_label   = profile_strength["label"],
            profile_strength_reasons = profile_strength["reasons"],
            hours_warning            = hours_warning,
            rewrite_count            = rewrite_count,
            trigger_match            = trigger_match,
            llm_model                = GROQ_MODEL,
            context                  = {k: v for k, v in context.items()
                                        if not k.startswith("_")},
        )

    def evaluate(
        self,
        held_out_path: str,
        output_path:   str = None,
    ) -> list:
        """
        Evaluate against held-out pairs. Computes RMSE on rating and
        authorship scores across the set.

        held_out_pairs.json format:
        [
          {
            "user_id": "...",
            "business_id": "...",
            "actual_rating": 4,
            "actual_text": "...",
            "context": {}
          }
        ]
        """
        with open(held_out_path) as f:
            pairs = json.load(f)

        results       = []
        rating_errors = []

        for i, pair in enumerate(pairs):
            print(f"\n[{i+1}/{len(pairs)}] Evaluating "
                  f"{pair['user_id'][:8]}... × {pair['business_id'][:8]}...")
            try:
                result = self.simulate(
                    user_id     = pair["user_id"],
                    business_id = pair["business_id"],
                    context     = pair.get("context", {}),
                )
                error = result.predicted_rating - pair["actual_rating"]
                rating_errors.append(error ** 2)
                results.append({
                    "user_id":                result.user_id,
                    "business_id":            result.business_id,
                    "predicted_rating":       result.predicted_rating,
                    "actual_rating":          pair["actual_rating"],
                    "rating_error":           error,
                    "review_text":            result.review_text,
                    "actual_text":            pair.get("actual_text", ""),
                    "authorship_score":       result.authorship_score,
                    "profile_strength":       result.profile_strength,
                    "profile_strength_label": result.profile_strength_label,
                    "rewrite_count":          result.rewrite_count,
                    "trigger_match":          result.trigger_match,
                })
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append({
                    "user_id":     pair["user_id"],
                    "business_id": pair["business_id"],
                    "error":       str(e),
                })

        if rating_errors:
            rmse        = (sum(rating_errors) / len(rating_errors)) ** 0.5
            auth_scores = [r["authorship_score"] for r in results
                           if "authorship_score" in r]
            print(f"\n{'='*60}")
            print(f"EVALUATION RESULTS")
            print(f"{'='*60}")
            print(f"  Pairs evaluated:      {len(results)}")
            print(f"  RMSE (rating):        {rmse:.3f}")
            print(f"  Avg authorship score: {statistics.mean(auth_scores):.1f}%")
            print(f"  Median authorship:    {statistics.median(auth_scores):.1f}%")
            print(f"  Pass rate (≥70%):     "
                  f"{sum(1 for s in auth_scores if s >= 70)/len(auth_scores)*100:.1f}%")

            if output_path:
                with open(output_path, "w") as f:
                    json.dump({
                        "results": results,
                        "summary": {
                            "rmse":           round(rmse, 4),
                            "avg_authorship": round(statistics.mean(auth_scores), 1),
                            "model":          GROQ_MODEL,
                            "embedder":       EMBEDDER_MODEL,
                            "sim_threshold":  SIMILARITY_THRESHOLD,
                        }
                    }, f, indent=2)
                print(f"\n  Results saved to {output_path}")

        return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, dataclasses

    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles",    required=True)
    parser.add_argument("--items",       required=True)
    parser.add_argument("--user_id",     default=None)
    parser.add_argument("--business_id", default=None)
    parser.add_argument("--time",        default="")
    parser.add_argument("--occasion",    default="")
    parser.add_argument("--evaluate",    default=None,
                        help="Path to held_out_pairs.json for evaluation mode")
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    sim = ReviewSimulator(profiles_path=args.profiles, items_path=args.items)

    if args.evaluate:
        sim.evaluate(held_out_path=args.evaluate, output_path=args.output)

    elif args.user_id and args.business_id:
        context = {}
        if args.time:     context["time"]     = args.time
        if args.occasion: context["occasion"] = args.occasion

        result = sim.simulate(
            user_id=args.user_id, business_id=args.business_id, context=context
        )

        print("\n" + "═"*60)
        print("CHOPLIFE — SIMULATION RESULT")
        print("═"*60)
        print(f"\nPredicted rating:  {'★'*result.predicted_rating}{'☆'*(5-result.predicted_rating)}")
        print(f"Profile strength:  {result.profile_strength}% ({result.profile_strength_label})")
        print(f"  → {result.profile_strength_reasons[0] if result.profile_strength_reasons else ''}")
        print(f"Authorship score:  {result.authorship_score}% (how well we simulated their voice)")
        if result.hours_warning:
            print(f"\n⚠ Hours warning: {result.hours_warning}")
        print(f"\nTrigger analysis:  {result.trigger_match['explanation']}")
        print(f"\nSimulated review:")
        print(f'  "{result.review_text}"')
        print(f"\n--- REASONING TRACE ---")
        print(result.reasoning_trace)
        print("═"*60)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(dataclasses.asdict(result), f, indent=2)
            print(f"Saved to {args.output}")