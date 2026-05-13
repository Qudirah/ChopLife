"""
item_profile_builder.py
=======================
ChopLife — Yelp Item Profile Builder (Task A)

Builds structured item profiles for every Yelp business that at least
one of our sampled users has reviewed. These profiles are what the Task A
review simulation agent uses alongside the user persona.

LLM mining (Groq llama-4-scout) extracts per business:
  - top_praised        what reviewers consistently love
  - top_complaints     what reviewers consistently dislike
  - menu_highlights    dishes and drinks mentioned most
  - flavour_tags       flavour qualities that appear in positive reviews
  - ambience_tags      atmosphere descriptors

Everything else (name, categories, price range, attributes, hours,
aggregate rating) comes directly from the Yelp business file.

Output:
  data/item_profiles.json          full item profiles keyed by business_id
  data/item_index.json             lightweight lookup (no LLM fields)
  data/item_checkpoint.json        tracks LLM-processed business IDs

Usage:
  # Full run
  python item_profile_builder.py --data_dir "./Yelp JSON/yelp_dataset" --output_dir ./data

  # Resume after interruption
  python item_profile_builder.py --data_dir "./Yelp JSON/yelp_dataset" --output_dir ./data --resume

  # No LLM — heuristic signals only (fast, for testing)
  python item_profile_builder.py --data_dir "./Yelp JSON/yelp_dataset" --output_dir ./data --no_llm

  # Single business test
  python item_profile_builder.py --data_dir "./Yelp JSON/yelp_dataset" --output_dir ./data --business_id <id>

"""

import json
import os
import re
import time
import argparse
import statistics
from collections import defaultdict, Counter
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from groq import Groq
 
load_dotenv()
 
# ─── CONFIG ───────────────────────────────────────────────────────────────────
 
GROQ_MODEL            = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_REVIEWS_FOR_LLM   = 20       # reviews to send per business
MAX_CHARS_PER_REVIEW  = 150      # truncate each — breadth over depth
MAX_TIPS_FOR_LLM      = 10       # tips are short, include more
RATE_LIMIT_WAIT_SECS  = 660
MIN_REVIEWS_FOR_LLM   = 3        # skip LLM if fewer reviews than this
MIN_SAMPLED_USERS     = 4        # skip LLM if fewer sampled users reviewed
                                  # Distribution from data:
                                  #   1 user:    2,432 businesses → heuristic
                                  #   2-3 users: 1,495 businesses → heuristic
                                  #   4-10 users: 1,020 businesses → LLM
                                  #   10+ users:   248 businesses → LLM
                                  # Total LLM scope: ~1,268 businesses
 
# ── Category filter — same as profile extractor ───────────────────────────────
PRIMARY_FOOD_CATS   = {"restaurants", "food"}
SECONDARY_FOOD_CATS = {
    "bakeries", "cafes", "coffee & tea", "food trucks", "bubble tea",
    "juice bars & smoothies", "desserts", "ice cream & frozen yogurt",
    "delis", "diners", "buffets", "street vendors", "grocery",
    "specialty food", "beer bar", "wine bars",
}
 
 
# ─── HELPERS ──────────────────────────────────────────────────────────────────
 
def load_jsonl(filepath: str) -> list:
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
 
 
def load_checkpoint(path: str) -> set:
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f).get("completed", []))
    return set()
 
 
def save_checkpoint(path: str, completed: set) -> None:
    with open(path, "w") as f:
        json.dump({"completed": list(completed), "count": len(completed)}, f)
 
 
def load_existing(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
 
 
def truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    last = max(chunk.rfind(". "), chunk.rfind("! "), chunk.rfind("? "))
    if last > max_chars // 2:
        return chunk[:last + 1].strip()
    last_space = chunk.rfind(" ")
    return (chunk[:last_space].strip() + "...") if last_space > 0 else chunk + "..."
 
 
# ─── ATTRIBUTE PARSING ────────────────────────────────────────────────────────
 
def parse_attributes(raw_attrs: dict) -> dict:
    """
    Clean and normalise Yelp business attributes into a usable dict.
    Yelp attributes are inconsistently formatted — string booleans,
    unicode artifacts, nested dicts serialised as strings.
    """
    if not raw_attrs:
        return {}
 
    def clean_val(v):
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in ("true", "yes"):
            return True
        if s in ("false", "no", "none"):
            return False
        # strip Yelp unicode artifacts like u'free'
        s = re.sub(r"^u'|^u\"|'$|\"$", "", s).strip("'\"")
        return s if s else None
 
    cleaned = {}
 
    field_map = {
        "RestaurantsPriceRange2":    "price_range",
        "OutdoorSeating":            "outdoor_seating",
        "WiFi":                      "wifi",
        "Alcohol":                   "alcohol",
        "RestaurantsReservations":   "takes_reservations",
        "GoodForGroups":             "good_for_groups",
        "NoiseLevel":                "noise_level",
        "RestaurantsGoodForGroups":  "good_for_groups",
        "HasTV":                     "has_tv",
        "RestaurantsTakeOut":        "takeout",
        "RestaurantsDelivery":       "delivery",
        "Caters":                    "catering",
        "GoodForKids":               "good_for_kids",
        "BikeParking":               "bike_parking",
        "WheelchairAccessible":      "wheelchair_accessible",
        "HappyHour":                 "happy_hour",
        "DogsAllowed":               "dogs_allowed",
        "DriveThru":                 "drive_thru",
    }
 
    for yelp_key, clean_key in field_map.items():
        if yelp_key in raw_attrs:
            val = clean_val(raw_attrs[yelp_key])
            if val is not None:
                cleaned[clean_key] = val
 
    # price range as int
    if "price_range" in cleaned:
        try:
            cleaned["price_range"] = int(cleaned["price_range"])
        except (ValueError, TypeError):
            cleaned.pop("price_range", None)
 
    return cleaned
 
 
def parse_hours(raw_hours: dict) -> str:
    """Produce a compact hours summary string."""
    if not raw_hours:
        return None
    days = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
    parts = []
    for day in days:
        if day in raw_hours:
            parts.append(f"{day[:3]}: {raw_hours[day]}")
    return ", ".join(parts) if parts else None
 
 
# ─── HEURISTIC SIGNAL EXTRACTION ──────────────────────────────────────────────
 
def heuristic_signals(reviews: list, tips: list) -> dict:
    """
    Fast keyword-based signal extraction when LLM is skipped or
    as a fallback. Cruder than LLM but always available.
    """
    PRAISE_WORDS = [
        "amazing", "excellent", "perfect", "best", "love", "delicious",
        "great", "fantastic", "wonderful", "outstanding", "fresh",
        "crispy", "tender", "flavourful", "flavorful", "authentic",
        "generous", "smoky", "rich", "juicy", "cozy", "friendly",
    ]
    COMPLAINT_WORDS = [
        "slow", "wait", "cold", "bland", "overpriced", "small",
        "rude", "disappointing", "terrible", "awful", "dry", "tough",
        "greasy", "loud", "noisy", "crowded", "dirty", "undercooked",
        "expensive", "mediocre", "inconsistent",
    ]
 
    praise_counts    = Counter()
    complaint_counts = Counter()
 
    for r in reviews:
        text = (r.get("text") or "").lower()
        stars = r.get("stars", 3)
        if stars >= 4:
            for w in PRAISE_WORDS:
                if w in text:
                    praise_counts[w] += 1
        elif stars <= 2:
            for w in COMPLAINT_WORDS:
                if w in text:
                    complaint_counts[w] += 1
 
    # mine potential dish names from tips (short, noun-heavy)
    dish_candidates = Counter()
    for t in tips:
        words = (t.get("text") or "").lower().split()
        for i, w in enumerate(words):
            if len(w) > 3 and w.isalpha():
                dish_candidates[w] += 1
 
    return {
        "top_praised":     [w for w, _ in praise_counts.most_common(5)],
        "top_complaints":  [w for w, _ in complaint_counts.most_common(5)],
        "menu_highlights": [w for w, _ in dish_candidates.most_common(8)],
        "flavour_tags":    [w for w, _ in praise_counts.most_common(3)],
        "ambience_tags":   [],
    }
 
 
# ─── LLM SIGNAL EXTRACTION ────────────────────────────────────────────────────
 
def extract_signals_llm(
    business:    dict,
    reviews:     list,
    tips:        list,
    groq_client: Groq,
) -> tuple:
    """
    Use Groq LLM to extract rich signals from a sample of reviews + tips.
    Returns (signals_dict, status).
 
    Sends more reviews per call than user profile extraction because we
    need aggregate signals across many different reviewers, not one person's
    behavioural pattern.
    """
    # sample: mix of high, low, and recent reviews for breadth
    sorted_by_stars = sorted(
        [r for r in reviews if r.get("text", "").strip()],
        key=lambda r: r.get("stars", 3)
    )
    low_reviews  = sorted_by_stars[:3]
    high_reviews = sorted_by_stars[-10:]
    mid_reviews  = sorted_by_stars[len(sorted_by_stars)//2 - 3:
                                   len(sorted_by_stars)//2 + 4]
 
    seen = set()
    sample_reviews = []
    for r in high_reviews + mid_reviews + low_reviews:
        rid = r.get("review_id", r.get("text","")[:30])
        if rid not in seen:
            seen.add(rid)
            sample_reviews.append(r)
        if len(sample_reviews) >= MAX_REVIEWS_FOR_LLM:
            break
 
    tip_sample = sorted(
        [t for t in tips if t.get("text","").strip()],
        key=lambda t: t.get("date",""),
        reverse=True
    )[:MAX_TIPS_FOR_LLM]
 
    formatted = []
    for r in sample_reviews:
        formatted.append(
            f"[{r.get('stars')}★] "
            f"{truncate_at_sentence(r.get('text',''), MAX_CHARS_PER_REVIEW)}"
        )
    for t in tip_sample:
        formatted.append(
            f"[TIP] {truncate_at_sentence(t.get('text',''), MAX_CHARS_PER_REVIEW)}"
        )
 
    biz_name = business.get("name", "this restaurant")
    cats     = business.get("categories", "")
 
    prompt = f"""You are analysing customer reviews for "{biz_name}" ({cats}).
 
Reviews and tips:
{chr(10).join(formatted)}
 
Extract aggregate signals and return ONLY valid JSON — no preamble, no markdown:
 
{{
  "top_praised": ["up to 5 specific things reviewers consistently love"],
  "top_complaints": ["up to 5 specific things reviewers consistently dislike"],
  "menu_highlights": ["up to 8 dishes, drinks, or menu items mentioned most"],
  "flavour_tags": ["up to 5 flavour or food quality descriptors from positive reviews"],
  "ambience_tags": ["up to 4 atmosphere or vibe descriptors"]
}}
 
Rules:
- Be specific — "wood-fired crust" not just "pizza"
- Use language from the actual reviews
- Only include signals that appear more than once
- If a field has insufficient signal, return an empty list"""
 
    def call_api():
        return groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
 
    def parse_response(resp):
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
 
    def attempt(retries_left: int, backoff: int) -> tuple:
        try:
            return parse_response(call_api()), "complete"
 
        except json.JSONDecodeError as e:
            raw = getattr(e, "doc", "") or ""
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start:end+1]), "complete"
                except Exception:
                    pass
            return heuristic_signals(reviews, tips), "fallback"
 
        except Exception as e:
            error_str = str(e)
 
            if "429" in error_str or "rate limit" in error_str.lower():
                wait = RATE_LIMIT_WAIT_SECS
                m = re.search(r"try again in (\d+)m(\d+)", error_str)
                if m:
                    wait = int(m.group(1)) * 60 + int(m.group(2)) + 10
                else:
                    m = re.search(r"try again in ([\d.]+)s", error_str)
                    if m:
                        wait = int(float(m.group(1))) + 5
                print(f"\n  [RATE LIMIT 429] Waiting {wait}s. "
                      f"Ctrl+C then --resume if needed.")
                time.sleep(wait)
                if retries_left > 0:
                    return attempt(retries_left - 1, backoff * 2)
                return heuristic_signals(reviews, tips), "fallback"
 
            if "503" in error_str or "over capacity" in error_str.lower():
                if retries_left > 0:
                    print(f"  [503] Backing off {backoff}s, "
                          f"{retries_left} retries left...")
                    time.sleep(backoff)
                    return attempt(retries_left - 1, backoff * 2)
                return heuristic_signals(reviews, tips), "fallback"
 
            return heuristic_signals(reviews, tips), "fallback"
 
    return attempt(retries_left=3, backoff=10)
 
 
# ─── ITEM PROFILE BUILDER ─────────────────────────────────────────────────────
 
def build_item_profile(
    business:    dict,
    reviews:     list,
    tips:        list,
    groq_client: Groq,
    use_llm:     bool = True,
) -> dict:
    """
    Build a structured item profile for a single Yelp business.
 
    Combines:
    - Static data from business.json (name, location, attributes, hours)
    - Aggregate rating stats from review data
    - LLM-mined signals (top_praised, complaints, menu, flavour, ambience)
    """
    bid = business["business_id"]
 
    # ── static business data ────────────────────────────────────────────────
    attrs    = parse_attributes(business.get("attributes") or {})
    hours    = parse_hours(business.get("hours") or {})
    cats_raw = business.get("categories") or ""
    cats     = [c.strip() for c in cats_raw.split(",") if c.strip()]
 
    # price range — from attributes first, fallback to None
    price_range = attrs.pop("price_range", None)
 
    # Halal/haram business classification
    # We do not attempt to certify halal — many compliant businesses
    # are unaware of the designation. Instead we flag haram signals.
    # If none are present the business is neutral (not confirmed halal,
    # not confirmed haram). Recommendation filter only excludes on
    # confirmed haram signals, not absence of halal certification.
    cats_lower = cats_raw.lower()
    attrs_str  = json.dumps(business.get("attributes") or {}).lower()
 
    haram_signals = []
    # alcohol in attributes
    alcohol_val = (business.get("attributes") or {}).get("Alcohol", "")
    if alcohol_val and str(alcohol_val).lower() not in ("none", "false", "no", "u'none'", ""):
        haram_signals.append("alcohol_served")
    # alcohol/pork/beer/wine in category tags
    haram_cats = {"bars", "beer", "wine", "cocktail", "pub", "brewery",
                  "pork", "bacon", "ham"}
    for hc in haram_cats:
        if hc in cats_lower:
            haram_signals.append(f"category:{hc}")
            break
    is_halal  = "halal" in cats_lower   # explicit halal certification only
    is_haram  = len(haram_signals) > 0
 
    # ── aggregate rating stats ───────────────────────────────────────────────
    stars_list = [r["stars"] for r in reviews if "stars" in r]
    if stars_list:
        agg_rating = round(statistics.mean(stars_list), 2)
        rating_dist = {
            str(k): stars_list.count(k) for k in range(1, 6)
        }
    else:
        # fall back to business-level aggregate
        agg_rating  = business.get("stars")
        rating_dist = {}
 
    # ── LLM or heuristic signal extraction ──────────────────────────────────
    llm_status = "skipped"
    if use_llm and len(reviews) >= MIN_REVIEWS_FOR_LLM and groq_client:
        signals, llm_status = extract_signals_llm(
            business, reviews, tips, groq_client
        )
    elif reviews:
        signals    = heuristic_signals(reviews, tips)
        llm_status = "heuristic"
    else:
        signals = {
            "top_praised":    [],
            "top_complaints": [],
            "menu_highlights":[],
            "flavour_tags":   [],
            "ambience_tags":  [],
        }
        llm_status = "skipped"
 
    return {
        "business_id":    bid,
        "name":           business.get("name", ""),
        "city":           business.get("city", ""),
        "state":          business.get("state", ""),
        "categories":     cats,
        "price_range":    price_range,
        "is_open":        bool(business.get("is_open", 0)),
        "is_halal":       is_halal,   # explicit halal certification in categories
        "is_haram":       is_haram,   # confirmed haram signal detected
        "haram_signals":  haram_signals,  # what triggered the haram flag
        "aggregate_rating": agg_rating,
        "review_count":   len(reviews),
        "rating_distribution": rating_dist,
        "attributes":     attrs,
        "hours_summary":  hours,
        # LLM-mined signals
        "top_praised":    signals.get("top_praised", []),
        "top_complaints": signals.get("top_complaints", []),
        "menu_highlights":signals.get("menu_highlights", []),
        "flavour_tags":   signals.get("flavour_tags", []),
        "ambience_tags":  signals.get("ambience_tags", []),
        "llm_status":     llm_status,
        "llm_model":      GROQ_MODEL if llm_status == "complete" else None,
        "profile_built":  datetime.now().isoformat(),
    }
 
 
def build_item_prompt_context(profile: dict) -> str:
    """
    Render an item profile as a plain-English context string
    ready to be injected into a Task A simulation prompt.
    """
    lines = [
        f"RESTAURANT: {profile['name']}",
        f"Location: {profile['city']}, {profile['state']}",
    ]
 
    if profile.get("categories"):
        lines.append(f"Type: {', '.join(profile['categories'][:4])}")
 
    if profile.get("price_range"):
        # $ symbol — these are American Yelp businesses
        # Nigerian restaurant pool (Task B) uses ₦ in its own builder
        price_str = "$" * int(profile["price_range"])
        lines.append(f"Price range: {price_str} ({profile['price_range']}/4 — "
                     f"{'budget' if profile['price_range'] == 1 else 'mid-range' if profile['price_range'] == 2 else 'upscale' if profile['price_range'] == 3 else 'luxury'})")
 
    lines.append(
        f"Aggregate rating: {profile['aggregate_rating']}★ "
        f"({profile['review_count']} reviews)"
    )
 
    if profile.get("is_halal"):
        lines.append("Halal certified: Yes")
 
    if profile.get("attributes"):
        attrs = profile["attributes"]
        attr_parts = []
        if attrs.get("outdoor_seating"):
            attr_parts.append("outdoor seating")
        if attrs.get("takes_reservations"):
            attr_parts.append("takes reservations")
        if attrs.get("alcohol") and attrs["alcohol"] not in (False, "none"):
            attr_parts.append(f"alcohol: {attrs['alcohol']}")
        if attrs.get("noise_level"):
            attr_parts.append(f"noise: {attrs['noise_level']}")
        if attr_parts:
            lines.append(f"Attributes: {', '.join(attr_parts)}")
 
    if profile.get("menu_highlights"):
        lines.append(f"Known for: {', '.join(profile['menu_highlights'][:6])}")
 
    if profile.get("flavour_tags"):
        lines.append(f"Flavour profile: {', '.join(profile['flavour_tags'])}")
 
    if profile.get("ambience_tags"):
        lines.append(f"Ambience: {', '.join(profile['ambience_tags'])}")
 
    if profile.get("top_praised"):
        lines.append(f"Reviewers love: {', '.join(profile['top_praised'])}")
 
    if profile.get("top_complaints"):
        lines.append(f"Common complaints: {', '.join(profile['top_complaints'])}")
 
    return "\n".join(lines)
 
 
# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--business_id", default=None,
                        help="Process a single business (for testing)")
    parser.add_argument("--no_llm",      action="store_true",
                        help="Use heuristic extraction only (fast, for testing)")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip businesses already in checkpoint")
    parser.add_argument("--min_users",   type=int, default=MIN_SAMPLED_USERS,
                        help=f"Min sampled users who reviewed a business for LLM "
                             f"extraction (default: {MIN_SAMPLED_USERS}). "
                             f"Below this threshold heuristic is used. "
                             f"Use --min_users 10 to process only high-coverage "
                             f"businesses first (248 businesses, ~112K tokens).")
    args = parser.parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)
 
    profiles_path   = os.path.join(args.output_dir, "item_profiles.json")
    index_path      = os.path.join(args.output_dir, "item_index.json")
    checkpoint_path = os.path.join(args.output_dir, "item_checkpoint.json")
 
    # ── Init Groq ────────────────────────────────────────────────────────────
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key and not args.no_llm:
        raise EnvironmentError("GROQ_API_KEY not in .env. Use --no_llm to skip.")
    groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
 
    # ── Resume ───────────────────────────────────────────────────────────────
    completed_ids     = load_checkpoint(checkpoint_path) if args.resume else set()
    existing_profiles = load_existing(profiles_path)     if args.resume else {}
    if args.resume:
        print(f"\n[RESUME] {len(completed_ids):,} already done, "
              f"{len(existing_profiles):,} profiles loaded")
 
    # ── [1/5] Load businesses ─────────────────────────────────────────────────
    print("\n[1/5] Loading businesses...")
    biz_path = os.path.join(args.data_dir, "yelp_academic_dataset_business.json")
    all_biz  = load_jsonl(biz_path)
    food_biz_list = [b for b in all_biz if is_food_business(b)]
    for b in food_biz_list:
        b["is_halal"] = "halal" in (b.get("categories") or "").lower()
    food_biz = {b["business_id"]: b for b in food_biz_list}
    print(f"  {len(all_biz):,} total → {len(food_biz):,} food businesses")
 
    # ── [2/5] Load sampled user profiles to find reviewed businesses ──────────
    print("\n[2/5] Loading sampled user profiles...")
    user_profiles_path = os.path.join(args.output_dir, "user_profiles.json")
    if not os.path.exists(user_profiles_path):
        raise FileNotFoundError(
            f"user_profiles.json not found at {user_profiles_path}. "
            "Run profile_extractor_v5.py first."
        )
    with open(user_profiles_path) as f:
        user_profiles = json.load(f)
    print(f"  {len(user_profiles):,} user profiles loaded")
 
    # ── [3/5] Load reviews — food businesses only ─────────────────────────────
    print("\n[3/5] Loading reviews...")
    review_path    = os.path.join(args.data_dir, "yelp_academic_dataset_review.json")
    reviews_raw    = load_jsonl(review_path)
    biz_reviews    = defaultdict(list)   # business_id -> list of reviews
    sampled_uids   = set(user_profiles.keys())
 
    # collect ALL reviews for businesses our sampled users have reviewed
    # we need the full review pool per business, not just sampled-user reviews,
    # to get good aggregate signals (top_praised, complaints, menu highlights)
    sampled_biz_ids = set()
    for r in reviews_raw:
        if (r.get("user_id") in sampled_uids
                and r.get("business_id") in food_biz):
            sampled_biz_ids.add(r["business_id"])
 
    # count how many sampled users reviewed each business
    # used to decide LLM vs heuristic per business
    biz_sampled_user_count = Counter()
    for r in reviews_raw:
        if (r.get("user_id") in sampled_uids
                and r.get("business_id") in sampled_biz_ids):
            biz_sampled_user_count[r["business_id"]] += 1
 
    # second pass — collect all reviews for those businesses
    for r in reviews_raw:
        if r.get("business_id") in sampled_biz_ids:
            biz_reviews[r["business_id"]].append(r)
 
    llm_eligible = sum(
        1 for c in biz_sampled_user_count.values()
        if c >= args.min_users
    )
    print(f"  {len(sampled_biz_ids):,} unique businesses reviewed by sampled users")
    print(f"  {sum(len(v) for v in biz_reviews.values()):,} total reviews across those businesses")
    print(f"  {llm_eligible:,} businesses meet --min_users {args.min_users} threshold for LLM")
    print(f"  {len(sampled_biz_ids) - llm_eligible:,} businesses will use heuristic extraction")
 
    # ── [4/5] Load tips ───────────────────────────────────────────────────────
    print("\n[4/5] Loading tips...")
    tip_path  = os.path.join(args.data_dir, "yelp_academic_dataset_tip.json")
    tips_raw  = load_jsonl(tip_path)
    biz_tips  = defaultdict(list)
    for t in tips_raw:
        if t.get("business_id") in sampled_biz_ids:
            biz_tips[t["business_id"]].append(t)
    print(f"  {sum(len(v) for v in biz_tips.values()):,} tips across target businesses")
 
    # ── [5/5] Determine target businesses ────────────────────────────────────
    print("\n[5/5] Building item profiles...")
    if args.business_id:
        target_ids = [args.business_id]
    else:
        target_ids = list(sampled_biz_ids)
 
    if args.resume:
        pending = [bid for bid in target_ids if bid not in completed_ids]
        print(f"  Pending after resume filter: {len(pending):,}")
    else:
        pending = target_ids
 
    use_llm = not args.no_llm
    print(f"  Processing {len(pending):,} businesses "
          f"({'with LLM' if use_llm else 'heuristic only'}) "
          f"— model: {GROQ_MODEL if use_llm else 'none'}")
 
    # ── Build profiles ────────────────────────────────────────────────────────
    profiles  = dict(existing_profiles)
    llm_stats = Counter()
 
    for bid in tqdm(pending, desc="Building item profiles"):
        business = food_biz.get(bid)
        if not business:
            continue
 
        sampled_user_count = biz_sampled_user_count.get(bid, 0)
        # only use LLM if business meets min_users threshold
        use_llm_for_this = use_llm and sampled_user_count >= args.min_users
        profile = build_item_profile(
            business    = business,
            reviews     = biz_reviews.get(bid, []),
            tips        = biz_tips.get(bid, []),
            groq_client = groq_client,
            use_llm     = use_llm_for_this,
        )
        # store sampled user count for reference
        profile["sampled_user_count"] = sampled_user_count
        profiles[bid] = profile
        llm_stats[profile["llm_status"]] += 1
 
        if use_llm and profile["llm_status"] == "complete":
            completed_ids.add(bid)
            save_checkpoint(checkpoint_path, completed_ids)
 
    # ── Build item index ──────────────────────────────────────────────────────
    item_index = [
        {
            "business_id":      bid,
            "name":             p["name"],
            "city":             p["city"],
            "state":            p["state"],
            "categories":       p["categories"][:3],
            "price_range":      p["price_range"],
            "aggregate_rating": p["aggregate_rating"],
            "review_count":     p["review_count"],
            "is_halal":         p["is_halal"],
            "is_haram":         p.get("is_haram", False),
            "haram_signals":    p.get("haram_signals", []),
            "is_open":          p["is_open"],
            "llm_status":          p["llm_status"],
            "sampled_user_count":  p.get("sampled_user_count", 0),
        }
        for bid, p in profiles.items()
    ]
 
    # ── Write outputs ─────────────────────────────────────────────────────────
    with open(profiles_path, "w") as f:
        json.dump(profiles, f, indent=2)
    with open(index_path, "w") as f:
        json.dump(item_index, f, indent=2)
 
    # ── Summary ───────────────────────────────────────────────────────────────
    halal_count  = sum(1 for p in profiles.values() if p.get("is_halal"))
    open_count   = sum(1 for p in profiles.values() if p.get("is_open"))
    cities       = Counter(p.get("city","") for p in profiles.values())
    price_dist   = Counter(
        p.get("price_range") for p in profiles.values()
        if p.get("price_range")
    )
 
    print("\n── COMPLETE ────────────────────────────────────────────────────")
    print(f"  Item profiles built:   {len(profiles):,}")
    print(f"  LLM extraction:        {dict(llm_stats)}")
    print(f"  Halal businesses:      {halal_count:,}")
    print(f"  Currently open:        {open_count:,}")
    print(f"  Price distribution:    {dict(price_dist)}")
    print(f"  Top cities:            "
          f"{', '.join(f'{c}({n})' for c,n in cities.most_common(5))}")
    print(f"  Outputs: {args.output_dir}/")
    print("────────────────────────────────────────────────────────────────\n")
 
    # ── Sample item context ───────────────────────────────────────────────────
    if profiles:
        # find a business with the most reviews for a clean demo
        sample_bid = max(
            profiles,
            key=lambda b: profiles[b].get("review_count", 0)
        )
        sample = profiles[sample_bid]
        print("── SAMPLE ITEM CONTEXT ─────────────────────────────────────────")
        print(build_item_prompt_context(sample))
        print("────────────────────────────────────────────────────────────────")
 
 
if __name__ == "__main__":
    main()
 