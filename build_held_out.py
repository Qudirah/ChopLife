"""
build_held_out.py
=================
ChopLife — Task A Evaluation Dataset Builder

Builds held_out_pairs.json for evaluating the review simulator.

Strategy:
  For each sampled user, hold out their LAST review on a business
  that has an item profile. This gives us ground truth:
    - actual_rating: what they really gave
    - actual_text: what they really wrote
  We then simulate that review and compare.

Selection criteria:
  - User must be in user_profiles.json (sampled)
  - Business must be in item_profiles.json (has item profile)
  - Review must have text (need actual_text for BERTScore)
  - Spread across user types: warm generous, warm balanced,
    warm harsh, thin, very_thin
  - Target: 50-60 pairs minimum

Output:
  data/held_out_pairs.json

Usage:
  python build_held_out.py \
    --data_dir "./Yelp JSON/yelp_dataset" \
    --output_dir ./data \
    --target 60
"""

import json
import os
import argparse
import random
from collections import defaultdict, Counter
from tqdm import tqdm

RANDOM_SEED = 42

# target pairs per bucket — spread ensures all user types evaluated
BUCKET_TARGETS = {
    "generous_casual":  10,
    "generous_formal":   2,
    "balanced_casual":  12,
    "balanced_formal":   4,
    "harsh_casual":      6,
    "harsh_formal":      1,
    "thin":              10,
    "very_thin":          5,
}


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def get_bucket(profile: dict) -> str:
    """Map a user profile to its evaluation bucket."""
    count    = profile.get("meta", {}).get("total_reviews", 0)
    tendency = profile.get("rating_profile", {}).get("tendency", "balanced")
    tone     = profile.get("writing_style", {}).get("tone", "casual")

    if count < 5:
        return "very_thin"
    if count < 15:
        return "thin"
    return f"{tendency}_{tone}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--target",      type=int, default=60,
                        help="Target number of held-out pairs (default: 60)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    # ── Load profiles and item profiles ──────────────────────────────────
    profiles_path = os.path.join(args.output_dir, "user_profiles.json")
    items_path    = os.path.join(args.output_dir, "item_profiles.json")

    if not os.path.exists(profiles_path):
        raise FileNotFoundError(f"user_profiles.json not found at {profiles_path}")
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"item_profiles.json not found at {items_path}")

    print("Loading profiles and item profiles...")
    user_profiles  = load_json(profiles_path)
    item_profiles  = load_json(items_path)

    sampled_uids   = set(user_profiles.keys())
    item_biz_ids   = set(item_profiles.keys())

    print(f"  {len(sampled_uids):,} user profiles")
    print(f"  {len(item_biz_ids):,} item profiles")

    # ── Load reviews — only for sampled users and profiled businesses ─────
    print("\nLoading reviews...")
    review_path = os.path.join(args.data_dir, "yelp_academic_dataset_review.json")
    reviews_raw = load_jsonl(review_path)

    # group reviews: user_id -> list of reviews for businesses we have profiles for
    user_eligible_reviews = defaultdict(list)
    for r in tqdm(reviews_raw, desc="Filtering reviews"):
        uid = r.get("user_id")
        bid = r.get("business_id")
        if (uid in sampled_uids
                and bid in item_biz_ids
                and r.get("text", "").strip()
                and r.get("stars")):
            user_eligible_reviews[uid].append(r)

    eligible_users = len(user_eligible_reviews)
    print(f"  {eligible_users:,} sampled users have reviews on profiled businesses")

    # ── Build held-out pairs by bucket ────────────────────────────────────
    print("\nBuilding held-out pairs by bucket...")

    # bucket each user
    user_buckets = defaultdict(list)
    for uid in sampled_uids:
        if uid not in user_eligible_reviews:
            continue
        bucket = get_bucket(user_profiles[uid])
        user_buckets[bucket].append(uid)

    print("  User distribution across buckets:")
    for bucket, uids in sorted(user_buckets.items()):
        print(f"    {bucket:<25} {len(uids):>4} users eligible")

    held_out_pairs = []
    bucket_counts  = Counter()

    for bucket, target_n in BUCKET_TARGETS.items():
        candidates = user_buckets.get(bucket, [])
        if not candidates:
            print(f"  ⚠ No eligible users for bucket: {bucket}")
            continue

        random.shuffle(candidates)
        added = 0

        for uid in candidates:
            if added >= target_n:
                break

            reviews = user_eligible_reviews[uid]
            if not reviews:
                continue

            # hold out the LAST review chronologically
            reviews_sorted = sorted(reviews, key=lambda r: r.get("date", ""))
            held_review    = reviews_sorted[-1]

            # make sure this review is not already in sample_reviews
            # (sample_reviews contains the user's highest and lowest rated reviews)
            # If it is, take the second-to-last instead
            sample_texts = [
                s.get("text", "")[:50]
                for s in user_profiles[uid].get("sample_reviews", [])
            ]
            held_text_preview = held_review.get("text", "")[:50]

            if held_text_preview in sample_texts and len(reviews_sorted) > 1:
                held_review = reviews_sorted[-2]

            pair = {
                "user_id":       uid,
                "business_id":   held_review["business_id"],
                "actual_rating": held_review["stars"],
                "actual_text":   held_review.get("text", ""),
                "actual_date":   held_review.get("date", "")[:10],
                "bucket":        bucket,
                "context": {}  # no context for baseline evaluation
            }
            held_out_pairs.append(pair)
            bucket_counts[bucket] += 1
            added += 1

    # ── Write output ──────────────────────────────────────────────────────
    output_path = os.path.join(args.output_dir, "held_out_pairs.json")
    with open(output_path, "w") as f:
        json.dump(held_out_pairs, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    rating_dist = Counter(p["actual_rating"] for p in held_out_pairs)

    print(f"\n── HELD-OUT DATASET BUILT ──────────────────────────────────────")
    print(f"  Total pairs:        {len(held_out_pairs)}")
    print(f"\n  By bucket:")
    for bucket, count in sorted(bucket_counts.items()):
        target = BUCKET_TARGETS.get(bucket, "?")
        print(f"    {bucket:<25} {count:>3} / {target}")
    print(f"\n  Rating distribution:")
    for star in range(1, 6):
        count = rating_dist.get(star, 0)
        bar   = "█" * count
        print(f"    {star}★  {bar} ({count})")
    print(f"\n  Output: {output_path}")
    print(f"────────────────────────────────────────────────────────────────")
    print(f"\nNext step:")
    print(f"  python review_simulator.py \\")
    print(f"    --profiles {os.path.join(args.output_dir, 'user_profiles.json')} \\")
    print(f"    --items {os.path.join(args.output_dir, 'item_profiles.json')} \\")
    print(f"    --evaluate {output_path} \\")
    print(f"    --output {os.path.join(args.output_dir, 'evaluation_results.json')}")


if __name__ == "__main__":
    main()