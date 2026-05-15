"""
stylometry_extractor.py
=======================
ChopLife — Deep Writing Pattern Analysis

Extracts structural writing DNA from a user's review history.
Heuristic-only module — no LLM calls here.
LLM structural DNA extraction is merged into profile_extractor.py
to halve Groq usage (one combined call per user instead of two).

Provides:
  - Sentence structure habits (length, fragments, rhythm)
  - Opening and closing patterns
  - Punctuation personality
  - Function word fingerprint
  - Transition habits
  - build_stylometry_prompt_section() — renders stylometry for simulation prompts

Used by:
  profile_extractor.py  — calls heuristic functions directly during profile build
  review_simulator.py   — calls build_stylometry_prompt_section() at simulation time
"""

import re
import statistics
from collections import Counter


# ─── HEURISTIC ANALYSIS (no LLM needed) ──────────────────────────────────────

def analyze_sentence_structure(texts: list) -> dict:
    """
    Measure sentence-level writing habits from review texts.
    Uses median (not mean) for length — robust against single long
    scene-setting sentences that inflate the average.
    """
    all_sentences = []
    fragment_count = 0
    total_sentences = 0

    for text in texts:
        if not text or not text.strip():
            continue
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        for s in sentences:
            words = s.split()
            word_count = len(words)
            all_sentences.append(word_count)
            total_sentences += 1
            if word_count <= 4:
                fragment_count += 1

    if not all_sentences:
        return {}

    return {
        "median_sentence_length":   round(statistics.median(all_sentences), 1),
        "mean_sentence_length":     round(statistics.mean(all_sentences), 1),
        "sentence_length_variance": round(statistics.stdev(all_sentences), 1)
                                    if len(all_sentences) > 1 else 0,
        "fragment_rate":            round(fragment_count / total_sentences, 2)
                                    if total_sentences > 0 else 0,
        "total_sentences_analyzed": total_sentences,
    }


def analyze_punctuation_personality(texts: list) -> dict:
    """
    Count punctuation habits across all texts.
    Punctuation is hard to fake consciously — strong stylometric signal.
    """
    full_text  = " ".join(texts)
    word_count = max(len(full_text.split()), 1)

    return {
        "exclamation_per_100_words": round(
            full_text.count("!") / word_count * 100, 2),
        "ellipsis_per_100_words":    round(
            full_text.count("...") / word_count * 100, 2),
        "dash_per_100_words":        round(
            (full_text.count(" - ") + full_text.count(" — ")) / word_count * 100, 2),
        "question_per_100_words":    round(
            full_text.count("?") / word_count * 100, 2),
        "parentheses_per_100_words": round(
            full_text.count("(") / word_count * 100, 2),
    }


def analyze_function_words(texts: list) -> dict:
    """
    Function word frequency — one of the most reliable stylometric signals.
    Writers rarely control these consciously.
    """
    FUNCTION_WORDS = [
        "and", "but", "so", "just", "really", "very", "pretty",
        "quite", "definitely", "absolutely", "actually", "honestly",
        "basically", "literally", "kind of", "sort of", "i think",
        "i mean", "you know", "overall", "however", "although",
    ]
    full_text  = " ".join(texts).lower()
    word_count = max(len(full_text.split()), 1)

    counts = {}
    for w in FUNCTION_WORDS:
        count = full_text.count(w)
        if count > 0:
            counts[w] = round(count / word_count * 1000, 2)  # per 1000 words

    return counts


def analyze_opening_closing_patterns(texts: list) -> dict:
    """
    Extract how writers start and end their reviews.
    Writers have strong habitual entry and exit patterns.
    """
    openings = []
    closings = []

    for text in texts:
        if not text or not text.strip():
            continue
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            openings.append(sentences[0][:80])

        if len(sentences) >= 2:
            closings.append(sentences[-1][:80])

    opening_types = Counter()
    for o in openings:
        o_lower = o.lower()
        if o_lower.startswith(("i stopped", "i visited", "i went", "i tried", "i came")):
            opening_types["action_opening"] += 1
        elif o_lower.startswith(("we ", "we went", "we stopped")):
            opening_types["group_action_opening"] += 1
        elif any(o_lower.startswith(w) for w in ["if you", "for anyone", "this is"]):
            opening_types["evaluative_opening"] += 1
        elif re.match(r'^[A-Z][a-z]+\'s|^The |^This ', o):
            opening_types["subject_opening"] += 1
        else:
            opening_types["other_opening"] += 1

    closing_types = Counter()
    for c in closings:
        c_lower = c.lower()
        if any(w in c_lower for w in ["would recommend", "highly recommend", "recommend"]):
            closing_types["recommendation_close"] += 1
        elif any(w in c_lower for w in ["will be back", "definitely return", "come back"]):
            closing_types["return_intent_close"] += 1
        elif any(w in c_lower for w in ["overall", "all in all", "in summary"]):
            closing_types["summary_close"] += 1
        elif any(w in c_lower for w in ["worth", "worth it", "worth a visit"]):
            closing_types["verdict_close"] += 1
        else:
            closing_types["other_close"] += 1

    return {
        "dominant_opening_type":      opening_types.most_common(1)[0][0]
                                      if opening_types else "unknown",
        "opening_type_distribution":  dict(opening_types),
        "dominant_closing_type":      closing_types.most_common(1)[0][0]
                                      if closing_types else "unknown",
        "closing_type_distribution":  dict(closing_types),
        "sample_openings":            openings[:3],
        "sample_closings":            closings[:3],
    }


def analyze_transition_habits(texts: list) -> dict:
    """Identify structural transitions the writer uses repeatedly."""
    TRANSITIONS = {
        "contrast":   ["however", "but", "although", "despite", "only downside",
                       "that said", "on the other hand", "except"],
        "addition":   ["also", "additionally", "and", "plus", "on top of that"],
        "summary":    ["overall", "all in all", "in summary", "bottom line",
                       "to sum up", "in conclusion"],
        "time":       ["first", "then", "after", "finally", "when we arrived",
                       "as soon as", "by the time"],
        "emphasis":   ["definitely", "absolutely", "really", "truly", "by far"],
        "concession": ["i mean", "granted", "admittedly", "to be fair",
                       "to be honest", "not gonna lie"],
    }

    full_text  = " ".join(texts).lower()
    word_count = max(len(full_text.split()), 1)
    hits = {}

    for category, words in TRANSITIONS.items():
        count = sum(full_text.count(w) for w in words)
        if count > 0:
            hits[category] = round(count / word_count * 1000, 2)

    return hits


# ─── PROMPT RENDERER ──────────────────────────────────────────────────────────

def build_stylometry_prompt_section(stylometry: dict) -> str:
    """
    Render stylometry profile as a concise instruction block
    for injection into the simulation prompt.
    Focus on decision patterns, not just vocabulary.
    Called by review_simulator.py at simulation time.
    """
    if not stylometry or not stylometry.get("available"):
        return ""

    ss  = stylometry.get("sentence_structure", {})
    pp  = stylometry.get("punctuation", {})
    oc  = stylometry.get("opening_closing", {})
    tr  = stylometry.get("transitions", {})
    dna = stylometry.get("structural_dna", {})
    fw  = stylometry.get("function_words", {})

    lines = ["WRITING DNA — match these structural patterns exactly:"]

    # sentence rhythm
    if ss.get("median_sentence_length"):
        lines.append(
            f"  Sentences: median {ss['median_sentence_length']:.0f} words "
            f"(variance: {ss.get('sentence_length_variance', 0):.1f})"
        )
    if ss.get("fragment_rate", 0) > 0.1:
        lines.append(
            f"  Uses fragments frequently ({ss['fragment_rate']*100:.0f}% of sentences) "
            f"— short punchy statements like 'Unreal.' or 'No such luck.'"
        )

    # narrative mode
    if dna.get("narrative_mode") and dna["narrative_mode"] != "unknown":
        lines.append(
            f"  Narrative mode: {dna['narrative_mode']} — "
            f"{dna.get('narrative_mode_explanation', '')}"
        )

    # structure pattern
    if dna.get("review_structure_pattern") and dna["review_structure_pattern"] != "unknown":
        lines.append(
            f"  Structure pattern: {dna['review_structure_pattern']}"
        )

    # semantic priority
    if dna.get("semantic_priority_order"):
        lines.append(
            f"  Mentions in this order: {' → '.join(dna['semantic_priority_order'][:4])}"
        )

    # opening style
    if oc.get("dominant_opening_type") and oc["dominant_opening_type"] != "unknown":
        lines.append(
            f"  Opens with: {oc['dominant_opening_type'].replace('_', ' ')}"
        )
    if oc.get("sample_openings"):
        lines.append(
            f"  Opening examples: \"{oc['sample_openings'][0]}\""
        )

    # closing style
    if oc.get("dominant_closing_type") and oc["dominant_closing_type"] != "unknown":
        lines.append(
            f"  Closes with: {oc['dominant_closing_type'].replace('_', ' ')}"
        )

    # emotional calibration
    if dna.get("emotional_calibration") and dna["emotional_calibration"] != "unknown":
        lines.append(
            f"  Emotional tone: {dna['emotional_calibration']} — "
            f"{dna.get('emotional_calibration_explanation', '')}"
        )

    # hedging
    if dna.get("hedging_style") and dna["hedging_style"] not in ("unknown", "none"):
        examples = ", ".join(f'"{e}"' for e in dna.get("hedging_examples", [])[:2])
        lines.append(f"  Hedging: {dna['hedging_style']} — e.g. {examples}")

    # certainty markers
    if dna.get("certainty_markers"):
        lines.append(
            f"  Certainty markers: {', '.join(dna['certainty_markers'][:3])}"
        )

    # food descriptors
    if dna.get("food_descriptor_style") and dna["food_descriptor_style"] != "unknown":
        examples = ", ".join(f'"{e}"' for e in dna.get("food_descriptor_examples", [])[:2])
        lines.append(
            f"  Food descriptions: {dna['food_descriptor_style']} — e.g. {examples}"
        )

    # favourite adjectives
    if dna.get("favourite_adjectives"):
        lines.append(
            f"  Overused adjectives: {', '.join(dna['favourite_adjectives'][:5])}"
        )

    # function word signals
    top_fw = sorted(fw.items(), key=lambda x: x[1], reverse=True)[:5]
    if top_fw:
        lines.append(
            f"  Function word fingerprint: "
            f"{', '.join(f'{w}({v:.1f}/1k)' for w, v in top_fw)}"
        )

    # transition habits
    if tr:
        dominant = sorted(tr.items(), key=lambda x: x[1], reverse=True)[:2]
        lines.append(
            f"  Transition style: heavy on "
            f"{', '.join(t[0] for t in dominant)} transitions"
        )

    # notable quirks
    if dna.get("notable_quirks"):
        for quirk in dna["notable_quirks"][:2]:
            lines.append(f"  Quirk: {quirk}")

    lines.append(
        "  CRITICAL: copy the STRUCTURE of their writing, not just the words."
    )

    return "\n".join(lines)