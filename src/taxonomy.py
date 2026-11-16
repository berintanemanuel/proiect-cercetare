"""Taxonomy of reasoning schemes, fallacy classes, verification questions, and label normalizers."""

import re

# All 11 target fallacy classes (8 from CoCoLoFa + 3 from Lab 3 / LOGIC)
FALLACY_CLASSES: tuple[str, ...] = (
    "Appeal to Authority",
    "Appeal to Majority",
    "False Dilemma",
    "Hasty Generalization",
    "Slippery Slope",
    "False Cause",
    "Faulty Analogy",
    "Ad Hominem",
    "Appeal to Nature",
    "Appeal to Tradition",
    "Appeal to Worse Problems",
)

# The 8 fallacy classes present in the CoCoLoFa benchmark
COCOLOFA_FALLACY_CLASSES: tuple[str, ...] = (
    "Appeal to Authority",
    "Appeal to Majority",
    "False Dilemma",
    "Hasty Generalization",
    "Slippery Slope",
    "Appeal to Nature",
    "Appeal to Tradition",
    "Appeal to Worse Problems",
)

# 1-to-1 mapping from neutral argumentation scheme ID to fallacy class
SCHEME_TO_FALLACY: dict[str, str] = {
    "authority_citation": "Appeal to Authority",
    "popular_consensus": "Appeal to Majority",
    "disjunctive_framing": "False Dilemma",
    "sample_generalization": "Hasty Generalization",
    "consequence_chain": "Slippery Slope",
    "causal_claim": "False Cause",
    "analogical_comparison": "Faulty Analogy",
    "speaker_critique": "Ad Hominem",
    "naturalness_appeal": "Appeal to Nature",
    "tradition_appeal": "Appeal to Tradition",
    "relative_privation": "Appeal to Worse Problems",
}

FALLACY_TO_SCHEME: dict[str, str] = {
    fallacy: scheme for scheme, fallacy in SCHEME_TO_FALLACY.items()
}

# Neutral descriptions of each reasoning pattern for Step 1 (avoids fallacy bias)
SCHEME_DESCRIPTIONS: dict[str, str] = {
    "authority_citation": "Citing an expert, specialist, institution, or authority figure to support a claim.",
    "popular_consensus": "Appealing to what most people do, believe, vote for, or adopt as evidence for a claim.",
    "disjunctive_framing": "Presenting an either-or choice between two options or outcomes.",
    "sample_generalization": "Drawing a general conclusion about a group or product based on observed examples or a sample.",
    "consequence_chain": "Predicting that an initial action will trigger a chain of subsequent events leading to an extreme outcome.",
    "causal_claim": "Arguing that one event or action caused another based on timing or observed correlation.",
    "analogical_comparison": "Comparing two different situations or domains to argue that what holds in one also holds in the other.",
    "speaker_critique": "Questioning or rejecting a claim by pointing to the personal attributes, past actions, or background of the speaker.",
    "naturalness_appeal": "Arguing that an option is better, safer, or healthier because it is natural rather than synthetic/artificial.",
    "tradition_appeal": "Arguing that a practice should be kept because it has traditionally been done that way for a long time.",
    "relative_privation": "Arguing that a complaint or issue should be set aside because a more severe problem exists.",
}

# The 2 critical verification questions checked against context in Step 2
SCHEME_QUESTIONS: dict[str, list[tuple[str, str]]] = {
    "authority_citation": [
        ("q_1_1", "Is the cited person or institution an objective, verified expert in the specific field of the claim, with credentials confirmed by the context rather than merely asserted by the speaker?"),
        ("q_1_2", "Is the citation free from conflicts of interest, personal bias, or misapplication of the source's actual scope in the context?"),
    ],
    "popular_consensus": [
        ("q_2_1", "Is majority preference or widespread adoption a logically valid indicator for this specific claim, and is the claimed consensus verified by concrete data in the context rather than asserted as a sweeping truism?"),
        ("q_2_2", "Does the argument rely on substantive evidence in the context rather than using popularity, numbers, or bandwagon pressure to prove a policy, moral, or factual conclusion?"),
    ],
    "disjunctive_framing": [
        ("q_3_1", "Are the two presented options genuinely the only possible choices under the factual, legal, or physical constraints in the context?"),
        ("q_3_2", "Does the context confirm there are no viable middle-ground compromises, gradations, or third alternatives being ignored?"),
    ],
    "sample_generalization": [
        ("q_4_1", "Is the observed sample or example in the context large and comprehensive enough to justify the broad generalization?"),
        ("q_4_2", "Is the generalization free from cherry-picking, isolated anecdotes, or stereotyping an entire group from a few cases?"),
    ],
    "consequence_chain": [
        ("q_5_1", "Is every step in the predicted chain of consequences backed by a concrete, proven causal or legal mechanism in the context rather than speculative escalation?"),
        ("q_5_2", "Does the context confirm there are no safeguards, policy limits, or stopping points that would prevent the extreme outcome?"),
    ],
    "causal_claim": [
        ("q_6_1", "Is there a verified causal mechanism or controlled evidence in the context showing the first event actually caused the second?"),
        ("q_6_2", "Have alternative confounding explanations (such as a third factor or coincidence) been ruled out by the context?"),
    ],
    "analogical_comparison": [
        ("q_7_1", "Do the two compared situations share the core structural properties that matter for the conclusion?"),
        ("q_7_2", "Are there no critical differences in the context that break or invalidate the analogy?"),
    ],
    "speaker_critique": [
        ("q_8_1", "Does the claim being critiqued rely purely on the speaker's uncorroborated personal word, making their personal credibility directly relevant to the truth of the claim?"),
        ("q_8_2", "Does the critique address the substance of the argument rather than dismissing it because of the speaker's personal traits, motives, or alleged hypocrisy?"),
    ],
    "naturalness_appeal": [
        ("q_9_1", "Does the provided Context contain explicit scientific, clinical, or empirical data proving that the 'natural' option or behavior is superior, rather than the speaker merely asserting that something is 'natural'?"),
        ("q_9_2", "Does the provided Context explicitly rule out risks or downsides of the 'natural' option via controlled testing, rather than relying on the assumption or metaphor that what is 'natural' is automatically right or inevitable?"),
    ],
    "tradition_appeal": [
        ("q_10_1", "Does the provided Context explicitly document a specific technical, engineering, or statutory requirement that mandates keeping the old historical practice, rather than the speaker appealing to 'time-honored tradition' or 'the old way'?"),
        ("q_10_2", "Does the provided Context explicitly confirm through comparative evaluation that the traditional practice outperforms modern reforms?"),
    ],
    "relative_privation": [
        ("q_11_1", "Does the provided Context explicitly state a fixed budget figure or shared personnel limit proving that addressing the first issue makes it mathematically impossible to address the second issue?"),
        ("q_11_2", "Does the speaker avoid minimizing, dismissing, or redirecting attention away from the article's topic by pointing to other severe problems (such as poverty, war, dictators, or the economy)?"),
    ],
}


def is_fallacy_label(label: str) -> bool:
    """Return True if label is one of the 11 fallacy classes."""
    return label in FALLACY_CLASSES


def to_coarse_label(fine_label: str) -> str:
    """Map both 'Valid' and 'Not Enough Context' to 'Valid' for standard evaluation."""
    if fine_label in ("Valid", "Not Enough Context"):
        return "Valid"
    return fine_label


def normalize_fallacy_label(raw: str) -> str:
    """Normalize noisy LLM label strings to canonical fallacy or non-fallacy names."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(raw)).strip().lower()
    cleaned = " ".join(cleaned.split())

    if "not enough context" in cleaned or "insufficient context" in cleaned:
        return "Not Enough Context"
    if cleaned in ("valid", "none", "no fallacy", "non fallacious") or "valid argument" in cleaned:
        return "Valid"

    if "authority" in cleaned:
        return "Appeal to Authority"
    if "majority" in cleaned or "bandwagon" in cleaned or "popularity" in cleaned or "ad populum" in cleaned:
        return "Appeal to Majority"
    if "dilemma" in cleaned or "dichotomy" in cleaned or "either or" in cleaned or "black and white" in cleaned:
        return "False Dilemma"
    if "generalization" in cleaned or "generalisation" in cleaned or "hasty" in cleaned:
        return "Hasty Generalization"
    if "slippery" in cleaned or "slope" in cleaned:
        return "Slippery Slope"
    if "false cause" in cleaned or "post hoc" in cleaned or "causal" in cleaned:
        return "False Cause"
    if "analogy" in cleaned or "analogical" in cleaned:
        return "Faulty Analogy"
    if "hominem" in cleaned or "personal attack" in cleaned or "tu quoque" in cleaned:
        return "Ad Hominem"
    if "nature" in cleaned or "natural" in cleaned:
        return "Appeal to Nature"
    if "tradition" in cleaned or "antiquity" in cleaned:
        return "Appeal to Tradition"
    if "worse problem" in cleaned or "relative privation" in cleaned or "whataboutism" in cleaned:
        return "Appeal to Worse Problems"

    if "valid" in cleaned or "non fallacious" in cleaned:
        return "Valid"

    return str(raw).strip()


def normalize_scheme_id(raw: str, allowed_schemes: set[str] | None = None) -> str:
    """Extract canonical scheme ID from LLM output, or 'none' if none matches."""
    cleaned = str(raw).strip().lower()
    candidates = allowed_schemes if allowed_schemes is not None else set(SCHEME_TO_FALLACY.keys())
    for scheme_id in SCHEME_TO_FALLACY:
        if scheme_id in candidates and scheme_id in cleaned:
            return scheme_id
    return "none"


def normalize_verification_state(raw: str) -> str:
    """Normalize verification answer to 'Satisfied', 'Violated', or 'Not Mentioned'."""
    cleaned = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    words = set(cleaned.split())

    if "not mentioned" in cleaned or "unmentioned" in cleaned or "unknown" in cleaned or "insufficient" in cleaned:
        return "Not Mentioned"
    if "violated" in cleaned or "failed" in cleaned or "contradicted" in cleaned or "no" in words or "false" in words:
        return "Violated"
    if "satisfied" in cleaned or "supported" in cleaned or "confirmed" in cleaned or "yes" in words or "true" in words:
        return "Satisfied"
    return "Not Mentioned"
