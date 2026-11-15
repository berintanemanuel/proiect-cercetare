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
        ("q_1_1", "Does the cited source possess relevant domain expertise for the specific claim being made?"),
        ("q_1_2", "Is the citation backed by reliable, unbiased evidence rather than blind deference to prestige, a conflict of interest, or an out-of-scope opinion?"),
    ],
    "popular_consensus": [
        ("q_2_1", "Is widespread adoption or majority opinion backed by verified objective evidence rather than assuming a claim is true or right merely because many people believe or do it?"),
        ("q_2_2", "Does the context confirm the consensus is free from bandwagon pressure, unverified assumptions, or safety/logical flaws?"),
    ],
    "disjunctive_framing": [
        ("q_3_1", "Are the two presented options genuinely the only possible choices under the constraints in the context?"),
        ("q_3_2", "Does the context confirm there are no viable middle-ground compromises or third alternatives being ignored?"),
    ],
    "sample_generalization": [
        ("q_4_1", "Is the observed sample large and comprehensive enough to justify the breadth of the conclusion?"),
        ("q_4_2", "Is the sample representative and properly controlled rather than anecdotal, cherry-picked, or confounded?"),
    ],
    "consequence_chain": [
        ("q_5_1", "Is each step in the predicted chain of consequences backed by a concrete, verified causal mechanism rather than speculative fear-mongering?"),
        ("q_5_2", "Does the evidence justify that the initial step will inevitably trigger the extreme outcome without plausible stopping points?"),
    ],
    "causal_claim": [
        ("q_6_1", "Is there a verified causal mechanism or controlled test showing the first event actually caused the second?"),
        ("q_6_2", "Have alternative confounding explanations (such as a simultaneous third factor or coincidence) been ruled out?"),
    ],
    "analogical_comparison": [
        ("q_7_1", "Do the two compared situations share the core structural properties that matter for the conclusion?"),
        ("q_7_2", "Are there no critical differences in the context (such as risk level, reversibility, or domain rules) that break the analogy?"),
    ],
    "speaker_critique": [
        ("q_8_1", "Does the claim being critiqued rely purely on the speaker's uncorroborated personal word or lifestyle testimonial, making their credibility directly relevant?"),
        ("q_8_2", "Does the speaker critique avoid dismissing independently verifiable data, proofs, or studies merely because of the author's personal flaws or hypocrisy?"),
    ],
    "naturalness_appeal": [
        ("q_9_1", "Is the preference for the 'natural' option or state supported by concrete, verified evidence rather than just assuming 'natural means good/right' or 'unnatural means bad/wrong'?"),
        ("q_9_2", "Does the context confirm that the natural option is genuinely safer, healthier, or more effective for this specific situation than the alternative?"),
    ],
    "tradition_appeal": [
        ("q_10_1", "Is the traditional practice maintained for a verified functional, legal, or practical reason rather than mere habit or 'it has always been done this way'?"),
        ("q_10_2", "Does the context confirm the traditional practice remains effective and appropriate rather than harmful, obsolete, or inferior to the modern alternative?"),
    ],
    "relative_privation": [
        ("q_11_1", "Do the two problems directly compete for the exact same limited resources, personnel, or immediate capacity at the same time?"),
        ("q_11_2", "Does addressing the worse problem genuinely require postponing the smaller issue, rather than dismissing an independent problem that could be addressed separately?"),
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
