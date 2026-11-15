"""Baseline and Verify-Before-Classify (VBC) fallacy classifiers."""

from src.llm_client import LLMClient
from src.taxonomy import (
    FALLACY_CLASSES,
    SCHEME_DESCRIPTIONS,
    SCHEME_QUESTIONS,
    SCHEME_TO_FALLACY,
    normalize_fallacy_label,
    normalize_scheme_id,
    normalize_verification_state,
    to_coarse_label,
)


def _format_input(item: dict) -> str:
    """Format an argument and its optional context for LLM prompts."""
    arg = item.get("argument", "").strip()
    ctx = item.get("context", "").strip()
    ctx_display = ctx if ctx else "[No additional context provided]"
    return f"Argument:\n{arg}\n\nContext:\n{ctx_display}"


def decide_from_verifications(
    scheme_id: str,
    verifications: list[dict],
    allow_abstention: bool = True,
) -> tuple[str, str | None]:
    """Formal Step 3 decision rule over the Step 2 verification states.

    Returns (predicted_fine_label, violated_question_id).
    - Any question 'Violated' -> corresponding fallacy class + that question's ID
    - All questions 'Satisfied' -> 'Valid'
    - When allow_abstention is True (bare argument with no context): 'Not Enough Context'
    - When allow_abstention is False (context is provided): unverified critical questions
      fail the burden of warrant -> corresponding fallacy class + first unverified question's ID
    """
    if scheme_id == "none" or scheme_id not in SCHEME_TO_FALLACY:
        return "Valid", None

    for v in verifications:
        if v["state"] == "Violated":
            return SCHEME_TO_FALLACY[scheme_id], v["question_id"]

    if verifications and all(v["state"] == "Satisfied" for v in verifications):
        return "Valid", None

    if not allow_abstention and verifications:
        first_unverified = next(
            (v["question_id"] for v in verifications if v["state"] != "Satisfied"),
            verifications[0]["question_id"],
        )
        return SCHEME_TO_FALLACY[scheme_id], first_unverified

    return "Not Enough Context", None


def classify_zero_shot(
    item: dict,
    llm: LLMClient,
    candidate_classes: tuple[str, ...] = FALLACY_CLASSES,
) -> dict:
    """Baseline 1: Direct one-step zero-shot fallacy classification."""
    labels_str = ", ".join(f'"{c}"' for c in candidate_classes) + ', or "Valid"'
    system_prompt = (
        "You are an expert in informal logic and argument evaluation. "
        "Respond strictly with a valid JSON object."
    )
    user_prompt = (
        f"[ZERO-SHOT CLASSIFICATION]\n"
        f"Classify the following argument into one of these labels: {labels_str}.\n\n"
        f"{_format_input(item)}\n\n"
        f'Return JSON: {{"predicted_label": "<label>", "explanation": "<1-sentence reason>"}}'
    )

    resp = llm.complete_json(system_prompt, user_prompt)
    fine_label = normalize_fallacy_label(resp.get("predicted_label", "Valid"))
    return {
        "instance_id": item["id"],
        "model_name": "ZeroShot",
        "predicted_fine_label": fine_label,
        "predicted_label": to_coarse_label(fine_label),
        "detected_scheme": None,
        "verifications": [],
        "violated_question_id": None,
        "explanation": str(resp.get("explanation", "")),
    }


def classify_cot(
    item: dict,
    llm: LLMClient,
    candidate_classes: tuple[str, ...] = FALLACY_CLASSES,
) -> dict:
    """Baseline 2: One-step Chain-of-Thought (CoT) reasoning before classification."""
    labels_str = ", ".join(f'"{c}"' for c in candidate_classes) + ', or "Valid"'
    system_prompt = (
        "You are an expert in informal logic. Reason step-by-step about the argument's "
        "premises, conclusion, and context before choosing a label. Respond strictly in JSON."
    )
    user_prompt = (
        f"[CHAIN-OF-THOUGHT CLASSIFICATION]\n"
        f"Analyze the following argument step-by-step and classify it into one of: {labels_str}.\n\n"
        f"{_format_input(item)}\n\n"
        f'Return JSON: {{"reasoning": "<step-by-step analysis>", "predicted_label": "<label>"}}'
    )

    resp = llm.complete_json(system_prompt, user_prompt)
    fine_label = normalize_fallacy_label(resp.get("predicted_label", "Valid"))
    return {
        "instance_id": item["id"],
        "model_name": "CoT",
        "predicted_fine_label": fine_label,
        "predicted_label": to_coarse_label(fine_label),
        "detected_scheme": None,
        "verifications": [],
        "violated_question_id": None,
        "explanation": str(resp.get("reasoning", "")),
    }


def run_vbc_step1(
    item: dict,
    llm: LLMClient,
    candidate_classes: tuple[str, ...] = FALLACY_CLASSES,
) -> tuple[str, str]:
    """VBC Step 1: Identify the neutral argumentation scheme without fallacy bias."""
    allowed_schemes = {
        s_id for s_id, fallacy in SCHEME_TO_FALLACY.items() if fallacy in candidate_classes
    }
    schemes_list = "\n".join(
        f'- "{s_id}": {desc}'
        for s_id, desc in SCHEME_DESCRIPTIONS.items()
        if s_id in allowed_schemes
    )
    system_prompt = (
        "You are an argumentation analyst. Identify the general reasoning structure used in the text. "
        "Do NOT judge whether the argument is valid or flawed yet. Respond strictly in JSON."
    )
    none_desc = (
        "None of these specific reasoning patterns is used."
        if candidate_classes == FALLACY_CLASSES
        else "None of these specific reasoning patterns is used by the speaker (e.g., general observation, balanced comment, or refuting someone else's claim without using one of the patterns above)."
    )
    user_prompt = (
        f"[VBC STEP 1: SCHEME IDENTIFICATION]\n"
        f"Which of the following general reasoning patterns is primarily used in the argument?\n"
        f"{schemes_list}\n"
        f'- "none": {none_desc}\n\n'
        f"{_format_input(item)}\n\n"
        f'Return JSON: {{"scheme_id": "<scheme_id or none>", "rationale": "<brief explanation>"}}'
    )

    resp = llm.complete_json(system_prompt, user_prompt)
    scheme_id = normalize_scheme_id(resp.get("scheme_id", "none"), allowed_schemes=allowed_schemes)
    rationale = str(resp.get("rationale", ""))
    return scheme_id, rationale


def run_vbc_step2(item: dict, scheme_id: str, llm: LLMClient) -> list[dict]:
    """VBC Step 2: Check the scheme's critical verification questions against the context."""
    questions = SCHEME_QUESTIONS.get(scheme_id, [])
    if not questions:
        return []

    has_context = bool(item.get("context", "").strip())
    q_lines = "\n".join(f'- {q_id}: "{q_text}"' for q_id, q_text in questions)
    system_prompt = (
        "You are a careful fact-and-context verifier. Evaluate each critical question strictly "
        "against the provided Argument and Context. Respond strictly in JSON."
    )
    if has_context:
        user_prompt = (
            f"[VBC STEP 2: CRITICAL QUESTION VERIFICATION]\n"
            f"Reasoning Scheme: {scheme_id}\n"
            f"Because surrounding Context IS provided, evaluate each critical question below using ONLY these two states:\n"
            f'- "Satisfied": The Argument and Context provide concrete justification confirming the condition holds (or the comment is a reasonable/hedged observation that does not make an unwarranted leap).\n'
            f'- "Violated": The Context contradicts the condition, OR the Argument makes an unwarranted fallacious leap by relying on a superficial shortcut (mere popularity, tradition, nature, authority, or worse problems) without supporting warrant in the Argument or Context.\n\n'
            f"Verification Questions:\n{q_lines}\n\n"
            f"{_format_input(item)}\n\n"
            f'Return JSON: {{"verifications": [{{"question_id": "<q_id>", "state": "Satisfied|Violated", "evidence": "<brief quote or reason>"}}]}}'
        )
    else:
        user_prompt = (
            f"[VBC STEP 2: CRITICAL QUESTION VERIFICATION]\n"
            f"Reasoning Scheme: {scheme_id}\n"
            f"Evaluate each verification question below using ONLY these three states:\n"
            f'- "Satisfied": Explicit evidence in the text/context confirms the condition holds.\n'
            f'- "Violated": Explicit evidence in the text/context shows the condition fails or is contradicted.\n'
            f'- "Not Mentioned": The text/context does not provide enough information to confirm or refute the condition.\n\n'
            f"Verification Questions:\n{q_lines}\n\n"
            f"{_format_input(item)}\n\n"
            f'Return JSON: {{"verifications": [{{"question_id": "<q_id>", "state": "Satisfied|Violated|Not Mentioned", "evidence": "<brief quote or reason>"}}]}}'
        )

    resp = llm.complete_json(system_prompt, user_prompt)
    raw_verifs = resp.get("verifications", [])
    by_id = {str(v.get("question_id", "")): v for v in raw_verifs if isinstance(v, dict)}

    default_state = "Violated" if has_context else "Not Mentioned"
    results: list[dict] = []
    for q_id, q_text in questions:
        matched = by_id.get(q_id, {})
        results.append(
            {
                "question_id": q_id,
                "question_text": q_text,
                "state": normalize_verification_state(matched.get("state", default_state)),
                "evidence": str(matched.get("evidence", "")),
            }
        )
    return results


def classify_vbc(
    item: dict,
    llm: LLMClient,
    candidate_classes: tuple[str, ...] = FALLACY_CLASSES,
) -> dict:
    """Proposed 3-Step Verify-Before-Classify (VBC) pipeline."""
    scheme_id, step1_rationale = run_vbc_step1(item, llm, candidate_classes=candidate_classes)
    if scheme_id == "none":
        return {
            "instance_id": item["id"],
            "model_name": "VBC",
            "predicted_fine_label": "Valid",
            "predicted_label": "Valid",
            "detected_scheme": "none",
            "verifications": [],
            "violated_question_id": None,
            "explanation": f"No target scheme detected ({step1_rationale}).",
        }

    has_context = bool(item.get("context", "").strip())
    verifications = run_vbc_step2(item, scheme_id, llm)
    fine_label, violated_q = decide_from_verifications(
        scheme_id, verifications, allow_abstention=not has_context
    )

    if violated_q:
        failed_ev = next((v["evidence"] for v in verifications if v["question_id"] == violated_q), "")
        explanation = f"Scheme '{scheme_id}' violated {violated_q}: {failed_ev}"
    elif fine_label == "Valid":
        explanation = f"Scheme '{scheme_id}' verified: all critical questions satisfied."
    else:
        explanation = f"Scheme '{scheme_id}' detected, but context does not mention required verification facts."

    return {
        "instance_id": item["id"],
        "model_name": "VBC",
        "predicted_fine_label": fine_label,
        "predicted_label": to_coarse_label(fine_label),
        "detected_scheme": scheme_id,
        "verifications": verifications,
        "violated_question_id": violated_q,
        "explanation": explanation,
    }


def classify_vbc_ablated(
    item: dict,
    llm: LLMClient,
    candidate_classes: tuple[str, ...] = FALLACY_CLASSES,
) -> dict:
    """Experiment 4 Ablation (VBC-NoStep2): Runs Step 1 scheme detection, skips Step 2 verification questions."""
    scheme_id, step1_rationale = run_vbc_step1(item, llm, candidate_classes=candidate_classes)
    if scheme_id == "none":
        return {
            "instance_id": item["id"],
            "model_name": "VBC_Ablated",
            "predicted_fine_label": "Valid",
            "predicted_label": "Valid",
            "detected_scheme": "none",
            "verifications": [],
            "violated_question_id": None,
            "explanation": f"No target scheme detected ({step1_rationale}).",
        }

    labels_str = ", ".join(f'"{c}"' for c in candidate_classes) + ', or "Valid"'
    system_prompt = (
        "You are an expert in informal logic. Given the argument, its context, and the detected "
        "reasoning scheme, classify the argument. Respond strictly in JSON."
    )
    user_prompt = (
        f"[VBC ABLATED STEP 3: DIRECT CLASSIFICATION FROM SCHEME]\n"
        f"Detected Reasoning Scheme: {scheme_id} ({SCHEME_DESCRIPTIONS.get(scheme_id, '')})\n"
        f"Classify the argument into one of: {labels_str}.\n\n"
        f"{_format_input(item)}\n\n"
        f'Return JSON: {{"predicted_label": "<label>", "explanation": "<1-sentence reason>"}}'
    )

    resp = llm.complete_json(system_prompt, user_prompt)
    fine_label = normalize_fallacy_label(resp.get("predicted_label", "Valid"))
    return {
        "instance_id": item["id"],
        "model_name": "VBC_Ablated",
        "predicted_fine_label": fine_label,
        "predicted_label": to_coarse_label(fine_label),
        "detected_scheme": scheme_id,
        "verifications": [],
        "violated_question_id": None,
        "explanation": str(resp.get("explanation", "")),
    }
