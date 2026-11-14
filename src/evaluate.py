"""Evaluation metrics for logical fallacy detection: FPR, CSS, Macro-F1, and Confusion Matrix."""

from src.taxonomy import FALLACY_CLASSES, is_fallacy_label, to_coarse_label


def compute_fpr(
    items: list[dict],
    predictions: list[dict],
    condition: str | None = None,
) -> float:
    """Compute False Positive Rate (FPR) on non-fallacious ('Valid' / 'Not Enough Context') items.

    If `condition` is specified (e.g. 'bare' or 'supporting'), only items with that condition are included.
    """
    pred_by_id = {p["instance_id"]: p for p in predictions}
    valid_items = [
        it
        for it in items
        if to_coarse_label(it["true_label"]) == "Valid"
        and (condition is None or it.get("condition") == condition)
        and it["id"] in pred_by_id
    ]
    if not valid_items:
        return 0.0

    false_positives = sum(
        1
        for it in valid_items
        if is_fallacy_label(pred_by_id[it["id"]]["predicted_label"])
    )
    return false_positives / len(valid_items)


def compute_css(items: list[dict], predictions: list[dict]) -> float:
    """Compute Context Sensitivity Score (CSS) across argument triplets.

    CSS is the fraction of triplets where the model:
      1) does NOT flag the 'supporting' version as a fallacy, AND
      2) predicts the exact ground-truth fallacy on the 'undermining' version.
    """
    pred_by_id = {p["instance_id"]: p for p in predictions}
    triplets: dict[str, dict[str, dict]] = {}
    for it in items:
        t_id = it.get("triplet_id")
        cond = it.get("condition")
        if t_id and cond in ("supporting", "undermining") and it["id"] in pred_by_id:
            triplets.setdefault(t_id, {})[cond] = it

    complete_triplets = [
        group for group in triplets.values() if "supporting" in group and "undermining" in group
    ]
    if not complete_triplets:
        return 0.0

    sensitive_count = 0
    for group in complete_triplets:
        supp_item = group["supporting"]
        und_item = group["undermining"]
        supp_pred = pred_by_id[supp_item["id"]]["predicted_label"]
        und_pred = pred_by_id[und_item["id"]]["predicted_label"]

        if not is_fallacy_label(supp_pred) and und_pred == und_item["true_label"]:
            sensitive_count += 1

    return sensitive_count / len(complete_triplets)


def compute_macro_f1(
    items: list[dict],
    predictions: list[dict],
    target_classes: tuple[str, ...] = FALLACY_CLASSES,
    condition: str | None = None,
) -> dict:
    """Compute Multi-Class Macro-F1, overall accuracy, and per-class F1 scores."""
    pred_by_id = {p["instance_id"]: p for p in predictions}
    filtered = [
        it
        for it in items
        if (condition is None or it.get("condition") == condition) and it["id"] in pred_by_id
    ]
    if not filtered:
        return {
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "per_class_f1": {cls: 0.0 for cls in target_classes},
        }

    y_true = [to_coarse_label(it["true_label"]) for it in filtered]
    y_pred = [to_coarse_label(pred_by_id[it["id"]]["predicted_label"]) for it in filtered]

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(filtered)

    per_class_f1: dict[str, float] = {}
    for cls in target_classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        denom = 2 * tp + fp + fn
        per_class_f1[cls] = (2 * tp / denom) if denom > 0 else 0.0

    macro_f1 = (
        sum(per_class_f1.values()) / len(target_classes) if target_classes else 0.0
    )
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "per_class_f1": per_class_f1,
    }


def compute_confusion_matrix(
    items: list[dict],
    predictions: list[dict],
    labels: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix dict[true_label][predicted_label] = count."""
    pred_by_id = {p["instance_id"]: p for p in predictions}
    if labels is None:
        labels = list(FALLACY_CLASSES) + ["Valid"]

    matrix: dict[str, dict[str, int]] = {
        row_label: {col_label: 0 for col_label in labels} for row_label in labels
    }

    for it in items:
        if it["id"] not in pred_by_id:
            continue
        t_label = to_coarse_label(it["true_label"])
        p_label = to_coarse_label(pred_by_id[it["id"]]["predicted_label"])
        if t_label not in matrix:
            matrix[t_label] = {col_label: 0 for col_label in labels}
        matrix[t_label][p_label] = matrix[t_label].get(p_label, 0) + 1

    return matrix


def evaluate_predictions(
    items: list[dict],
    predictions: list[dict],
    target_classes: tuple[str, ...] = FALLACY_CLASSES,
) -> dict:
    """Compute a complete evaluation summary for a model's predictions on a dataset."""
    has_triplets = any(it.get("condition") == "undermining" for it in items)
    f1_cond = "undermining" if has_triplets else None
    f1_stats = compute_macro_f1(items, predictions, target_classes=target_classes, condition=f1_cond)
    overall_f1 = compute_macro_f1(items, predictions, target_classes=target_classes, condition=None)

    return {
        "fpr_bare": round(compute_fpr(items, predictions, condition="bare"), 4),
        "fpr_supporting": round(compute_fpr(items, predictions, condition="supporting"), 4),
        "fpr_all_valid": round(compute_fpr(items, predictions, condition=None), 4),
        "css": round(compute_css(items, predictions), 4),
        "macro_f1": round(f1_stats["macro_f1"], 4),
        "accuracy": round(f1_stats["accuracy"], 4),
        "overall_accuracy": round(overall_f1["accuracy"], 4),
        "per_class_f1": {k: round(v, 4) for k, v in f1_stats["per_class_f1"].items()},
        "confusion_matrix": compute_confusion_matrix(items, predictions, list(target_classes) + ["Valid"]),
    }
