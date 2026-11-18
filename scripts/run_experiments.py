"""Run the 4 fallacy detection models on Context Triplets and CoCoLoFa and save metrics."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifiers import (
    classify_cot,
    classify_vbc,
    classify_vbc_ablated,
    classify_zero_shot,
)
from src.evaluate import evaluate_predictions
from src.llm_client import LLMClient
from src.taxonomy import COCOLOFA_FALLACY_CLASSES, FALLACY_CLASSES


def run_models_on_dataset(
    items: list[dict],
    llm: LLMClient,
    candidate_classes: tuple[str, ...],
    dataset_name: str,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Run ZeroShot, CoT, VBC_Ablated, and VBC on a list of items."""
    models = {
        "ZeroShot": lambda it: classify_zero_shot(it, llm, candidate_classes),
        "CoT": lambda it: classify_cot(it, llm, candidate_classes),
        "VBC_Ablated": lambda it: classify_vbc_ablated(it, llm, candidate_classes),
        "VBC": lambda it: classify_vbc(it, llm, candidate_classes),
    }

    metrics_by_model: dict[str, dict] = {}
    preds_by_model: dict[str, list[dict]] = {}

    for model_name, classify_fn in models.items():
        print(f"[{dataset_name}] Running {model_name} on {len(items)} items...")
        preds: list[dict] = []
        for idx, item in enumerate(items, start=1):
            pred = classify_fn(item)
            preds.append(pred)
            if idx % 20 == 0 or idx == len(items):
                print(f"  {model_name}: {idx}/{len(items)} completed")

        preds_by_model[model_name] = preds
        metrics_by_model[model_name] = evaluate_predictions(
            items, preds, target_classes=candidate_classes
        )

    return metrics_by_model, preds_by_model


def print_summary_table(title: str, metrics_by_model: dict[str, dict], is_triplets: bool) -> None:
    """Print a clean Markdown table of results to the terminal."""
    print(f"\n### {title}")
    if is_triplets:
        print("| Model | FPR (Bare) | FPR (Supporting) | FPR (All Valid) | CSS | Undermining Macro-F1 | Overall Acc |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, m in metrics_by_model.items():
            print(
                f"| {name} | {m['fpr_bare']:.1%} | {m['fpr_supporting']:.1%} | "
                f"{m['fpr_all_valid']:.1%} | {m['css']:.1%} | "
                f"{m['macro_f1']:.1%} | {m['overall_accuracy']:.1%} |"
            )
    else:
        print("| Model | FPR (Valid Comments) | 8-Class Macro-F1 | Overall Accuracy |")
        print("|---|---:|---:|---:|")
        for name, m in metrics_by_model.items():
            print(
                f"| {name} | {m['fpr_all_valid']:.1%} | "
                f"{m['macro_f1']:.1%} | {m['overall_accuracy']:.1%} |"
            )


def backup_existing_results(
    results_dir: Path,
    backup_root: Path,
    current_model: str,
) -> Path | None:
    """Back up existing metrics/predictions files if they were produced by a different model."""
    metrics_file = results_dir / "experiment_metrics.json"
    preds_file = results_dir / "predictions.json"
    if not metrics_file.exists() and not preds_file.exists():
        return None

    existing_model: str | None = None
    if metrics_file.exists():
        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("model"), str):
                existing_model = data["model"]
        except (OSError, json.JSONDecodeError):
            existing_model = None

    if existing_model == current_model:
        return None

    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", existing_model or "unknown").strip("_") or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"{timestamp}_{safe_model}"
    suffix = 1
    while backup_dir.exists():
        backup_dir = backup_root / f"{timestamp}_{safe_model}_{suffix}"
        suffix += 1
    backup_dir.mkdir(parents=True, exist_ok=True)

    for src_file in (metrics_file, preds_file):
        if src_file.exists():
            shutil.copy2(src_file, backup_dir / src_file.name)

    print(f"[Backup] Existing results for '{existing_model or 'unknown'}' saved to {backup_dir}")
    return backup_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VBC fallacy detection experiments.")
    parser.add_argument("--model", type=str, default=None, help="LLM model name (overrides .env)")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL (e.g. http://127.0.0.1:1234/v1)",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "triplets", "cocolofa"],
        default="all",
        help="Which dataset to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of items to run (for quick smoke testing)",
    )
    parser.add_argument(
        "--strict-model",
        action="store_true",
        help="Disable cross-model Gemini cache sharing and automatic fallback rotation",
    )
    args = parser.parse_args()

    llm = LLMClient(
        model=args.model,
        base_url=args.base_url,
        cache_path=str(PROJECT_ROOT / "cache" / "llm_cache.db"),
        strict_model=args.strict_model,
    )
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    backup_existing_results(results_dir, PROJECT_ROOT / "results_backup", llm.model)

    metrics_file = results_dir / "experiment_metrics.json"
    preds_file = results_dir / "predictions.json"

    all_metrics: dict[str, dict] = {"model": llm.model}
    all_predictions: dict[str, dict] = {}
    if args.dataset != "all" and not args.limit:
        if metrics_file.exists():
            try:
                existing_m = json.loads(metrics_file.read_text(encoding="utf-8"))
                if isinstance(existing_m, dict):
                    all_metrics.update(existing_m)
                    all_metrics["model"] = llm.model
            except (OSError, json.JSONDecodeError):
                pass
        if preds_file.exists():
            try:
                existing_p = json.loads(preds_file.read_text(encoding="utf-8"))
                if isinstance(existing_p, dict):
                    all_predictions.update(existing_p)
            except (OSError, json.JSONDecodeError):
                pass

    if args.dataset in ("all", "triplets"):
        triplets_path = PROJECT_ROOT / "data" / "context_triplets.json"
        triplets = json.loads(triplets_path.read_text(encoding="utf-8"))
        if args.limit:
            triplets = triplets[: args.limit]
        t_metrics, t_preds = run_models_on_dataset(
            triplets, llm, FALLACY_CLASSES, "Context Triplets"
        )
        all_metrics["triplets"] = t_metrics
        all_predictions["triplets"] = t_preds
        print_summary_table("Context Triplets Results", t_metrics, is_triplets=True)

    if args.dataset in ("all", "cocolofa"):
        cocolofa_path = PROJECT_ROOT / "data" / "cocolofa_sample.json"
        cocolofa = json.loads(cocolofa_path.read_text(encoding="utf-8"))
        if args.limit:
            cocolofa = cocolofa[: args.limit]
        c_metrics, c_preds = run_models_on_dataset(
            cocolofa, llm, COCOLOFA_FALLACY_CLASSES, "CoCoLoFa Sample"
        )
        all_metrics["cocolofa"] = c_metrics
        all_predictions["cocolofa"] = c_preds
        print_summary_table("CoCoLoFa Sample Results", c_metrics, is_triplets=False)

    metrics_file = results_dir / "experiment_metrics.json"
    preds_file = results_dir / "predictions.json"
    metrics_file.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    preds_file.write_text(json.dumps(all_predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved metrics to {metrics_file}")
    print(f"Saved detailed predictions to {preds_file}")


if __name__ == "__main__":
    main()
