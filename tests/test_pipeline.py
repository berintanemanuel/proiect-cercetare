"""Unit tests for the VBC fallacy detection pipeline, datasets, and evaluation metrics."""

from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import threading
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifiers import (
    classify_cot,
    classify_vbc,
    classify_vbc_ablated,
    classify_zero_shot,
    decide_from_verifications,
)
from src.evaluate import compute_css, compute_fpr, compute_macro_f1
from src.llm_client import LLMClient, extract_json_object
from src.taxonomy import (
    COCOLOFA_FALLACY_CLASSES,
    FALLACY_CLASSES,
    FALLACY_TO_SCHEME,
    SCHEME_QUESTIONS,
    SCHEME_TO_FALLACY,
    is_fallacy_label,
    normalize_fallacy_label,
    normalize_scheme_id,
    normalize_verification_state,
    to_coarse_label,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_taxonomy_bijection_and_normalizers():
    """Verify all 11 schemes map 1-to-1 to the 11 fallacy classes and have 2 questions each."""
    assert len(FALLACY_CLASSES) == 11
    assert len(COCOLOFA_FALLACY_CLASSES) == 8
    assert len(SCHEME_TO_FALLACY) == 11

    for scheme_id, fallacy in SCHEME_TO_FALLACY.items():
        assert FALLACY_TO_SCHEME[fallacy] == scheme_id
        questions = SCHEME_QUESTIONS[scheme_id]
        assert len(questions) == 2
        for q_id, q_text in questions:
            assert q_id.startswith("q_")
            assert q_text.endswith("?")

    assert normalize_fallacy_label("ad-hominem") == "Ad Hominem"
    assert normalize_fallacy_label("whataboutism / worse problems") == "Appeal to Worse Problems"
    assert normalize_fallacy_label("not enough context") == "Not Enough Context"
    assert to_coarse_label("Not Enough Context") == "Valid"
    assert is_fallacy_label("Slippery Slope") is True
    assert is_fallacy_label("Valid") is False
    assert normalize_scheme_id("authority_citation") == "authority_citation"
    assert normalize_verification_state("not mentioned") == "Not Mentioned"
    assert normalize_verification_state("Violated") == "Violated"
    assert normalize_verification_state("Satisfied") == "Satisfied"


def test_datasets_integrity_and_balance():
    """Verify context_triplets.json (132 items) and cocolofa_sample.json (200 items)."""
    triplets_path = PROJECT_ROOT / "data" / "context_triplets.json"
    cocolofa_path = PROJECT_ROOT / "data" / "cocolofa_sample.json"

    triplets = json.loads(triplets_path.read_text(encoding="utf-8"))
    assert len(triplets) == 132
    assert Counter(t["condition"] for t in triplets) == {
        "bare": 44,
        "supporting": 44,
        "undermining": 44,
    }

    undermining_counts = Counter(t["true_label"] for t in triplets if t["condition"] == "undermining")
    for cls in FALLACY_CLASSES:
        assert undermining_counts[cls] == 4

    cocolofa = json.loads(cocolofa_path.read_text(encoding="utf-8"))
    assert len(cocolofa) == 200
    assert len({c["id"] for c in cocolofa}) == 200
    cocolofa_counts = Counter(c["true_label"] for c in cocolofa)
    for cls in COCOLOFA_FALLACY_CLASSES:
        assert cocolofa_counts[cls] == 18
    assert cocolofa_counts["Valid"] == 56


def test_llm_client_cache_and_json_extraction(tmp_path: Path):
    """Verify markdown code-fence stripping and SQLite response caching."""
    raw = 'Here is the JSON:\n```json\n{"scheme_id": "causal_claim"}\n```'
    assert extract_json_object(raw) == {"scheme_id": "causal_claim"}

    calls = []

    def fake_transport(messages, model, temperature):
        calls.append(messages)
        return '```json\n{"predicted_label": "False Cause", "explanation": "Coincidence"}\n```'

    client = LLMClient(
        model="test-model",
        cache_path=str(tmp_path / "cache.db"),
        transport=fake_transport,
    )
    res1 = client.complete_json("sys", "prompt A")
    res2 = client.complete_json("sys", "prompt A")
    assert res1["predicted_label"] == "False Cause"
    assert res2 == res1
    assert len(calls) == 1


def test_vbc_decision_rule_and_classifiers(tmp_path: Path):
    """Verify Step 3 decision logic and VBC vs. Baseline behavior on a triplet."""
    # Violated takes precedence over Not Mentioned
    verifs = [
        {"question_id": "q_1_1", "state": "Not Mentioned", "evidence": ""},
        {"question_id": "q_1_2", "state": "Violated", "evidence": "Conflict of interest"},
    ]
    assert decide_from_verifications("authority_citation", verifs) == ("Appeal to Authority", "q_1_2")

    def mock_transport(messages, model, temperature):
        prompt = messages[-1]["content"]
        if "VBC STEP 1" in prompt:
            return '{"scheme_id": "authority_citation", "rationale": "Cites Dr. Meyer"}'
        if "VBC STEP 2" in prompt:
            if "quantum chemist" in prompt:
                return '{"verifications": [{"question_id": "q_1_1", "state": "Violated", "evidence": "Quantum chemist, not virologist"}, {"question_id": "q_1_2", "state": "Violated", "evidence": "Contradicted by trials"}]}'
            if "clinical immunology" in prompt:
                return '{"verifications": [{"question_id": "q_1_1", "state": "Satisfied", "evidence": "Immunologist"}, {"question_id": "q_1_2", "state": "Satisfied", "evidence": "Replicated meta-analysis"}]}'
            return '{"verifications": [{"question_id": "q_1_1", "state": "Not Mentioned", "evidence": "No background"}, {"question_id": "q_1_2", "state": "Not Mentioned", "evidence": "No context"}]}'
        return '{"predicted_label": "Appeal to Authority", "reasoning": "Cites authority", "explanation": "Cites authority"}'

    llm = LLMClient(cache_path=str(tmp_path / "cache.db"), transport=mock_transport)
    bare = {"id": "t01_bare", "triplet_id": "t01", "condition": "bare", "argument": "Dr. Meyer says vitamin C cures viruses.", "context": "", "true_label": "Not Enough Context"}
    supp = {"id": "t01_supp", "triplet_id": "t01", "condition": "supporting", "argument": "Dr. Meyer says vitamin C cures viruses.", "context": "Specialized in clinical immunology.", "true_label": "Valid"}
    und = {"id": "t01_und", "triplet_id": "t01", "condition": "undermining", "argument": "Dr. Meyer says vitamin C cures viruses.", "context": "He is a quantum chemist.", "true_label": "Appeal to Authority"}

    assert classify_vbc(bare, llm)["predicted_fine_label"] == "Not Enough Context"
    assert classify_vbc(bare, llm)["predicted_label"] == "Valid"
    assert classify_vbc(supp, llm)["predicted_label"] == "Valid"
    assert classify_vbc(und, llm)["predicted_label"] == "Appeal to Authority"

    assert classify_zero_shot(bare, llm)["predicted_label"] == "Appeal to Authority"
    assert classify_cot(bare, llm)["predicted_label"] == "Appeal to Authority"
    assert classify_vbc_ablated(bare, llm)["predicted_label"] == "Appeal to Authority"


def test_evaluation_metrics_fpr_css_and_f1():
    """Verify FPR, Context Sensitivity Score (CSS), and Macro-F1 math."""
    items = [
        {"id": "1b", "triplet_id": "t1", "condition": "bare", "true_label": "Not Enough Context"},
        {"id": "1s", "triplet_id": "t1", "condition": "supporting", "true_label": "Valid"},
        {"id": "1u", "triplet_id": "t1", "condition": "undermining", "true_label": "Appeal to Authority"},
        {"id": "2b", "triplet_id": "t2", "condition": "bare", "true_label": "Not Enough Context"},
        {"id": "2s", "triplet_id": "t2", "condition": "supporting", "true_label": "Valid"},
        {"id": "2u", "triplet_id": "t2", "condition": "undermining", "true_label": "False Cause"},
    ]
    preds = [
        {"instance_id": "1b", "predicted_label": "Valid"},
        {"instance_id": "1s", "predicted_label": "Valid"},
        {"instance_id": "1u", "predicted_label": "Appeal to Authority"},
        {"instance_id": "2b", "predicted_label": "False Cause"},
        {"instance_id": "2s", "predicted_label": "Valid"},
        {"instance_id": "2u", "predicted_label": "False Cause"},
    ]

    assert compute_fpr(items, preds, condition="bare") == pytest.approx(0.5)
    assert compute_fpr(items, preds, condition="supporting") == pytest.approx(0.0)
    assert compute_fpr(items, preds) == pytest.approx(0.25)
    assert compute_css(items, preds) == pytest.approx(1.0)

    f1 = compute_macro_f1(
        items,
        preds,
        target_classes=("Appeal to Authority", "False Cause"),
        condition="undermining",
    )
    assert f1["macro_f1"] == pytest.approx(1.0)


def test_extract_json_strips_think_tags():
    """Verify <think>...</think> reasoning blocks with braces are ignored before JSON parsing."""
    raw = '<think>Maybe {"wrong": 1}?</think>\n{"predicted_label": "Valid", "explanation": "Ok"}'
    assert extract_json_object(raw) == {"predicted_label": "Valid", "explanation": "Ok"}


def test_extract_json_repairs_missing_opening_quote():
    """Verify small-LLM JSON missing an opening quote on a string value is recovered."""
    raw = (
        '{"predicted_label": "Appeal to Authority", '
        '"explanation": The argument relies on Dr. Petrescu without independent evidence."}'
    )
    assert extract_json_object(raw) == {
        "predicted_label": "Appeal to Authority",
        "explanation": "The argument relies on Dr. Petrescu without independent evidence.",
    }


def test_complete_json_retries_and_does_not_cache_invalid_json(tmp_path: Path):
    """Verify complete_json evicts unparseable output and caches only valid JSON after retry."""
    calls = []

    def flaky_transport(messages, model, temperature):
        calls.append(messages[-1]["content"])
        if len(calls) == 1:
            return "Sorry, I cannot output JSON right now."
        return '{"predicted_label": "Valid", "explanation": "Fixed"}'

    cache_file = tmp_path / "retry_cache.db"
    client = LLMClient(model="test-model", cache_path=str(cache_file), transport=flaky_transport)
    res1 = client.complete_json("sys", "prompt A")
    res2 = client.complete_json("sys", "prompt A")
    assert res1 == {"predicted_label": "Valid", "explanation": "Fixed"}
    assert res2 == res1
    assert len(calls) == 2


def test_lmstudio_endpoint_resolution_and_http_call(tmp_path: Path, monkeypatch):
    """Verify lmstudio/ prefix and local base_url resolution, plus real HTTP payload."""
    from src.llm_client import _resolve_endpoint_and_key

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    url, key = _resolve_endpoint_and_key("lmstudio/qwen3.5-4b", None, None)
    assert url == "http://127.0.0.1:1234/v1/chat/completions"
    assert key

    url2, key2 = _resolve_endpoint_and_key("qwen3.5:4b", None, "http://127.0.0.1:1234")
    assert url2 == "http://127.0.0.1:1234/v1/chat/completions"
    assert key2

    url3, key3 = _resolve_endpoint_and_key("qwen3.5-4b", None, "http://127.0.0.1:1234/v1")
    assert url3 == "http://127.0.0.1:1234/v1/chat/completions"
    assert key3

    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            received["path"] = self.path
            received["body"] = json.loads(self.rfile.read(length).decode("utf-8"))
            resp = json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = LLMClient(
            model="lmstudio/qwen3.5-4b",
            base_url=f"http://127.0.0.1:{port}",
            cache_path=str(tmp_path / "cache.db"),
        )
        out = client.complete_json("sys", "user")
        assert out == {"ok": True}
        assert received["path"] == "/v1/chat/completions"
        assert received["body"]["model"] == "qwen3.5-4b"
    finally:
        server.shutdown()
        server.server_close()


def test_backup_existing_results_on_different_model(tmp_path: Path):
    """Verify existing results are backed up when starting a run with a different model."""
    from scripts.run_experiments import backup_existing_results

    results_dir = tmp_path / "results"
    backup_root = tmp_path / "results_backup"
    results_dir.mkdir()
    metrics_payload = {"model": "lmstudio/qwen3.5:4b", "triplets": {"ZeroShot": {}}}
    preds_payload = {"triplets": {"ZeroShot": []}}
    (results_dir / "experiment_metrics.json").write_text(json.dumps(metrics_payload), encoding="utf-8")
    (results_dir / "predictions.json").write_text(json.dumps(preds_payload), encoding="utf-8")

    backup_dir = backup_existing_results(results_dir, backup_root, current_model="gemini-3.5-flash-lite")
    assert backup_dir is not None
    assert backup_dir.parent == backup_root
    assert ":" not in backup_dir.name and "/" not in backup_dir.name
    assert json.loads((backup_dir / "experiment_metrics.json").read_text(encoding="utf-8")) == metrics_payload
    assert json.loads((backup_dir / "predictions.json").read_text(encoding="utf-8")) == preds_payload


def test_backup_existing_results_skips_same_model(tmp_path: Path):
    """Verify no backup is created when running with the same model as existing results."""
    from scripts.run_experiments import backup_existing_results

    results_dir = tmp_path / "results"
    backup_root = tmp_path / "results_backup"
    results_dir.mkdir()
    (results_dir / "experiment_metrics.json").write_text(
        json.dumps({"model": "gemini-3.5-flash-lite"}), encoding="utf-8"
    )

    assert backup_existing_results(results_dir, backup_root, current_model="gemini-3.5-flash-lite") is None
    assert not backup_root.exists()


def test_gemma_endpoint_resolution_and_cache_isolation(tmp_path: Path, monkeypatch):
    """Verify gemma-* models route to Gemini endpoint and do not reuse gemini-* fallback cache."""
    from src.llm_client import _resolve_endpoint_and_key

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    url, key = _resolve_endpoint_and_key("gemma-4-31b-it", None, None)
    assert url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert key == "test-gemini-key"

    cache_file = tmp_path / "shared_cache.db"
    gemini_client = LLMClient(model="gemini-3.5-flash-lite", cache_path=str(cache_file))
    gemini_client._save_cached_response(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "prompt"}],
        '{"predicted_label": "False Cause"}',
    )

    gemma_calls = []
    gemma_client = LLMClient(model="gemma-4-31b-it", cache_path=str(cache_file))
    gemma_client._invoke_model = lambda msgs: (
        gemma_calls.append(msgs) or '{"predicted_label": "Valid"}'
    )
    assert gemma_client.complete_json("sys", "prompt") == {"predicted_label": "Valid"}
    assert len(gemma_calls) == 1


def test_vbc_candidate_classes_filtering_and_no_abstention_with_context(tmp_path: Path):
    """Verify VBC filters schemes by candidate_classes and forbids Not Enough Context when context is present."""
    seen_prompts = []

    def mock_transport(messages, model, temperature):
        prompt = messages[-1]["content"]
        seen_prompts.append(prompt)
        if "VBC STEP 1" in prompt:
            return '{"scheme_id": "naturalness_appeal", "rationale": "Appeals to nature"}'
        if "VBC STEP 2" in prompt:
            return '{"verifications": [{"question_id": "q_9_1", "state": "Not Mentioned", "evidence": "No warrant provided"}, {"question_id": "q_9_2", "state": "Satisfied", "evidence": "Ok"}]}'
        return '{"predicted_label": "Valid"}'

    llm = LLMClient(cache_path=str(tmp_path / "cache.db"), transport=mock_transport)
    cocolofa_item = {
        "id": "cocolofa_1",
        "condition": "benchmark",
        "argument": "Censorship is unnatural, so it is always wrong.",
        "context": "News Article: Debate over internet regulations.",
        "true_label": "Appeal to Nature",
    }
    res = classify_vbc(cocolofa_item, llm, candidate_classes=COCOLOFA_FALLACY_CLASSES)
    assert "analogical_comparison" not in seen_prompts[0]
    assert "causal_claim" not in seen_prompts[0]
    assert "speaker_critique" not in seen_prompts[0]
    assert res["predicted_fine_label"] == "Appeal to Nature"
    assert res["predicted_label"] == "Appeal to Nature"
    assert res["violated_question_id"] == "q_9_1"


def test_strict_model_disables_cross_model_cache_and_fallback(tmp_path: Path, monkeypatch):
    """Verify strict_model=True only hits the requested model's cache and refuses fallback rotation."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    cache_file = tmp_path / "strict_cache.db"

    fallback_client = LLMClient(model="gemini-3.1-flash-lite", cache_path=str(cache_file))
    fallback_client._save_cached_response(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "prompt"}],
        '{"predicted_label": "False Cause"}',
    )

    # Default non-strict mode reads fallback cache entry
    non_strict = LLMClient(model="gemini-3.5-flash-lite", cache_path=str(cache_file))
    assert non_strict.complete_json("sys", "prompt") == {"predicted_label": "False Cause"}

    # Strict mode ignores 3.1 cache entry, invokes 3.5 directly, and blocks fallback switching
    calls = []
    strict = LLMClient(
        model="gemini-3.5-flash-lite",
        cache_path=str(cache_file),
        strict_model=True,
    )
    strict._invoke_model = lambda msgs: (
        calls.append(msgs) or '{"predicted_label": "Valid"}'
    )
    assert strict.complete_json("sys", "prompt") == {"predicted_label": "Valid"}
    assert len(calls) == 1
    assert strict._switch_to_next_gemini_model("daily quota reached") is False
    assert strict.model == "gemini-3.5-flash-lite"

