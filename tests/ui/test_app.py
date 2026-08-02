from __future__ import annotations

import ast
import importlib
from pathlib import Path


def _static_file(name: str) -> Path:
    ui_module = importlib.import_module("paper_search.ui.app")
    return Path(ui_module.__file__).parent / "static" / name


def test_browser_assets_are_present_and_semantic() -> None:
    html = _static_file("index.html").read_text(encoding="utf-8")

    for expected in (
        '<label for="query">',
        'id="query"',
        'id="mode"',
        'value="replay" selected',
        'role="status"',
        'id="results"',
        'id="provenance"',
        'id="diagnostics"',
        'src="/static/app.js"',
        'href="/static/styles.css"',
    ):
        assert expected in html


def test_browser_script_uses_only_canonical_request_and_safe_dom_rendering() -> None:
    script = _static_file("app.js").read_text(encoding="utf-8")

    assert 'fetch("/v1/search"' in script
    assert 'method: "POST"' in script
    assert 'headers: { "Content-Type": "application/json" }' in script
    assert 'query_id: `ui-${crypto.randomUUID()}`' in script
    assert "query: queryInput.value" in script
    assert 'budget_profile: "balanced"' in script
    assert "include_trace: true" in script
    assert "mode: modeSelect.value" in script
    assert "AbortController" in script
    assert "abort()" in script
    assert "requestSequence" in script
    assert "textContent" in script
    assert "innerHTML" not in script
    assert "insertAdjacentHTML" not in script
    assert "manifest" not in script.lower()
    assert "snapshot_path" not in script
    assert "new Set(payload.selected_paper_ids" in script
    assert "selectedIds.has(paper.canonical_id)" in script


def test_browser_script_renders_success_partial_and_typed_error_state() -> None:
    script = _static_file("app.js").read_text(encoding="utf-8")

    for expected in (
        "selected_paper_ids",
        "high_relevance",
        "partial_relevance",
        "citation_edges",
        "source_ranks",
        "fusion_score",
        "usage",
        "is_partial",
        "planner_fallback",
        "dependency_status",
        "warnings",
        "stop_reason",
        "execution_mode",
        "snapshot_captured_at",
        "snapshot_set_id",
        "config_hash",
        "run_id",
        "payload.code",
        "payload.detail",
        "Loading search results",
    ):
        assert expected in script


def test_ui_module_only_installs_routes_and_never_imports_evaluation_pipeline() -> None:
    ui_module = importlib.import_module("paper_search.ui.app")
    source = Path(ui_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "PipelineResult" not in source
    assert "evaluation.runner" not in source
    assert "def install_ui" in source
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "search"
        for node in ast.walk(tree)
    )
