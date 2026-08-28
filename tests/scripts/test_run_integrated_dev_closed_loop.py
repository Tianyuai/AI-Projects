from pathlib import Path

from paper_search.learning.gold_retrievability_audit import (
    build_frozen_audit_manifest,
    freeze_audit_manifest,
)
from scripts.run_integrated_dev_closed_loop import (
    _capture_root,
    _load_development_rows,
)


def test_closed_loop_capture_root_stays_bounded_on_windows() -> None:
    workspace = Path(r"D:\AI Projects\.worktrees\week3")
    response_path = (
        _capture_root(workspace)
        / "captures"
        / ("a" * 64)
        / "dependency-snapshot"
        / "responses"
        / "openalex"
        / (("b" * 64) + ".bin")
    )

    assert len(str(response_path)) < 240


def test_closed_loop_loads_exact_frozen_manifest_sample(tmp_path: Path) -> None:
    partition = tmp_path / "pasa_auto_dev.jsonl"
    partition.write_text(
        "".join(
            '{"dataset":"pasa","split":"auto_dev","role":"development",'
            '"revision":"r1","query_id":"q-%d","query":"query %d",'
            '"gold_paper_ids":["arxiv:%d"],"source_components":[]}\n'
            % (index, index, index)
            for index in range(8)
        ),
        encoding="utf-8",
    )
    manifest = build_frozen_audit_manifest(
        partition,
        sample_size=3,
        seed="closed-loop-test",
        excluded_query_ids=frozenset({"q-0"}),
    )
    manifest_path = tmp_path / "sample.json"
    freeze_audit_manifest(manifest_path, manifest)

    rows = _load_development_rows(
        partition,
        limit=3,
        sample_manifest=manifest_path,
    )

    assert [row["query_id"] for row in rows] == [
        item.query_id for item in manifest.sample
    ]
