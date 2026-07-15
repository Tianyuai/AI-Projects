import json
from importlib import import_module
from importlib.util import find_spec


def test_health_module_is_available() -> None:
    assert find_spec("paper_search.health") is not None


def test_collect_local_health_validates_gpu_and_retrieval_stack() -> None:
    health = import_module("paper_search.health")
    collector = getattr(health, "collect_local_health", None)

    assert callable(collector)

    report = collector(matrix_size=64)

    assert report["status"] == "ready"
    assert report["python"]["major_minor"] == "3.11"
    assert report["torch"]["version"] == "2.5.1+cu121"
    assert report["torch"]["cuda_build"] == "12.1"
    assert report["torch"]["cuda_available"] is True
    assert "RTX 3050 Ti" in report["torch"]["device"]
    assert report["torch"]["matrix_smoke"]["shape"] == [64, 64]
    assert report["torch"]["matrix_smoke"]["finite"] is True
    assert all(item["available"] for item in report["dependencies"].values())


def test_health_cli_outputs_json_without_environment_secrets(
    monkeypatch, capsys
) -> None:
    health = import_module("paper_search.health")
    main = getattr(health, "main", None)

    assert callable(main)

    sentinel = "must-not-appear-in-health-output"
    monkeypatch.setenv("LLM_API_KEY", sentinel)
    exit_code = main()
    output = capsys.readouterr().out
    report = json.loads(output)

    assert exit_code == 0
    assert report["status"] == "ready"
    assert sentinel not in output
