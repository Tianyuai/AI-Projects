from __future__ import annotations

from pathlib import Path

from paper_search.learning.data_isolation import load_dataset_role_registry


def test_training_scope_contains_only_pasa_partitions() -> None:
    registry = load_dataset_role_registry(
        Path("configs/training/dataset-roles.yaml")
    )
    roles = {
        partition.identity: partition.role for partition in registry.partitions
    }

    assert roles == {
        "pasa/auto_dev": "development",
        "pasa/auto_test": "final_test",
        "pasa/auto_train": "training",
        "pasa/real_test": "final_test",
    }
    assert registry.partition_counts() == {
        identity: 0 for identity in sorted(roles)
    }
