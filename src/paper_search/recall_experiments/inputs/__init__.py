"""Replaceable frozen inputs and sealed Gold-document catalogs."""

from paper_search.recall_experiments.inputs.base import (
    FrozenInputSource,
    FrozenRecallDataset,
    HistoricalRecallBaseline,
    OpaqueEvaluationMaterials,
)
from paper_search.recall_experiments.inputs.formal_run import FormalRunInputSource
from paper_search.recall_experiments.inputs.gold_catalog import (
    GoldDocumentCatalogBuilder,
    GoldDocumentCatalogSource,
    OracleCatalogStatus,
    SealedGoldDocumentCatalog,
    SealedGoldDocumentRecord,
    SourceManifestEntry,
)

__all__ = [
    "FormalRunInputSource",
    "FrozenInputSource",
    "FrozenRecallDataset",
    "GoldDocumentCatalogBuilder",
    "GoldDocumentCatalogSource",
    "HistoricalRecallBaseline",
    "OpaqueEvaluationMaterials",
    "OracleCatalogStatus",
    "SealedGoldDocumentCatalog",
    "SealedGoldDocumentRecord",
    "SourceManifestEntry",
]
