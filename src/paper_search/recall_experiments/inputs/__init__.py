"""Replaceable frozen inputs and sealed Gold-document catalogs."""

from paper_search.recall_experiments.inputs.base import (
    FrozenInputSource,
    FrozenRecallDataset,
    HistoricalRecallBaseline,
    OpaqueEvaluationMaterials,
)
from paper_search.recall_experiments.inputs.formal_run import FormalRunInputSource
from paper_search.recall_experiments.inputs.gold_catalog import (
    BoundPaperSource,
    GoldDocumentCatalogBuilder,
    GoldDocumentCatalogSource,
    OracleCatalogStatus,
    SealedGoldDocumentCatalog,
    SealedGoldDocumentRecord,
)

__all__ = [
    "BoundPaperSource",
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
]
