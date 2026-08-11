"""Generation adapters for recall experiments."""

from paper_search.recall_experiments.generation.base import GenerationResult, QueryGenerator
from paper_search.recall_experiments.generation.fixed import FixedActionGenerator
from paper_search.recall_experiments.generation.manual import ManualActionGenerator
from paper_search.recall_experiments.generation.deepseek import (
    DeepSeekPromptGenerator,
    RecallGenerationFailure,
    RecallPromptArtifact,
    build_generation_payload,
    build_repair_payload,
    render_recall_prompt,
)

from paper_search.recall_experiments.generation.backends import (
    BudgetedLLMBackend,
    LLMBackend,
    LLMBackendResult,
    LLMGenerationRequest,
)

__all__ = [
    "BudgetedLLMBackend",
    "DeepSeekPromptGenerator",
    "LLMBackend",
    "LLMBackendResult",
    "LLMGenerationRequest",
    "FixedActionGenerator",
    "GenerationResult",
    "ManualActionGenerator",
    "QueryGenerator",
    "RecallGenerationFailure",
    "RecallPromptArtifact",
    "build_generation_payload",
    "build_repair_payload",
    "render_recall_prompt",
]
