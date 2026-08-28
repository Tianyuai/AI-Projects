"""Training-safe query-policy foundations."""

from paper_search.learning.data_isolation import (
    DatasetExample,
    DatasetIsolationIssue,
    DatasetIsolationReport,
    DatasetPartition,
    DatasetRoleRegistry,
    assert_training_safe,
    audit_dataset_isolation,
    load_dataset_role_registry,
)
from paper_search.learning.contracts import (
    PolicyActionCandidate,
    QueryPolicyInput,
    QueryPolicyOutput,
)
from paper_search.learning.policy import BoundedQueryPolicy, RuleActionScorer
from paper_search.learning.routing import RuleQueryRouter
from paper_search.learning.adapters import (
    QueryPolicyAnalyzerAdapter,
    RecallQueryPolicyGenerator,
)
from paper_search.learning.model_bundle import ModelBundleManifest, verify_model_bundle
from paper_search.learning.promotion import (
    ActionRankerEvaluationSummary,
    ActionRankerPromotionCriteria,
    evaluate_action_ranker_promotion,
)
from paper_search.learning.training_samples import (
    ActionObservation,
    ActionTrainingExample,
    build_action_training_examples,
)
from paper_search.learning.training_freeze import (
    TrainingFreezeManifest,
    freeze_pasa_training_data,
)
from paper_search.learning.action_labels import (
    ActionWeakLabel,
    build_action_labels,
    freeze_action_labels,
)
from paper_search.learning.candidates import DeterministicActionCandidateGenerator
from paper_search.learning.cpu_action_ranker import (
    CpuActionRanker,
    run_cpu_action_experiment,
)
from paper_search.learning.deployment import (
    build_cpu_action_analyzer_decorator,
    load_cpu_action_policy,
)
from paper_search.learning.pasa_training_augmentation import (
    write_pasa_augmented_handoff,
    write_pasa_supplement_receipt,
)

__all__ = [
    "DatasetExample",
    "DatasetIsolationIssue",
    "DatasetIsolationReport",
    "DatasetPartition",
    "DatasetRoleRegistry",
    "assert_training_safe",
    "audit_dataset_isolation",
    "load_dataset_role_registry",
    "ActionObservation",
    "ActionTrainingExample",
    "PolicyActionCandidate",
    "QueryPolicyInput",
    "QueryPolicyOutput",
    "BoundedQueryPolicy",
    "RuleActionScorer",
    "RuleQueryRouter",
    "QueryPolicyAnalyzerAdapter",
    "RecallQueryPolicyGenerator",
    "ModelBundleManifest",
    "verify_model_bundle",
    "ActionRankerEvaluationSummary",
    "ActionRankerPromotionCriteria",
    "evaluate_action_ranker_promotion",
    "build_action_training_examples",
    "TrainingFreezeManifest",
    "freeze_pasa_training_data",
    "ActionWeakLabel",
    "build_action_labels",
    "freeze_action_labels",
    "DeterministicActionCandidateGenerator",
    "CpuActionRanker",
    "run_cpu_action_experiment",
    "build_cpu_action_analyzer_decorator",
    "load_cpu_action_policy",
    "write_pasa_augmented_handoff",
    "write_pasa_supplement_receipt",
]
