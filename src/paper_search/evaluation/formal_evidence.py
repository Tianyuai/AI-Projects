"""Shared reconstruction of formal Gate audit measures from published evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from urllib.parse import unquote

from paper_search.control.ledger import LedgerReport
from paper_search.control.pricing import QualityGatePolicy
from paper_search.domain.models import (
    CitationEdge,
    Paper,
    ProviderPaperId,
    QueryAnalysisResult,
    QuerySpec,
)
from paper_search.evaluation.business_results import BusinessResultRecord
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    IdentifierMap,
    PredictionRecord,
    normalize_paper_id,
)
from paper_search.evaluation.execution_adapter import (
    EvaluationExecutionRecord,
    EvaluationFailureRecord,
)
from paper_search.evaluation.gates import MeasureValue
from paper_search.evaluation.metrics import EvaluationResult, evaluate
from paper_search.llm.client import LLMResponseDecoder
from paper_search.processing import apply_hard_filters
from paper_search.query.parser import rule_fallback
from paper_search.retrieval.openalex import decode_openalex_page
from paper_search.retrieval.semantic_scholar import (
    decode_semantic_scholar_batch,
    decode_semantic_scholar_expansion,
    decode_semantic_scholar_search,
)
from paper_search.storage.dependency_snapshot import (
    DependencyRequestIdentity,
    DependencySnapshotManifestV2,
    DependencySnapshotReader,
    SnapshotEntryV2,
)


def _measure(numerator: int | Decimal, denominator: int | Decimal) -> MeasureValue:
    numerator_value = Decimal(numerator)
    denominator_value = Decimal(denominator)
    return MeasureValue(
        numerator=numerator_value,
        denominator=denominator_value,
        value=(
            numerator_value / denominator_value if denominator_value else None
        ),
    )


def configured_retrieval_endpoints(config: object) -> dict[str, str]:
    """Derive configured providers from immutable retrieval endpoint fields."""
    payload = config.model_dump() if hasattr(config, "model_dump") else config
    if not isinstance(payload, Mapping):
        raise TypeError("retrieval configuration must be a mapping or model")
    suffix = "_endpoint"
    return {
        str(name)[: -len(suffix)]: str(endpoint)
        for name, endpoint in payload.items()
        if str(name).endswith(suffix) and isinstance(endpoint, str) and endpoint
    }


def _source_edge_hash(provider: str, citing: str, cited: str) -> str:
    return "sha256:" + hashlib.sha256(
        f"{provider}|{citing}|{cited}".encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _DecodedResponse:
    papers: tuple[Paper, ...]
    candidate_papers: tuple[Paper, ...]
    raw_edges: tuple[CitationEdge, ...]


@dataclass(frozen=True)
class _RetrievalEvidence:
    parseable_search: bool
    candidate_papers: tuple[Paper, ...]
    all_papers: tuple[Paper, ...]
    bound_edges: frozenset[tuple[str, str, str]]
    openalex_candidate_ids: frozenset[str]


def _decode_provider_response(
    entry: SnapshotEntryV2,
    response_bytes: bytes,
) -> _DecodedResponse | None:
    dependency = entry.request.dependency
    operation = entry.request.operation
    try:
        if dependency == "openalex" and operation == "search":
            decoded = decode_openalex_page(response_bytes, limit=10_000)
            if decoded.errors:
                return None
            papers = tuple(decoded.papers)
            return _DecodedResponse(
                papers=papers,
                candidate_papers=papers,
                raw_edges=(),
            )
        if dependency != "semantic_scholar":
            return None
        if operation == "search":
            decoded_papers = decode_semantic_scholar_search(
                response_bytes,
                limit=10_000,
            )
        elif operation == "batch":
            decoded_papers = decode_semantic_scholar_batch(response_bytes)
        elif operation in {"references", "citations"}:
            endpoint_parts = entry.request.endpoint.strip("/").split("/")
            if len(endpoint_parts) != 3 or endpoint_parts[0] != "paper":
                return None
            decoded_expansion = decode_semantic_scholar_expansion(
                response_bytes,
                direction=operation,
                paper_id=ProviderPaperId(
                    provider="semantic_scholar",
                    value=unquote(endpoint_parts[1]),
                ),
                limit=10_000,
            )
            if decoded_expansion.errors:
                return None
            return _DecodedResponse(
                papers=tuple(decoded_expansion.expansion.papers),
                candidate_papers=(),
                raw_edges=tuple(decoded_expansion.expansion.raw_edges),
            )
        else:
            return None
        if decoded_papers.errors:
            return None
        papers = tuple(decoded_papers.papers)
        return _DecodedResponse(
            papers=papers,
            candidate_papers=papers,
            raw_edges=(),
        )
    except ValueError:
        return None


def _provider_id_bindings(papers: Sequence[Paper]) -> dict[tuple[str, str], str]:
    bindings: dict[tuple[str, str], str] = {}
    for paper in papers:
        if paper.openalex_id is not None:
            bindings[("openalex", paper.openalex_id)] = paper.canonical_id
        if paper.semantic_scholar_id is not None:
            bindings[("semantic_scholar", paper.semantic_scholar_id)] = paper.canonical_id
    return bindings


def _resolved_provider_id(
    provider_id: ProviderPaperId,
    *,
    bindings: Mapping[tuple[str, str], str],
    resolve: Callable[[str], str],
) -> str | None:
    identifier = bindings.get((provider_id.provider, provider_id.value))
    if identifier is None:
        try:
            identifier = normalize_paper_id(provider_id.value, kind=provider_id.provider)
        except ValueError:
            return None
    return resolve(identifier)


def _snapshot_retrieval_evidence(
    execution: EvaluationExecutionRecord,
    *,
    configured_endpoints: Mapping[str, str],
    snapshot_manifest: DependencySnapshotManifestV2 | None,
    snapshot_reader: DependencySnapshotReader | None,
    resolve: Callable[[str], str],
) -> _RetrievalEvidence:
    if snapshot_manifest is None or snapshot_reader is None:
        return _RetrievalEvidence(False, (), (), frozenset(), frozenset())
    entries = {
        (entry.request.dependency, entry.cache_key): entry
        for entry in snapshot_manifest.entries
    }
    parseable_search = False
    candidate_papers: list[Paper] = []
    all_papers: list[Paper] = []
    raw_edges: list[CitationEdge] = []
    openalex_candidate_ids: set[str] = set()
    for diagnostic in execution.diagnostics:
        configured_endpoint = configured_endpoints.get(diagnostic.dependency)
        if configured_endpoint is None or diagnostic.errors:
            continue
        for ref in diagnostic.snapshot_refs:
            entry = entries.get((ref.dependency, ref.cache_key))
            if (
                entry is None
                or entry.request.dependency != diagnostic.dependency
                or entry.request.operation == "search"
                and not configured_endpoint.endswith(entry.request.endpoint)
            ):
                continue
            try:
                snapshot = snapshot_reader.read(entry.request)
            except (KeyError, OSError, ValueError):
                continue
            if snapshot.ref != ref:
                continue
            decoded = _decode_provider_response(entry, snapshot.response_bytes)
            if decoded is None:
                continue
            candidate_papers.extend(decoded.candidate_papers)
            all_papers.extend(decoded.papers)
            raw_edges.extend(decoded.raw_edges)
            if entry.request.dependency == "openalex":
                openalex_candidate_ids.update(
                    resolve(paper.canonical_id)
                    for paper in decoded.candidate_papers
                )
            parseable_search |= entry.request.operation == "search"
    bindings = _provider_id_bindings(all_papers)
    bound_edges: set[tuple[str, str, str]] = set()
    for edge in raw_edges:
        citing = _resolved_provider_id(
            edge.citing_provider_id,
            bindings=bindings,
            resolve=resolve,
        )
        cited = _resolved_provider_id(
            edge.cited_provider_id,
            bindings=bindings,
            resolve=resolve,
        )
        if citing is not None and cited is not None:
            bound_edges.add((edge.provider, citing, cited))
    return _RetrievalEvidence(
        parseable_search=parseable_search,
        candidate_papers=tuple(candidate_papers),
        all_papers=tuple(all_papers),
        bound_edges=frozenset(bound_edges),
        openalex_candidate_ids=frozenset(openalex_candidate_ids),
    )


def _frozen_query_spec(query: EvaluationQuery) -> QuerySpec:
    payload = query.metadata.get("query_spec")
    if payload is None:
        return rule_fallback(query.query)
    if not isinstance(payload, Mapping):
        raise ValueError("frozen query_spec metadata must be an object")
    return QuerySpec.model_validate(
        {
            **payload,
            "original_query": query.query,
            "research_goal": payload.get("research_goal", query.query),
        }
    )


def _bound_llm_query_specs(
    execution: EvaluationExecutionRecord,
    *,
    query: EvaluationQuery,
    prompt_version: str | None,
    prompt_name: str | None,
    llm_model_allowlist: frozenset[str] | None,
    snapshot_manifest: DependencySnapshotManifestV2 | None,
    snapshot_reader: DependencySnapshotReader | None,
) -> tuple[QuerySpec, ...]:
    if (
        prompt_version is None
        or prompt_name is None
        or not llm_model_allowlist
        or snapshot_manifest is None
        or snapshot_reader is None
    ):
        return ()
    entries = {
        (entry.request.dependency, entry.cache_key): entry
        for entry in snapshot_manifest.entries
    }
    decoder = LLMResponseDecoder(prompt_version=prompt_version)
    specs: list[QuerySpec] = []
    for diagnostic in execution.diagnostics:
        if diagnostic.dependency != "llm" or diagnostic.errors:
            continue
        for ref in diagnostic.snapshot_refs:
            entry = entries.get((ref.dependency, ref.cache_key))
            if (
                entry is None
                or entry.request.dependency != "llm"
                or entry.request.operation != "generate_json"
                or entry.request.model_or_adapter not in llm_model_allowlist
            ):
                continue
            expected_request = DependencyRequestIdentity.from_canonical_request(
                dependency="llm",
                operation="generate_json",
                method="POST",
                endpoint="/chat/completions",
                model_or_adapter=entry.request.model_or_adapter,
                canonical_request={
                    "prompt_name": prompt_name,
                    "payload": {"query": query.query},
                    "prompt_version": prompt_version,
                },
            )
            if entry.request != expected_request:
                continue
            try:
                snapshot = snapshot_reader.read(entry.request)
            except (KeyError, OSError, ValueError):
                continue
            if snapshot.ref != ref:
                continue
            decoded = decoder.decode(
                snapshot.response_bytes,
                model_id=entry.request.model_or_adapter,
                captured_at=entry.captured_at,
                cache_hit=diagnostic.cache_hit,
                snapshot_ref=snapshot.ref,
            )
            if decoded.errors:
                continue
            try:
                analysis = QueryAnalysisResult.model_validate(decoded.data)
            except ValueError:
                continue
            specs.append(
                analysis.query_spec.model_copy(
                    update={"original_query": " ".join(query.query.split())}
                )
            )
    return tuple(specs)


def _filter_query_spec(
    query: EvaluationQuery,
    *,
    record: BusinessResultRecord | None,
    execution: EvaluationExecutionRecord,
    prompt_version: str | None,
    prompt_name: str | None,
    llm_model_allowlist: frozenset[str] | None,
    snapshot_manifest: DependencySnapshotManifestV2 | None,
    snapshot_reader: DependencySnapshotReader | None,
) -> tuple[QuerySpec, bool]:
    frozen_spec = _frozen_query_spec(query)
    if record is None or record.query_analysis is None:
        return frozen_spec, False
    published_spec = record.query_analysis.query_spec
    bound_specs = _bound_llm_query_specs(
        execution,
        query=query,
        prompt_version=prompt_version,
        prompt_name=prompt_name,
        llm_model_allowlist=llm_model_allowlist,
        snapshot_manifest=snapshot_manifest,
        snapshot_reader=snapshot_reader,
    )
    if published_spec in bound_specs:
        return published_spec, False
    if "query_spec" in query.metadata and published_spec == frozen_spec:
        return frozen_spec, False
    if record.planner_fallback and published_spec == frozen_spec:
        return frozen_spec, False
    return frozen_spec, True


def formal_audit_measures(
    *,
    frozen_queries: Sequence[EvaluationQuery],
    executions: Sequence[EvaluationExecutionRecord],
    business_results: Sequence[BusinessResultRecord],
    failures: Sequence[EvaluationFailureRecord],
    ledger_report: LedgerReport,
    identifier_map: IdentifierMap | None = None,
    metrics: EvaluationResult | None = None,
    configured_endpoints: Mapping[str, str] | None = None,
    snapshot_manifest: DependencySnapshotManifestV2 | None = None,
    snapshot_reader: DependencySnapshotReader | None = None,
    prompt_version: str | None = None,
    prompt_name: str | None = None,
    llm_model_allowlist: frozenset[str] | None = None,
) -> dict[str, MeasureValue]:
    """Derive all applicable enforced and core reporting measures."""
    count = len(frozen_queries)
    diagnostics = [item for execution in executions for item in execution.diagnostics]
    error_codes = [error.code for item in diagnostics for error in item.errors]
    hard_failure_count = len(failures)
    resolve = identifier_map.resolve if identifier_map is not None else lambda value: value
    execution_by_query = {execution.query_id: execution for execution in executions}
    structured_by_query = {record.query_id: record for record in business_results}
    relevant_count = 0
    retrieved_relevant_count = 0
    post_filter_relevant_count = 0
    parseable_retrieval_query_count = 0
    response_ids_by_query: dict[str, set[str]] = {}
    bound_edges_by_query: dict[str, frozenset[tuple[str, str, str]]] = {}
    openalex_ids_by_query: dict[str, frozenset[str]] = {}
    query_spec_binding_failures = 0
    configured = configured_endpoints or {}
    for query in frozen_queries:
        relevant = {resolve(identifier) for identifier in query.relevant_paper_ids}
        relevant_count += len(relevant)
        execution = execution_by_query.get(query.query_id)
        if execution is None:
            continue
        evidence = _snapshot_retrieval_evidence(
            execution,
            configured_endpoints=configured,
            snapshot_manifest=snapshot_manifest,
            snapshot_reader=snapshot_reader,
            resolve=resolve,
        )
        response_ids_by_query[query.query_id] = {
            resolve(paper.canonical_id) for paper in evidence.all_papers
        }
        bound_edges_by_query[query.query_id] = evidence.bound_edges
        openalex_ids_by_query[query.query_id] = evidence.openalex_candidate_ids
        parseable_retrieval_query_count += evidence.parseable_search
        retrieved = {
            resolve(paper.canonical_id) for paper in evidence.candidate_papers
        }
        record = structured_by_query.get(query.query_id)
        query_spec, binding_failed = _filter_query_spec(
            query,
            record=record,
            execution=execution,
            prompt_version=prompt_version,
            prompt_name=prompt_name,
            llm_model_allowlist=llm_model_allowlist,
            snapshot_manifest=snapshot_manifest,
            snapshot_reader=snapshot_reader,
        )
        query_spec_binding_failures += binding_failed
        filtered = apply_hard_filters(evidence.candidate_papers, query_spec)
        post_filter = {
            resolve(item.paper.canonical_id) for item in filtered.accepted
        }
        retrieved_relevant_count += len(relevant & retrieved)
        post_filter_relevant_count += len(relevant & post_filter)
    actual_matches = all(
        getattr(ledger_report.actual, field)
        == sum(getattr(execution.usage, field) for execution in executions)
        for field in (
            "search_api_calls",
            "llm_calls",
            "input_tokens",
            "output_tokens",
            "elapsed_ms",
        )
    )
    execution_costs = [execution.usage.cost_cny for execution in executions]
    expected_cost = (
        sum((value for value in execution_costs if value is not None), Decimal("0"))
        if all(value is not None for value in execution_costs)
        else None
    )
    actual_matches = actual_matches and ledger_report.actual.cost_cny == expected_cost
    clean_diagnostics = sum(
        item.endpoint == "dependency"
        and all(
            error.request_id is None
            and error.provider == item.dependency
            and error.message == "Dependency execution reported an error"
            for error in item.errors
        )
        for item in diagnostics
    )
    latency_values = sorted(execution.usage.elapsed_ms for execution in executions)
    p50 = latency_values[(len(latency_values) - 1) // 2] if latency_values else 0
    p95 = latency_values[max(0, (95 * len(latency_values) + 99) // 100 - 1)] if latency_values else 0
    cached_latency_values = sorted(
        item.latency_ms for item in diagnostics if item.cache_hit
    )

    def canonical_id(identifier: str) -> bool:
        try:
            return normalize_paper_id(identifier) == identifier
        except ValueError:
            return False

    schema_valid_queries = sum(
        query.query_id in structured_by_query for query in frozen_queries
    )
    valid_link_queries = 0
    reason_complete_queries = 0
    verifiable_edge_queries = 0
    fabricated_count = 0
    for query in frozen_queries:
        record = structured_by_query.get(query.query_id)
        execution = execution_by_query.get(query.query_id)
        if record is None:
            continue
        ranked = [*record.high_relevance, *record.partial_relevance]
        ranked_ids = {item.paper.canonical_id for item in ranked}
        linked_ids = [
            *record.selected_paper_ids,
            *ranked_ids,
            *(edge.citing_canonical_id for edge in record.citation_edges),
            *(edge.cited_canonical_id for edge in record.citation_edges),
        ]
        links_valid = all(canonical_id(identifier) for identifier in linked_ids)
        valid_link_queries += links_valid
        reason_complete_queries += set(record.selected_paper_ids) <= ranked_ids
        trusted_ids = {
            resolve(identifier) for identifier in query.relevant_paper_ids
        } | response_ids_by_query.get(query.query_id, set())
        bound_edges = [
            canonical_id(edge.citing_canonical_id)
            and canonical_id(edge.cited_canonical_id)
            and resolve(edge.citing_canonical_id) in trusted_ids
            and resolve(edge.cited_canonical_id) in trusted_ids
            and edge.source_edge_hash
            == _source_edge_hash(
                edge.provider,
                edge.citing_canonical_id,
                edge.cited_canonical_id,
            )
            and (
                edge.provider,
                resolve(edge.citing_canonical_id),
                resolve(edge.cited_canonical_id),
            )
            in bound_edges_by_query.get(query.query_id, frozenset())
            for edge in record.citation_edges
        ]
        edges_valid = all(bound_edges)
        verifiable_edge_queries += edges_valid
        published_paper_ids = set(record.selected_paper_ids) | ranked_ids
        fabricated_count += sum(
            not canonical_id(identifier) or resolve(identifier) not in trusted_ids
            for identifier in published_paper_ids
        )
        fabricated_count += sum(not bound for bound in bound_edges)
    measures = {
        "integrity_failures": _measure(
            sum(failure.error_code == "integrity_failure" for failure in failures),
            1,
        ),
        "provenance_failures": _measure(
            sum(
                code in {"integrity_failure", "snapshot_unavailable"}
                for code in error_codes
            )
            + query_spec_binding_failures,
            1,
        ),
        "sanitization_failures": _measure(len(diagnostics) - clean_diagnostics, 1),
        "unaccounted_usage_failures": _measure(0 if actual_matches else 1, 1),
        "valid_model_produced_query_analysis_rate": _measure(
            sum(
                record.query_analysis is not None and record.planner_status == "primary"
                for record in business_results
            ),
            count,
        ),
        "parseable_configured_retrieval_response_rate": _measure(
            parseable_retrieval_query_count,
            count,
        ),
        "hard_filter_absolute_recall_loss": _measure(
            retrieved_relevant_count - post_filter_relevant_count,
            relevant_count,
        ),
        "hard_failure_rate": _measure(hard_failure_count, count),
        "partial_result_rate": _measure(
            sum(record.is_partial for record in business_results), count
        ),
        "planner_fallback_rate": _measure(
            sum(record.planner_fallback for record in business_results), count
        ),
        "latency_p50_ms": _measure(p50, 1),
        "latency_p95_ms": _measure(p95, 1),
        "external_calls": _measure(ledger_report.actual.search_api_calls, 1),
        "actual_tokens": _measure(
            ledger_report.actual.input_tokens + ledger_report.actual.output_tokens, 1
        ),
        "valued_cost_cny": _measure(ledger_report.actual.cost_cny or 0, 1),
        "cache_hit_rate": _measure(
            sum(item.cache_hit for item in diagnostics), len(diagnostics)
        ),
        "schema_valid_rate": _measure(schema_valid_queries, count),
        "valid_paper_link_rate": _measure(valid_link_queries, count),
        "reason_complete_rate": _measure(reason_complete_queries, count),
        "verifiable_citation_edge_rate": _measure(verifiable_edge_queries, count),
        "fabricated_paper_or_relation_count": _measure(fabricated_count, 1),
    }
    if cached_latency_values:
        cached_p50 = cached_latency_values[(len(cached_latency_values) - 1) // 2]
        measures["cached_repeat_latency_p50_ms"] = _measure(cached_p50, 1)
    if metrics is not None:
        splits = {query.metadata.get("split") for query in frozen_queries}
        if len(splits) != 1 or not splits <= {"dev", "validation"}:
            raise ValueError("formal reporting requires one dev or validation split")
        split = str(next(iter(splits)))
        macro_f1 = metrics.measures["macro_f1"]
        measures[f"{split}_macro_f1"] = MeasureValue.model_validate(macro_f1.model_dump())
        raw_predictions = [
            PredictionRecord(
                query_id=query.query_id,
                predicted_paper_ids=sorted(
                    openalex_ids_by_query.get(query.query_id, frozenset())
                ),
            )
            for query in frozen_queries
        ]
        raw_macro_f1 = evaluate(
            frozen_queries, raw_predictions, id_map=identifier_map
        ).measures["macro_f1"].value
        delta = (macro_f1.value or Decimal(0)) - (raw_macro_f1 or Decimal(0))
        measures[f"{split}_macro_f1_delta_vs_raw_openalex"] = _measure(delta, 1)
    measures.update(
        {
            f"hard_failed_query:{failure.query_id}": _measure(1, 1)
            for failure in failures
        }
    )
    return measures


def complete_policy_measures(
    measures: dict[str, MeasureValue],
    *,
    policy: QualityGatePolicy,
    split: str,
) -> dict[str, MeasureValue]:
    """Preserve evidence without manufacturing rows that can mask real metrics."""
    completed = measures.copy()
    del policy, split
    return completed


__all__ = [
    "complete_policy_measures",
    "configured_retrieval_endpoints",
    "formal_audit_measures",
]
