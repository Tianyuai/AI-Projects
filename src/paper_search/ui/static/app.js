"use strict";

const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const modeSelect = document.querySelector("#mode");
const submitButton = document.querySelector("#submit-search");
const cancelButton = document.querySelector("#cancel-search");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const resultSummary = document.querySelector("#result-summary");
const selectedGroup = document.querySelector("#selected-group");
const selectedResults = document.querySelector("#selected-results");
const selectedCount = document.querySelector("#selected-count");
const highGroup = document.querySelector("#high-group");
const highResults = document.querySelector("#high-results");
const partialGroup = document.querySelector("#partial-group");
const partialResults = document.querySelector("#partial-results");
const highCount = document.querySelector("#high-count");
const partialCount = document.querySelector("#partial-count");
const emptyState = document.querySelector("#empty-state");
const evidencePanel = document.querySelector("#evidence-panel");
const provenance = document.querySelector("#provenance");
const diagnostics = document.querySelector("#diagnostics");
const queryUnderstanding = document.querySelector("#query-understanding");
const systemPipeline = document.querySelector("#system-pipeline");
const executionTrace = document.querySelector("#execution-trace");
const costSummary = document.querySelector("#cost-summary");

let activeController = null;
let requestSequence = 0;

function textElement(tagName, value, className) {
  const element = document.createElement(tagName);
  element.textContent = String(value ?? "Unknown");
  if (className) element.className = className;
  return element;
}

function listText(values, fallback = "Not provided") {
  return Array.isArray(values) && values.length ? values.join(", ") : fallback;
}

function formatScore(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "Not scored";
}

function appendDefinition(label, value) {
  provenance.append(textElement("dt", label), textElement("dd", value));
}

function appendDefinitionTo(container, label, value) {
  container.append(textElement("dt", label), textElement("dd", value));
}

function displayValue(value) {
  if (Array.isArray(value)) return listText(value);
  if (value && typeof value === "object") return JSON.stringify(value);
  return value ?? "Not provided";
}

function clearOutput() {
  selectedResults.replaceChildren();
  highResults.replaceChildren();
  partialResults.replaceChildren();
  provenance.replaceChildren();
  diagnostics.replaceChildren();
  queryUnderstanding.replaceChildren();
  systemPipeline.replaceChildren();
  executionTrace.replaceChildren();
  costSummary.replaceChildren();
  selectedGroup.hidden = true;
  highGroup.hidden = true;
  partialGroup.hidden = true;
  results.hidden = true;
  emptyState.hidden = true;
  evidencePanel.hidden = true;
}

function pipelineStage(index, title, detail, statusLabel, tone = "complete") {
  const item = document.createElement("li");
  item.className = `pipeline-stage ${tone}`;
  item.append(textElement("span", String(index).padStart(2, "0"), "pipeline-index"));
  item.append(textElement("strong", title));
  item.append(textElement("small", detail, "pipeline-detail"));
  item.append(textElement("span", statusLabel, "pipeline-status"));
  return item;
}

function renderSystemPipeline(payload) {
  const trace = payload.search_trace ?? [];
  const analysis = payload.query_analysis ?? {};
  const spec = analysis.query_spec ?? {};
  const plan = analysis.search_plan ?? {};
  const dependencies = payload.dependency_status ?? [];
  const selectedPaperIds = payload.selected_paper_ids ?? [];
  const firstTrace = (step) => trace.find((entry) => entry.step === step);
  const allTrace = (step) => trace.filter((entry) => entry.step === step);

  const analyze = firstTrace("analyze");
  const llmState = dependencies.find((item) => item.dependency === "llm")?.state;
  const plannerFallback = Boolean(payload.planner_fallback);
  const plannedActions = plan.subqueries?.length ?? 0;
  const understandingDetail = [
    "QuerySpec + controlled search plan",
    analyze?.model_id ?? "Bound LLM analyzer",
    payload.prompt_version ?? "bound prompt",
    `${plannedActions} finalized actions`,
  ].join(" · ");

  const bridge = firstTrace("supervised_query_expansion");
  const bridgeStatus = bridge?.status ?? "not_run";
  const bridgeAppended = bridgeStatus === "appended";
  const bridgeDetail = bridge
    ? [
        bridge.model_id ?? "frozen supervised bridge",
        bridge.training_query_count
          ? `${bridge.training_query_count} training queries`
          : "hash-bound model",
        `${bridge.action_count_before ?? 0}→${bridge.action_count_after ?? 0} actions`,
      ].join(" · ")
    : "No supervised bridge receipt was returned";

  const retrievals = allTrace("retrieve");
  const retrievalProviders = new Set(retrievals.map((entry) => entry.provider));
  const providerStates = Object.fromEntries(
    dependencies.map((item) => [item.dependency, item.state])
  );
  const retrievalCount = retrievals.reduce(
    (total, entry) => total + Number(entry.result_count ?? 0),
    0
  );
  const retrievalDetail = [
    `OpenAlex: ${retrievalProviders.has("openalex") ? "used" : providerStates.openalex ?? "not used"}`,
    `Semantic Scholar: ${retrievalProviders.has("semantic_scholar") ? "used" : providerStates.semantic_scholar ?? "not used"}`,
    `${retrievals.length} calls · ${retrievalCount} returned rows`,
  ].join(" · ");

  const boundedSupplement = allTrace("retrieve_supplement");
  const confidence = firstTrace("assess_low_confidence_recall");
  const llmSupplement = allTrace("retrieve_llm_supplement");
  const supplementSkip = firstTrace("skip_llm_supplement");
  const supplementFallback = firstTrace("llm_supplement_fallback");
  const adaptiveParts = [];
  if (boundedSupplement.length) {
    adaptiveParts.push(`${boundedSupplement.length} bounded cross-vocabulary call`);
  }
  if (confidence) {
    adaptiveParts.push(
      confidence.low_confidence ? "low confidence detected" : "production pool adequate"
    );
  }
  if (llmSupplement.length) {
    adaptiveParts.push(`${llmSupplement.length} protected LLM supplement call`);
  } else if (supplementFallback) {
    adaptiveParts.push(`baseline retained: ${supplementFallback.reason}`);
  } else if (supplementSkip) {
    adaptiveParts.push(`abstained: ${supplementSkip.reason}`);
  }
  const adaptiveActive = boundedSupplement.length > 0 || llmSupplement.length > 0;
  const adaptiveDetail = adaptiveParts.length
    ? adaptiveParts.join(" · ")
    : "No supplemental recall action was required";

  const cap = firstTrace("candidate_cap");
  const deduplicate = firstTrace("deduplicate");
  const filter = firstTrace("filter");
  const fusion = firstTrace("fuse");
  const candidateDetail = [
    `${cap?.deduplicated_after ?? deduplicate?.count ?? 0} deduplicated candidates`,
    `${deduplicate?.identifier_alias_count ?? 0} PASA-derived production aliases available`,
    `${filter?.accepted ?? 0} passed hard filters`,
    fusion ? "source evidence fused" : "fusion receipt unavailable",
  ].join(" · ");

  const rank = firstTrace("document_rank");
  const rankRole = rank?.deployment_role ?? "deterministic fallback";
  const failoverCount = Array.isArray(rank?.failover_receipt)
    ? rank.failover_receipt.length
    : 0;
  const rankDetail = [
    rank?.model_id ?? "bound document ranker",
    failoverCount ? `${failoverCount} recorded failover` : "no failover",
    `${selectedPaperIds.length} selected papers`,
  ].join(" · ");
  const rankFallback = rankRole !== "F5-gated-fusion";

  const deliveryDetail = [
    "One structured response for HTTP API, browser UI, CLI, and JSONL batch",
    `${payload.execution_mode ?? "unknown"} execution`,
    payload.snapshot_set_id ? `snapshot ${payload.snapshot_set_id}` : "snapshot pending",
  ].join(" · ");

  const stages = [
    pipelineStage(
      1,
      "Natural-language query",
      spec.research_goal ?? spec.original_query ?? "Research question accepted",
      "accepted"
    ),
    pipelineStage(
      2,
      "LLM understanding",
      understandingDetail,
      plannerFallback ? "rules fallback" : llmState ?? "complete",
      plannerFallback ? "fallback" : "complete"
    ),
    pipelineStage(
      3,
      "Supervised vocabulary bridge",
      bridgeDetail,
      bridgeStatus.replaceAll("_", " "),
      bridgeAppended ? "active" : bridge ? "skipped" : "fallback"
    ),
    pipelineStage(
      4,
      "OpenAlex + Semantic Scholar",
      retrievalDetail,
      retrievals.length ? "retrieved" : "no retrieval receipt",
      retrievals.length ? "complete" : "fallback"
    ),
    pipelineStage(
      5,
      "Adaptive supplemental recall",
      adaptiveDetail,
      adaptiveActive ? "activated" : supplementFallback ? "baseline retained" : "abstained",
      supplementFallback ? "fallback" : adaptiveActive ? "active" : "skipped"
    ),
    pipelineStage(
      6,
      "Identity, filtering, and fusion",
      candidateDetail,
      "candidate pool finalized"
    ),
    pipelineStage(
      7,
      "F5 → F4 → B0 ranking",
      rankDetail,
      rankRole,
      rankFallback ? "fallback" : "complete"
    ),
    pipelineStage(
      8,
      "Structured delivery + replay",
      deliveryDetail,
      payload.execution_mode === "replay" ? "zero-network replay" : "captured live run"
    ),
  ];
  systemPipeline.append(...stages);
}

function renderQueryUnderstanding(payload) {
  const analysis = payload.query_analysis ?? {};
  const spec = analysis.query_spec ?? {};
  const plan = analysis.search_plan ?? {};
  const definitions = document.createElement("dl");
  definitions.className = "definition-grid";
  appendDefinitionTo(definitions, "Research goal", spec.research_goal);
  appendDefinitionTo(definitions, "Retrieval rationale", plan.rationale);
  appendDefinitionTo(definitions, "Topics", listText(spec.topics));
  appendDefinitionTo(definitions, "Tasks", listText(spec.tasks));
  appendDefinitionTo(definitions, "Methods", listText(spec.methods));
  appendDefinitionTo(definitions, "Datasets", listText(spec.datasets));
  appendDefinitionTo(definitions, "Domains", listText(spec.domains));
  appendDefinitionTo(
    definitions,
    "Year range",
    spec.year_from || spec.year_to ? `${spec.year_from ?? "…"}–${spec.year_to ?? "…"}` : "Not constrained"
  );
  appendDefinitionTo(definitions, "Required", listText(spec.must_have));
  appendDefinitionTo(definitions, "Excluded", listText(spec.exclusions));
  appendDefinitionTo(definitions, "Planner status", analysis.planner_status ?? payload.planner_status);
  queryUnderstanding.append(definitions);

  const heading = textElement("h3", "Planned retrieval actions");
  const actionList = document.createElement("ol");
  actionList.className = "action-list";
  (plan.subqueries ?? []).forEach((action) => {
    const item = document.createElement("li");
    item.append(textElement("strong", action.text));
    item.append(
      textElement(
        "small",
        `${action.provider_hint ?? "either"} · ${action.search_mode ?? "lexical"} · ${action.action_type ?? "text_search"} · anchors: ${listText(action.target_constraints)}`
      )
    );
    actionList.append(item);
  });
  queryUnderstanding.append(heading, actionList);
}

function renderExecutionTrace(payload) {
  const labels = {
    analyze: "Understand query",
    supervised_query_expansion: "Apply supervised vocabulary bridge",
    retrieve: "Retrieve candidates",
    retrieve_supplement: "Run bounded supplement",
    merge_supplement: "Merge supplement fairly",
    assess_low_confidence_recall: "Assess recall confidence",
    analyze_low_confidence_supplement: "Generate supplemental retrieval action",
    select_low_confidence_supplement: "Select safe supplemental action",
    retrieve_llm_supplement: "Retrieve low-confidence supplement",
    merge_llm_supplement: "Merge independent supplement quota",
    skip_llm_supplement: "Skip low-confidence supplement",
    llm_supplement_fallback: "Keep the production candidate pool",
    candidate_cap: "Apply candidate limits",
    deduplicate: "Deduplicate identities",
    filter: "Apply constraints",
    uncertainty_adjustment: "Apply uncertainty score",
    fuse: "Fuse source evidence",
    document_rank: "Rank candidates",
    truncate: "Select final output",
  };
  (payload.search_trace ?? []).forEach((entry, index) => {
    const item = document.createElement("li");
    item.className = "trace-item";
    const title = labels[entry.step] ?? entry.step ?? `Step ${index + 1}`;
    item.append(textElement("strong", `${index + 1}. ${title}`));
    const details = document.createElement("dl");
    details.className = "trace-grid";
    Object.entries(entry).forEach(([key, value]) => {
      if (key === "step") return;
      const fieldLabels = {
        deployment_role: "Ranker role",
        failover_receipt: "Fallback receipt",
        pricing_receipt: "Pricing receipt",
      };
      appendDefinitionTo(details, fieldLabels[key] ?? key.replaceAll("_", " "), displayValue(value));
    });
    item.append(details);
    executionTrace.append(item);
  });
  if (!executionTrace.childElementCount) {
    executionTrace.append(textElement("li", "No execution trace was returned.", "muted-text"));
  }
}

function renderCost(payload) {
  const usage = payload.usage ?? {};
  const pricingStep = (payload.search_trace ?? []).find((entry) => entry.pricing_receipt);
  let pricingReceipt = pricingStep?.pricing_receipt ?? null;
  if (typeof pricingReceipt === "string") {
    try {
      pricingReceipt = JSON.parse(pricingReceipt);
    } catch (_error) {
      pricingReceipt = null;
    }
  }
  appendDefinitionTo(costSummary, "Total cost (CNY)", usage.cost_cny ?? "Not available");
  appendDefinitionTo(costSummary, "LLM calls", usage.llm_calls ?? 0);
  appendDefinitionTo(costSummary, "Search calls", usage.search_api_calls ?? 0);
  appendDefinitionTo(costSummary, "Input tokens", usage.input_tokens ?? 0);
  appendDefinitionTo(costSummary, "Cached input tokens", usage.cached_input_tokens ?? 0);
  appendDefinitionTo(costSummary, "Uncached input tokens", usage.uncached_input_tokens ?? usage.input_tokens ?? 0);
  appendDefinitionTo(costSummary, "Output tokens", usage.output_tokens ?? 0);
  appendDefinitionTo(costSummary, "Price band", pricingReceipt?.time_band ?? "Not available");
  appendDefinitionTo(costSummary, "Pricing source", pricingReceipt?.source_identity ?? "Not available");
  appendDefinitionTo(
    costSummary,
    "Rates (CNY / 1M tokens)",
    displayValue(pricingReceipt?.rates_cny_per_million_tokens)
  );
  appendDefinitionTo(costSummary, "Price time", pricingReceipt?.valued_at ?? "Not available");
  appendDefinitionTo(costSummary, "Model concurrency", pricingReceipt?.concurrency_limit ?? "Not provided");
}

function setBusy(isBusy) {
  form.setAttribute("aria-busy", String(isBusy));
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "Searching…" : "Search papers";
  cancelButton.hidden = !isBusy;
}

function addChip(container, value, variant) {
  if (value === null || value === undefined || value === "") return;
  const chip = textElement("span", value, `chip${variant ? ` ${variant}` : ""}`);
  container.append(chip);
}

function evidenceRow(label, values, tone) {
  const row = document.createElement("div");
  row.className = "evidence-row";
  row.append(textElement("span", label, `evidence-label ${tone ?? ""}`));
  row.append(textElement("span", listText(values)));
  return row;
}

function renderPaper(item, group, selectedIds) {
  const paper = item.paper ?? item;
  const evidence = item.evidence ?? {};
  const article = document.createElement("article");
  article.className = `paper-card ${group}`;

  const top = document.createElement("div");
  top.className = "card-topline";
  const groupLabels = {
    high: "High relevance",
    partial: "Partial relevance",
    selected: "Selected result",
  };
  addChip(top, groupLabels[group] ?? "Result", group);
  if (group !== "selected" && selectedIds.has(paper.canonical_id)) {
    addChip(top, "Selected", "selected");
  }
  addChip(top, `Score ${formatScore(evidence.final_score ?? evidence.fusion_score ?? item.score)}`, "score");
  article.append(top);

  const title = textElement("h4", paper.title ?? "Untitled paper");
  if (paper.url) {
    const link = document.createElement("a");
    link.href = paper.url;
    link.target = "_blank";
    link.rel = "noreferrer noopener";
    link.textContent = title.textContent;
    title.replaceChildren(link);
  }
  article.append(title);

  const metadata = document.createElement("div");
  metadata.className = "metadata";
  addChip(metadata, paper.publication_year, "neutral");
  addChip(metadata, paper.venue, "neutral");
  addChip(metadata, paper.citation_count === null || paper.citation_count === undefined ? null : `${paper.citation_count} citations`, "neutral");
  (paper.sources ?? []).forEach((source) => addChip(metadata, source, "source"));
  if (!metadata.childElementCount) addChip(metadata, "Metadata unavailable", "muted");
  article.append(metadata);

  if (paper.authors?.length) article.append(textElement("p", listText(paper.authors), "authors"));
  article.append(textElement("p", paper.abstract || "Abstract not available from the current provider response.", `abstract${paper.abstract ? "" : " muted-text"}`));

  if (item.evidence) {
    const evidenceBox = document.createElement("div");
    evidenceBox.className = "paper-evidence";
    evidenceBox.append(
      evidenceRow("Matched query parts", evidence.matched_subqueries),
      evidenceRow("Matched constraints", evidence.matched_constraints, "positive"),
      evidenceRow("Open constraints", evidence.unmatched_constraints, "warning"),
      evidenceRow("Filter notes", evidence.filter_reasons)
    );
    article.append(evidenceBox);
  }

  const rankEntries = Object.entries(evidence.source_ranks ?? item.source_ranks ?? {});
  if (rankEntries.length) {
    article.append(textElement("p", `Source ranks: ${rankEntries.map(([source, rank]) => `${source} #${rank}`).join(" · ")}`, "source-ranks"));
  }
  return article;
}

function appendDiagnosticSection(title, entries, formatter, emptyMessage) {
  const section = document.createElement("section");
  section.append(textElement("h3", title));
  if (!entries.length) {
    section.append(textElement("p", emptyMessage, "muted-text"));
  } else {
    entries.forEach((entry) => section.append(textElement("p", formatter(entry))));
  }
  diagnostics.append(section);
}

function renderSuccess(payload) {
  clearOutput();
  const selectedIds = new Set(payload.selected_paper_ids ?? []);
  const high = payload.high_relevance ?? [];
  const partial = payload.partial_relevance ?? [];
  const rankedIds = new Set(
    [...high, ...partial].map((item) => (item.paper ?? item).canonical_id)
  );
  const selectedFallback = (payload.fused_papers ?? []).filter((item) => {
    const paper = item.paper ?? item;
    return selectedIds.has(paper.canonical_id) && !rankedIds.has(paper.canonical_id);
  });

  selectedFallback.forEach((item) => selectedResults.append(renderPaper(item, "selected", selectedIds)));
  high.forEach((item) => highResults.append(renderPaper(item, "high", selectedIds)));
  partial.forEach((item) => partialResults.append(renderPaper(item, "partial", selectedIds)));
  selectedCount.textContent = String(selectedFallback.length);
  highCount.textContent = String(high.length);
  partialCount.textContent = String(partial.length);
  selectedGroup.hidden = selectedFallback.length === 0;
  highGroup.hidden = high.length === 0;
  partialGroup.hidden = partial.length === 0;
  const displayedCount = selectedFallback.length + high.length + partial.length;
  resultSummary.textContent = `${selectedIds.size} selected · ${displayedCount} shown`;
  results.hidden = displayedCount === 0;
  emptyState.hidden = displayedCount !== 0;

  appendDefinition("Execution mode", payload.execution_mode);
  appendDefinition("Selected paper IDs", listText(payload.selected_paper_ids, "None"));
  appendDefinition("Snapshot set", payload.snapshot_set_id);
  appendDefinition("Snapshot captured", payload.snapshot_captured_at ?? "Not captured");
  appendDefinition("Config hash", payload.config_hash);
  appendDefinition("Run ID", payload.run_id);
  appendDefinition("Usage", JSON.stringify(payload.usage ?? {}));
  appendDefinition("Stop reason", payload.stop_reason);
  appendDefinition("Partial result", payload.is_partial);
  appendDefinition("Planner fallback", payload.planner_fallback);
  appendDefinition("Planner status", payload.planner_status);

  renderQueryUnderstanding(payload);
  renderSystemPipeline(payload);
  renderExecutionTrace(payload);
  renderCost(payload);

  appendDiagnosticSection(
    "Dependency statuses",
    payload.dependency_status ?? [],
    (dependency) => `${dependency.dependency}: ${dependency.state} (cache: ${dependency.cache_hit})`,
    "No dependency diagnostics were returned."
  );
  appendDiagnosticSection("Safe warnings", payload.warnings ?? [], (warning) => warning, "No warnings.");
  appendDiagnosticSection(
    "Citation edges",
    payload.citation_edges ?? [],
    (edge) => `${edge.citing_canonical_id} → ${edge.cited_canonical_id} (${edge.provider})`,
    "No citation edges were returned."
  );
  evidencePanel.hidden = false;
  status.textContent = payload.is_partial ? "Search completed with partial results." : "Search completed.";
  status.className = payload.is_partial ? "status warning" : "status success";
}

function renderError(payload) {
  clearOutput();
  status.textContent = `Search failed: ${payload.code}: ${payload.detail}`;
  status.className = "status error";
}

document.querySelectorAll(".example-query").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query ?? "";
    queryInput.focus();
  });
});

cancelButton.addEventListener("click", () => {
  activeController?.abort();
  requestSequence += 1;
  setBusy(false);
  status.textContent = "Search cancelled.";
  status.className = "status warning";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  activeController?.abort();
  const controller = new AbortController();
  activeController = controller;
  const sequence = ++requestSequence;
  const request = {
    query_id: `ui-${crypto.randomUUID()}`,
    query: queryInput.value,
    budget_profile: "balanced",
    include_trace: true,
    mode: modeSelect.value,
  };
  clearOutput();
  setBusy(true);
  status.textContent = "Loading search results…";
  status.className = "status loading";

  try {
    const response = await fetch("/v1/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: controller.signal,
    });
    const payload = await response.json();
    if (sequence !== requestSequence) return;
    if (response.ok) renderSuccess(payload);
    else renderError(payload);
  } catch (error) {
    if (error.name === "AbortError" || sequence !== requestSequence) return;
    renderError({ code: "request_failed", detail: "The search request could not be completed safely." });
  } finally {
    if (sequence === requestSequence) {
      activeController = null;
      setBusy(false);
    }
  }
});
