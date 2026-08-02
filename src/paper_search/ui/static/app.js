"use strict";

const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const modeSelect = document.querySelector("#mode");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const provenance = document.querySelector("#provenance");
const diagnostics = document.querySelector("#diagnostics");

let activeController = null;
let requestSequence = 0;

function appendText(parent, tagName, value) {
  const element = document.createElement(tagName);
  element.textContent = String(value ?? "Unknown");
  parent.append(element);
  return element;
}

function appendDefinition(label, value) {
  appendText(provenance, "dt", label);
  appendText(provenance, "dd", value);
}

function clearOutput() {
  results.replaceChildren();
  provenance.replaceChildren();
  diagnostics.replaceChildren();
}

function renderPaper(item, group) {
  const paper = item.paper ?? item;
  const evidence = item.evidence ?? {};
  const row = document.createElement("li");
  appendText(row, "h3", paper.title);
  appendText(row, "p", `Group: ${group}`);
  appendText(row, "p", `ID: ${paper.canonical_id}`);
  appendText(row, "p", `Fusion/RRF score: ${evidence.fusion_score ?? item.score ?? "Unknown"}`);
  appendText(row, "p", `Source ranks: ${JSON.stringify(evidence.source_ranks ?? item.source_ranks ?? {})}`);
  appendText(row, "p", `Relevance: ${evidence.relevance_level ?? group}`);
  appendText(row, "p", `Matched subqueries: ${(evidence.matched_subqueries ?? []).join(", ")}`);
  appendText(row, "p", `Matched constraints: ${(evidence.matched_constraints ?? []).join(", ")}`);
  appendText(row, "p", `Unmatched constraints: ${(evidence.unmatched_constraints ?? []).join(", ")}`);
  appendText(row, "p", `Filter reasons: ${(evidence.filter_reasons ?? []).join(", ")}`);
  results.append(row);
}

function renderSuccess(payload) {
  clearOutput();
  appendText(results, "li", `Selected paper IDs: ${(payload.selected_paper_ids ?? []).join(", ")}`);
  const selectedIds = new Set(payload.selected_paper_ids ?? []);
  const selectedPapers = new Map();
  [
    ...(payload.fused_papers ?? []),
    ...(payload.high_relevance ?? []),
    ...(payload.partial_relevance ?? []),
  ].forEach((item) => {
    const paper = item.paper ?? item;
    if (selectedIds.has(paper.canonical_id)) selectedPapers.set(paper.canonical_id, item);
  });
  selectedPapers.forEach((paper) => renderPaper(paper, "selected"));
  (payload.high_relevance ?? []).forEach((paper) => renderPaper(paper, "high"));
  (payload.partial_relevance ?? []).forEach((paper) => renderPaper(paper, "partial"));

  appendDefinition("Execution mode", payload.execution_mode);
  appendDefinition("Snapshot set", payload.snapshot_set_id);
  appendDefinition("Snapshot captured", payload.snapshot_captured_at);
  appendDefinition("Config hash", payload.config_hash);
  appendDefinition("Run ID", payload.run_id);
  appendDefinition("Usage", JSON.stringify(payload.usage ?? {}));
  appendDefinition("Stop reason", payload.stop_reason);
  appendDefinition("Partial result", payload.is_partial);
  appendDefinition("Planner fallback", payload.planner_fallback);
  appendDefinition("Planner status", payload.planner_status);

  appendText(diagnostics, "h3", "Dependency statuses");
  (payload.dependency_status ?? []).forEach((dependency) => {
    appendText(diagnostics, "p", `${dependency.dependency}: ${dependency.state} (cache: ${dependency.cache_hit})`);
  });
  appendText(diagnostics, "h3", "Safe warnings");
  (payload.warnings ?? []).forEach((warning) => appendText(diagnostics, "p", warning));
  appendText(diagnostics, "h3", "Citation edges");
  (payload.citation_edges ?? []).forEach((edge) => {
    appendText(diagnostics, "p", `${edge.citing_canonical_id} → ${edge.cited_canonical_id} (${edge.provider})`);
  });
  status.textContent = payload.is_partial ? "Search completed with partial results." : "Search completed.";
  status.className = "";
}

function renderError(payload) {
  clearOutput();
  status.textContent = `Search failed: ${payload.code}: ${payload.detail}`;
  status.className = "error";
}

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
  status.textContent = "Loading search results…";
  status.className = "";

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
  }
});
