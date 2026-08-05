"use strict";

const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const modeSelect = document.querySelector("#mode");
const submitButton = document.querySelector("#submit-search");
const cancelButton = document.querySelector("#cancel-search");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const resultSummary = document.querySelector("#result-summary");
const highResults = document.querySelector("#high-results");
const partialResults = document.querySelector("#partial-results");
const highCount = document.querySelector("#high-count");
const partialCount = document.querySelector("#partial-count");
const emptyState = document.querySelector("#empty-state");
const evidencePanel = document.querySelector("#evidence-panel");
const provenance = document.querySelector("#provenance");
const diagnostics = document.querySelector("#diagnostics");

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

function clearOutput() {
  highResults.replaceChildren();
  partialResults.replaceChildren();
  provenance.replaceChildren();
  diagnostics.replaceChildren();
  results.hidden = true;
  emptyState.hidden = true;
  evidencePanel.hidden = true;
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
  addChip(top, group === "high" ? "High relevance" : "Partial relevance", group);
  if (selectedIds.has(paper.canonical_id)) addChip(top, "Selected", "selected");
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

  const evidenceBox = document.createElement("div");
  evidenceBox.className = "paper-evidence";
  evidenceBox.append(
    evidenceRow("Matched query parts", evidence.matched_subqueries),
    evidenceRow("Matched constraints", evidence.matched_constraints, "positive"),
    evidenceRow("Open constraints", evidence.unmatched_constraints, "warning"),
    evidenceRow("Filter notes", evidence.filter_reasons)
  );
  article.append(evidenceBox);

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

  high.forEach((item) => highResults.append(renderPaper(item, "high", selectedIds)));
  partial.forEach((item) => partialResults.append(renderPaper(item, "partial", selectedIds)));
  highCount.textContent = String(high.length);
  partialCount.textContent = String(partial.length);
  resultSummary.textContent = `${selectedIds.size} selected · ${high.length + partial.length} ranked`;
  results.hidden = high.length + partial.length === 0;
  emptyState.hidden = high.length + partial.length !== 0;

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
