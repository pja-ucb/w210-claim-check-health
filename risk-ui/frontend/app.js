/** API base for outpatient NN scoring (/risk-score-batch) and policy context (/policy-review). */
const API_BASE_URL = "http://localhost:8000";
const fileInput = document.getElementById("claims-file");
const scoreBtn = document.getElementById("score-btn");
const resultsDiv = document.getElementById("results");
const summaryDiv = document.getElementById("summary");
const progressDiv = document.getElementById("progress");
const exportBtn = document.getElementById("export-btn");
const policyClaimIdInput = document.getElementById("policy-claim-id");
const policyRunBtn = document.getElementById("policy-run-btn");
const policyStatusEl = document.getElementById("policy-status");
const policySummaryEl = document.getElementById("policy-summary");
const policyRawWrap = document.getElementById("policy-raw-wrap");
const policyRawEl = document.getElementById("policy-raw");
const policyHealthBtn = document.getElementById("policy-health-btn");
const policyHealthMsg = document.getElementById("policy-health-msg");
const policyElapsedEl = document.getElementById("policy-elapsed");
const policyCopyJsonBtn = document.getElementById("policy-copy-json-btn");
const policyToolsAfter = document.getElementById("policy-tools-after");

/** Last successful policy-review JSON string (for copy) */
let lastPolicyReviewJsonText = null;
let policyElapsedTimerId = null;

// Per-claim review state: claim_id -> { notes: string, decision: 'approved'|'denied'|null }
let reviewState = {};
// Last scoring result for export and re-renders
let lastScoredData = null;
// Which claim's review panel is expanded (only one at a time)
let expandedClaimId = null;

const RESULTS_PAGE_SIZE = 10;
/** When true, table lists only rows with flag === true */
let resultsFlaggedOnly = false;
/** 1-based current page */
let resultsPage = 1;

/** Wall-clock ms for last finished batch score (shown in Summary) */
let lastScoringDurationMs = null;
let scoreElapsedTimerId = null;

function formatElapsed(totalSeconds) {
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

/** Progress % shown in steps of 5 until complete (100%). */
function percentCompleteChunked(completed, total) {
  if (!total) return 100;
  if (completed >= total) return 100;
  const raw = (completed / total) * 100;
  return Math.floor(raw / 5) * 5;
}

function stopScoreElapsedTimer() {
  if (scoreElapsedTimerId != null) {
    clearInterval(scoreElapsedTimerId);
    scoreElapsedTimerId = null;
  }
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map(h => h.trim());

  return lines.slice(1).map(line => {
    const values = line.split(",").map(v => v.trim());
    const row = {};
    headers.forEach((h, i) => { row[h] = values[i] ?? ""; });
    return row;
  });
}

function buildClaims(rows) {
  return rows.map(row => {
    const claimId = row.claim_id || row.claimId || row.CLAIM_ID || row.CLM_ID || "";
    const policyId = row.policy_id || row.policyId || row.POLICY_ID || "";

    const fields = { ...row };
    delete fields.claim_id;
    delete fields.claimId;
    delete fields.CLAIM_ID;
    delete fields.CLM_ID;
    delete fields.policy_id;
    delete fields.policyId;
    delete fields.POLICY_ID;
    delete fields.claim_type;
    delete fields.CLAIM_TYPE;

    return {
      claim_id: String(claimId),
      policy_id: policyId ? String(policyId) : null,
      claim_type: "outpatient",
      fields,
    };
  }).filter(c => c.claim_id);
}

async function scoreClaims(claims) {
  const res = await fetch(`${API_BASE_URL}/risk-score-batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ claims, return_evidence: false }),
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg);
  }
  return res.json();
}

async function scoreClaimsInChunks(claims, chunkSize = 50) {
  const total = claims.length;
  const results = [];
  let flagged = 0;
  const started = Date.now();

  const paintProgress = (completed) => {
    const pct = percentCompleteChunked(completed, total);
    const sec = Math.floor((Date.now() - started) / 1000);
    const line = `Running NN batch… ${completed} / ${total} · ${pct}% complete · Runtime: ${formatElapsed(sec)}`;
    if (progressDiv) progressDiv.textContent = line;
  };

  stopScoreElapsedTimer();
  paintProgress(0);
  let lastCompleted = 0;
  scoreElapsedTimerId = setInterval(() => {
    paintProgress(lastCompleted);
  }, 250);

  for (let i = 0; i < claims.length; i += chunkSize) {
    const chunk = claims.slice(i, i + chunkSize);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const data = await scoreClaims(chunk);
    results.push(...data.results);
    flagged += data.summary.flagged;
    lastCompleted = results.length;
    paintProgress(lastCompleted);
  }

  stopScoreElapsedTimer();
  paintProgress(total);

  return {
    results,
    summary: {
      total,
      flagged,
      flagged_rate: total ? flagged / total : 0,
    },
  };
}

function getReviewCounts(data) {
  const claimIds = new Set((data && data.results) ? data.results.map(r => r.claim_id) : []);
  let reviewed = 0, approved = 0, denied = 0;
  for (const cid of claimIds) {
    const s = reviewState[cid];
    if (!s) continue;
    if (s.notes || s.decision) reviewed++;
    if (s.decision === "approved") approved++;
    if (s.decision === "denied") denied++;
  }
  return { reviewed, approved, denied };
}

function ensureReviewState(claimId) {
  if (!reviewState[claimId]) reviewState[claimId] = { notes: "", decision: null };
  return reviewState[claimId];
}

function getFilteredSortedResults() {
  if (!lastScoredData || !lastScoredData.results) return [];
  let rows = [...lastScoredData.results].sort((a, b) => {
    if (a.flag !== b.flag) return b.flag - a.flag;
    return b.risk_score - a.risk_score;
  });
  if (resultsFlaggedOnly) {
    rows = rows.filter((r) => r.flag);
  }
  return rows;
}

const colCount = 6;

function renderResultsTable() {
  const toolbar = document.getElementById("results-toolbar");
  const pageInfo = document.getElementById("results-page-info");
  const paginationEl = document.getElementById("results-pagination");

  if (!lastScoredData || !lastScoredData.results.length) {
    if (toolbar) toolbar.classList.add("hidden");
    if (paginationEl) paginationEl.classList.add("hidden");
    resultsDiv.innerHTML = "No results yet.";
    return;
  }

  if (toolbar) toolbar.classList.remove("hidden");
  const flaggedCb = document.getElementById("filter-flagged-only");
  if (flaggedCb) flaggedCb.checked = resultsFlaggedOnly;

  const filtered = getFilteredSortedResults();
  const totalFiltered = filtered.length;
  const totalPages = Math.max(1, Math.ceil(totalFiltered / RESULTS_PAGE_SIZE));
  if (resultsPage > totalPages) resultsPage = totalPages;
  if (resultsPage < 1) resultsPage = 1;
  const startIdx = (resultsPage - 1) * RESULTS_PAGE_SIZE;
  const pageRows = filtered.slice(startIdx, startIdx + RESULTS_PAGE_SIZE);
  const showingFrom = totalFiltered === 0 ? 0 : startIdx + 1;
  const showingTo = startIdx + pageRows.length;

  if (pageInfo) {
    const scope = resultsFlaggedOnly ? "flagged" : "all";
    pageInfo.textContent = `Showing ${showingFrom}–${showingTo} of ${totalFiltered} (${scope}) · Page ${resultsPage} of ${totalPages}`;
  }

  if (paginationEl) {
    if (totalPages <= 1) {
      paginationEl.classList.add("hidden");
      paginationEl.innerHTML = "";
    } else {
      paginationEl.classList.remove("hidden");
      paginationEl.innerHTML = `
        <button type="button" class="page-btn" data-page-action="prev" ${resultsPage <= 1 ? "disabled" : ""}>Previous</button>
        <span class="page-indicator">${resultsPage} / ${totalPages}</span>
        <button type="button" class="page-btn" data-page-action="next" ${resultsPage >= totalPages ? "disabled" : ""}>Next</button>
      `;
    }
  }

  const rows = pageRows.map((r) => {
    const state = ensureReviewState(r.claim_id);
    const decisionLabel = state.decision === "approved" ? "Approved" : state.decision === "denied" ? "Denied" : "—";
    const decisionClass = state.decision === "approved" ? "decision-approved" : state.decision === "denied" ? "decision-denied" : "";
    const hasReview = state.notes || state.decision;
    const expandClass = expandedClaimId === r.claim_id ? "review-row" : "review-row hidden";
    const notesForHtml = (state.notes || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return `
    <tr class="${decisionClass}" data-claim-id="${r.claim_id}">
      <td>${r.claim_id}</td>
      <td>${r.risk_score.toFixed(3)}</td>
      <td>${r.model_score.toFixed(3)}</td>
      <td class="${r.flag ? "flag-true" : "flag-false"}">${r.flag}</td>
      <td class="review-cell">
        <span class="decision-label">${decisionLabel}</span>
        <button type="button" class="review-action-btn" data-action="toggle-review" data-claim-id="${r.claim_id}">${hasReview ? "Edit review" : "Add review"}</button>
        <button type="button" class="review-action-btn approve-btn" data-action="approve" data-claim-id="${r.claim_id}">Approve</button>
        <button type="button" class="review-action-btn deny-btn" data-action="deny" data-claim-id="${r.claim_id}">Deny</button>
        <button type="button" class="review-action-btn clear-btn" data-action="clear-decision" data-claim-id="${r.claim_id}">Clear</button>
      </td>
      <td>
        <button type="button" class="btn-generate-context" data-action="generate-context" data-claim-id="${r.claim_id}" title="Run policy pipeline for this claim">Generate Context</button>
      </td>
    </tr>
    <tr class="${expandClass}" data-claim-id="${r.claim_id}" data-review-row="true">
      <td colspan="${colCount}">
        <label for="review-notes-${r.claim_id}">Review notes</label>
        <textarea id="review-notes-${r.claim_id}" data-claim-id="${r.claim_id}" rows="3" placeholder="Reason for approval/denial…">${notesForHtml}</textarea>
      </td>
    </tr>`;
  }).join("");

  resultsDiv.innerHTML = `
    <div class="results-table-wrap">
      <table id="results-table">
        <thead>
          <tr>
            <th title="Claim identifier (CLM_ID)">Claim ID</th>
            <th title="0.6 × NN probability + 0.4 × rule score (API)">Combined score</th>
            <th title="Raw neural network output P(high risk)">NN probability</th>
            <th title="True if NN ≥ threshold or combined ≥ 0.7 or rules ≥ 0.8">Flag</th>
            <th>Review</th>
            <th>Context</th>
          </tr>
        </thead>
        <tbody>${rows.length ? rows : `<tr><td colspan="${colCount}" class="empty-page">No claims match this filter.</td></tr>`}</tbody>
      </table>
    </div>
  `;
}

/**
 * @param {{ resetPage?: boolean }} [options] — set resetPage: false when re-rendering after review edits
 */
function renderResults(data, options = {}) {
  const resetPage = options.resetPage !== false;
  lastScoredData = data;
  if (resetPage) resultsPage = 1;
  if (exportBtn) exportBtn.disabled = !data;

  const counts = getReviewCounts(data);
  let summaryLine = `Total: ${data.summary.total}, Flagged: ${data.summary.flagged} (${(data.summary.flagged_rate * 100).toFixed(1)}%) | Reviewed: ${counts.reviewed} | Approved: ${counts.approved} | Denied: ${counts.denied}`;
  if (lastScoringDurationMs != null) {
    summaryLine += ` | Batch runtime: ${formatElapsed(Math.round(lastScoringDurationMs / 1000))}`;
  }
  summaryDiv.textContent = summaryLine;

  renderResultsTable();
}

scoreBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please upload a CSV file.");
    return;
  }
  const text = await file.text();
  const rows = parseCsv(text);
  const claims = buildClaims(rows);

  if (!claims.length) {
    alert("No valid claims found. Ensure claim_id is present.");
    return;
  }

  try {
    scoreBtn.disabled = true;
    lastScoringDurationMs = null;
    const scoreStarted = Date.now();
    const data = await scoreClaimsInChunks(claims);
    lastScoringDurationMs = Date.now() - scoreStarted;
    if (progressDiv) {
      const sec = Math.round(lastScoringDurationMs / 1000);
      progressDiv.textContent = `NN batch complete · 100% · Total runtime: ${formatElapsed(sec)}`;
    }
    renderResults(data);
  } catch (err) {
    stopScoreElapsedTimer();
    lastScoringDurationMs = null;
    if (progressDiv) {
      progressDiv.textContent = "";
    }
    alert(`NN scoring failed: ${err && err.message ? err.message : err}`);
  } finally {
    stopScoreElapsedTimer();
    scoreBtn.disabled = false;
  }
});

// Event delegation for review actions
if (resultsDiv) {
  resultsDiv.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-action][data-claim-id]");
    if (!btn) return;
    const action = btn.getAttribute("data-action");
    const claimId = btn.getAttribute("data-claim-id");
    if (!claimId) return;
    ensureReviewState(claimId);
    if (action === "toggle-review") {
      expandedClaimId = expandedClaimId === claimId ? null : claimId;
      renderResults(lastScoredData, { resetPage: false });
      return;
    }
    if (action === "approve") {
      reviewState[claimId].decision = "approved";
      renderResults(lastScoredData, { resetPage: false });
      return;
    }
    if (action === "deny") {
      reviewState[claimId].decision = "denied";
      renderResults(lastScoredData, { resetPage: false });
      return;
    }
    if (action === "clear-decision") {
      reviewState[claimId].decision = null;
      renderResults(lastScoredData, { resetPage: false });
      return;
    }
    if (action === "generate-context") {
      const panel = document.getElementById("policy-review");
      if (panel) panel.scrollIntoView({ behavior: "smooth", block: "start" });
      void runPolicyReview({ claimId });
      return;
    }
  });

  resultsDiv.addEventListener("blur", (e) => {
    const textarea = e.target.closest("textarea[data-claim-id]");
    if (!textarea) return;
    const claimId = textarea.getAttribute("data-claim-id");
    if (!claimId) return;
    ensureReviewState(claimId);
    reviewState[claimId].notes = textarea.value;
    renderResults(lastScoredData, { resetPage: false });
  }, true);
}

function escapeCsvCell(val) {
  const s = String(val == null ? "" : val);
  if (/[",\r\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function exportResultsWithReviews() {
  if (!lastScoredData || !lastScoredData.results.length) {
    alert("No results to export. Run scoring first.");
    return;
  }
  const sorted = getFilteredSortedResults();
  const headers = ["Claim ID", "Combined score", "NN probability", "Flag", "Review notes", "Decision"];
  const lines = [headers.map(escapeCsvCell).join(",")];
  for (const r of sorted) {
    const state = reviewState[r.claim_id] || { notes: "", decision: null };
    const decision = state.decision === "approved" ? "Approved" : state.decision === "denied" ? "Denied" : "—";
    lines.push([
      r.claim_id,
      r.risk_score.toFixed(3),
      r.model_score.toFixed(3),
      r.flag,
      state.notes || "",
      decision,
    ].map(escapeCsvCell).join(","));
  }
  const csv = lines.join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "claim-results-with-reviews.csv";
  a.click();
  URL.revokeObjectURL(url);
}

if (exportBtn) {
  exportBtn.disabled = true;
  exportBtn.addEventListener("click", exportResultsWithReviews);
}

document.getElementById("filter-flagged-only")?.addEventListener("change", (e) => {
  resultsFlaggedOnly = e.target.checked;
  resultsPage = 1;
  renderResultsTable();
});

const resultsPaginationEl = document.getElementById("results-pagination");
if (resultsPaginationEl) {
  resultsPaginationEl.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-page-action]");
    if (!btn || btn.disabled) return;
    const filtered = getFilteredSortedResults();
    const totalPages = Math.max(1, Math.ceil(filtered.length / RESULTS_PAGE_SIZE));
    const act = btn.getAttribute("data-page-action");
    if (act === "prev") resultsPage = Math.max(1, resultsPage - 1);
    if (act === "next") resultsPage = Math.min(totalPages, resultsPage + 1);
    renderResultsTable();
  });
}

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderPolicyReviewPayload(data) {
  if (!policySummaryEl || !policyRawEl || !policyRawWrap) return;

  const ra = data.recommended_action || {};
  const primary = ra.primary_action || "—";
  const secondary = Array.isArray(ra.secondary_actions) ? ra.secondary_actions.join(", ") : "";
  const rationale = ra.decision_rationale || "";
  const summary = ra.claim_summary || "";
  const confidence = ra.claim_action_confidence || ra.action_confidence || "—";
  const drivers = Array.isArray(ra.action_drivers) ? ra.action_drivers.join(", ") : "";

  const dx = (data.diagnosis_descriptions || []).slice(0, 6).join("; ");
  const proc = (data.procedure_descriptions || []).slice(0, 6).join("; ");
  const selectedHcpcs = (data.selected_primary_hcpcs || data.hcpcs || []).join(", ");

  let hcpcHtml = "";
  const per = data.per_hcpc_results || [];
  for (const row of per) {
    const summ = row.summary || {};
    const code = escapeHtml(row.hcpc || summ.hcpc || "");
    hcpcHtml += `<div class="hcpc-block"><h4>HCPCS ${code}</h4>`;
    hcpcHtml += `<p class="meta">${escapeHtml(summ.service_summary || "")}</p>`;
    hcpcHtml += `<p class="meta"><strong>Medical necessity:</strong> ${escapeHtml(summ.medical_necessity_findings || "")}</p>`;
    hcpcHtml += `<p class="meta"><strong>Documentation:</strong> ${escapeHtml(summ.documentation_findings || "")}</p>`;
    hcpcHtml += `<p class="meta"><strong>Billing/coding:</strong> ${escapeHtml(summ.billing_coding_findings || "")}</p>`;
    hcpcHtml += `<p class="meta"><strong>Limitations:</strong> ${escapeHtml(summ.limitations_findings || "")}</p>`;
    const ev = summ.evidence_strength || "—";
    const amb = summ.policy_ambiguity || "—";
    hcpcHtml += `<p class="meta policy-meta-tags"><span class="tag">Evidence: ${escapeHtml(ev)}</span> <span class="tag">Ambiguity: ${escapeHtml(amb)}</span></p>`;
    hcpcHtml += `</div>`;
  }

  policySummaryEl.classList.remove("hidden");
  policySummaryEl.innerHTML = `
    <h3>Recommended action</h3>
    <div class="primary-action">${escapeHtml(primary)}</div>
    ${secondary ? `<p class="meta"><strong>Secondary:</strong> ${escapeHtml(secondary)}</p>` : ""}
    <p class="meta"><strong>Confidence:</strong> ${escapeHtml(String(confidence))}</p>
    ${drivers ? `<p class="meta"><strong>Drivers:</strong> ${escapeHtml(drivers)}</p>` : ""}
    <p class="meta"><strong>Rationale:</strong> ${escapeHtml(rationale)}</p>
    <p class="meta"><strong>Claim summary:</strong> ${escapeHtml(summary)}</p>
    <p class="meta"><strong>Policy pipeline risk label:</strong> ${escapeHtml(String(data.risk_label ?? ""))} ${data.risk_probability != null ? `(${data.risk_probability})` : ""}</p>
    <p class="meta"><strong>Claim year:</strong> ${escapeHtml(String(data.claim_year ?? "—"))}</p>
    <p class="meta"><strong>Selected HCPCS:</strong> ${escapeHtml(selectedHcpcs || "—")}</p>
    ${dx ? `<p class="meta"><strong>Diagnoses:</strong> ${escapeHtml(dx)}</p>` : ""}
    ${proc ? `<p class="meta"><strong>Procedures:</strong> ${escapeHtml(proc)}</p>` : ""}
    ${hcpcHtml}
  `;

  policyRawWrap.classList.remove("hidden");
  const jsonText = JSON.stringify(data, null, 2);
  policyRawEl.textContent = jsonText;
  lastPolicyReviewJsonText = jsonText;
  if (policyToolsAfter) policyToolsAfter.classList.remove("hidden");
}

function stopPolicyElapsedTimer() {
  if (policyElapsedTimerId != null) {
    clearInterval(policyElapsedTimerId);
    policyElapsedTimerId = null;
  }
}

async function checkPolicyRagHealth() {
  if (!policyHealthMsg) return;
  policyHealthMsg.textContent = "Checking…";
  policyHealthMsg.classList.remove("error", "ok");
  try {
    const res = await fetch(`${API_BASE_URL}/policy-review/health`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    if (data.rag_ready) {
      policyHealthMsg.textContent = "Policy data paths OK.";
      policyHealthMsg.classList.add("ok");
    } else {
      policyHealthMsg.textContent = data.message || "Not ready.";
      policyHealthMsg.classList.add("error");
    }
  } catch (err) {
    policyHealthMsg.textContent = err && err.message ? err.message : String(err);
    policyHealthMsg.classList.add("error");
  }
}

function setGenerateContextButtonsDisabled(disabled) {
  document.querySelectorAll('[data-action="generate-context"]').forEach((b) => {
    b.disabled = disabled;
  });
}

/**
 * Run the policy context pipeline for a claim. Pass { claimId } from the table, or leave empty to use the input field.
 * @param {{ claimId?: string }} [options]
 */
async function runPolicyReview(options = {}) {
  if (!policyClaimIdInput || !policyRunBtn) return;
  const fromOpts = options.claimId != null ? String(options.claimId).trim() : "";
  const fromInput = policyClaimIdInput.value.trim();
  const claimId = fromOpts || fromInput;
  if (!claimId) {
    alert("Enter a claim ID.");
    return;
  }
  policyClaimIdInput.value = claimId;

  stopPolicyElapsedTimer();
  if (policyStatusEl) {
    policyStatusEl.textContent = "Running pipeline…";
    policyStatusEl.classList.remove("error");
  }
  if (policyElapsedEl) {
    policyElapsedEl.textContent = "";
    const started = Date.now();
    policyElapsedTimerId = setInterval(() => {
      const sec = Math.floor((Date.now() - started) / 1000);
      if (policyElapsedEl) policyElapsedEl.textContent = `Elapsed: ${formatElapsed(sec)}`;
    }, 1000);
  }
  if (policySummaryEl) policySummaryEl.classList.add("hidden");
  if (policyRawWrap) policyRawWrap.classList.add("hidden");
  if (policyToolsAfter) policyToolsAfter.classList.add("hidden");
  lastPolicyReviewJsonText = null;

  policyRunBtn.disabled = true;
  setGenerateContextButtonsDisabled(true);
  try {
    const res = await fetch(`${API_BASE_URL}/policy-review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim_id: claimId }),
    });
    const text = await res.text();
    let body;
    try {
      body = JSON.parse(text);
    } catch {
      body = { detail: text };
    }
    if (!res.ok) {
      const detail = body.detail != null ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : text;
      throw new Error(detail || `HTTP ${res.status}`);
    }
    renderPolicyReviewPayload(body);
    if (policyStatusEl) policyStatusEl.textContent = "Complete.";
  } catch (err) {
    if (policyStatusEl) {
      policyStatusEl.textContent = err && err.message ? err.message : String(err);
      policyStatusEl.classList.add("error");
    }
  } finally {
    stopPolicyElapsedTimer();
    policyRunBtn.disabled = false;
    setGenerateContextButtonsDisabled(false);
  }
}

if (policyRunBtn) {
  policyRunBtn.addEventListener("click", () => void runPolicyReview({}));
}

if (policyHealthBtn) {
  policyHealthBtn.addEventListener("click", checkPolicyRagHealth);
}

if (policyCopyJsonBtn) {
  policyCopyJsonBtn.addEventListener("click", async () => {
    if (!lastPolicyReviewJsonText) {
      alert("Generate context first (table or panel).");
      return;
    }
    try {
      await navigator.clipboard.writeText(lastPolicyReviewJsonText);
      const t = policyCopyJsonBtn.textContent;
      policyCopyJsonBtn.textContent = "Copied!";
      setTimeout(() => { policyCopyJsonBtn.textContent = t; }, 2000);
    } catch {
      alert("Could not copy to clipboard.");
    }
  });
}
