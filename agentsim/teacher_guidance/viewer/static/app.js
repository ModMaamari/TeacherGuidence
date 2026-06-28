"use strict";

const state = { runs: [], currentRun: null, episodes: [], currentQid: null };

// ---------- utilities ----------
function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function pct(x) { return (Number(x || 0) * 100).toFixed(0) + "%"; }
function num(x) { return Number(x || 0).toFixed(2); }
function json(o) { return esc(JSON.stringify(o == null ? {} : o, null, 2)); }
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return r.json();
}
function badge(text, cls) { return `<span class="badge ${cls || ""}">${esc(text)}</span>`; }

// ---------- runs ----------
async function init() {
  document.getElementById("private-toggle").addEventListener("change", (e) => {
    document.body.classList.toggle("hide-private", !e.target.checked);
    if (state.currentQid) loadEpisode(state.currentQid); // re-render gold gating
  });
  document.getElementById("run-select").addEventListener("change", (e) => loadRun(e.target.value));
  document.getElementById("episode-search").addEventListener("input", (e) => renderEpisodeList(e.target.value));

  try {
    state.runs = await getJSON("/api/runs");
  } catch (err) {
    document.getElementById("detail").innerHTML = errorState(err.message);
    return;
  }
  const sel = document.getElementById("run-select");
  if (!state.runs.length) {
    sel.innerHTML = `<option>No runs found</option>`;
    document.getElementById("detail").innerHTML = errorState("No teacher_guidance_episodes.jsonl found under the output root.");
    return;
  }
  sel.innerHTML = state.runs.map((r) =>
    `<option value="${esc(r.run_id)}">${esc(r.run_id)} (${r.num_episodes})</option>`).join("");
  loadRun(state.runs[0].run_id);
}

async function loadRun(runId) {
  state.currentRun = state.runs.find((r) => r.run_id === runId) || null;
  renderRunStats(state.currentRun);
  state.episodes = await getJSON("/api/episodes?run=" + encodeURIComponent(runId));
  state.currentQid = null;
  renderEpisodeList("");
  document.getElementById("detail").innerHTML =
    `<div class="empty-state"><h2>${state.episodes.length} episodes</h2><p>Select a question on the left.</p></div>`;
}

function renderRunStats(run) {
  const box = document.getElementById("run-stats");
  if (!run) { box.innerHTML = ""; return; }
  const stops = Object.entries(run.stop_reasons || {})
    .map(([k, v]) => `${k}: ${v}`).join(" · ") || "—";
  box.innerHTML = [
    stat(run.num_episodes, "episodes"),
    stat("G" + (run.guidance_level ?? "?"), "guidance"),
    stat(pct(run.mean_exact_match), "mean EM"),
    stat(num(run.mean_f1), "mean F1"),
    stat(pct(run.mean_doc_recall), "doc recall"),
    `<div class="stat" style="min-width:auto"><span class="v" style="font-size:13px">${esc(run.student_model)}</span><span class="k">student</span></div>`,
    `<div class="stat" style="min-width:auto"><span class="v" style="font-size:13px">${esc(run.teacher_model)}</span><span class="k">teacher</span></div>`,
    `<div class="stat" style="min-width:auto"><span class="v" style="font-size:12px">${esc(stops)}</span><span class="k">stop reasons</span></div>`,
  ].join("");
}
function stat(v, k) { return `<div class="stat"><span class="v">${esc(v)}</span><span class="k">${esc(k)}</span></div>`; }

// ---------- episode list ----------
function renderEpisodeList(filter) {
  const q = (filter || "").toLowerCase();
  const items = state.episodes.filter((e) => !q || (e.query || "").toLowerCase().includes(q));
  document.getElementById("episode-count").textContent = `${items.length}/${state.episodes.length}`;
  const ul = document.getElementById("episode-list");
  ul.innerHTML = items.map((e) => {
    const emBadge = e.exact_match ? badge("EM ✓", "good") : badge("EM ✗", "bad");
    const active = e.qid === state.currentQid ? "active" : "";
    return `<li class="episode-item ${active}" data-qid="${esc(e.qid)}">
      <div class="q">${esc(e.query)}</div>
      <div class="meta">${emBadge}${badge("F1 " + num(e.f1))}${badge(e.num_steps + " steps")}${badge(e.stop_reason || "—")}</div>
    </li>`;
  }).join("");
  ul.querySelectorAll(".episode-item").forEach((li) =>
    li.addEventListener("click", () => loadEpisode(li.dataset.qid)));
}

// ---------- episode detail ----------
async function loadEpisode(qid) {
  state.currentQid = qid;
  document.querySelectorAll(".episode-item").forEach((li) =>
    li.classList.toggle("active", li.dataset.qid === qid));
  const ep = await getJSON(
    "/api/episode?run=" + encodeURIComponent(state.currentRun.run_id) + "&qid=" + encodeURIComponent(qid));
  renderEpisodeDetail(ep);
}

function goldValue(text) {
  // Gold is teacher-private: show it only when the reveal toggle is on.
  return `<span class="when-revealed">${esc(text)}</span><span class="when-hidden redacted">[hidden — enable “Show teacher-private”]</span>`;
}

function renderEpisodeDetail(ep) {
  const fm = ep.final_metrics || {};
  const matchCls = fm.exact_match ? "match" : "nomatch";
  const header = `
    <div class="ep-header">
      <div class="qid">${esc(ep.qid)}</div>
      <h2>${esc(ep.query)}</h2>
      <div class="answers">
        <div class="answer-card">
          <div class="lbl">Gold answer · teacher-private</div>
          <div class="val">${goldValue(ep.gold_answer)}</div>
        </div>
        <div class="answer-card final ${matchCls}">
          <div class="lbl">Final answer · ${fm.exact_match ? "exact match" : "no match"}</div>
          <div class="val">${esc(ep.final_answer) || "<span class='redacted'>(none)</span>"}</div>
        </div>
      </div>
      <div class="chips">
        ${metric(fm.exact_match ? "1" : "0", "EM")}
        ${metric(num(fm.f1), "F1")}
        ${metric(pct(fm.supporting_doc_recall), "doc recall")}
        ${metric(pct(fm.supporting_fact_recall), "fact recall")}
      </div>
      <div class="meta-line">
        <span>budget <b>${esc(ep.budget)}</b></span>
        <span>guidance <b>G${esc(ep.guidance_level)}</b></span>
        <span>stop <b>${esc(ep.stop_reason)}</b></span>
        <span>student <code>${esc(ep.student_model)}</code></span>
        <span>teacher <code>${esc(ep.teacher_model)}</code></span>
      </div>
    </div>`;

  const planHtml = renderPlanReview(ep.plan_review || {});
  const stepsHtml = `
    <div class="section">
      <h3>Trajectory · ${(ep.steps || []).length} steps</h3>
      <div class="timeline">${(ep.steps || []).map(renderStep).join("")}</div>
    </div>`;

  document.getElementById("detail").innerHTML = header + planHtml + stepsHtml;
}

function metric(v, k) { return `<span class="metric"><span class="mv">${esc(v)}</span><span class="mk">${esc(k)}</span></span>`; }

function planStepsList(steps) {
  if (!steps || !steps.length) return `<div class="redacted">—</div>`;
  return `<ul class="plan-steps">${steps.map((s) =>
    `<li><span class="n">${esc(s.step_id)}</span>${badge(s.intended_tool || "?", "tool")}<span>${esc(s.goal || "")}</span></li>`).join("")}</ul>`;
}

function renderPlanReview(pr) {
  if (!pr || !pr.enabled) {
    return `<div class="section"><h3>Plan review</h3><div class="panel redacted">Disabled for this run.</div></div>`;
  }
  const initial = pr.initial_student_plan || {};
  const revised = pr.revised_student_plan || {};
  const feedback = pr.student_visible_plan_feedback || {};
  const leak = pr.leakage_check || {};
  const leaked = ["gold_answer_leaked", "hidden_title_leaked", "hidden_doc_id_leaked", "hidden_span_leaked"]
    .some((k) => leak[k]);
  const leakBadge = leaked
    ? badge("guard fired · sanitized", "warn")
    : badge("no leakage", "good");

  return `
  <div class="section">
    <h3>Plan review ${leakBadge}</h3>
    <div class="panel">
      <div class="plan-grid">
        <div class="plan-col">
          <h4>1 · Initial plan (student)</h4>
          <div class="plan-summary">${esc(initial.plan_summary || "")}</div>
          ${planStepsList(initial.steps)}
        </div>
        <div class="plan-col">
          <h4>2 · Teacher feedback (student-visible)</h4>
          <div class="guidance-box">
            <div class="score">score ${esc(num(feedback.score))}</div>
            <div class="fb">${esc(feedback.feedback || "—")}</div>
          </div>
        </div>
        <div class="plan-col">
          <h4>3 · Revised plan (student)</h4>
          <div class="plan-summary">${esc(revised.plan_summary || "")}</div>
          ${planStepsList(revised.steps)}
        </div>
      </div>
      <div class="private-only" style="margin-top:12px">
        <div class="private-card">
          <div class="blk-label">${badge("teacher-private", "private")} plan diagnosis</div>
          <pre class="code">${json((pr.teacher_plan_review_full || {}).private_diagnosis)}</pre>
        </div>
      </div>
    </div>
  </div>`;
}

function renderObservation(obs) {
  obs = obs || {};
  if (obs.tool === "search" && Array.isArray(obs.results)) {
    if (!obs.results.length) return `<div class="redacted">no results</div>`;
    return `<ul class="search-results">${obs.results.map((r, i) =>
      `<li><span class="rank">#${i + 1}</span><span><span class="title">${esc(r.title)}</span>
        <div class="muted">${esc(r.doc_id)} — ${esc(r.text_preview || "")}</div></span>
        <span class="score">${esc(r.score)}</span></li>`).join("")}</ul>`;
  }
  if (obs.tool === "extract") {
    const ex = (obs.extracted || []).map((e) =>
      `<li class="fact"><div class="doc">${esc(e.doc_id)}</div>${esc(e.span)}</li>`).join("");
    const bad = (obs.invalid_spans || []).map((e) =>
      `<li class="fact" style="border-color:#fecaca;background:#fef2f2">${esc(e.span)} <span class="muted">(${esc(e.reason)})</span></li>`).join("");
    return `<ul class="facts">${ex || ""}${bad || ""}</ul>` || `<div class="redacted">—</div>`;
  }
  return `<pre class="code">${json(obs)}</pre>`;
}

function renderStep(s) {
  const action = (s.student_action || {}).action || {};
  const decision = (s.student_action || {}).decision || {};
  const facts = (s.student_action || {}).new_facts_extracted || [];
  const g = s.student_visible_guidance || {};
  const m = s.metrics || {};
  const leak = s.leakage_check || {};
  const leaked = ["gold_answer_leaked", "hidden_title_leaked", "hidden_doc_id_leaked", "hidden_span_leaked"]
    .filter((k) => leak[k]);
  const td = (s.teacher_full_decision) || "";

  const factsHtml = facts.length
    ? `<ul class="facts">${facts.map((f) =>
        `<li class="fact"><div class="doc">${esc(f.doc_id)}</div>${esc(f.span)}</li>`).join("")}</ul>` : "";

  const hint = g.hint
    ? `<div class="hint">hint: ${esc(JSON.stringify(g.hint))}</div>` : "";

  const guidanceHtml = ("feedback" in g || "score" in g)
    ? `<div class="guidance-box">
         <div class="score">score ${esc(g.score)}</div>
         ${g.feedback ? `<div class="fb">${esc(g.feedback)}</div>` : ""}
         ${hint}
       </div>`
    : `<div class="redacted">reward only (no feedback at this guidance level)</div>`;

  const metricsBadges = [
    m.json_valid === false ? badge("invalid JSON", "bad") : "",
    m.retrieved_gold_doc ? badge("retrieved gold doc", "good") : "",
    m.span_validation_failed ? badge("invalid spans", "warn") : "",
    m.parametric_knowledge_used ? badge("parametric knowledge", "warn") : "",
  ].filter(Boolean).join("");

  const leakBadge = leaked.length ? badge("guard sanitized: " + leaked.join(", "), "warn") : "";

  return `
  <div class="step">
    <div class="step-head">
      <span class="t">Step ${esc(s.t)}</span>
      ${badge(action.tool || "?", "tool")}
      ${decision.category ? badge(decision.category) : ""}
      ${s.stop_condition === "FINISH" ? badge("FINISH", "accent") : ""}
      ${leakBadge}
    </div>
    <div class="step-body">
      <div class="block">
        <div class="blk-label">Student · thought</div>
        <div class="thought">${esc((s.student_action || {}).thought || "—")}</div>
      </div>
      <div class="block">
        <div class="blk-label">Action params</div>
        <pre class="code">${json(action.params)}</pre>
        ${factsHtml ? `<div style="margin-top:8px">${factsHtml}</div>` : ""}
      </div>
      <div class="block">
        <div class="blk-label">Tool observation</div>
        ${renderObservation(s.tool_observation)}
      </div>
      <div class="block">
        <div class="blk-label">Student-visible guidance</div>
        ${guidanceHtml}
      </div>
      <div class="block">
        <div class="blk-label">Step labels</div>
        <div class="chips">${metricsBadges || badge("—")}</div>
      </div>
      <div class="block private-only">
        <div class="private-card">
          <div class="blk-label">${badge("teacher-private", "private")} diagnosis</div>
          <pre class="code">${json((s.teacher_private_diagnosis))}</pre>
        </div>
      </div>
    </div>
  </div>`;
}

function errorState(msg) {
  return `<div class="empty-state"><h2>Nothing to show</h2><p>${esc(msg)}</p></div>`;
}

init();
