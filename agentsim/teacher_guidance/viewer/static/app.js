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
function badge(text, cls, tip) {
  const t = tip ? ` title="${esc(tip)}"` : "";
  return `<span class="badge ${cls || ""}"${t}>${esc(text)}</span>`;
}
function cleanModel(m) { return String(m == null ? "" : m).replace(/^custom\//, ""); }

// hh:mm:ss:mmm, or null if ms is missing (older runs recorded before this feature existed).
function formatElapsed(ms) {
  if (ms == null || Number.isNaN(Number(ms))) return null;
  const total = Math.max(0, Math.round(Number(ms)));
  const pad = (n, len) => String(n).padStart(len, "0");
  const h = Math.floor(total / 3600000);
  const m = Math.floor((total % 3600000) / 60000);
  const s = Math.floor((total % 60000) / 1000);
  const rem = total % 1000;
  return `${pad(h, 2)}:${pad(m, 2)}:${pad(s, 2)}:${pad(rem, 3)}`;
}
function timingBadge(ms, label, startedAt, endedAt) {
  const f = formatElapsed(ms);
  if (!f) return "";
  const tip = startedAt && endedAt ? `${startedAt} → ${endedAt}` : undefined;
  return badge(`⏱ ${label ? label + " " : ""}${f}`, "timing", tip);
}

// One HTTP call's raw input/output, and the raw provider response body when present
// (e.g. OpenRouter/Ollama's full JSON -- finish_reason, usage, eval_count, etc.).
function callAttempt(role, call) {
  const label = call.attempt > 1 ? `${role} · attempt ${call.attempt} (repair)` : `${role} · attempt ${call.attempt}`;
  const tsTip = call.started_at && call.ended_at ? `${call.started_at} → ${call.ended_at}` : "";
  const rawResponse = call.raw_response != null
    ? `<details><summary>${esc(role)} · Show raw provider response</summary><pre class="code">${json(call.raw_response)}</pre></details>`
    : "";
  return `<div class="raw-blocks">
    <div class="chips" style="margin-bottom:4px"><span class="muted"${tsTip ? ` title="${esc(tsTip)}"` : ""}>${esc(label)}</span>${timingBadge(call.elapsed_ms)}</div>
    <details><summary>${esc(role)} · Show raw input</summary><pre class="code">${esc(call.prompt || "(empty)")}</pre></details>
    <details><summary>${esc(role)} · Show raw output</summary><pre class="code">${esc(call.response_text || "(empty)")}</pre></details>
    ${rawResponse}
  </div>`;
}

// Renders raw model I/O for one logical turn (student action, teacher evaluation, plan
// draft, plan review, ...). Prefers the native per-call log (one entry per HTTP
// request, including failed repair attempts, with the raw provider response) when
// present; falls back to a single prompt/raw pair (covers runs recorded after the
// raw-I/O fix but before per-call logging existed, and backfilled legacy runs); shows
// an explicit "not recorded" note rather than silently rendering nothing when there's
// truly no data for this run.
function rawSection(role, calls, prompt, raw, backfilled) {
  if (Array.isArray(calls) && calls.length) {
    return calls.map((c) => callAttempt(role, c)).join("");
  }
  if (prompt || raw) {
    const note = backfilled
      ? `<div class="raw-missing">recovered from ${esc(role.toLowerCase())}_sft.jsonl — per-call timing and raw provider response not available for this run</div>`
      : "";
    return `<div class="raw-blocks">
      <details><summary>${esc(role)} · Show raw input</summary><pre class="code">${esc(prompt || "(empty)")}</pre></details>
      <details><summary>${esc(role)} · Show raw output</summary><pre class="code">${esc(raw || "(empty)")}</pre></details>
    </div>${note}`;
  }
  return `<div class="raw-missing">${esc(role)} raw input/output not recorded for this run</div>`;
}

// Hover explanations for UI elements (shown as native tooltips).
const TIP = {
  episodes: "Number of question trajectories collected in this run.",
  guidance: "Guidance level 0–4: how much of the teacher's evaluation the student was shown. G3 = diagnostic feedback (score + explanation, no explicit next action).",
  correct: "Answer correct: the gold answer appears in the student's final answer (handles 'answer + explanation'). Robust alternative to strict exact match.",
  em: "Exact match: the student's normalized final answer equals the gold answer exactly.",
  f1: "Token-level F1 overlap between the student's final answer and the gold answer.",
  doc_recall: "Supporting-document recall: fraction of the gold supporting documents the student retrieved.",
  fact_recall: "Supporting-fact recall: fraction of gold supporting sentences appearing verbatim in the student's extracted spans.",
  budget: "Budget: the maximum number of tool-use steps the student was allowed for this question.",
  used_steps: "Used steps: how many tool-use steps the student actually took before finishing.",
  stop_reason: "Why the episode ended — teacher_accept (teacher accepted a finish) or budget_forced_finish (ran out of budget).",
  student: "Student model: solves the task with tools and never sees the gold answer.",
  teacher: "Teacher model: sees gold metadata and scores each step; how much it can tell the student is gated by the guidance level.",
  plan_review: "Plan review: before acting, the student drafts a plan, the teacher reviews it, and the student revises it (revision is skipped if the teacher accepts the plan).",
  leakage: "Leakage guard: detects and sanitizes any gold answer / title / doc-id that the teacher's student-visible feedback tried to reveal.",
  step_tool: "The tool the student invoked this step (search, extract, verify, synthesize, decompose, reformulate, finish).",
  guidance_box: "Exactly what the student saw after this step — the rendered guidance allowed at this guidance level.",
};

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
  const meanCorrect = run.mean_correct != null ? run.mean_correct : run.mean_exact_match;
  box.innerHTML = [
    stat(run.num_episodes, "episodes", TIP.episodes),
    stat("G" + (run.guidance_level ?? "?"), "guidance", TIP.guidance),
    stat(pct(meanCorrect), "mean correct", TIP.correct),
    stat(pct(run.mean_exact_match), "mean EM", TIP.em),
    stat(num(run.mean_f1), "mean F1", TIP.f1),
    stat(pct(run.mean_doc_recall), "doc recall", TIP.doc_recall),
    `<div class="stat" style="min-width:auto" title="${esc(TIP.student)}"><span class="v" style="font-size:13px">${esc(cleanModel(run.student_model))}</span><span class="k">student</span></div>`,
    `<div class="stat" style="min-width:auto" title="${esc(TIP.teacher)}"><span class="v" style="font-size:13px">${esc(cleanModel(run.teacher_model))}</span><span class="k">teacher</span></div>`,
    `<div class="stat" style="min-width:auto" title="${esc(TIP.stop_reason)}"><span class="v" style="font-size:12px">${esc(stops)}</span><span class="k">stop reasons</span></div>`,
  ].join("");
}
function stat(v, k, tip) {
  const t = tip ? ` title="${esc(tip)}"` : "";
  return `<div class="stat"${t}><span class="v">${esc(v)}</span><span class="k">${esc(k)}</span></div>`;
}

// ---------- episode list ----------
function renderEpisodeList(filter) {
  const q = (filter || "").toLowerCase();
  const items = state.episodes.filter((e) => !q || (e.query || "").toLowerCase().includes(q));
  document.getElementById("episode-count").textContent = `${items.length}/${state.episodes.length}`;
  const ul = document.getElementById("episode-list");
  ul.innerHTML = items.map((e) => {
    const correct = e.answer_correct != null ? e.answer_correct : e.exact_match;
    const okBadge = correct ? badge("✓ correct", "good", TIP.correct) : badge("✗ incorrect", "bad", TIP.correct);
    const active = e.qid === state.currentQid ? "active" : "";
    return `<li class="episode-item ${active}" data-qid="${esc(e.qid)}">
      <div class="q">${esc(e.query)}</div>
      <div class="meta">${okBadge}${badge("F1 " + num(e.f1), "", TIP.f1)}${badge(e.num_steps + " steps", "", TIP.used_steps)}${badge(e.stop_reason || "—", "", TIP.stop_reason)}</div>
    </li>`;
  }).join("");
  ul.querySelectorAll(".episode-item").forEach((li) => {
    li.addEventListener("click", () => loadEpisode(li.dataset.qid));
    li.addEventListener("dblclick", () => window.scrollTo({ top: 0, behavior: "smooth" }));
  });
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
  const correct = fm.answer_correct != null ? fm.answer_correct : fm.exact_match;
  const matchCls = correct ? "match" : "nomatch";
  const usedSteps = ep.used_steps != null ? ep.used_steps : (ep.steps || []).length;
  const header = `
    <div class="ep-header">
      <div class="qid">${esc(ep.qid)}</div>
      <h2>${esc(ep.query)}</h2>
      <div class="answers">
        <div class="answer-card">
          <div class="lbl">Gold answer · teacher-private</div>
          <div class="val">${goldValue(ep.gold_answer)}</div>
        </div>
        <div class="answer-card final ${matchCls}" title="${esc(TIP.correct)}">
          <div class="lbl">Final answer · ${correct ? "correct" : "incorrect"}</div>
          <div class="val">${esc(ep.final_answer) || "<span class='redacted'>(none)</span>"}</div>
        </div>
      </div>
      <div class="chips">
        ${metric(correct ? "✓" : "✗", "correct", TIP.correct)}
        ${metric(fm.exact_match ? "1" : "0", "EM", TIP.em)}
        ${metric(num(fm.f1), "F1", TIP.f1)}
        ${metric(pct(fm.supporting_doc_recall), "doc recall", TIP.doc_recall)}
        ${metric(pct(fm.supporting_fact_recall), "fact recall", TIP.fact_recall)}
      </div>
      <div class="meta-line">
        <span title="${esc(TIP.budget)}">budget <b>${esc(ep.budget)}</b></span>
        <span title="${esc(TIP.used_steps)}">used steps <b>${esc(usedSteps)}</b></span>
        <span title="${esc(TIP.guidance)}">guidance <b>G${esc(ep.guidance_level)}</b></span>
        <span title="${esc(TIP.stop_reason)}">stop <b>${esc(ep.stop_reason)}</b></span>
        <span title="${esc(TIP.student)}">student <code>${esc(cleanModel(ep.student_model))}</code></span>
        <span title="${esc(TIP.teacher)}">teacher <code>${esc(cleanModel(ep.teacher_model))}</code></span>
      </div>
    </div>`;

  const planHtml = renderPlanReview(ep.plan_review || {});
  const stepsHtml = `
    <div class="section">
      <h3 title="${esc(TIP.used_steps)}">Trajectory · ${usedSteps} used steps of budget ${esc(ep.budget)}</h3>
      <div class="timeline">${(ep.steps || []).map(renderStep).join("")}</div>
    </div>`;

  document.getElementById("detail").innerHTML = header + planHtml + stepsHtml;
}

function metric(v, k, tip) {
  const t = tip ? ` title="${esc(tip)}"` : "";
  return `<span class="metric"${t}><span class="mv">${esc(v)}</span><span class="mk">${esc(k)}</span></span>`;
}

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
    ? badge("guard fired · sanitized", "warn", TIP.leakage)
    : badge("no leakage", "good", TIP.leakage);
  const skipBadge = pr.revision_skipped ? badge("revision skipped · plan accepted", "accent", TIP.plan_review) : "";
  const rounds = pr.rounds || [];

  return `
  <div class="section">
    <h3 title="${esc(TIP.plan_review)}">Plan review ${leakBadge}${skipBadge}${timingBadge(pr.plan_review_elapsed_ms, "total", pr.plan_review_started_at, pr.plan_review_ended_at)}</h3>
    <div class="panel">
      <div class="plan-grid">
        <div class="plan-col">
          <h4>1 · Initial plan (student)</h4>
          <div class="plan-summary">${esc(initial.plan_summary || "")}</div>
          ${planStepsList(initial.steps)}
          <div class="chips" style="margin-top:8px">${timingBadge(pr.initial_plan_call_ms, "call")}</div>
          ${rawSection("Initial plan", pr.initial_plan_calls, pr.initial_student_plan_prompt, pr.initial_student_plan_raw)}
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
      ${renderPlanReviewRounds(rounds)}
      <div class="private-only" style="margin-top:12px">
        <div class="private-card">
          <div class="blk-label">${badge("teacher-private", "private")} plan diagnosis</div>
          <pre class="code">${json((pr.teacher_plan_review_full || {}).private_diagnosis)}</pre>
        </div>
      </div>
    </div>
  </div>`;
}

function renderPlanReviewRounds(rounds) {
  if (!rounds || !rounds.length) return "";
  return `
  <div class="panel" style="margin-top:12px">
    <div class="blk-label">Rounds · raw teacher review / student revision</div>
    ${rounds.map((r) => `
      <div style="${r.round > 1 ? "margin-top:12px;padding-top:12px;border-top:1px solid var(--border)" : ""}">
        <div class="chips" style="margin-bottom:6px">
          ${badge("round " + esc(r.round))}
          ${r.accepted ? badge("plan accepted", "good") : badge("plan revised")}
          ${timingBadge(r.review_call_ms, "teacher review")}
          ${r.revision_call_ms != null ? timingBadge(r.revision_call_ms, "student revision") : ""}
        </div>
        ${rawSection("Teacher review", r.review_calls, r.teacher_plan_review_prompt, r.teacher_plan_review_raw)}
        ${r.accepted ? "" : rawSection("Student revision", r.revision_calls, r.revised_student_plan_prompt, r.revised_student_plan_raw)}
      </div>
    `).join("")}
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
      ${badge(action.tool || "?", "tool", TIP.step_tool)}
      ${decision.category ? badge(decision.category) : ""}
      ${s.stop_condition === "FINISH" ? badge("FINISH", "accent") : ""}
      ${leakBadge}
      ${timingBadge(s.step_elapsed_ms, undefined, s.step_started_at, s.step_ended_at)}
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
        <div class="blk-label" title="${esc(TIP.guidance_box)}">Student-visible guidance</div>
        ${guidanceHtml}
      </div>
      <div class="block">
        <div class="blk-label">Raw model I/O</div>
        <div class="chips" style="margin-bottom:6px">${timingBadge(s.student_call_ms, "student call")}${timingBadge(s.teacher_call_ms, "teacher call")}</div>
        ${rawSection("Student", s.student_calls, s.student_prompt, s.student_raw, s.student_io_backfilled)}
        ${rawSection("Teacher", s.teacher_calls, s.teacher_prompt, s.teacher_raw, s.teacher_io_backfilled)}
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
