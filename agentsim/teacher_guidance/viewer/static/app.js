"use strict";

/*
 * Trajectory Explorer SPA.
 *
 * Hash-routed single-page app over three views:
 *   #/                      runs dashboard (sortable, searchable)
 *   #/run/<id>              run view: episode sidebar + empty detail
 *   #/run/<id>/ep/<qid>     run view with an episode's full trajectory
 *
 * All API responses are cached in-memory for the session (the refresh button clears
 * the cache); the server additionally serves gzip + ETag/304 so even cold loads are
 * cheap. Rendering is plain template strings — no framework, no build step.
 */

// ---------- utilities ----------
function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function pct(x) { return (Number(x || 0) * 100).toFixed(0) + "%"; }
function num(x) { return Number(x || 0).toFixed(2); }
function json(o) { return esc(JSON.stringify(o == null ? {} : o, null, 2)); }
function badge(text, cls, tip) {
  const t = tip ? ` title="${esc(tip)}"` : "";
  return `<span class="badge ${cls || ""}"${t}>${esc(text)}</span>`;
}
function cleanModel(m) { return String(m == null ? "" : m).replace(/^custom\//, ""); }

// Session-scoped API cache: repeated navigation never refetches (the server's ETag
// makes even a forced refetch a 304).
const apiCache = new Map();
async function getJSON(url) {
  if (apiCache.has(url)) return apiCache.get(url);
  const p = fetch(url).then((r) => {
    if (!r.ok) throw new Error(url + " -> " + r.status);
    return r.json();
  });
  apiCache.set(url, p);
  try { return await p; } catch (e) { apiCache.delete(url); throw e; }
}

function relTime(mtime) {
  if (!mtime) return "—";
  const s = Date.now() / 1000 - mtime;
  if (s < 90) return "just now";
  if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 86400) return Math.round(s / 3600) + " h ago";
  if (s < 7 * 86400) return Math.round(s / 86400) + " d ago";
  return new Date(mtime * 1000).toISOString().slice(0, 10);
}
function fmtDate(mtime) {
  if (!mtime) return "—";
  return new Date(mtime * 1000).toISOString().slice(0, 16).replace("T", " ");
}

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

// Tools get a stable color identity across the whole UI (mini-map, badges, timeline).
const TOOL_CLASS = {
  search: "t-search", extract: "t-extract", verify: "t-verify",
  synthesize: "t-synth", decompose: "t-decomp", reformulate: "t-reform",
  finish: "t-finish", wiki_read: "t-wiki", wiki_write: "t-wiki",
};
function toolBadge(tool, tip) {
  return badge(tool || "?", "tool " + (TOOL_CLASS[tool] || ""), tip || TIP.step_tool);
}

// Hover explanations for UI elements (shown as native tooltips).
const TIP = {
  episodes: "Number of question trajectories collected in this run.",
  guidance: "Guidance level 0–4: how much of the teacher's evaluation the student was shown. G3 = diagnostic feedback (score + explanation, no explicit next action).",
  correct: "Answer correct: the teacher (which can see the gold answer) scored the student's final answer ≥ 0.40 on its 0.0–1.0 scale. Falls back to cover-match for runs with no teacher verdict.",
  teacher_correct: "Teacher verdict: the teacher (which can see the gold answer) judged whether the student's final answer is correct — a binary 0/1 and a continuous 0.0–1.0 score, compared to the deterministic cover-match verdict.",
  em: "Exact match: the student's normalized final answer equals the gold answer exactly.",
  f1: "Token-level F1 overlap between the student's final answer and the gold answer.",
  doc_recall: "Supporting-document recall: fraction of the gold supporting documents the student retrieved.",
  fact_recall: "Supporting-fact recall: fraction of gold supporting sentences appearing verbatim in the student's extracted spans.",
  budget: "Budget: the maximum number of tool-use steps the student was allowed for this question.",
  used_steps: "Used steps: how many tool-use steps the student actually took before finishing.",
  mean_steps: "Mean tool-use steps per episode — lower is more efficient.",
  stop_reason: "Why the episode ended — teacher_accept (teacher accepted a finish) or budget_forced_finish (ran out of budget).",
  student: "Student model: solves the task with tools and never sees the gold answer.",
  teacher: "Teacher model: sees gold metadata and scores each step; how much it can tell the student is gated by the guidance level.",
  teacher_source: "Which provider actually served each teacher call, as a percentage — FAU (free), OpenRouter free, OpenRouter paid — reflecting the cost router's fallthrough.",
  plan_review: "Plan review: before acting, the student drafts a plan, the teacher reviews it, and the student revises it (revision is skipped if the teacher accepts the plan).",
  leakage: "Leakage guard: detects and sanitizes any gold answer / title / doc-id that the teacher's student-visible feedback tried to reveal.",
  step_tool: "The tool the student invoked this step (search, extract, verify, synthesize, decompose, reformulate, finish).",
  guidance_box: "Exactly what the student saw after this step — the rendered guidance allowed at this guidance level.",
  wiki: "Agent wiki (wiki.md): the student's private, per-episode notes file. In 'auto' mode it is read into every step's prompt and rewritten by a dedicated call after every step; in 'tools' mode the student calls wiki_read/wiki_write itself (each costs a step).",
  wiki_diff: "How wiki.md changed at this step relative to the previous step's version — additions highlighted green, removals struck through red.",
  wiki_final: "The content of wiki.md when the episode ended.",
  minimap: "One chip per step, colored by tool — click to jump to that step.",
};

// ---------- theme ----------
function initTheme() {
  const btn = document.getElementById("theme-toggle");
  btn.addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = cur;
    localStorage.setItem("tg-theme", cur);
  });
}

// ---------- router ----------
const state = { runs: null, currentRun: null, episodes: [], currentQid: null, epFilter: "all", epSearch: "" };

function navigate(hash) { location.hash = hash; }
function routeParts() {
  // "#/run/<encoded id>/ep/<encoded qid>" -> ["run", id, "ep", qid]
  const raw = location.hash.replace(/^#\/?/, "");
  return raw ? raw.split("/").map(decodeURIComponent) : [];
}
function runHash(runId) { return "#/run/" + encodeURIComponent(runId); }
function epHash(runId, qid) { return runHash(runId) + "/ep/" + encodeURIComponent(qid); }

async function route() {
  const parts = routeParts();
  try {
    if (parts[0] === "run" && parts[1]) {
      await showRunView(parts[1], parts[2] === "ep" ? parts[3] : null);
    } else {
      await showDashboard();
    }
  } catch (err) {
    app().innerHTML = errorState(err.message);
  }
}
function app() { return document.getElementById("app"); }

function setBreadcrumb(items) {
  document.getElementById("breadcrumb").innerHTML = items
    .map((it) => it.href
      ? `<a href="${esc(it.href)}">${esc(it.label)}</a>`
      : `<span class="crumb-here">${esc(it.label)}</span>`)
    .join(`<span class="crumb-sep">/</span>`);
}

// ---------- dashboard (runs) ----------
const dash = { sortKey: "mtime", sortDir: -1, search: "" };

const DASH_COLUMNS = [
  { key: "run_id", label: "Run", sortable: true },
  { key: "mtime", label: "Date", sortable: true },
  { key: "num_episodes", label: "Episodes", sortable: true, tip: TIP.episodes },
  { key: "mean_correct", label: "Correct", sortable: true, tip: TIP.correct },
  { key: "mean_f1", label: "F1", sortable: true, tip: TIP.f1 },
  { key: "mean_doc_recall", label: "Doc recall", sortable: true, tip: TIP.doc_recall },
  { key: "mean_steps", label: "Steps", sortable: true, tip: TIP.mean_steps },
  { key: "student_model", label: "Student", sortable: true, tip: TIP.student },
  { key: "teacher_source_pct", label: "Teacher source", sortable: false, tip: TIP.teacher_source },
];

async function showDashboard() {
  setBreadcrumb([{ label: "Runs" }]);
  document.title = "Runs — Trajectory Explorer";
  if (!state.runs) {
    app().innerHTML = `<section class="dashboard">${skeleton(6)}</section>`;
    state.runs = await getJSON("/api/runs");
  }
  renderDashboard();
}

function dashStats(runs) {
  const eps = runs.reduce((a, r) => a + r.num_episodes, 0);
  const correct = runs.reduce((a, r) => a + r.mean_correct * r.num_episodes, 0);
  return [
    { v: runs.length, k: "runs" },
    { v: eps, k: "episodes", tip: TIP.episodes },
    { v: eps ? pct(correct / eps) : "—", k: "overall correct", tip: TIP.correct },
  ];
}

function renderDashboard() {
  const q = dash.search.toLowerCase();
  let runs = (state.runs || []).filter((r) =>
    !q || r.run_id.toLowerCase().includes(q) || cleanModel(r.student_model).toLowerCase().includes(q));
  runs = sortRuns(runs);

  const stats = dashStats(state.runs || []).map((s) =>
    `<div class="stat"${s.tip ? ` title="${esc(s.tip)}"` : ""}><span class="v">${esc(s.v)}</span><span class="k">${esc(s.k)}</span></div>`).join("");

  const head = DASH_COLUMNS.map((c) => {
    const active = dash.sortKey === c.key;
    const arrow = active ? (dash.sortDir > 0 ? " ▲" : " ▼") : "";
    const cls = (c.sortable ? "sortable" : "") + (active ? " sorted" : "");
    return `<th class="${cls}" data-key="${esc(c.key)}"${c.tip ? ` title="${esc(c.tip)}"` : ""}>${esc(c.label)}${arrow}</th>`;
  }).join("");

  const rows = runs.map((r) => `<tr class="run-row" data-run="${esc(r.run_id)}">
    <td class="run-id">${esc(r.run_id)}${r.wiki_mode ? " " + badge("wiki·" + r.wiki_mode, "wiki", TIP.wiki) : ""}${r.guidance_level != null ? " " + badge("G" + r.guidance_level, "", TIP.guidance) : ""}</td>
    <td class="nowrap muted" title="${esc(fmtDate(r.mtime))}">${esc(relTime(r.mtime))}</td>
    <td>${r.num_episodes}</td>
    <td>${rateBar(r.mean_correct)}</td>
    <td>${num(r.mean_f1)}</td>
    <td>${pct(r.mean_doc_recall)}</td>
    <td>${r.mean_steps ? Number(r.mean_steps).toFixed(1) : "—"}</td>
    <td><code>${esc(cleanModel(r.student_model))}</code></td>
    <td class="src">${sourcePctStr(r.teacher_source_pct)}</td>
  </tr>`).join("");

  app().innerHTML = `
  <section class="dashboard">
    <div class="dash-head">
      <div class="dash-stats">${stats}</div>
      <input type="search" id="dash-search" class="search-input" placeholder="Filter runs or models…" value="${esc(dash.search)}" />
    </div>
    <div class="table-scroll panel-flat">
      <table class="runs-table">
        <thead><tr>${head}</tr></thead>
        <tbody>${rows || `<tr><td colspan="9" class="muted" style="text-align:center;padding:32px">No matching runs</td></tr>`}</tbody>
      </table>
    </div>
  </section>`;

  document.querySelectorAll(".run-row").forEach((tr) =>
    tr.addEventListener("click", () => navigate(runHash(tr.dataset.run))));
  document.querySelectorAll("th.sortable").forEach((th) =>
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (dash.sortKey === key) dash.sortDir *= -1;
      else { dash.sortKey = key; dash.sortDir = key === "run_id" || key === "student_model" ? 1 : -1; }
      renderDashboard();
    }));
  const search = document.getElementById("dash-search");
  search.addEventListener("input", (e) => { dash.search = e.target.value; renderDashboard(); refocusSearch(); });
  function refocusSearch() {
    const el = document.getElementById("dash-search");
    el.focus(); el.setSelectionRange(el.value.length, el.value.length);
  }
}

function sortRuns(runs) {
  const k = dash.sortKey, dir = dash.sortDir;
  return [...runs].sort((a, b) => {
    const va = a[k], vb = b[k];
    if (typeof va === "string" || typeof vb === "string")
      return String(va || "").localeCompare(String(vb || "")) * dir;
    return ((va || 0) - (vb || 0)) * dir;
  });
}

function rateBar(rate) {
  const p = Math.round((rate || 0) * 100);
  const cls = p >= 70 ? "hi" : p >= 40 ? "mid" : "lo";
  return `<div class="rate" title="${esc(TIP.correct)}">
    <div class="rate-bar"><div class="rate-fill ${cls}" style="width:${p}%"></div></div>
    <span class="rate-num">${p}%</span>
  </div>`;
}

function sourcePctStr(pcts) {
  const e = Object.entries(pcts || {});
  if (!e.length) return "—";
  return e.map(([k, v]) => `${esc(k)} ${v}%`).join(" · ");
}

// ---------- run view ----------
async function showRunView(runId, qid) {
  if (!state.runs) state.runs = await getJSON("/api/runs");
  state.currentRun = state.runs.find((r) => r.run_id === runId) || { run_id: runId };
  document.title = runId + " — Trajectory Explorer";

  const isNewRun = !app().querySelector(".layout") || state._loadedRun !== runId;
  if (isNewRun) {
    setBreadcrumb([{ label: "Runs", href: "#/" }, { label: runId }]);
    app().innerHTML = `
      ${renderRunStats(state.currentRun)}
      <div class="layout">
        <aside class="sidebar">
          <div class="sidebar-head">
            <input type="search" id="episode-search" class="search-input" placeholder="Filter questions…" value="${esc(state.epSearch)}" />
            <div class="filter-chips" id="ep-filters"></div>
            <span id="episode-count" class="muted"></span>
          </div>
          <ul id="episode-list" class="episode-list">${skeleton(4)}</ul>
        </aside>
        <section id="detail" class="detail"></section>
      </div>`;
    document.getElementById("episode-search").addEventListener("input", (e) => {
      state.epSearch = e.target.value; renderEpisodeList();
    });
    state.episodes = await getJSON("/api/episodes?run=" + encodeURIComponent(runId));
    state._loadedRun = runId;
  }

  state.currentQid = qid;
  renderEpisodeList();
  if (qid) {
    setBreadcrumb([{ label: "Runs", href: "#/" }, { label: runId, href: runHash(runId) }, { label: qid }]);
    await loadEpisode(runId, qid);
  } else {
    document.getElementById("detail").innerHTML = `
      <div class="empty-state">
        <h2>${state.episodes.length} episodes</h2>
        <p>Select a question on the left — or use <kbd>j</kbd>/<kbd>k</kbd> to move and <kbd>Enter</kbd> to open.</p>
      </div>`;
  }
}

function renderRunStats(run) {
  if (!run || run.num_episodes == null) return "";
  const stops = Object.entries(run.stop_reasons || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") || "—";
  const cells = [
    st(run.num_episodes, "episodes", TIP.episodes),
    st("G" + (run.guidance_level ?? "?"), "guidance", TIP.guidance),
    st(pct(run.mean_correct), "correct", TIP.correct),
    st(num(run.mean_f1), "F1", TIP.f1),
    st(pct(run.mean_doc_recall), "doc recall", TIP.doc_recall),
    run.mean_steps ? st(Number(run.mean_steps).toFixed(1), "mean steps", TIP.mean_steps) : "",
    run.wiki_mode ? st("wiki·" + run.wiki_mode, "agent wiki", TIP.wiki) : "",
    stWide(cleanModel(run.student_model), "student", TIP.student),
    stWide(cleanModel(run.teacher_model), "teacher", TIP.teacher),
    stWide(stops, "stop reasons", TIP.stop_reason),
  ].filter(Boolean).join("");
  return `<div class="run-stats">${cells}</div>`;
  function st(v, k, tip) {
    return `<div class="stat" title="${esc(tip || "")}"><span class="v">${esc(v)}</span><span class="k">${esc(k)}</span></div>`;
  }
  function stWide(v, k, tip) {
    return `<div class="stat wide" title="${esc(tip || "")}"><span class="v">${esc(v)}</span><span class="k">${esc(k)}</span></div>`;
  }
}

const EP_FILTERS = [
  { id: "all", label: "All" },
  { id: "correct", label: "✓ correct" },
  { id: "incorrect", label: "✗ incorrect" },
  { id: "forced", label: "forced finish" },
];

function filteredEpisodes() {
  const q = state.epSearch.toLowerCase();
  return state.episodes.filter((e) => {
    if (q && !(e.query || "").toLowerCase().includes(q) && !(e.qid || "").toLowerCase().includes(q)) return false;
    const correct = e.answer_correct != null ? e.answer_correct : e.exact_match;
    if (state.epFilter === "correct") return correct;
    if (state.epFilter === "incorrect") return !correct;
    if (state.epFilter === "forced") return e.stop_reason === "budget_forced_finish";
    return true;
  });
}

function renderEpisodeList() {
  const items = filteredEpisodes();
  const countEl = document.getElementById("episode-count");
  if (countEl) countEl.textContent = `${items.length}/${state.episodes.length}`;

  const filters = document.getElementById("ep-filters");
  if (filters) {
    filters.innerHTML = EP_FILTERS.map((f) =>
      `<button class="chip-btn ${state.epFilter === f.id ? "on" : ""}" data-f="${f.id}" type="button">${esc(f.label)}</button>`).join("");
    filters.querySelectorAll(".chip-btn").forEach((b) =>
      b.addEventListener("click", () => { state.epFilter = b.dataset.f; renderEpisodeList(); }));
  }

  const ul = document.getElementById("episode-list");
  if (!ul) return;
  ul.innerHTML = items.map((e) => {
    const correct = e.answer_correct != null ? e.answer_correct : e.exact_match;
    const active = e.qid === state.currentQid ? "active" : "";
    return `<li class="episode-item ${active} ${correct ? "ok" : "ko"}" data-qid="${esc(e.qid)}">
      <div class="q">${esc(e.query)}</div>
      <div class="meta">
        ${correct ? badge("✓", "good", TIP.correct) : badge("✗", "bad", TIP.correct)}
        ${badge("F1 " + num(e.f1), "", TIP.f1)}
        ${badge(e.num_steps + " steps", "", TIP.used_steps)}
        ${e.stop_reason === "budget_forced_finish" ? badge("forced", "warn", TIP.stop_reason) : ""}
      </div>
    </li>`;
  }).join("") || `<li class="muted" style="padding:16px">No matching episodes</li>`;
  ul.querySelectorAll(".episode-item").forEach((li) =>
    li.addEventListener("click", () => navigate(epHash(state.currentRun.run_id, li.dataset.qid))));
}

// Keyboard navigation: j/k move through the sidebar, Enter opens, Esc goes up a level.
function initKeyboard() {
  document.addEventListener("keydown", (ev) => {
    if (ev.target.matches("input, textarea, select")) return;
    const parts = routeParts();
    if (parts[0] !== "run") return;
    const items = filteredEpisodes();
    if (ev.key === "j" || ev.key === "k") {
      if (!items.length) return;
      let i = items.findIndex((e) => e.qid === state.currentQid);
      i = ev.key === "j" ? Math.min(i + 1, items.length - 1) : Math.max(i - 1, 0);
      navigate(epHash(state.currentRun.run_id, items[i].qid));
      ev.preventDefault();
    } else if (ev.key === "Escape") {
      navigate(state.currentQid ? runHash(state.currentRun.run_id) : "#/");
    }
  });
}

// ---------- episode detail ----------
async function loadEpisode(runId, qid) {
  state.currentQid = qid;
  document.querySelectorAll(".episode-item").forEach((li) =>
    li.classList.toggle("active", li.dataset.qid === qid));
  const detail = document.getElementById("detail");
  detail.innerHTML = skeleton(5);
  const ep = await getJSON(
    "/api/episode?run=" + encodeURIComponent(runId) + "&qid=" + encodeURIComponent(qid));
  renderEpisodeDetail(ep);
  detail.scrollTop = 0;
  window.scrollTo({ top: 0 });
}

function goldValue(text) {
  // Gold is teacher-private: show it only when the reveal toggle is on.
  return `<span class="when-revealed">${esc(text)}</span><span class="when-hidden redacted">[hidden — enable “Teacher-private”]</span>`;
}

function renderEpisodeDetail(ep) {
  const fm = ep.final_metrics || {};
  const correct = fm.answer_correct != null ? fm.answer_correct : fm.exact_match;
  const matchCls = correct ? "match" : "nomatch";
  const usedSteps = ep.used_steps != null ? ep.used_steps : (ep.steps || []).length;
  const tCorrect = fm.teacher_answer_correct;
  const tScore = fm.teacher_answer_score;
  const teacherChip = tCorrect != null
    ? metric(`${tCorrect ? "✓" : "✗"}${tScore != null ? " " + Number(tScore).toFixed(2) : ""}`,
             "teacher verdict", TIP.teacher_correct)
    : "";
  const steps = ep.steps || [];

  const header = `
    <div class="ep-header">
      <div class="qid mono">${esc(ep.qid)}</div>
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
        ${teacherChip}
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
        ${ep.wiki_enabled ? `<span title="${esc(TIP.wiki)}">wiki <b>${esc(ep.wiki_mode || "tools")}</b></span>` : ""}
        <span title="${esc(TIP.student)}">student <code>${esc(cleanModel(ep.student_model))}</code></span>
        <span title="${esc(TIP.teacher)}">teacher <code>${esc(cleanModel(ep.teacher_model))}</code></span>
      </div>
      ${renderMinimap(steps)}
    </div>`;

  const planHtml = renderPlanReview(ep.plan_review || {});
  const wikiHtml = renderWikiSection(ep);
  const stepsHtml = `
    <div class="section">
      <h3 title="${esc(TIP.used_steps)}">Trajectory · ${usedSteps} used steps of budget ${esc(ep.budget)}</h3>
      <div class="timeline">${steps.map((s, i) =>
        renderStep(s, i > 0 ? steps[i - 1].wiki_after : "")).join("")}</div>
    </div>`;

  document.getElementById("detail").innerHTML = header + planHtml + wikiHtml + stepsHtml;
  document.querySelectorAll(".minimap .mm-step").forEach((el) =>
    el.addEventListener("click", () => {
      const target = document.getElementById("step-" + el.dataset.t);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    }));
}

function renderMinimap(steps) {
  if (!steps.length) return "";
  return `<div class="minimap" title="${esc(TIP.minimap)}">
    ${steps.map((s) => {
      const tool = ((s.student_action || {}).action || {}).tool || "?";
      return `<button type="button" class="mm-step ${TOOL_CLASS[tool] || ""}" data-t="${esc(s.t)}" title="step ${esc(s.t)} · ${esc(tool)}">${esc(s.t)}</button>`;
    }).join("")}
  </div>`;
}

function metric(v, k, tip) {
  const t = tip ? ` title="${esc(tip)}"` : "";
  return `<span class="metric"${t}><span class="mv">${esc(v)}</span><span class="mk">${esc(k)}</span></span>`;
}

// ---------- raw model I/O ----------
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

// Renders raw model I/O for one logical turn. Prefers the native per-call log when
// present; falls back to a single prompt/raw pair; shows an explicit "not recorded"
// note rather than silently rendering nothing.
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

// ---------- agent wiki (wiki.md) ----------
// Word-level LCS diff between two wiki versions. Wikis are hard-capped at 2000 chars,
// so the O(n·m) table stays tiny.
function diffTokens(a, b) {
  const tok = (s) => String(s || "").match(/\S+\s*|\s+/g) || [];
  const A = tok(a), B = tok(b);
  const n = A.length, m = B.length;
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const ops = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) { ops.push(["=", A[i]]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { ops.push(["-", A[i]]); i++; }
    else { ops.push(["+", B[j]]); j++; }
  }
  while (i < n) ops.push(["-", A[i++]]);
  while (j < m) ops.push(["+", B[j++]]);
  return ops;
}

function wikiDiffHtml(prev, cur) {
  return diffTokens(prev, cur).map(([op, t]) =>
    op === "=" ? esc(t)
      : op === "+" ? `<ins>${esc(t)}</ins>`
      : `<del>${esc(t)}</del>`).join("");
}

// Per-step wiki.md panel (auto mode): the version after this step, with a collapsible
// word-level diff against the previous step's version and the raw update call.
function renderStepWiki(s, prevWiki) {
  if (s.wiki_after == null) return "";
  const cur = String(s.wiki_after || "");
  const prev = String(prevWiki || "");
  const changed = prev !== cur;
  const chip = changed
    ? badge("updated", "wiki", TIP.wiki_diff)
    : badge("unchanged", "", TIP.wiki_diff);
  const diff = changed
    ? `<details class="wiki-details"><summary>Show changes from previous version</summary>
         <div class="wiki-diff">${wikiDiffHtml(prev, cur)}</div>
       </details>`
    : "";
  const updCall = s.wiki_update_call
    ? `<details class="wiki-details"><summary>Wiki update · raw model call</summary>
         ${callAttempt("Wiki update", s.wiki_update_call)}
       </details>`
    : "";
  return `<div class="block">
    <div class="blk-label" title="${esc(TIP.wiki)}">wiki.md · after this step ${chip}</div>
    <pre class="wiki-content">${esc(cur) || "<span class='redacted'>(empty)</span>"}</pre>
    ${diff}${updCall}
  </div>`;
}

// Episode-level wiki section: mode + the final wiki.md state.
function renderWikiSection(ep) {
  if (!ep.wiki_enabled) return "";
  const mode = ep.wiki_mode || "tools";
  const modeNote = mode === "auto"
    ? "read into every step's prompt · rewritten after every step — per-step versions and diffs are in the trajectory below"
    : "student-managed via wiki_read/wiki_write tool calls";
  return `
  <div class="section">
    <h3 title="${esc(TIP.wiki)}">Agent wiki ${badge("wiki.md · " + mode, "wiki", TIP.wiki)}</h3>
    <div class="panel">
      <div class="muted" style="margin-bottom:8px">${esc(modeNote)}</div>
      <div class="blk-label" title="${esc(TIP.wiki_final)}">Final state</div>
      <pre class="wiki-content">${esc(ep.wiki_final || "") || "<span class='redacted'>(empty — never written)</span>"}</pre>
    </div>
  </div>`;
}

// ---------- plan review ----------
function planStepsList(steps) {
  if (!steps || !steps.length) return `<div class="redacted">—</div>`;
  return `<ul class="plan-steps">${steps.map((s) =>
    `<li><span class="n">${esc(s.step_id)}</span>${toolBadge(s.intended_tool)}<span>${esc(s.goal || "")}</span></li>`).join("")}</ul>`;
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
  <div class="panel inner" style="margin-top:12px">
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

// ---------- steps ----------
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
      `<li class="fact invalid">${esc(e.span)} <span class="muted">(${esc(e.reason)})</span></li>`).join("");
    return `<ul class="facts">${ex || ""}${bad || ""}</ul>` || `<div class="redacted">—</div>`;
  }
  return `<pre class="code">${json(obs)}</pre>`;
}

function renderStep(s, prevWiki) {
  const action = (s.student_action || {}).action || {};
  const decision = (s.student_action || {}).decision || {};
  const facts = (s.student_action || {}).new_facts_extracted || [];
  const g = s.student_visible_guidance || {};
  const m = s.metrics || {};
  const leak = s.leakage_check || {};
  const leaked = ["gold_answer_leaked", "hidden_title_leaked", "hidden_doc_id_leaked", "hidden_span_leaked"]
    .filter((k) => leak[k]);

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
  <div class="step ${TOOL_CLASS[action.tool] || ""}" id="step-${esc(s.t)}">
    <div class="step-head">
      <span class="t">Step ${esc(s.t)}</span>
      ${toolBadge(action.tool)}
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
      ${renderStepWiki(s, prevWiki)}
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

// ---------- shared ----------
function skeleton(n) {
  return `<div class="skeleton">${Array.from({ length: n }, () => `<div class="sk-row"></div>`).join("")}</div>`;
}
function errorState(msg) {
  return `<div class="empty-state"><h2>Nothing to show</h2><p>${esc(msg)}</p></div>`;
}

// ---------- boot ----------
function init() {
  initTheme();
  initKeyboard();

  const priv = document.getElementById("private-toggle");
  document.body.classList.toggle("hide-private", !priv.checked);
  priv.addEventListener("change", (e) => {
    document.body.classList.toggle("hide-private", !e.target.checked);
  });

  document.getElementById("refresh-btn").addEventListener("click", () => {
    apiCache.clear();
    state.runs = null;
    state._loadedRun = null;
    route();
  });

  window.addEventListener("hashchange", route);
  route();
}

init();
