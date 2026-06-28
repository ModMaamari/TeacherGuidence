# Teacher Guidance — Trajectory Explorer (web UI)

A local, dependency-free web UI for exploring the trajectories a Teacher Guidance run
produces. It reads the `teacher_guidance_episodes.jsonl` files under an output root and
lets you browse runs, pick a question, and inspect the plan-review phase and every step.

## Launch

From the repo root, in your environment (the `.venv` you set up in `TEACHER_GUIDANCE.md`):

```powershell
# via the CLI
agentsim viewer --output-root data/simulation_output --port 8000

# or the module
python -m agentsim.teacher_guidance.viewer --output-root data/simulation_output --port 8000

# or the script
python scripts/run_trajectory_viewer.py --output-root data/simulation_output --port 8000
```

It opens `http://127.0.0.1:8000/` in your browser (pass `--no-browser` to skip).

Options: `--output-root` (default `data/simulation_output`), `--host` (default
`127.0.0.1`), `--port` (default `8000`), `--no-browser`.

## What you see

- **Top bar** — pick a run; aggregate chips show episode count, guidance level, mean
  exact match / F1 / supporting-doc recall, the student/teacher models, and the
  stop-reason breakdown. A **"Show teacher-private"** toggle reveals data the student
  never saw.
- **Left list** — every question in the run with EM ✓/✗, F1, step count, and stop
  reason. Filter by typing.
- **Detail view**
  - **Header** — query, gold vs final answer (gold is hidden until you enable
    "Show teacher-private"), metric chips, models, budget, guidance level, stop reason.
  - **Plan review** — the student's initial plan, the student-visible teacher feedback,
    and the revised plan side by side, with a leakage badge. The teacher-private plan
    diagnosis appears only when the toggle is on.
  - **Trajectory** — one card per step: the chosen tool, the student's thought and
    action params, the tool observation (search results render as a ranked list), the
    student-visible guidance box, deterministic step labels, and a badge whenever the
    leakage guard sanitized the teacher's feedback. The teacher-private diagnosis is
    gated behind the toggle.

## Student view vs analyst view

By default the UI emulates **what the student saw**: the gold answer and the teacher's
private diagnosis are hidden. Toggle **"Show teacher-private"** to switch to the
**analyst view** and reveal them — useful for auditing whether the guidance the student
received was appropriate and leakage-free.

## Notes

- The viewer only **reads** run outputs; it never modifies them.
- It uses the Python standard library only (no Flask/FastAPI/Node) and serves a vanilla
  HTML/CSS/JS frontend — nothing to build.
- Point `--output-root` at any directory that contains
  `**/teacher_guidance_episodes.jsonl`; multiple runs are listed in the run selector.
