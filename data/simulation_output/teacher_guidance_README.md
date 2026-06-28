# Teacher Guidance simulation output

Each `agentsim simulate hotpot_teacher_guidance_b5_g<N>` run creates:

```
<this dir>/hotpot_teacher_guidance_b5_g<N>/<run_uuid>/hotpot_questions/sample_XXX/
  traces.jsonl
  supervised.jsonl
  trajectories.jsonl
  teacher_guidance_episodes.jsonl
  student_sft.jsonl
  teacher_sft.jsonl
  student_visible_guidance.jsonl
  plan_review_rows.jsonl            # only when plan review is enabled
  teacher_guidance_metrics.json
```

See `TEACHER_GUIDANCE.md` at the repo root for the full setup and run guide.

These output files are git-ignored; only this README is tracked.
