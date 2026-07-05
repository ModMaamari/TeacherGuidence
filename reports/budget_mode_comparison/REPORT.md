# Budget disclosure comparison — disclosed vs hidden budget

Two runs on the **same 10 HotpotQA questions**, student `ollama/qwen3.5:2b`, teacher
`gpt-oss-120b` via the cost router, **plan budget = 3 rounds, run budget = 9 steps**.
The only difference is whether the student is told its step budget.

- **Disclosed** (default): every step shows `Step N of budget 9`; the plan prompt states
  the 9-step budget.
- **Hidden** (`--hidden-budget`): the student never sees the budget — steps show only
  `This is step N` plus a nudge to retrieve only what it needs and finish as soon as it is
  confident; the plan prompt asks for the *shortest efficient plan*. The budget is revealed
  only on the forced final step so the episode still terminates.

## Results (n = 10, same questions)

| Metric | Disclosed | Hidden |
|---|---|---|
| Answer correct | 2 | **3** |
| Grounded | 4 | **5** |
| Natural finish (`teacher_accept`) | 1 | **2** |
| Mean steps used | 8.4 | **8.0** |
| Median steps | 9 | 9 |

## Per-question steps used (correct = ✓)

| Question | Disclosed | Hidden |
|---|---|---|
| PDC tournament at The O2 | 9 | 9 |
| British singer-songwriter, 16th Young Hollywood Awards | 9 ✓ | **6 ✓** |
| W. Tresper Clarke High School county | 9 | **9 ✓** |
| General Survey Act / Supreme Court decision | 9 ✓ | 9 |
| Adventist World & Girls' Life HQ (yes/no) | 9 | 9 |
| 2005 Pixar animated short | 9 | **2 ✓** |
| Who was born first (Echegaray / O'Rourke) | 9 | 9 |
| Jim Price / Rolling Stones album month | 9 | 9 |
| Russian ballet choreographer | 3 | 9 |
| Marianas Trench vs Tse Tse Fly | 9 | 9 |

## Takeaways

- **Hidden budget nudges efficient, earlier answers.** The clearest wins: the 2005 Pixar
  short (**2 steps, correct** vs 9 steps, wrong when disclosed) and the singer-songwriter
  (**6 vs 9 steps**, both correct). Telling the model to "answer as soon as you are
  confident" instead of handing it a step budget it tends to spend in full produced more
  natural finishes (2 vs 1) and a slightly higher correct/grounded count with fewer steps.
- **Disclosing the budget encourages the student to use it all** — 9/10 disclosed episodes
  ran to the cap (only one natural finish), consistent with the earlier 100-sample runs
  where students almost never stopped on their own.
- **Caveat:** n = 10 (the hardest first-10 questions at a tight budget of 9), so absolute
  accuracy is low and the difference is within noise on any single metric; the *direction*
  — hidden budget → fewer steps, more natural finishes, no accuracy loss — is the
  reportable signal. A larger run would be needed to quantify the effect precisely.
