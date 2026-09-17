# CLAUDE.md — DPCN Assignment 1: Opinion Network Formation

Read this in full before writing or editing any code. This file is the single source of
truth for the pipeline. If a teammate's instruction conflicts with this file, flag the
conflict and ask — do not silently pick one.

---

## 0. Project facts

- **Course:** Dynamical Processes on Complex Networks, Assignment 1.
- **Team size:** 3. Graded **individually** — every commit must be attributable to one
  person. Do not commit on someone else's behalf.
- **Build window:** 2 days.
- **Deliverable:** ~8-page PDF report (LaTeX) + public, well-organised GitHub repo.
- **Data:** `data/Survey_Results_UC.csv` — 96 respondents × 60 Likert items across four
  topic blocks (T = Technology, E = Education, S = Society/Ethics, V = Environment).
- **AI tool use is explicitly permitted** by the assignment.

Required report sections, in this order: Team Name; GitHub link; Dataset Documentation;
Pipeline Followed; Analysis and Visualizations; Results and Discussion; Individual
Contribution.

Instructor's graded expectations: state what the nodes and edges are and justify it; state
binary vs weighted and justify it; apply multiple metrics and infer significant results;
show findings via graphs and heatmaps. **Effort is explicitly graded** — document methods
tried and rejected, not only the one kept.

---

## 1. Dataset handling

- Read with `encoding='utf-8-sig'`. The file has a UTF-8 BOM on the first column header;
  without this flag that column name will not match anything.
- Shape: 96 rows × 61 columns (1 respondent-ID column + 60 item columns).
- Column headers are full question text with an item-code prefix, e.g. `"T01. AI-assisted..."`.
  Parse the code with a regex; never rename columns by hand.
- Four blocks of 15 items each: T, E, S, V.
- Scale: Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree, **No Comments**.
  Map to −2, −1, 0, +1, +2 and **No Comments → NaN** (37 occurrences). "No Comments" is a
  decline-to-answer, not a neutral opinion — do not map it to 0.
- 505 blanks + 37 "No Comments" = 542 missing entries.

### Missingness structure (measured — verify against this)

| Respondents | Missing items |
|---|---|
| 68 | 0 |
| 17 | 1–5 |
| 1 | 6–12 |
| 5 | 13–59 |
| 5 | all 60 (empty rows) |

**495 of 542 missing cells sit in the 10 respondents with >12 missing.** Drop those 10.
Among the remaining **86 respondents only 47 cells (0.9%) are missing.** Fill with column
median. Do not build a regression, ordinal, or Bayesian imputer — at 0.9% missingness it
cannot change any downstream result. This is a documented decision, not a shortcut.

---

## 2. Frozen pipeline decisions

Do not change these once Person 1 has delivered the centred matrix. If a change seems
necessary, stop and ask the team.

| Step | Decision |
|---|---|
| Filter | Drop respondents with >12 missing → **86 remain** |
| Imputation | Column-median on the remaining 47 cells |
| Baseline correction | **Row-centre**: subtract each respondent's own mean across all 60 items |
| Similarity | Pearson correlation on the row-centred matrix, **signs retained** (never absolute value) |
| Primary network | **Item–item** (60 nodes). The respondent network is secondary / negative-result only |
| Edge rule | Threshold on \|r\|, chosen from the percolation sweep (Graph 7), not picked arbitrarily |
| Null model | **Column-permutation** is primary. ER and degree-preserving rewiring appear only as weaker comparisons the report explains are insufficient |
| Weighting | Signed and weighted by default; binarise only where a metric requires it |

### Why row-centring is mandatory
Grand mean across all responses is **+1.14** on a −2…+2 scale — near-universal agreement.
Raw-score similarity would mostly measure who is agreeable, not what they believe.

### Why the permutation null is mandatory
A correlation-derived network has transitivity built into its construction: if A correlates
with B and B with C, A and C are pulled toward correlation automatically. This alone
produces clustering and modularity in meaningless data. Rewiring and ER nulls take the
*finished graph* as given and cannot detect this. The permutation null regenerates the whole
pipeline (centre → correlate → threshold → measure) from scrambled responses, which is the
only test matching the claim being made.

**This matters empirically.** On the respondent network, modularity clears the rewiring null
easily (0.423 vs 0.329 ± 0.011, z = 7.04) but does **not** clear the permutation null
(0.403 ± 0.017, z = 1.15, p = 0.08). That community structure is a construction artefact.
Report it as a finding, not a failure.

---

## 3. Reference values — regression-test against these

If your code doesn't reproduce these, something drifted. Investigate before proceeding.

- **PCA on row-centred item matrix:** PC1–PC5 = 8.3%, 7.2%, 6.7%, 5.5%, 5.1%.
- **Marchenko–Pastur:** T = 86, N = 60, Q = 1.433, λ₊ = (1 + 1/√Q)² ≈ **3.368**. Exactly
  **3 eigenvalues** exceed it: **4.05, 3.98, 3.53**, carrying **19.3%** of total variance.
- **Item SD range:** 0.50 (E15, near-unanimous) to 1.17 (E04, most contested). Also high:
  E03 (1.12), E02 (1.12), T08 (1.09).
- **Real vs shuffled item correlations:** real SD = 0.136, shuffled SD = 0.108 (matches
  theoretical 1/√(n−3) ≈ 0.109). At |r| > 0.15: 467 real edges vs 309 chance (66%).
  At |r| > 0.20: 258 vs 118 (46%). At |r| > 0.25: 111 vs 37 (33%). At |r| > 0.30: 49 vs 9 (18%).
- **Block-mean correlations:** T–E 0.29, T–S 0.54, T–V 0.44, E–S 0.51, E–V 0.49, **S–V 0.68**.
- **Respondent network** (kNN k = 4, symmetrised — **not** mutual): 86 nodes, 286 edges,
  1 component, 7 Louvain communities sized 19/13/12/11/11/11/9. Modularity 0.408–0.423,
  clustering 0.202. Rewiring null: 0.329 ± 0.011 (z ≈ 7) and 0.091 ± 0.013 (z ≈ 8.9) — passes.
  Permutation null: 0.403 ± 0.017 (z ≈ 1.15) — **fails**. Mutual kNN fragments the graph
  (58 edges, 38 components) — do not use it.
- **Structural balance** (signed item network): balanced-triad fraction 83% at |r| > 0.15,
  96% at 0.20, 100% at 0.25, against ~50% under a sign-permutation null. The trend
  strengthening as the threshold tightens is itself part of the evidence.
- **Deffuant–Weisbuch** (seeded on E03, rescaled to [0,1], 20 reps): 5.0 clusters for
  ε ≤ 0.25, 4.2 at ε = 0.30, 3.0 at 0.40, 1.6 at 0.50. Consensus near ε ≈ 0.5. The plateau
  at low ε is partly a discretisation artefact of the 5-point scale — say so.

---

## 4. The 12 graphs, effort-weighted, with owners

Effort 1 (quick plot) to 5 (heaviest). Total 36, ~12 per person.

| # | Graph | Effort | Owner | Depends on |
|---|---|---|---|---|
| 1 | Missingness + response distribution | 1 | P1 | raw data |
| 2 | Item variance ranking | 1 | P1 | raw data |
| 3 | Row-centring effect | 2 | P1 | clean matrix |
| 4 | Real vs shuffled correlation distributions | 4 | P2 | centred matrix |
| 5 | Eigenvalue spectrum vs Marchenko–Pastur band | 3 | P2 | correlation matrix |
| 7 | Threshold / percolation sweep | 4 | P2 | denoised matrix |
| 9 | Topic-block connectivity (4×4) | 2 | P1 | centred matrix |
| 8 | Main item network | 3 | P1 | chosen threshold |
| 12 | Opinion dynamics (Deffuant–Weisbuch) | 4 | P1 | final graph |
| 10 | Communities vs 3 null models | 5 | P3 | final graph + null machinery |
| 11 | Structural balance vs threshold | 5 | P3 | signed network |
| 6 | Reordered correlation heatmap | 2 | P3 | matrix + communities |

**Effort totals:** P1 = 13 (graphs 1, 2, 3, 9, 8, 12); P2 = 11 (graphs 4, 5, 7);
P3 = 12 (graphs 10, 11, 6).

**Critical path:** 1 → 3 → 4 → 5 → 7 → 8 → {10, 11, 12} → 6. Graphs 2 and 9 branch off and
block nobody.

**Two hard handoffs:**
- **P1 → P2**, end of Day 1 morning: the cleaned, row-centred 86 × 60 matrix.
- **P2 → P1 and P3**, end of Day 1 afternoon: denoised correlation matrix, chosen threshold,
  and the null-model framework with cached replicates.

---

## 5. Repository structure

```
opinion-network/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── run_all.py
├── CONTRIBUTIONS.md
├── data/
│   └── Survey_Results_UC.csv       (read-only, never edit in place)
├── src/
│   ├── common/
│   │   ├── config.py               SEED, paths, palette — import, never redefine
│   │   ├── loader.py               P1: parsing, encoding, filtering, imputation, centring
│   │   └── nulls.py                P2: permutation / rewiring / ER generators
│   ├── person1/
│   │   ├── graphs_1_2_3.py
│   │   ├── graph_9.py
│   │   ├── graph_8.py
│   │   └── graph_12.py
│   ├── person2/
│   │   ├── graph_4.py
│   │   ├── graph_5.py
│   │   └── graph_7.py
│   └── person3/
│       ├── graph_10.py
│       ├── graph_11.py
│       └── graph_6.py
├── figures/
│   ├── person1/  person2/  person3/
├── notebooks/                      exploration only, nothing load-bearing
└── report/
    ├── report.tex
    └── figures/                    final copies used in the PDF
```

**Rules:**
- Modules in `src/personX/` are importable and side-effect free — no top-level plotting or
  file writes at import time. `run_all.py` orchestrates.
- All three import `src/common/config.py` for seed, paths, and colour palette. Never
  hardcode a seed inline.
- Figures go to `figures/<owner>/<name>.png` at ≥150 DPI, copied to `report/figures/` only
  once finalised.
- No absolute paths. Resolve relative to repo root via `pathlib`.

---

## 6. Git workflow

- **Branches:** `person1-work`, `person2-work`, `person3-work`. Never commit directly to
  `main`. PR into `main` when a graph is finished.
- **Commit format:** `[P<n>][G<graph#>] <what changed>`, e.g.
  `[P2][G5] Add Marchenko-Pastur eigenvalue filtering`. Makes individual contribution
  auditable from `git log` alone.
- **Frequency:** one commit per meaningful step, not one per graph. A single squashed
  commit at the end looks like one person did everything.
- **CONTRIBUTIONS.md:** append `YYYY-MM-DD | P<n> | <task> | <commit-hash>` per completed
  task. This becomes the report's Individual Contribution section — don't reconstruct it
  from memory later.
- **At handoffs:** push, open the PR, and tell the team. Don't let a handoff sit unmerged
  while downstream work is blocked.
- **Before submission:** repo public, README has working clone + run instructions,
  `python run_all.py` succeeds from a fresh clone. Verify it yourself.

---

## 7. Conventions

- Python 3: pandas, numpy, networkx, scikit-learn, matplotlib, scipy — pinned in
  `requirements.txt`.
- **Seed everything stochastic** — Louvain, rewiring, permutation, layouts, DW — via the
  shared `SEED`. Unseeded randomness breaks reproducibility and makes your numbers differ
  from your teammates'.
- One qualitative palette for T/E/S/V, identical in all 12 figures. Agree the hex codes
  before anyone plots.
- Every figure: title, axis labels with units, legend if multiple series, legible at the
  size it appears in the PDF. Test by compiling the report, not by viewing the PNG
  full-screen.

---

## 8. Do not

- Do not claim polarisation — PC1 explains 8.3%; the data does not support a two-camp story.
- Do not report modularity or any structure metric without its null beside it.
- Do not use rewiring or ER as primary evidence.
- Do not use mutual kNN for the respondent network.
- Do not build an imputation model beyond column-median.
- Do not report a single threshold without the sweep.
- Do not present the DW low-ε plateau without noting the discretisation artefact.
- Do not silently change Section 2. Ask first.
- Do not commit on a teammate's behalf or squash multiple people's work into one commit.
