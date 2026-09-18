# Person 3 — Sprint Plan
## Graphs 10, 11, 6 · Communities, Structural Balance, Heatmap · Report Assembly

> **Branch:** `person3-work` — never commit to `main` directly.  
> **Commit format:** `[P3][G<n>] <what changed>`  
> **Log all completions** to `CONTRIBUTIONS.md` as you go — don't reconstruct at the end.

---

## Context (what P1 and P2 already delivered)

| Artifact | Path | Used by |
|---|---|---|
| Imputed 91×60 matrix (encoded, pre-centring) | `artifacts/X_encoded.npy` | G10, G11 null seeds |
| Centred 91×60 matrix | `artifacts/X_centred.npy` | G10, G11, G6 |
| Item metadata (codes, block labels, question text) | `artifacts/items.json` | G10, G11, G6 |
| Raw 60×60 correlation matrix | `artifacts/corr_raw.npy` | G6 (comparison only) |
| **MP-denoised 60×60 correlation matrix** | `artifacts/corr_denoised.npy` | G10, G11, G6 ← primary |
| Chosen threshold `0.22` + stats | `artifacts/threshold.json` | G10, G11 |
| Permutation null replicates (200 × 1770 correlations) | `artifacts/null_replicates.npz` | G10 (pull cached) |
| Modularity under all 3 nulls (cached) | `artifacts/null_modularity_replicates.npz` + `null_comparison.json` | G10 |
| Percolation sweep table | `artifacts/threshold_sweep.csv` | G11 |
| Chance-edge table | `artifacts/chance_edges_by_threshold.csv` | G11 (context) |

> [!IMPORTANT]
> All artifacts above are **already written to disk** by P2 — do not re-run those stages. Simply load them.

---

## Sprint 0 — Setup & Orientation *(do this first)*

**Goal:** get the branch, verify artifacts exist, confirm environment works.

## Sprint Plan (Person 3)

| Sprint | Task | Status | Notes / Output |
| :--- | :--- | :--- | :--- |
| **S0** | **Setup & Orientation** | [x] | Created `src/person3/`, updated `FIGDIR_P3` config, validated artifacts. |
| **S1** | **Graph 10: Communities vs Nulls** | [x] | Implemented Leiden (previously Louvain) on item network, ran permutation/rewiring/ER nulls, computed NMI vs topic blocks. Saved to `graph_10.py` and `graph10_communities.json`. |
| **S2** | **Graph 11: Structural Balance** | [x] | Computed balanced triads fraction against sign-permutation null. Saved to `graph_11.py` and `graph11_balance_stats.json`. |
| **S3** | **Graph 6: Reordered Heatmap** | [x] | Plotted MP-denoised correlations ordered by Graph 10 communities. Diverging map + block border strips. Saved to `graph_6.py`. |
| **S4** | **Exploration Notebook** | [x] | Created `notebooks/person3_exploration.ipynb` referencing P3 modules and showing figures/stats inline. |
| **S5** | **Pipeline Integration** | [x] | Replaced `_p3_pending()` in `run_all.py` with the 3 distinct Graph 6, 10, 11 stages. All runs pass. |
| **S6** | **Report Writing** | [x] | Replaced P3 `\todoblock`s in `report/report.tex` with analysis for G6, G10, G11, Discussion, Conclusion, and Individual Contribution. |
| **S7** | **Final Polish & PR** | [ ] | (User instructed to skip this sprint) |

| # | Task | Notes |
|---|---|---|
| 0.1 | Checkout `person3-work` branch (or create it) | `git checkout -b person3-work` from latest main |
| 0.2 | Verify all P2 artifacts exist in `artifacts/` | Run quick Python check: load each `.npy` / `.json` / `.npz`, print shapes |
| 0.3 | Verify `python run_all.py --list` shows all stages | Confirm P3 stage is still listed as pending |
| 0.4 | Create notebook at `notebooks/person3_exploration.ipynb` | Scratch space — exploration only, nothing load-bearing |
| 0.5 | Confirm `src/common/config.py` has `FIGDIR_P3 = FIGDIR / "person3"` | If missing, add it (it only defines P2 currently) |
| 0.6 | Create the `src/person3/` package | `src/person3/__init__.py` (empty, 1 line) |

**Commit:** `[P3][setup] Branch scaffold, artifact verification, package skeleton`

---

## Sprint 1 — Graph 10: Communities vs Three Null Models *(effort 5, heaviest)*

**Goal:** Two-panel figure. Left: item network coloured by Leiden (previously Louvain) community. Right: modularity of the real network versus three null distributions (ER, rewiring, permutation) shown as violins/histograms.

> [!NOTE]
> P2's `src/person2/null_comparison.py` already ran the full null experiment on the **respondent network** and cached the results. For Graph 10 you run Leiden (previously Louvain) on the **item network** (60 nodes, threshold 0.22) and plot its modularity against the same three nulls — some recomputation is needed for the item network specifically.

### Sub-tasks

| # | Task | Details |
|---|---|---|
| 1.1 | **Load artifacts** | `corr_denoised.npy`, `threshold.json`, `null_modularity_replicates.npz` (P2 cached), `items.json` |
| 1.2 | **Build signed item graph** | Use the same `build_signed_graph()` logic from `src/person2/graph_7.py`. Threshold = 0.22. Edges get `weight=abs(r)`, `r=signed_r`, `sign=±1` attributes |
| 1.3 | **Run Leiden (previously Louvain) on item network** | `nx.community.louvain_communities(G, weight="weight", seed=SEED)`. Record `Q_observed` and community assignments |
| 1.4 | **Compute modularity under all 3 nulls on the item network** | Pull rewiring + ER from P2's cached `.npz` if they cover the item network; otherwise recompute 200 reps each using `src/common/nulls.py`. Check if `null_modularity_replicates.npz` covers item-network or only respondent-network (it is the respondent network — so recompute for item network) |
| 1.5 | **Pull permutation null replicates** | Load `artifacts/null_replicates.npz` (200 × 1770 permuted correlations). For each replicate: threshold at 0.22 → build graph → run Leiden (previously Louvain) → get Q. This gives the permutation null distribution for the item network |
| 1.6 | **Compute z-scores and p-values** | Use `src/common/nulls.zscore(Q_observed, null_samples)` for each of the 3 nulls |
| 1.7 | **Second analysis — community vs topic label comparison** | Map each of 60 items to its detected community and its T/E/S/V block. Build a 4×n_communities contingency table. Check specifically whether the S block splits. Compute NMI (normalized mutual information) between community labels and block labels |
| 1.8 | **Left panel** | Spring or Kamada-Kawai layout (seeded). Nodes coloured by Leiden (previously Louvain) community. Edge colour by sign (+/−). Node size uniform or by degree. Topic-block symbol or border encoding. Save layout coordinates so Graph 6 can reference them if needed |
| 1.9 | **Right panel** | Observed Q as vertical line. Three distributions as violin plots or overlapping histograms. Label each null with its mean ± SD and z. Colour: real=`REAL_COLOR`, null=`NULL_COLOR` (from config) |
| 1.10 | **Tabular summary** | Print and save to `artifacts/graph10_null_table.json`: observed Q, each null's mean/SD/z/p |
| 1.11 | **Write `src/person3/graph_10.py`** | Side-effect-free module. Main logic in functions, a `run()` entry point. Saves figure to `figures/person3/graph_10_communities_nulls.{png,pdf}` at DPI 200 |
| 1.12 | **Regression check** | Verify community count is plausible given `threshold.json` reports 15 communities at threshold 0.22. If wildly different, debug before moving on |

**Expected numbers (from CLAUDE.md §3, respondent network context):**
- Permutation null should sit *close* to Q observed (item network behaviour may differ)
- Rewiring and ER nulls should be clearly lower

**Commit cadence:**
- `[P3][G10] Load artifacts, build signed item graph at threshold 0.22`
- `[P3][G10] Leiden (previously Louvain) communities + three-null distributions computed`
- `[P3][G10] Community vs topic-block NMI analysis`
- `[P3][G10] Two-panel figure: network layout + null comparison`
- `[P3][G10] graph_10.py module, figure saved`

---

## Sprint 2 — Graph 11: Structural Balance vs Threshold *(effort 5)*

**Goal:** Line plot. X-axis = |r| threshold. Y-axis = fraction of closed triads that are balanced (sign product = +1). Observed line + shaded null band (~50%). Annotate triad count at each threshold.

> [!NOTE]
> Structural balance theory (Heider 1946): a triad of three mutually connected nodes is **balanced** when the product of its three edge signs is positive. Pattern `+++` and `+--` are balanced; `++-` and `---` are not.

### Sub-tasks

| # | Task | Details |
|---|---|---|
| 2.1 | **Load artifacts** | `corr_denoised.npy`, `items.json`, `threshold_sweep.csv` (for the threshold grid) |
| 2.2 | **Implement triad enumeration** | For each threshold τ in the grid: build signed graph, enumerate all closed triads using `itertools.combinations(nodes, 3)` checking all three edges exist (feasible at 60 nodes), compute `sign_product = s_ij * s_jk * s_ik` |
| 2.3 | **Compute balanced fraction** | `balanced_fraction = count(sign_product > 0) / total_triads` per threshold |
| 2.4 | **Sign-permutation null** | For each threshold: shuffle edge signs (keep graph structure fixed) 500 times, recompute balanced fraction each time → get null distribution → mean and ±1SD band |
| 2.5 | **Cross-check reference numbers** | CLAUDE.md §3 + Person3 PDF §2.6 give: 83% balanced at |r|>0.15 (664 triads), 96% at 0.20 (137 triads), 100% at 0.25 (19 triads), null ≈ 50% |
| 2.6 | **Plot** | Observed fraction line (solid, `REAL_COLOR`). Null band as shaded region (mean ± SD, `NULL_COLOR`, alpha 0.25). Annotate triad count at key thresholds. Mark the operating threshold (0.22) with a vertical line |
| 2.7 | **Write `src/person3/graph_11.py`** | Side-effect-free module. Saves to `figures/person3/graph_11_structural_balance.{png,pdf}` |
| 2.8 | **Save summary JSON** | `artifacts/graph11_balance_stats.json` — balanced fraction + triad count + z-score at each report threshold |

**Key point to make in report:** The *trend* (83% → 96% → 100% as threshold tightens) is the validation, not any single number. At low threshold two-thirds of edges are noise (Graph 4); at high threshold only the strongest correlations remain and all triads are balanced.

**Commit cadence:**
- `[P3][G11] Triad enumeration + balanced fraction across threshold sweep`
- `[P3][G11] Sign-permutation null (500 reps), null band computed`
- `[P3][G11] Structural balance plot with null band, triad count annotations`
- `[P3][G11] graph_11.py module, figure saved`

---

## Sprint 3 — Graph 6: Reordered Correlation Heatmap *(effort 2, quickest)*

**Goal:** 60×60 heatmap of the MP-denoised correlation matrix. Rows/columns reordered by community assignment from Graph 10. Diverging colour scale centred at zero. Topic-block colour strip on both margins. Separator lines between communities.

> [!NOTE]
> This is the last graph — it depends on Graph 10's community assignments. Do it after Sprint 1 is done.

### Sub-tasks

| # | Task | Details |
|---|---|---|
| 3.1 | **Load community assignment** | Import from Graph 10's output or reload from a saved artifact (save community labels to `artifacts/graph10_communities.json` in Sprint 1) |
| 3.2 | **Reorder indices** | Sort items by (community_id, block_letter, item_number). Apply same permutation to rows and columns of `corr_denoised` |
| 3.3 | **Draw heatmap** | `matplotlib.pyplot.imshow` or `ax.pcolormesh`. Use `DIVERGING_CMAP = "RdBu_r"` from config, `vmin=-1, vmax=1`. Do NOT use sequential colourmap |
| 3.4 | **Block colour strip** | A thin strip of rectangles on both the top and left margins using `BLOCK_COLORS` from config (T=#2E86AB, E=#7B9E4A, S=#8E5EA2, V=#D17A22) |
| 3.5 | **Community separator lines** | Draw thin horizontal and vertical lines at community boundaries |
| 3.6 | **Colourbar** | Attach a properly labelled colourbar (range −1 to +1, tick at 0) |
| 3.7 | **Tick labels** | Item codes on axes at readable font size; may need to reduce to 7pt given 60 items |
| 3.8 | **Write `src/person3/graph_6.py`** | Side-effect-free module. Saves to `figures/person3/graph_6_reordered_heatmap.{png,pdf}` |

**What the figure must show:** Block structure on the diagonal (communities cluster together). Margin strip lets reader immediately see whether communities = topic blocks or cut across them.

**Commit cadence:**
- `[P3][G6] Reordered heatmap with community sort order, block strip, separators`
- `[P3][G6] graph_6.py module, figure saved`

---

## Sprint 4 — Notebook (Jupyter)

**Goal:** A clean, narrative Jupyter notebook at `notebooks/person3_exploration.ipynb` that documents the analysis interactively. This is *in addition* to the `src/person3/` production modules.

| # | Task |
|---|---|
| 4.1 | Section 0: Environment check — load all P2 artifacts, print shapes, verify reference numbers |
| 4.2 | Section 1: Graph 10 — community analysis walkthrough with inline plots |
| 4.3 | Section 2: Graph 11 — balance sweep walkthrough with inline plots |
| 4.4 | Section 3: Graph 6 — heatmap construction walkthrough |
| 4.5 | Section 4: Report numbers — print all numbers that go into the report text (Q observed, z-scores, balance fractions, NMI) |
| 4.6 | Ensure notebook runs top-to-bottom with `Kernel → Restart & Run All` |

> [!NOTE]
> The notebook is for exploration and grader readability. The production code in `src/person3/` is what `run_all.py` actually calls.

---

## Sprint 5 — Wire into `run_all.py` and Integration

**Goal:** Replace the `_p3_pending()` stub with real function calls. Verify the full pipeline runs end-to-end.

| # | Task | Details |
|---|---|---|
| 5.1 | Add `FIGDIR_P3` to `src/common/config.py` if not already there | `FIGDIR_P3 = FIGDIR / "person3"` |
| 5.2 | Split P3 stub into 3 separate stages in `run_all.py` | One stage per graph (G10, G11, G6), in dependency order |
| 5.3 | Add produces/requires annotations | G10 produces `artifacts/graph10_communities.json`; G6 requires it |
| 5.4 | Run `python run_all.py --only P3` | Confirm all 3 stages complete without errors |
| 5.5 | Run `python run_all.py` (full pipeline) | Confirm end-to-end works from a clean state |
| 5.6 | Verify figures land in `figures/person3/` | 6 files: `graph_6_*.{png,pdf}`, `graph_10_*.{png,pdf}`, `graph_11_*.{png,pdf}` |

**Commit:** `[P3][infra] Wire graphs 6, 10, 11 into run_all.py; replace pending stub`

---

## Sprint 6 — Report Writing

**Goal:** Write Person 3's sections in `report/report.tex`.

> [!NOTE]
> The report already has a skeleton (`report/report.tex` exists at 62 KB) and P2 has written `report/sections/pipeline_followed.tex`. Check what's already there before writing.

### Sections to write

| Section | Notes |
|---|---|
| **Team Name** | Fill in team name if blank |
| **GitHub link** | Confirm repo URL is in the report |
| **Analysis and Visualizations** | For each of your 3 figures: what question it answers, what the figure shows, key numbers. Every figure must be referenced in the text. Use `\includegraphics` on `.pdf` versions |
| **Results and Discussion** | Distinguish facts from interpretation. Use the pattern from P3 PDF §5: "At threshold τ, GCC contains X of 60 questions. Communities contain [blocks]. This indicates [interpretation]." |
| **Individual Contribution** | Table built from `CONTRIBUTIONS.md` with commit hashes. The assignment is graded individually — this is the evidence |

### The headline to write toward (from P3 PDF §5.1)
> The class appears to agree on almost everything, but that consensus is superficial. Once each respondent's agreement baseline is removed, exactly three statistically real opinion dimensions survive Marchenko-Pastur filtering (19.3% of variance). The opinion space is internally coherent: 83–100% of closed triads are balanced against ≈50% by chance. Topic domains couple unevenly — Ethics–Environment r = 0.68, Technology–Education r = 0.29. The apparent student factions do not survive a properly constructed null, so we report them as a construction artefact rather than a finding.

### LaTeX duties

| Task |
|---|
| Use `\includegraphics` on `.pdf` versions, not PNGs |
| Check all 12 figures are referenced in the text |
| Compile and verify page count is close to 8 |
| Check no figure is illegible at print size (view in PDF viewer, not full-screen PNG) |
| Verify GitHub link in report resolves and repo is public |

**Commit cadence:**
- `[P3][report] Analysis and Visualizations section — graphs 6, 10, 11`
- `[P3][report] Results and Discussion`
- `[P3][report] Individual Contribution table from CONTRIBUTIONS.md`
- `[P3][report] Final compile check, figure legibility verified`

---

## Sprint 7 — Final Polish & Submission Checklist

| # | Task |
|---|---|
| 7.1 | Copy finalised figures to `report/figures/` |
| 7.2 | Compile `report/report.tex` → PDF, check page count ~8 |
| 7.3 | Verify all 12 figures are cited in the body text |
| 7.4 | Run `python run_all.py` from a clean state — must succeed |
| 7.5 | Verify commit log: every P3 commit follows `[P3][G<n>] …` format |
| 7.6 | Check `CONTRIBUTIONS.md` has every P3 task logged with hash |
| 7.7 | Open a PR from `person3-work` → `main` |
| 7.8 | Confirm GitHub repo is public and URL in report resolves |

---

## Dependency Map

```
P2 artifacts (already done)
    ↓
Sprint 0: Setup & verification
    ↓
Sprint 1: Graph 10 (communities + nulls)      ← effort 5, do first
    ↓                    ↓
Sprint 2: Graph 11       Sprint 3: Graph 6     ← can be done in parallel after S1
    ↓
Sprint 4: Notebook (can be done alongside S1–S3)
    ↓
Sprint 5: Wire into run_all.py
    ↓
Sprint 6: Report writing
    ↓
Sprint 7: Final polish & submission
```

---

## Quick Reference: Key Numbers to Reproduce

| Metric | Expected value | Source |
|---|---|---|
| Chosen threshold | 0.22 | `artifacts/threshold.json` |
| Graph at τ=0.22: edges | 131 | `threshold.json` |
| Graph at τ=0.22: GCC size | 50/60 items (83%) | `threshold.json` |
| Graph at τ=0.22: communities (Leiden (previously Louvain)) | 15 | `threshold.json` |
| Balanced triads at τ=0.15 | 83% (664 triads) | CLAUDE.md §3 |
| Balanced triads at τ=0.20 | 96% (137 triads) | CLAUDE.md §3 |
| Balanced triads at τ=0.25 | 100% (19 triads) | CLAUDE.md §3 |
| Null balanced fraction | ~50% | CLAUDE.md §3 |
| S–V block correlation | 0.68 (strongest block pair) | CLAUDE.md §3 |
| T–E block correlation | 0.29 (weakest block pair) | CLAUDE.md §3 |
| Permutation null modularity (respondent net) | 0.403 ± 0.017, z=1.15 | `null_comparison.json` |
| Rewiring null modularity (respondent net) | 0.337 ± 0.011, z=7.45 | `null_comparison.json` |

---

## Things NOT to do (from CLAUDE.md §8 + P3 PDF §9)

- ❌ Do not claim polarisation — PC1 = 8.3%, no two-camp story
- ❌ Do not report modularity without its null beside it in the same table/figure
- ❌ Do not use rewiring or ER as *primary* evidence
- ❌ Do not hardcode the seed — use `SEED` from `src/common/config.py`
- ❌ Do not use absolute paths — use `pathlib` relative to repo root
- ❌ Do not write plots or files at import time — `run_all.py` orchestrates
- ❌ Do not present the DW low-ε plateau without noting the Likert-discretisation artefact
- ❌ Do not commit on behalf of a teammate
- ❌ Do not skip logging to CONTRIBUTIONS.md — it is the grader's evidence

