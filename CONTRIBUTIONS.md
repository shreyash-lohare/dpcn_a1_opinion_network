# Contributions — Team Edge Runners

Per-task log kept as work completed, and the source for the report's Individual
Contribution section. Authorship is verifiable from `git log`, where every commit
carries its author.

| Member | GitHub | Owns |
|---|---|---|
| Arijeet Paul | `Arijeet1729` | `src/preprocessing/`, `src/viz/` |
| Shreyash Lohare | `shreyash-lohare` | `src/common/`, `src/analysis/`, `run_all.py` |
| Dev Patel | `devpatel301` | `src/communities/` |

---

## Arijeet Paul — data preparation, imputation, dynamics

| Task | Commit |
|---|---|
| Row-centring in the item-network pipeline; split into modules | `74f4673` |
| Figure 1: missingness map and response distribution | `c8b9ac7` |
| Figure 2: item variance ranking | `661133d` |
| Figure 3: row-centring effect on item–item correlations | `eed3418` |
| Method and artefact notes for Figures 1–3 | `6a5ff8e` |
| Figure 12: Deffuant–Weisbuch dynamics and respondent kNN network | `8e7264b` |
| Report skeleton: data handling, Figures 1–3 and 12, introduction | `90d52f4` |
| Figure 9: topic-block connectivity with block-label permutation null | `eaf158d` |
| Figure 9 report section; title block and column-overflow fixes | `04824ea` |

**Deliverables.** Parsing, encoding and missingness analysis; per-item
regularised proportional-odds ordinal regression for the 242 missing cells, with
artificial-masking validation under both scattered and structural mechanisms;
row-centring; Graphs 1, 2, 3 and 8.

## Shreyash Lohare — noise floor, spectral denoising, null framework

| Task | Commit |
|---|---|
| Shared config: seed, paths, palette, threshold grid | `2d676c2` |
| Analysis matrix builder (filter, imputation handoff, centring) | `2d676c2` |
| Null framework: permutation / rewiring / ER, with synthetic validation | `dee9d34` |
| Figure 4: noise floor and chance-edge table | `15dfec2` |
| Figure 5: Marchenko–Pastur denoising and eigenvalue spectrum | `3c85918` |
| Figure 7: threshold sweep and selection rule | `ef572cd` |
| Null comparison: permutation vs rewiring vs ER on modularity | `9058372` |
| Sensitivity analysis: methods tried and rejected | `875ac84` |
| `run_all.py` orchestration, pinned requirements, README | `c43b217` |
| Report sections for Figures 4, 5 and 7 | `4b312df` |
| Switch analysis onto the 91-respondent ordinal-imputed matrix | `005b994` |

**Deliverables.** The permutation-null framework (`src/common/nulls.py`) that
both other analyses are tested against; the noise floor (observed correlation SD
0.139 against 0.106 under the null); Marchenko–Pastur filtering, which leaves
exactly four significant modes carrying 25.6% of variance, and the denoised
matrix all downstream networks are built from; the threshold sweep and its
stated selection rule at |r| > 0.22; the null-model comparison showing the item
network's community structure clears the rewiring and Erdős–Rényi nulls but
fails the column-permutation null; Graphs 4, 5 and 7.

## Dev Patel — community detection, structural balance

| Task | Commit |
|---|---|
| Package scaffold, artefact verification, figure config | `8ed6ae7` |
| Figure 10: communities vs three null models, with NMI analysis | `175423f` |
| Figure 11: structural balance sweep with sign-permutation null | `14ba86a` |
| Figure 6: reordered correlation heatmap with community strips | `d1402d7` |
| Wire community stages into `run_all.py` | `2ebf764` |
| Report sections for Figures 6, 10 and 11 | `5f98523` |
| Migrate community detection from Louvain to Leiden | `203841e` |
| Girvan–Newman and the three-way algorithm comparison | `7569fa4` |
| Team name and documentation updates | `73702f4` |

**Deliverables.** Community detection on the item network and the
Louvain / Leiden / Girvan–Newman comparison (all three converge on k = 15,
Leiden best at Q = 0.491); null-model evaluation of the item partition, which
independently reproduces the permutation-null failure; the reordered
MP-denoised heatmap showing communities cut across the survey's thematic blocks
(NMI 0.205); Graphs 6 and 9.

---

## Shared handoff artefacts

Written to `artifacts/` by `python run_all.py`, consumed across the team.

| File | Produced by | Consumed by |
|---|---|---|
| `X_centred.npy`, `X_encoded.npy`, `items.json` | Shreyash | all |
| `corr_raw.npy`, `corr_denoised.npy`, `mp_dimensions.json` | Shreyash | Dev |
| `threshold.json`, `threshold_sweep.csv` | Shreyash | Dev |
| `chance_edges_by_threshold.csv`, `null_replicates.npz` | Shreyash | Dev |
| `null_comparison.json`, `null_modularity_replicates.npz` | Shreyash | Dev |
| `sensitivity.json` | Shreyash | report |
| `graph10_communities.json` | Dev | report |
| `outputs/sanitised_data/` | Arijeet | all |
