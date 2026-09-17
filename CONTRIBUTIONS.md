# Contributions

Per-task log, kept as work completes rather than reconstructed afterwards. This is the
source for the report's Individual Contribution section.

Format: `YYYY-MM-DD | P<n> | <task> | <commit-hash>`

> All Person 2 tasks below are committed on `main`; hashes are the real commits.
> Person 3's section is filled in as that work lands.

## Person 1 — data preparation, imputation, graphs 1–3

```
2026-09-16 | P1 | Row-centring added to item-network pipeline; split into modules | 74f4673
2026-09-17 | P1 | Graph 1: missingness map + response distribution                | c8b9ac7
2026-09-17 | P1 | Graph 2: item variance ranking                                  | 661133d
2026-09-17 | P1 | Graph 3: row-centring effect on item-item correlations          | eed3418
2026-09-17 | P1 | Method/artifact report for graphs 1-3                           | 6a5ff8e
```

## Person 2 — statistical spine, null framework, graphs 4/5/7

```
2026-09-17 | P2 | Adopt item-network spec as shared CLAUDE.md                     | 1bb881a
2026-09-17 | P2 | Shared config: seed, paths, palette, threshold grid             | 2d676c2
2026-09-17 | P2 | Spec-conformant matrix builder (>12 filter, mean fill, centring)| 2d676c2
2026-09-17 | P2 | Null framework: permutation / rewiring / ER + synthetic test    | dee9d34
2026-09-17 | P2 | Graph 4: noise floor, chance-edge table, figure                 | 15dfec2
2026-09-17 | P2 | Graph 5: Marchenko-Pastur denoising, spectrum figure            | 3c85918
2026-09-17 | P2 | Graph 7: threshold sweep, selection at |r| > 0.20               | ef572cd
2026-09-17 | P2 | Null comparison: permutation vs rewiring vs ER on modularity    | 9058372
2026-09-17 | P2 | Sensitivity annex: methods tried and rejected                   | 875ac84
2026-09-17 | P2 | run_all.py orchestration, pinned requirements, README           | c43b217
2026-09-17 | P2 | Report section: Pipeline Followed                               | 4b312df
```

### Person 2 deliverables

| Deliverable | Path |
|---|---|
| Shared config | `src/common/config.py` |
| Analysis matrix | `src/common/matrix.py` |
| Null framework | `src/common/nulls.py` |
| Graph 4 | `src/person2/graph_4.py` → `figures/person2/graph_4_noise_floor.{png,pdf}` |
| Graph 5 | `src/person2/graph_5.py` → `figures/person2/graph_5_mp_spectrum.{png,pdf}` |
| Graph 7 | `src/person2/graph_7.py` → `figures/person2/graph_7_threshold_sweep.{png,pdf}` |
| Null comparison | `src/person2/null_comparison.py` |
| Rejected methods | `src/person2/sensitivity.py` |
| Orchestration | `run_all.py` |
| Report section | `report/sections/pipeline_followed.tex` |

### Handoff artefacts written for teammates

| File | Consumed by | Contents |
|---|---|---|
| `artifacts/corr_raw.npy` | P3 (graph 6) | 60×60 raw correlation matrix |
| `artifacts/corr_denoised.npy` | P1 (graph 8), P3 (graphs 6, 10, 11) | 60×60 MP-denoised matrix |
| `artifacts/threshold.json` | P1 (graph 8), P3 (graphs 10, 11) | chosen threshold, rule, resulting stats |
| `artifacts/chance_edges_by_threshold.csv` | graph 7, P3 (graph 11) | expected chance edges per threshold |
| `artifacts/null_replicates.npz` | P3 (graph 10) | 200 × 1770 permuted correlations |
| `artifacts/mp_dimensions.json` | P3 (results section) | 3 dimensions, loadings, names |
| `artifacts/X_centred.npy`, `X_encoded.npy`, `items.json` | P1, P3 | 86×60 centred / pre-centring matrices, item metadata |
| `artifacts/null_comparison.json`, `null_modularity_replicates.npz` | P3 (graph 10) | modularity under all three nulls |
| `artifacts/sensitivity.json` | report | rejected-methods numbers |

## Person 3 — communities, structural balance, reordered heatmap

```
(not started)
```
