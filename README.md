# Opinion Network Formation — DPCN Assignment 1

**Team: Edge Runners**

Constructing and analysing an opinion network from a 96-respondent, 60-item Likert survey
covering Technology, Education, Society/Ethics and Environment.

**Nodes** are the 60 survey questions. **Edges** are signed Pearson correlations between
questions, computed on row-centred responses and thresholded at `|r| > 0.22`. Edges are
weighted by default and binarised only where a specific metric requires it.

## Setup

Requires **Python 3.13**. Python 3.14 is not supported — several pinned dependencies have
no 3.14 wheels yet, and `run_all.py` fails at import.

```bash
git clone <repo-url>
cd dpcn_a1_opinion_network
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import sys; print(sys.version)"   # confirm 3.13.x
```

## Running

```bash
python run_all.py            # everything, in dependency order
python run_all.py --list     # show the stage table
python run_all.py --only Shreyash  # one owner's stages
python run_all.py --keep-going   # don't stop at the first failure
```

Stage 1 fits per-item ordinal regression models and dominates the runtime. Everything in
`src/analysis/` completes in under 10 seconds. A stage whose inputs are missing is skipped
with a message naming the person who owes the artefact.

## Layout

Modules are grouped by what they do; ownership is recorded in `CONTRIBUTIONS.md`
and in `git log`.

```
data/                    Survey_Results_UC.csv  (read-only, never edited in place)
src/
  common/                shared foundations
    config.py            SEED, paths, palette, threshold grid — import, never redefine
    matrix.py            loads the imputed matrix and row-centres it
    nulls.py             permutation / rewiring / Erdos-Renyi generators
  preprocessing/         raw responses -> clean, complete, centred matrix
    settings.py          encoding map and PipelineConfig
    loader.py            parsing, encoding, missingness identification
    imputation.py        per-item proportional-odds ordinal regression
    similarity.py        row-centring and rank correlation
    pipeline.py          end-to-end sanitisation run
    metrics.py           correlation network and node statistics
    reporting.py         run summary written to outputs/
  analysis/              is the structure real?
    noise_floor.py       observed vs permuted correlations      (Figure 4)
    spectrum.py          Marchenko-Pastur denoising             (Figure 5)
    threshold.py         threshold and percolation sweep        (Figure 7)
    null_models.py       permutation vs rewiring vs ER
    sensitivity.py       methods tried and rejected
  communities/           what structure is there?
    detection.py         Louvain / Leiden / Girvan-Newman + nulls (Figure 10)
    balance.py           structural balance sweep               (Figure 11)
    heatmap.py           reordered correlation heatmap          (Figure 6)
  viz/                   figures owned by the preprocessing side
    diagnostics.py       missingness, item variance, centring   (Figures 1-3)
    block_connectivity.py  topic-block connectivity             (Figure 9)
    dynamics.py          Deffuant-Weisbuch opinion dynamics     (Figure 12)
    networks.py          network drawing helpers
artifacts/               handoff files between team members
figures/                 generated PNG + PDF, named figNN_*
outputs/                 sanitisation and per-analysis outputs
report/                  report.tex, references.bib, report.pdf
run_all.py               reproduces every figure and number, in dependency order
```

## Reproducibility

Every stochastic call — permutation, rewiring, Erdős–Rényi, Louvain, layouts — is seeded
from the single `SEED` in `src/common/config.py`. All paths resolve relative to the repo
root via `pathlib`; there are no absolute paths. Reported numbers regenerate exactly.

All analysis runs on one matrix: **91 respondents** (every non-empty record), with the 242
missing cells recovered by question-wise regularised proportional-odds ordinal regression
(`src/preprocessing/imputation.py`). `src/common/matrix.py` consumes that matrix and row-centres it; it
does not re-impute anything. Shreyash's stages use Pearson correlation throughout, because
the Marchenko–Pastur eigenvalue band has no closed form for rank correlations.

`build_frozen_variant()` rebuilds a superseded alternative — drop the 10 respondents with
more than 12 missing items, fill the rest with a column mean — purely so the report can
quantify what the respondent filter changes. It yields three significant eigenvalue modes
rather than four. See `artifacts/sensitivity.json` and *Methods considered and rejected*
in the report.

## Team

See `CONTRIBUTIONS.md` for the per-task log with commit hashes.
