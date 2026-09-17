# Opinion Network Formation — DPCN Assignment 1

Constructing and analysing an opinion network from a 96-respondent, 60-item Likert survey
covering Technology, Education, Society/Ethics and Environment.

**Nodes** are the 60 survey questions. **Edges** are signed Pearson correlations between
questions, computed on row-centred responses and thresholded at `|r| > 0.20`. Edges are
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
python run_all.py --only P2  # one owner's stages
python run_all.py --keep-going   # don't stop at the first failure
```

Stage 1 fits per-item ordinal regression models and dominates the runtime. Everything in
`src/person2/` completes in under 10 seconds. A stage whose inputs are missing is skipped
with a message naming the person who owes the artefact.

## Layout

```
data/                  Survey_Results_UC.csv   (read-only, never edited in place)
src/
  config.py            P1: encoding map, PipelineConfig
  loader.py            P1: parsing, encoding, missingness
  imputation.py        P1: ordinal-regression imputer
  pipeline.py          P1: end-to-end sanitisation run
  diagnostics.py       P1: graphs 1-3
  similarity.py        P1: row-centring, Spearman matrix
  metrics.py           P1: correlation network + node statistics
  common/
    config.py          P2: SEED, paths, palette, threshold grid — import, never redefine
    matrix.py          P2: frozen filter + mean fill + row-centring
    nulls.py           P2: permutation / rewiring / ER generators
  person2/
    graph_4.py         noise floor
    graph_5.py         Marchenko-Pastur denoising
    graph_7.py         threshold sweep
    null_comparison.py permutation vs rewiring vs ER
    sensitivity.py     methods tried and rejected
artifacts/             handoff files between team members
figures/               generated PNG + PDF output
report/                LaTeX source; sections/ holds one file per owner
outputs/sanitised_data/  P1's sanitisation outputs
```

## Reproducibility

Every stochastic call — permutation, rewiring, Erdős–Rényi, Louvain, layouts — is seeded
from the single `SEED` in `src/common/config.py`. All paths resolve relative to the repo
root via `pathlib`; there are no absolute paths. Reported numbers regenerate exactly.

Two pipelines coexist in this repo, deliberately:

- **Frozen pipeline** (`src/common/matrix.py`) — 86 respondents after dropping the 10 with
  more than 12 missing items, column-mean fill, Pearson. This produces every number in the
  report.
- **Person 1's variant** (`src/pipeline.py`) — 91 respondents, ordinal-regression
  imputation, Spearman. Retained and reported as a documented sensitivity check; it yields
  four significant dimensions rather than three. See `artifacts/sensitivity.json` and the
  *Methods tried and rejected* table in the report.

## Team

See `CONTRIBUTIONS.md` for the per-task log with commit hashes.
