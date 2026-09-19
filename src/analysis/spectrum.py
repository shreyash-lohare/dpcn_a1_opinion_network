"""Graph 5 -- eigenvalue spectrum against the Marchenko-Pastur band.

PCA alone is unsatisfying here: PC1 explains only 8.3% of variance, and from
that number alone you cannot tell whether the data has weak structure or none
at all. Marchenko-Pastur answers it exactly. For a correlation matrix built
from T observations of N pure-noise variables, the eigenvalues fall inside an
analytically known band [lambda_-, lambda_+]. Anything above lambda_+ carries
more structure than T observations could have produced by chance.

At T = 91, N = 60 the band ends at lambda_+ = 3.283 and exactly four
eigenvalues clear it. There are precisely four real opinion dimensions in this
survey, and the denoised correlation matrix is rebuilt from those four alone.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.config import (
    ARTIFACTS,
    BLOCK_COLORS,
    BLOCK_NAMES,
    DPI,
    FIGDIR,
    MARK_COLOR,
    NULL_COLOR,
    REAL_COLOR,
    ensure_dirs,
)
from src.common.matrix import build_matrices, correlation

N_LOADINGS = 8

# Named by the analyst from the loadings printed below, not derived by the code.
# PC1 and PC3 split cleanly along topic blocks; PC2 and PC4 cut across them and
# are the weaker readings, which the report states rather than glosses over.
DIMENSION_NAMES = {
    1: "Environmental and social commitment vs. technology governance and assessment",
    2: "Developmental optimism vs. regulatory caution",
    3: "AI-enabled education and industry vs. environmental stewardship",
    4: "Systemic technological action vs. individual and community obligation",
}


def mp_bounds(n_obs: int, n_vars: int, sigma_sq: float = 1.0):
    """Marchenko-Pastur band edges for a T x N noise correlation matrix."""
    q = n_obs / n_vars
    spread = 1.0 / np.sqrt(q)
    return q, sigma_sq * (1 - spread) ** 2, sigma_sq * (1 + spread) ** 2


def mp_density(lam: np.ndarray, q: float, lam_minus: float, lam_plus: float, sigma_sq: float = 1.0):
    """Analytic MP density, zero outside the band."""
    out = np.zeros_like(lam, dtype=float)
    inside = (lam > lam_minus) & (lam < lam_plus)
    x = lam[inside]
    out[inside] = q / (2 * np.pi * sigma_sq * x) * np.sqrt((lam_plus - x) * (x - lam_minus))
    return out


def spectrum(corr: np.ndarray):
    """Eigen-decomposition, descending. eigvalsh/eigh because corr is symmetric.

    ``np.linalg.eig`` would return complex values with tiny imaginary parts,
    which silently break every ``>`` comparison downstream.
    """
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def denoise(vals: np.ndarray, vecs: np.ndarray, n_sig: int) -> np.ndarray:
    """Rebuild the correlation matrix from the significant components only.

    C_denoised = sum_i lambda_i v_i v_i^T over significant i, then the diagonal
    is restored to unity (Laloux et al. cleaning): a rank-3 reconstruction does
    not preserve unit self-correlation on its own.
    """
    c = (vecs[:, :n_sig] * vals[:n_sig]) @ vecs[:, :n_sig].T
    np.fill_diagonal(c, 1.0)
    return c


def component_loadings(vecs, vals, codes, question_text, n_sig, n_top=N_LOADINGS):
    """Top +/- loadings per significant component, for naming the dimensions."""
    dims = []
    for i in range(n_sig):
        v = vecs[:, i]
        # Sign of an eigenvector is arbitrary; orient so the heaviest loading is positive.
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        order = np.argsort(v)
        neg = [
            {"code": codes[j], "loading": float(v[j]), "text": question_text.get(codes[j], "")}
            for j in order[:n_top]
        ]
        pos = [
            {"code": codes[j], "loading": float(v[j]), "text": question_text.get(codes[j], "")}
            for j in order[::-1][:n_top]
        ]
        dims.append(
            {
                "component": i + 1,
                "eigenvalue": float(vals[i]),
                "variance_pct": float(vals[i] / len(codes) * 100),
                "proposed_name": DIMENSION_NAMES.get(i + 1),
                "top_positive": pos,
                "top_negative": neg,
            }
        )
    return dims


def report_numbers(q, lam_minus, lam_plus, vals, n_sig, n_items) -> None:
    print("=" * 78)
    print("GRAPH 5 ACCEPTANCE CRITERIA")
    print("=" * 78)
    print(f"  T respondents                {int(q * n_items):>8}    (91, ordinal-imputed)")
    print(f"  Q = T/N                      {q:>8.4f}    (ref 1.5167)")
    print(f"  lambda_plus                  {lam_plus:>8.4f}    (ref 3.283)")
    print(f"  lambda_minus                 {lam_minus:>8.4f}    (ref 0.035)")
    print(f"  eigenvalues above lambda_+   {n_sig:>8}    (ref exactly 4)")
    print(f"  their values                 {np.round(vals[:n_sig], 3).tolist()}")
    print(f"                               (ref 4.38, 4.06, 3.55, 3.35)")
    print(f"  variance carried             {vals[:n_sig].sum() / n_items * 100:>7.1f}%    (ref 25.6%)")
    print(f"  next eigenvalue below band   {vals[n_sig]:>8.3f}")
    print("=" * 78)


def print_loadings(dims) -> None:
    for d in dims:
        print()
        print("-" * 78)
        print(
            f"COMPONENT {d['component']}   eigenvalue {d['eigenvalue']:.3f}   "
            f"{d['variance_pct']:.1f}% of variance"
        )
        if d["proposed_name"]:
            print(f"  name: {d['proposed_name']}")
        print("-" * 78)
        for side, key in (("POSITIVE pole", "top_positive"), ("NEGATIVE pole", "top_negative")):
            print(f"  {side}:")
            for item in d[key]:
                text = item["text"]
                text = text if len(text) <= 84 else text[:81] + "..."
                print(f"    {item['loading']:+.3f}  {item['code']}  {text}")


def plot_spectrum(vals, q, lam_minus, lam_plus, n_sig, n_items, path_stem) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.3))

    bins = np.linspace(0, max(vals.max() * 1.05, lam_plus * 1.2), 46)
    ax.hist(
        vals,
        bins=bins,
        density=True,
        color=REAL_COLOR,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.5,
        label=f"Observed eigenvalues (N = {n_items})",
    )
    grid = np.linspace(lam_minus * 0.98, lam_plus * 1.02, 600)
    ax.plot(
        grid,
        mp_density(grid, q, lam_minus, lam_plus),
        color=NULL_COLOR,
        linewidth=2.2,
        label="Marchenko-Pastur density (pure noise)",
    )
    ax.axvline(lam_plus, color=MARK_COLOR, linestyle="--", linewidth=1.6)
    ax.annotate(
        f"$\\lambda_+ = {lam_plus:.3f}$",
        xy=(lam_plus, ax.get_ylim()[1] * 0.80),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=9.5,
        color=MARK_COLOR,
        fontweight="bold",
    )
    for i in range(n_sig):
        ax.axvline(vals[i], color=MARK_COLOR, linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Eigenvalue $\\lambda$ of the item-item correlation matrix")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Exactly {n_sig} eigenvalues exceed the noise band "
        f"({vals[:n_sig].sum() / n_items * 100:.1f}% of total variance)",
        fontsize=11.5,
        loc="left",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)

    # --- inset: the top 6 eigenvalues, so the 3-vs-rest gap is visible -----
    inset = ax.inset_axes((0.46, 0.30, 0.50, 0.40))
    top = vals[:6]
    colors = [NULL_COLOR if i < n_sig else "#9AA5B1" for i in range(6)]
    inset.bar(np.arange(1, 7), top, color=colors, edgecolor="white", linewidth=0.6)
    inset.axhline(lam_plus, color=MARK_COLOR, linestyle="--", linewidth=1.3)
    inset.text(
        6.35, lam_plus, "$\\lambda_+$", va="center", ha="left", fontsize=9.5, color=MARK_COLOR
    )
    for i, v in enumerate(top):
        inset.text(i + 1, v + 0.07, f"{v:.2f}", ha="center", fontsize=8.5, color=MARK_COLOR)
    inset.set_xticks(np.arange(1, 7))
    inset.set_xticklabels([f"PC{i}" for i in range(1, 7)], fontsize=8.5)
    inset.tick_params(axis="y", labelsize=8.5)
    inset.set_ylim(0, top.max() * 1.20)
    inset.set_xlim(0.4, 6.9)
    inset.set_title("Top 6 eigenvalues", fontsize=9.5)
    inset.grid(alpha=0.18, axis="y", linewidth=0.5)
    inset.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    ensure_dirs()
    am = build_matrices()
    corr = correlation(am.centred)
    n_obs, n_items = am.centred.shape

    q, lam_minus, lam_plus = mp_bounds(n_obs, n_items)
    vals, vecs = spectrum(corr)
    n_sig = int((vals > lam_plus).sum())

    report_numbers(q, lam_minus, lam_plus, vals, n_sig, n_items)
    dims = component_loadings(vecs, vals, am.codes, am.question_text, n_sig)
    print_loadings(dims)

    corr_denoised = denoise(vals, vecs, n_sig)
    np.save(ARTIFACTS / "corr_raw.npy", corr)
    np.save(ARTIFACTS / "corr_denoised.npy", corr_denoised)
    payload = {
        "n_observations": int(n_obs),
        "n_items": int(n_items),
        "Q": float(q),
        "lambda_plus": float(lam_plus),
        "lambda_minus": float(lam_minus),
        "n_significant": int(n_sig),
        "significant_eigenvalues": [float(v) for v in vals[:n_sig]],
        "variance_pct": float(vals[:n_sig].sum() / n_items * 100),
        "eigenvalues_all": [float(v) for v in vals],
        "dimensions": dims,
    }
    (ARTIFACTS / "mp_dimensions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_spectrum(vals, q, lam_minus, lam_plus, n_sig, n_items, FIGDIR / "fig05_eigenvalue_spectrum")
    print(f"\n  wrote {FIGDIR / 'fig05_eigenvalue_spectrum'}.png / .pdf")
    print("  wrote artifacts/corr_raw.npy, corr_denoised.npy, mp_dimensions.json")
    print("\n  NOTE: the three dimension names are an analyst judgement read off the")
    print("        loadings above, not a code-derived label.")
    return payload


if __name__ == "__main__":
    main()
