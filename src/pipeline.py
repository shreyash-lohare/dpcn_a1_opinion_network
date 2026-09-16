"""End-to-end orchestration: load -> impute -> row-centre -> correlate -> report.

Importable and side-effect free aside from writing to `config.output_dir`.
`run_all.py` at the repo root is the entry point that calls `run_pipeline`.
"""

from __future__ import annotations

import json
from typing import Dict

import pandas as pd

from src.config import PipelineConfig
from src.figures import visualize_network
from src.imputation import (
    impute_missing_responses,
    save_models,
    validate_models_nc_like,
    validate_models_structural,
)
from src.loader import (
    build_label_dataset,
    build_numeric_dataset,
    completion_summary,
    drop_entirely_empty_respondents,
    encode_responses,
    identify_missingness,
    load_data,
    parse_question_columns,
)
from src.metrics import build_correlation_network, compute_network_statistics, network_edges_frame
from src.reporting import save_markdown_report
from src.similarity import compute_spearman_matrix, row_centre, summarize_correlation_matrix


def run_pipeline(config: PipelineConfig = PipelineConfig()) -> Dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw_df_all = load_data(config.raw_csv)
    question_cols, question_codes, question_text = parse_question_columns(raw_df_all)
    structural_blank_all, nc_all = identify_missingness(raw_df_all, question_cols)
    encoded_all = encode_responses(raw_df_all, question_cols, question_codes)
    raw_summary = completion_summary(encoded_all, structural_blank_all, nc_all)

    raw_df, encoded, structural_blank, nc, keep_mask = drop_entirely_empty_respondents(
        raw_df_all, encoded_all, structural_blank_all, nc_all
    )
    analysis_summary = completion_summary(encoded, structural_blank, nc)
    summary = {
        **analysis_summary,
        "raw_respondents": raw_summary["respondents"],
        "excluded_entirely_empty_rows": int((~keep_mask).sum()),
        "raw_structural_blank_cells": raw_summary["structural_blank_cells"],
        "raw_nc_cells": raw_summary["nc_cells"],
        "raw_nc_users": raw_summary["nc_users"],
    }
    missing_mask = structural_blank | nc

    nc_like_perf = validate_models_nc_like(encoded, config)
    structural_perf = validate_models_structural(encoded, config)
    performance = pd.concat([nc_like_perf, structural_perf], ignore_index=True)
    performance.to_csv(config.output_dir / "model_performance.csv", index=False)

    imputed_numeric, imputation_log, models = impute_missing_responses(
        encoded, raw_df, question_cols, question_codes, missing_mask, config
    )

    # Row-centring: mandatory (see src/similarity.py). Computed here, after
    # imputation, so the network is built on a bias-corrected but complete matrix.
    centred_numeric = row_centre(imputed_numeric)
    network_matrix = centred_numeric if config.row_centre else imputed_numeric
    centred_export = centred_numeric.copy()
    centred_export.insert(0, raw_df.columns[0], raw_df.iloc[:, 0].to_numpy())
    centred_export.to_csv(config.output_dir / "row_centred_data.csv", index=False)

    corr, ns, pvals = compute_spearman_matrix(network_matrix, config.min_pairwise_n)
    raw_corr, _, _ = compute_spearman_matrix(imputed_numeric, config.min_pairwise_n)
    centred_corr, _, _ = compute_spearman_matrix(centred_numeric, config.min_pairwise_n)
    raw_comparison = summarize_correlation_matrix(raw_corr, config.edge_threshold)
    centred_comparison = summarize_correlation_matrix(centred_corr, config.edge_threshold)

    graph = build_correlation_network(corr, ns, pvals, question_text)
    graph_summary, node_stats = compute_network_statistics(graph, corr)
    edges = network_edges_frame(graph)

    corr.to_csv(config.output_dir / "correlation_matrix.csv")
    ns.to_csv(config.output_dir / "pairwise_sample_size.csv")
    edges.to_csv(config.output_dir / "network_edges.csv", index=False)
    node_stats.to_csv(config.output_dir / "node_statistics.csv", index=False)

    centring_label = "row-centred" if config.row_centre else "raw (uncentred)"
    full_drawn = visualize_network(
        graph,
        config.output_dir / "full_network",
        f"Full questionnaire network ({centring_label}): all valid Spearman edges",
        edge_threshold=None,
        random_state=config.random_state,
        save_pdf=False,
    )
    thresholded_drawn = visualize_network(
        graph,
        config.output_dir / "thresholded_network",
        f"Thresholded questionnaire network ({centring_label}): |Spearman r| >= {config.edge_threshold:.2f}",
        edge_threshold=config.edge_threshold,
        random_state=config.random_state,
    )

    numeric_dataset = build_numeric_dataset(raw_df, question_cols, question_codes, imputed_numeric)
    label_dataset = build_label_dataset(raw_df, question_cols, question_codes, imputed_numeric)
    numeric_dataset.to_csv(config.output_dir / "dataset_imputed.csv", index=False)
    numeric_dataset.to_csv(config.output_dir / "sanitised_data.csv", index=False)
    label_dataset.to_csv(config.output_dir / "dataset_imputed_labels.csv", index=False)
    imputation_log.to_csv(config.output_dir / "imputation_log.csv", index=False)
    save_models(models, config.output_dir / "ordinal_models.pkl")

    combined_summary = {
        **summary,
        **graph_summary,
        "edge_threshold": config.edge_threshold,
        "row_centre": config.row_centre,
        "thresholded_edge_count": thresholded_drawn.number_of_edges(),
        "cells_imputed": int(len(imputation_log)),
        "validation_accuracy": float(performance["accuracy"].mean()),
        "validation_mae": float(performance["mae"].mean()),
        "raw_mean_abs_corr": raw_comparison["mean_abs_corr"],
        "raw_median_abs_corr": raw_comparison["median_abs_corr"],
        "raw_edge_count_at_threshold": raw_comparison["edge_count_at_threshold"],
        "centred_mean_abs_corr": centred_comparison["mean_abs_corr"],
        "centred_median_abs_corr": centred_comparison["median_abs_corr"],
        "centred_edge_count_at_threshold": centred_comparison["edge_count_at_threshold"],
    }
    (config.output_dir / "summary.json").write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")
    save_markdown_report(config.output_dir / "sanitised_data_report.md", combined_summary, performance, len(imputation_log))

    print("Sanitised data pipeline complete")
    print(f"Raw respondents: {combined_summary['raw_respondents']}")
    print(f"Excluded entirely empty respondents: {combined_summary['excluded_entirely_empty_rows']}")
    print(f"Respondents analysed/exported: {combined_summary['respondents']}")
    print(f"Questions: {combined_summary['questions']}")
    print(f"Structural blanks: {combined_summary['structural_blank_cells']}")
    print(f"NC cells: {combined_summary['nc_cells']}")
    print(f"Usable respondents: {combined_summary['usable_respondents']}")
    print(f"Row-centred: {combined_summary['row_centre']}")
    print(f"Raw mean |r|: {combined_summary['raw_mean_abs_corr']:.3f}  "
          f"(edges @ {config.edge_threshold:.2f}: {combined_summary['raw_edge_count_at_threshold']})")
    print(f"Centred mean |r|: {combined_summary['centred_mean_abs_corr']:.3f}  "
          f"(edges @ {config.edge_threshold:.2f}: {combined_summary['centred_edge_count_at_threshold']})")
    print(f"Network nodes: {combined_summary['node_count']}")
    print(f"Network edges: {combined_summary['edge_count']}")
    print(f"Chosen edge threshold: {combined_summary['edge_threshold']}")
    print(f"Mean |correlation|: {combined_summary['mean_abs_correlation']:.3f}")
    print(f"Median |correlation|: {combined_summary['median_abs_correlation']:.3f}")
    print(f"Ordinal validation accuracy: {combined_summary['validation_accuracy']:.3f}")
    print(f"Ordinal validation MAE: {combined_summary['validation_mae']:.3f}")
    print(f"Cells imputed: {combined_summary['cells_imputed']}")
    print(f"Outputs written to: {config.output_dir}")

    return {
        "summary": combined_summary,
        "graph": graph,
        "full_drawn_graph": full_drawn,
        "thresholded_drawn_graph": thresholded_drawn,
        "performance": performance,
        "imputation_log": imputation_log,
    }
