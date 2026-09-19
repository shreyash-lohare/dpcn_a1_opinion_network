"""Question-wise regularised proportional-odds ordinal logistic regression.

Used to recover NC and structural-blank cells on the original -2..+2 scale.
This runs strictly before row-centring: centring is a network-construction
step, not part of how missing responses are recovered.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.preprocessing.settings import CATEGORIES, PipelineConfig


class OrdinalLogitImputerModel:
    """Regularized proportional-odds ordinal logistic regression.

    Missing predictors are imputed with training-column medians before
    standardization. This treats missing predictors as unknown covariates,
    not as neutral responses.
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 500):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.feature_names: List[str] = []
        self.medians_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.scales_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.threshold_base_: Optional[float] = None
        self.threshold_deltas_: Optional[np.ndarray] = None
        self.success_: bool = False
        self.message_: str = ""

    def _prepare_x(self, x: pd.DataFrame, fit: bool) -> np.ndarray:
        values = x.to_numpy(dtype=float)
        if fit:
            medians = np.nanmedian(values, axis=0)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            filled = np.where(np.isnan(values), medians, values)
            means = filled.mean(axis=0)
            scales = filled.std(axis=0)
            scales = np.where(scales > 1e-8, scales, 1.0)
            self.medians_ = medians
            self.means_ = means
            self.scales_ = scales
        else:
            if self.medians_ is None or self.means_ is None or self.scales_ is None:
                raise ValueError("Model has not been fitted.")
            filled = np.where(np.isnan(values), self.medians_, values)
        return (filled - self.means_) / self.scales_

    @staticmethod
    def _unpack(params: np.ndarray, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        beta = params[:n_features]
        base = params[n_features]
        raw_deltas = params[n_features + 1 :]
        thresholds = [base]
        for raw_delta in raw_deltas:
            thresholds.append(thresholds[-1] + np.exp(raw_delta))
        return beta, np.array(thresholds)

    @staticmethod
    def _probabilities_from_params(x: np.ndarray, beta: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        eta = x @ beta
        cumulative = expit(thresholds[None, :] - eta[:, None])
        probs = np.column_stack(
            [
                cumulative[:, 0],
                cumulative[:, 1] - cumulative[:, 0],
                cumulative[:, 2] - cumulative[:, 1],
                cumulative[:, 3] - cumulative[:, 2],
                1.0 - cumulative[:, 3],
            ]
        )
        return np.clip(probs, 1e-12, 1.0)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "OrdinalLogitImputerModel":
        self.feature_names = list(x.columns)
        observed = y.notna()
        x_fit = self._prepare_x(x.loc[observed], fit=True)
        y_codes = pd.Categorical(y.loc[observed].astype(int), categories=CATEGORIES, ordered=True).codes
        if len(np.unique(y_codes)) < 2:
            raise ValueError("At least two target classes are required.")

        n_features = x_fit.shape[1]
        start_thresholds = np.array([-1.5, -0.5, 0.5, 1.5])
        start = np.concatenate(
            [
                np.zeros(n_features),
                [start_thresholds[0]],
                np.log(np.diff(start_thresholds)),
            ]
        )

        def objective(params: np.ndarray) -> float:
            beta, thresholds = self._unpack(params, n_features)
            probs = self._probabilities_from_params(x_fit, beta, thresholds)
            nll = -np.log(probs[np.arange(len(y_codes)), y_codes]).sum()
            penalty = 0.5 * self.l2 * np.dot(beta, beta)
            return float(nll + penalty)

        result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": self.max_iter})
        beta, thresholds = self._unpack(result.x, n_features)
        self.beta_ = beta
        self.threshold_base_ = thresholds[0]
        self.threshold_deltas_ = np.diff(thresholds)
        self.success_ = bool(result.success)
        self.message_ = str(result.message)
        return self

    @property
    def thresholds_(self) -> np.ndarray:
        if self.threshold_base_ is None or self.threshold_deltas_ is None:
            raise ValueError("Model has not been fitted.")
        thresholds = [self.threshold_base_]
        for delta in self.threshold_deltas_:
            thresholds.append(thresholds[-1] + delta)
        return np.array(thresholds)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.beta_ is None:
            raise ValueError("Model has not been fitted.")
        x_prepared = self._prepare_x(x.loc[:, self.feature_names], fit=False)
        return self._probabilities_from_params(x_prepared, self.beta_, self.thresholds_)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(x)
        return CATEGORIES[np.argmax(probs, axis=1)]


def prepare_ordinal_training_data(
    encoded: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    features = [col for col in encoded.columns if col != target]
    return encoded.loc[:, features], encoded[target]


def train_ordinal_models(
    encoded: pd.DataFrame, config: PipelineConfig
) -> Dict[str, OrdinalLogitImputerModel]:
    models: Dict[str, OrdinalLogitImputerModel] = {}
    for target in encoded.columns:
        x, y = prepare_ordinal_training_data(encoded, target)
        model = OrdinalLogitImputerModel(l2=config.ordinal_l2, max_iter=config.max_iter)
        model.fit(x, y)
        models[target] = model
    return models


def performance_row(scenario: str, repeat: int, actual: Sequence[int], predicted: Sequence[int]) -> Dict[str, object]:
    actual_arr = np.array(actual, dtype=int)
    pred_arr = np.array(predicted, dtype=int)
    row: Dict[str, object] = {
        "scenario": scenario,
        "repeat": repeat,
        "n_masked": int(len(actual_arr)),
        "accuracy": float(np.mean(actual_arr == pred_arr)) if len(actual_arr) else np.nan,
        "mae": float(np.mean(np.abs(actual_arr - pred_arr))) if len(actual_arr) else np.nan,
    }
    for actual_value in CATEGORIES:
        for pred_value in CATEGORIES:
            row[f"confusion_actual_{actual_value}_pred_{pred_value}"] = int(
                ((actual_arr == actual_value) & (pred_arr == pred_value)).sum()
            )
    return row


def validate_models_nc_like(encoded: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)
    observed_positions = np.argwhere(encoded.notna().to_numpy())
    rows = []
    mask_count = max(1, int(len(observed_positions) * config.validation_mask_rate))

    for repeat in range(config.validation_repeats):
        chosen = observed_positions[rng.choice(len(observed_positions), size=mask_count, replace=False)]
        masked = encoded.copy()
        for row_idx, col_idx in chosen:
            masked.iat[row_idx, col_idx] = np.nan
        models = train_ordinal_models(masked, config)
        actual = []
        predicted = []
        for row_idx, col_idx in chosen:
            target = encoded.columns[col_idx]
            feature_cols = [col for col in encoded.columns if col != target]
            pred = models[target].predict(masked.iloc[[row_idx]][feature_cols])[0]
            actual.append(int(encoded.iat[row_idx, col_idx]))
            predicted.append(int(pred))
        rows.append(performance_row("NC-like scattered masking", repeat, actual, predicted))
    return pd.DataFrame(rows)


def validate_models_structural(encoded: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state + 1000)
    eligible_rows = np.where(encoded.notna().sum(axis=1).to_numpy() >= 55)[0]
    if len(eligible_rows) == 0:
        return pd.DataFrame()
    n_eval_rows = min(config.structural_validation_rows, len(eligible_rows))
    rows = []

    for repeat in range(config.validation_repeats):
        chosen_rows = rng.choice(eligible_rows, size=n_eval_rows, replace=False)
        stop_points = rng.integers(low=15, high=55, size=n_eval_rows)
        masked = encoded.copy()
        masked_positions = []
        for row_idx, stop in zip(chosen_rows, stop_points):
            for col_idx in range(stop, encoded.shape[1]):
                if pd.notna(encoded.iat[row_idx, col_idx]):
                    masked.iat[row_idx, col_idx] = np.nan
                    masked_positions.append((row_idx, col_idx, stop))
        models = train_ordinal_models(masked, config)
        actual = []
        predicted = []
        for row_idx, col_idx, stop in masked_positions:
            target = encoded.columns[col_idx]
            feature_cols = [col for col in encoded.columns if col != target]
            row_features = masked.iloc[[row_idx]][feature_cols].copy()
            future_cols = [encoded.columns[k] for k in range(stop, encoded.shape[1]) if encoded.columns[k] != target]
            row_features.loc[:, future_cols] = np.nan
            pred = models[target].predict(row_features)[0]
            actual.append(int(encoded.iat[row_idx, col_idx]))
            predicted.append(int(pred))
        rows.append(performance_row("Structural suffix masking", repeat, actual, predicted))
    return pd.DataFrame(rows)


def impute_missing_responses(
    encoded: pd.DataFrame,
    raw_df: pd.DataFrame,
    question_cols: Sequence[str],
    question_codes: Dict[str, str],
    missing_mask: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, OrdinalLogitImputerModel]]:
    from src.preprocessing.loader import decode_response

    models = train_ordinal_models(encoded, config)
    imputed_numeric = encoded.copy()
    imputation_log = []

    for original_col in question_cols:
        target = question_codes[original_col]
        feature_cols = [col for col in encoded.columns if col != target]
        target_missing = missing_mask[original_col].to_numpy()
        if not target_missing.any():
            continue
        predictions = models[target].predict(imputed_numeric.loc[target_missing, feature_cols])
        imputed_numeric.loc[target_missing, target] = predictions
        for row_idx, value in zip(np.where(target_missing)[0], predictions):
            imputation_log.append(
                {
                    "row_index": int(row_idx),
                    "response_id": raw_df.iloc[row_idx, 0],
                    "question_id": target,
                    "imputed_value": int(value),
                    "imputed_label": decode_response(int(value)),
                }
            )

    return imputed_numeric, pd.DataFrame(imputation_log), models


def save_models(models: Dict[str, OrdinalLogitImputerModel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(models, file)
