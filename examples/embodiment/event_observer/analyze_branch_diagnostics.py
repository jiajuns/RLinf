#!/usr/bin/env python3
"""Audit whether matched-state intervention labels contain usable signal.

Input files are compact ``.npz`` artifacts emitted by
``EmbodiedFSDPActor._write_event_branch_diagnostics``.  They intentionally
store returns/scores only, not RGB, so that all statistics remain auditable
without turning branch collection into an image-data dump.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def _pairwise_and_tau(true_scores: np.ndarray, predicted_scores: np.ndarray, tie_eps: float) -> dict[str, float | int | None]:
    """Tie-aware candidate ordering metrics, calculated state by state."""
    concordant = discordant = true_ties = predicted_ties = eligible_pairs = 0
    for truth, prediction in zip(true_scores, predicted_scores, strict=True):
        for left in range(truth.size):
            for right in range(left + 1, truth.size):
                true_delta = float(truth[left] - truth[right])
                pred_delta = float(prediction[left] - prediction[right])
                true_sign = 0 if abs(true_delta) <= tie_eps else int(np.sign(true_delta))
                pred_sign = 0 if abs(pred_delta) <= tie_eps else int(np.sign(pred_delta))
                if true_sign == 0:
                    # Near-identical true returns do not define a useful
                    # ordering target.  They are excluded from pairwise
                    # accuracy but still count as an x-tie in tau-b when the
                    # prediction tries to order them.
                    if pred_sign != 0:
                        true_ties += 1
                    continue
                eligible_pairs += 1
                if pred_sign == 0:
                    predicted_ties += 1
                elif true_sign == pred_sign:
                    concordant += 1
                else:
                    discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + true_ties)
        * (concordant + discordant + predicted_ties)
    )
    return {
        "tie_eps": tie_eps,
        "tie_aware_pairwise_accuracy": (
            float(concordant / eligible_pairs) if eligible_pairs else None
        ),
        "ordered_pairs": int(eligible_pairs),
        "concordant": int(concordant),
        "discordant": int(discordant),
        "true_ties": int(true_ties),
        "predicted_ties": int(predicted_ties),
        "kendall_tau_b": (
            float((concordant - discordant) / denominator)
            if denominator > 0
            else None
        ),
    }


def _metrics(true_scores: np.ndarray, predicted_scores: np.ndarray, tie_eps: float) -> dict[str, Any]:
    if true_scores.size == 0:
        return {"states": 0}
    centered = true_scores - true_scores.mean(axis=1, keepdims=True)
    candidate0_target = centered[:, 0]
    candidate0_prediction = predicted_scores[:, 0]
    best_true = true_scores.max(axis=1)
    selected = np.argmax(predicted_scores, axis=1)
    chosen_true = true_scores[np.arange(true_scores.shape[0]), selected]
    regrets = best_true - chosen_true
    spread_std = true_scores.std(axis=1)
    spread_range = true_scores.max(axis=1) - true_scores.min(axis=1)
    predicted_std = predicted_scores.std(axis=1)
    result: dict[str, Any] = {
        "states": int(true_scores.shape[0]),
        "candidates_per_state": int(true_scores.shape[1]),
        "branch_return_std_per_state": _summary(spread_std),
        "branch_return_range_per_state": _summary(spread_range),
        "near_zero_spread_fraction": float(np.mean(spread_range <= tie_eps)),
        "candidate0_zero_predictor_mse": float(np.mean(candidate0_target**2)),
        "candidate0_model_mse": float(np.mean((candidate0_prediction - candidate0_target) ** 2)),
        "candidate0_model_vs_zero_mse_ratio": float(
            np.mean((candidate0_prediction - candidate0_target) ** 2)
            / max(np.mean(candidate0_target**2), np.finfo(np.float64).eps)
        ),
        "predicted_score_std_per_state": _summary(predicted_std),
        "predicted_near_constant_fraction": float(np.mean(predicted_std <= tie_eps)),
        "top1_regret": _summary(regrets),
        "top1_exact_best_fraction": float(np.mean(regrets <= tie_eps)),
    }
    result.update(_pairwise_and_tau(true_scores, predicted_scores, tie_eps))
    return result


def _same_action_repeatability(returns: np.ndarray, actions: np.ndarray | None) -> dict[str, Any]:
    """Measure branch-return variation for exact duplicate action chunks."""
    if actions is None:
        return {"pairs": 0, "absolute_return_difference": {"count": 0}}
    differences = []
    for state_returns, state_actions in zip(returns, actions, strict=True):
        for left in range(len(state_returns)):
            for right in range(left + 1, len(state_returns)):
                if np.array_equal(state_actions[left], state_actions[right]):
                    differences.append(abs(float(state_returns[left] - state_returns[right])))
    return {"pairs": len(differences), "absolute_return_difference": _summary(np.asarray(differences))}


def _markdown(report: dict[str, Any]) -> str:
    global_metrics = report["global"]
    lines = [
        "# Matched-state branch influence diagnostic",
        "",
        f"- files: {report['files']}",
        f"- states: {global_metrics['states']}",
        f"- dropped states with incomplete candidate validity: {report['invalid_candidate_states']}",
        f"- candidates/state: {global_metrics.get('candidates_per_state', 0)}",
        f"- tie epsilon: {report['tie_eps']}",
        "",
        "## Global signal",
        "",
        f"- branch-return std mean / median: {global_metrics['branch_return_std_per_state'].get('mean')} / {global_metrics['branch_return_std_per_state'].get('median')}",
        f"- near-zero return-spread fraction: {global_metrics['near_zero_spread_fraction']}",
        f"- candidate-0 model MSE / zero MSE / ratio: {global_metrics['candidate0_model_mse']} / {global_metrics['candidate0_zero_predictor_mse']} / {global_metrics['candidate0_model_vs_zero_mse_ratio']}",
        f"- tie-aware pairwise accuracy: {global_metrics['tie_aware_pairwise_accuracy']}",
        f"- Kendall tau-b: {global_metrics['kendall_tau_b']}",
        f"- top-1 regret mean: {global_metrics['top1_regret'].get('mean')}",
        f"- predicted near-constant fraction: {global_metrics['predicted_near_constant_fraction']}",
        f"- actual branch duration mean: {report['branch_duration_steps'].get('mean')}",
        f"- bootstrap discount units mean: {report['bootstrap_discount_units'].get('mean')}",
        f"- terminal / truncated candidate fraction: {report['terminal_candidate_fraction']} / {report['truncated_candidate_fraction']}",
        f"- same-action repeat pairs / |return difference| mean: {report['same_action_repeatability']['pairs']} / {report['same_action_repeatability']['absolute_return_difference'].get('mean')}",
        "",
        "## Per-event return spread",
        "",
        "| event id | states | return std mean | range median |",
        "|---:|---:|---:|---:|",
    ]
    for event_id, metrics in sorted(report["per_event"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {event_id} | {metrics['states']} | {metrics['branch_return_std_per_state'].get('mean')} | {metrics['branch_return_range_per_state'].get('median')} |"
        )
    lines.extend(["", "## Branch-success strata", ""])
    for name, metrics in report["success_strata"].items():
        lines.append(
            f"- {name}: states={metrics['states']}, pairwise={metrics.get('tie_aware_pairwise_accuracy')}, tau_b={metrics.get('kendall_tau_b')}, regret_mean={metrics.get('top1_regret', {}).get('mean')}"
        )
    lines.extend([
        "",
        "## Interpretation guardrail",
        "",
        "This report is evidence about the supplied diagnostic files only. It supports a causal-credit claim only when files were collected from states held out from Influence training (normally `record_only=true`) and when branch transitions are counted in the interaction budget.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostic_dir", type=Path)
    parser.add_argument("--tie-eps", type=float, default=1e-4)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    args = parser.parse_args()
    if args.tie_eps < 0:
        raise ValueError("--tie-eps must be non-negative")

    files = sorted(args.diagnostic_dir.glob("branch_diag_rank*_opt*_seen*.npz"))
    if not files:
        raise FileNotFoundError(f"no diagnostic artifacts under {args.diagnostic_dir}")
    returns, predictions, event_ids, success = [], [], [], []
    success_available = []
    durations, discount_units, terminals, truncations, action_rows = [], [], [], [], []
    invalid_candidate_states = 0
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            true_values = np.asarray(payload["branch_returns"], dtype=np.float64)
            predicted_values = np.asarray(payload["predicted_scores"], dtype=np.float64)
            if true_values.ndim != 2 or true_values.shape != predicted_values.shape or true_values.shape[1] < 2:
                raise ValueError(f"invalid candidate matrices in {path}")
            valid = np.asarray(payload["candidate_valid_mask"], dtype=bool) if "candidate_valid_mask" in payload.files else np.ones_like(true_values, dtype=bool)
            if valid.shape != true_values.shape:
                raise ValueError(f"candidate_valid_mask is misaligned in {path}")
            complete = valid.all(axis=1)
            invalid_candidate_states += int((~complete).sum())
            returns.append(true_values[complete])
            predictions.append(predicted_values[complete])
            event_ids.append(np.asarray(payload["event_ids"]).reshape(-1)[complete])
            # ``success_available`` was added with the persistent diagnostic
            # schema.  Treat legacy artifacts as unavailable rather than
            # rejecting an otherwise useful return/ranking audit.
            success.append(np.asarray(payload["branch_success"], dtype=bool)[complete])
            if "actual_duration_steps" in payload.files:
                durations.append(np.asarray(payload["actual_duration_steps"], dtype=np.float64)[complete])
            if "bootstrap_discount_units" in payload.files:
                discount_units.append(np.asarray(payload["bootstrap_discount_units"], dtype=np.float64)[complete])
            if "branch_terminations" in payload.files:
                terminals.append(np.asarray(payload["branch_terminations"], dtype=bool)[complete])
            if "branch_truncations" in payload.files:
                truncations.append(np.asarray(payload["branch_truncations"], dtype=bool)[complete])
            if "candidate_actions" in payload.files:
                candidate_actions = np.asarray(payload["candidate_actions"])
                if candidate_actions.shape[:2] != true_values.shape:
                    raise ValueError(f"candidate_actions is misaligned in {path}")
                action_rows.append(candidate_actions[complete])
            success_available.append(
                bool(np.asarray(payload["success_available"]).item())
                if "success_available" in payload.files
                else False
            )
    true_values = np.concatenate(returns, axis=0)
    if not len(true_values):
        raise ValueError("all diagnostic states have invalid candidates")
    predicted_values = np.concatenate(predictions, axis=0)
    ids = np.concatenate(event_ids, axis=0)
    branch_success = np.concatenate(success, axis=0)
    candidate_actions = np.concatenate(action_rows, axis=0) if action_rows else None

    per_event: dict[str, Any] = {}
    for event_id in np.unique(ids):
        mask = ids == event_id
        per_event[str(int(event_id))] = _metrics(true_values[mask], predicted_values[mask], args.tie_eps)

    strata_masks = {
        "candidate0_success": branch_success[:, 0],
        "candidate0_failure": ~branch_success[:, 0],
        "any_candidate_success": branch_success.any(axis=1),
        "no_candidate_success": ~branch_success.any(axis=1),
    }
    report = {
        "files": len(files),
        "states": int(true_values.shape[0]),
        "tie_eps": args.tie_eps,
        "branch_success_available_for_all_files": bool(all(success_available)),
        "invalid_candidate_states": invalid_candidate_states,
        "branch_duration_steps": _summary(np.concatenate(durations) if durations else np.asarray([])),
        "bootstrap_discount_units": _summary(np.concatenate(discount_units) if discount_units else np.asarray([])),
        "same_action_repeatability": _same_action_repeatability(true_values, candidate_actions),
        "terminal_candidate_fraction": float(np.concatenate(terminals).mean()) if terminals else None,
        "truncated_candidate_fraction": float(np.concatenate(truncations).mean()) if truncations else None,
        "global": _metrics(true_values, predicted_values, args.tie_eps),
        "per_event": per_event,
        "success_strata": {
            name: _metrics(true_values[mask], predicted_values[mask], args.tie_eps)
            for name, mask in strata_masks.items()
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(_markdown(report))
    print(rendered)


if __name__ == "__main__":
    main()
