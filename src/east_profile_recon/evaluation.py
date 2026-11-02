import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde


@dataclass
class EvaluationResult:
    time: np.ndarray
    selected_indices: np.ndarray
    r2: np.ndarray
    mean_relative_error: np.ndarray
    normalized_integrated_absolute_error: np.ndarray
    rmse: np.ndarray
    max_relative_error: np.ndarray
    mean_pointwise_accuracy: np.ndarray
    mean_density_accuracy: np.ndarray
    target_points: np.ndarray
    candidate_points: np.ndarray
    total_frames: int
    selected_frames: int
    processed_frames: int
    skipped_frames: dict

    def metric_arrays(self):
        return {
            "mean_density_accuracy": self.mean_density_accuracy,
            "r2": self.r2,
            "rmse": self.rmse,
            "mean_relative_error": self.mean_relative_error,
            "normalized_integrated_absolute_error": self.normalized_integrated_absolute_error,
            "max_relative_error": self.max_relative_error,
            "mean_pointwise_accuracy": self.mean_pointwise_accuracy,
        }

    def summary(self):
        metrics = {
            name: _summary_statistics(values)
            for name, values in self.metric_arrays().items()
        }
        finite_relative_error = self.mean_relative_error[
            np.isfinite(self.mean_relative_error)
        ]
        metrics["mean_relative_error"]["p90"] = (
            float(np.percentile(finite_relative_error, 90))
            if finite_relative_error.size
            else None
        )
        return {
            "total_frames": self.total_frames,
            "selected_frames": self.selected_frames,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
            "metrics": metrics,
        }


def _summary_statistics(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _load_struct_array(mat_file_path, variable_name=None):
    path = Path(mat_file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MAT file not found: {path}")

    mat_data = sio.loadmat(path)
    public_keys = [key for key in mat_data if not key.startswith("__")]
    if variable_name is None:
        if not public_keys:
            raise ValueError(f"No public variables found in MAT file: {path}")
        variable_name = public_keys[0]
    elif variable_name not in mat_data:
        raise KeyError(
            f"Variable {variable_name!r} not found in {path}. "
            f"Available variables: {public_keys}"
        )

    profiles = np.asarray(mat_data[variable_name]).reshape(-1)
    if profiles.size == 0:
        raise ValueError(f"Variable {variable_name!r} is empty in {path}")
    return profiles, variable_name


def _field(record, field_name):
    if isinstance(record, np.void) and record.dtype.names:
        if field_name not in record.dtype.names:
            raise KeyError(
                f"Field {field_name!r} not found. Available fields: "
                f"{record.dtype.names}"
            )
        return record[field_name]
    if hasattr(record, field_name):
        return getattr(record, field_name)
    try:
        return record[field_name]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise KeyError(f"Field {field_name!r} not found in profile record") from exc


def _field_array(record, field_name):
    return np.asarray(_field(record, field_name), dtype=float).reshape(-1)


def _field_scalar(record, field_name):
    values = _field_array(record, field_name)
    if values.size == 0:
        raise ValueError(f"Field {field_name!r} is empty")
    return float(values[0])


def _r2_score(target, candidate):
    residual_sum = np.sum((target - candidate) ** 2)
    centered_sum = np.sum((target - np.mean(target)) ** 2)
    if centered_sum <= np.finfo(float).eps:
        return np.nan
    return 1.0 - residual_sum / centered_sum


def evaluate_profile_series(
    reference_profiles,
    candidate_profiles,
    *,
    max_profiles=2000,
    min_valid_points=10,
    min_density=1e-9,
    radius_atol=1e-8,
    time_start=0.0,
    time_interval=1.0,
):
    reference_profiles = np.asarray(reference_profiles).reshape(-1)
    candidate_profiles = np.asarray(candidate_profiles).reshape(-1)
    frame_count = min(reference_profiles.size, candidate_profiles.size)
    if frame_count == 0:
        raise ValueError("No overlapping profile frames are available")

    reference_profiles = reference_profiles[:frame_count]
    candidate_profiles = candidate_profiles[:frame_count]
    reference_nwidth = np.array(
        [_field_scalar(record, "nwidth") for record in reference_profiles],
        dtype=float,
    )
    candidate_nwidth = np.array(
        [_field_scalar(record, "nwidth") for record in candidate_profiles],
        dtype=float,
    )
    valid_nwidth = np.flatnonzero(
        np.isfinite(reference_nwidth)
        & np.isfinite(candidate_nwidth)
        & (reference_nwidth >= 0)
        & (candidate_nwidth >= 0)
    )
    if max_profiles is not None:
        valid_nwidth = valid_nwidth[:max_profiles]

    metrics = {
        "r2": np.full(frame_count, np.nan),
        "mean_relative_error": np.full(frame_count, np.nan),
        "normalized_integrated_absolute_error": np.full(frame_count, np.nan),
        "rmse": np.full(frame_count, np.nan),
        "max_relative_error": np.full(frame_count, np.nan),
        "mean_pointwise_accuracy": np.full(frame_count, np.nan),
        "mean_density_accuracy": np.full(frame_count, np.nan),
    }
    skipped = {
        "shape_mismatch": 0,
        "insufficient_valid_points": 0,
        "radius_mismatch": 0,
        "non_monotonic_radius": 0,
        "zero_density_integral": 0,
    }
    pooled_target = []
    pooled_candidate = []

    for index in valid_nwidth:
        radius_reference = _field_array(reference_profiles[index], "R_fit")
        density_reference = _field_array(reference_profiles[index], "ne_fit")
        radius_candidate = _field_array(candidate_profiles[index], "R_fit")
        density_candidate = _field_array(candidate_profiles[index], "ne_fit")

        sizes = {
            radius_reference.size,
            density_reference.size,
            radius_candidate.size,
            density_candidate.size,
        }
        if len(sizes) != 1:
            skipped["shape_mismatch"] += 1
            continue

        valid = (
            np.isfinite(radius_reference)
            & np.isfinite(density_reference)
            & np.isfinite(radius_candidate)
            & np.isfinite(density_candidate)
            & (density_reference > min_density)
            & (density_candidate > min_density)
        )
        if np.count_nonzero(valid) < min_valid_points:
            skipped["insufficient_valid_points"] += 1
            continue

        radius_ref = radius_reference[valid]
        radius_test = radius_candidate[valid]
        if not np.allclose(radius_ref, radius_test, rtol=0.0, atol=radius_atol):
            skipped["radius_mismatch"] += 1
            continue
        radius_delta = np.diff(radius_ref)
        if not (np.all(radius_delta > 0) or np.all(radius_delta < 0)):
            skipped["non_monotonic_radius"] += 1
            continue

        target = density_reference[valid]
        candidate = density_candidate[valid]
        relative_error = np.abs(candidate - target) / target
        target_integral = trapezoid(target, radius_ref)
        if abs(target_integral) <= np.finfo(float).eps:
            skipped["zero_density_integral"] += 1
            continue

        metrics["r2"][index] = _r2_score(target, candidate)
        metrics["mean_relative_error"][index] = np.mean(relative_error)
        metrics["normalized_integrated_absolute_error"][index] = (
            trapezoid(np.abs(candidate - target), radius_ref) / target_integral
        )
        metrics["rmse"][index] = np.sqrt(np.mean((candidate - target) ** 2))
        metrics["max_relative_error"][index] = np.max(relative_error)
        metrics["mean_pointwise_accuracy"][index] = np.mean(1.0 - relative_error)
        metrics["mean_density_accuracy"][index] = 1.0 - (
            abs(np.mean(candidate) - np.mean(target)) / np.mean(target)
        )
        pooled_target.append(target)
        pooled_candidate.append(candidate)

    time = time_start + np.arange(frame_count, dtype=float) * time_interval
    target_points = (
        np.concatenate(pooled_target) if pooled_target else np.array([], dtype=float)
    )
    candidate_points = (
        np.concatenate(pooled_candidate)
        if pooled_candidate
        else np.array([], dtype=float)
    )
    processed_frames = int(np.count_nonzero(np.isfinite(metrics["rmse"])))

    return EvaluationResult(
        time=time,
        selected_indices=valid_nwidth,
        target_points=target_points,
        candidate_points=candidate_points,
        total_frames=frame_count,
        selected_frames=int(valid_nwidth.size),
        processed_frames=processed_frames,
        skipped_frames=skipped,
        **metrics,
    )


def evaluate_mat_files(
    reference_path,
    candidate_path,
    *,
    reference_variable=None,
    candidate_variable=None,
    **evaluation_options,
):
    reference_profiles, reference_variable = _load_struct_array(
        reference_path,
        reference_variable,
    )
    candidate_profiles, candidate_variable = _load_struct_array(
        candidate_path,
        candidate_variable,
    )
    result = evaluate_profile_series(
        reference_profiles,
        candidate_profiles,
        **evaluation_options,
    )
    return result, {
        "reference_path": str(Path(reference_path).expanduser()),
        "candidate_path": str(Path(candidate_path).expanduser()),
        "reference_variable": reference_variable,
        "candidate_variable": candidate_variable,
    }


def save_evaluation_data(result, output_dir, metadata=None):
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path / "evaluation_metrics.npz",
        time=result.time,
        selected_indices=result.selected_indices,
        target_points=result.target_points,
        candidate_points=result.candidate_points,
        **result.metric_arrays(),
    )
    summary = result.summary()
    if metadata:
        summary["input"] = metadata
    with (output_path / "evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return output_path


def _finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _trim_upper_outliers(values, percentile=99.5):
    finite = _finite_values(values)
    if finite.size == 0:
        return finite
    upper = np.percentile(finite, percentile)
    return np.clip(finite, None, upper)


def _plot_style():
    return {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.8,
    }


def _save_figure(fig, output_path):
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")


def plot_accuracy_time_series(
    result,
    output_path,
    *,
    candidate_label="Candidate",
    event_times=(),
):
    import matplotlib.pyplot as plt

    series_color = "#2563EB"
    event_color = "#64748B"
    with plt.rc_context(_plot_style()):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for axis in axes:
            for event_time in event_times:
                axis.axvline(
                    event_time,
                    color=event_color,
                    linestyle=":",
                    alpha=0.65,
                    linewidth=1.2,
                )
        axes[0].plot(
            result.time,
            result.r2 * 100.0,
            color=series_color,
            linewidth=2.0,
        )
        axes[0].set_ylabel("Profile-wise $R^2$ (%)")
        axes[0].set_title(f"{candidate_label}: profile agreement over time")
        axes[1].plot(
            result.time,
            result.mean_relative_error * 100.0,
            color=series_color,
            linewidth=2.0,
        )
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Mean relative error (%)")
        fig.tight_layout()
        _save_figure(fig, output_path)
    return fig


def plot_metric_distributions(result, output_path, *, candidate_label="Candidate"):
    import matplotlib.pyplot as plt

    series_color = "#2563EB"
    distributions = (
        (
            _trim_upper_outliers(result.mean_relative_error * 100.0),
            "Mean relative error (%)",
        ),
        (_trim_upper_outliers(result.r2 * 100.0), "$R^2$ (%)"),
    )
    with plt.rc_context(_plot_style()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for axis, (values, label) in zip(axes, distributions):
            if values.size == 0:
                axis.text(
                    0.5,
                    0.5,
                    "No valid values",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_xlabel(label)
                continue
            bins = min(60, max(10, int(np.sqrt(values.size))))
            axis.hist(
                values,
                bins=bins,
                density=True,
                color=series_color,
                alpha=0.3,
                edgecolor="white",
                linewidth=0.8,
            )
            if values.size > 1 and np.std(values) > np.finfo(float).eps:
                density = gaussian_kde(values)
                x_values = np.linspace(np.min(values), np.max(values), 300)
                axis.plot(
                    x_values,
                    density(x_values),
                    color=series_color,
                    linewidth=2.0,
                )
            axis.axvline(
                np.mean(values),
                color="#111827",
                linestyle="--",
                linewidth=1.2,
            )
            axis.set_xlabel(label)
            axis.set_ylabel("Density")
        fig.suptitle(f"{candidate_label}: metric distributions")
        fig.tight_layout()
        _save_figure(fig, output_path)
    return fig


def plot_relative_error_cdf(result, output_path, *, candidate_label="Candidate"):
    import matplotlib.pyplot as plt

    values = np.sort(_trim_upper_outliers(result.mean_relative_error * 100.0))
    with plt.rc_context(_plot_style()):
        fig, axis = plt.subplots(figsize=(8, 5))
        if values.size:
            cumulative = np.arange(1, values.size + 1) / values.size * 100.0
            axis.plot(values, cumulative, color="#2563EB", linewidth=2.0)
            p90 = np.percentile(values, 90)
            axis.axvline(p90, color="#64748B", linestyle="--", linewidth=1.2)
            axis.text(
                p90,
                10,
                f"90th percentile: {p90:.3f}%",
                rotation=90,
                va="bottom",
                ha="right",
                color="#374151",
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No valid values",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_xlabel("Mean relative error (%)")
        axis.set_ylabel("Cumulative fraction of profiles (%)")
        axis.set_title(f"{candidate_label}: relative-error CDF")
        fig.tight_layout()
        _save_figure(fig, output_path)
    return fig


def plot_density_scatter(
    result,
    output_path,
    *,
    reference_label="Reference",
    candidate_label="Candidate",
    max_points=20000,
    random_seed=0,
):
    import matplotlib.pyplot as plt

    target = result.target_points
    candidate = result.candidate_points
    with plt.rc_context(_plot_style()):
        fig, axis = plt.subplots(figsize=(6.8, 6.2))
        if target.size:
            random = np.random.default_rng(random_seed)
            sample = random.choice(
                target.size,
                size=min(max_points, target.size),
                replace=False,
            )
            lower = min(np.min(target), np.min(candidate))
            upper = max(np.max(target), np.max(candidate))
            axis.plot(
                (lower, upper),
                (lower, upper),
                color="#111827",
                linestyle="--",
                linewidth=1.2,
                label="1:1",
            )
            axis.scatter(
                target[sample],
                candidate[sample],
                s=10,
                alpha=0.25,
                color="#2563EB",
                edgecolors="none",
                rasterized=True,
            )
            pooled_r2 = _r2_score(target, candidate)
            axis.text(
                0.04,
                0.96,
                f"Pooled $R^2$ = {pooled_r2:.4f}",
                transform=axis.transAxes,
                va="top",
                color="#111827",
            )
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.legend(loc="lower right", frameon=False)
        else:
            axis.text(
                0.5,
                0.5,
                "No valid profile points",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_xlabel(f"{reference_label} $n_e$ ($10^{{19}}$ m$^{{-3}}$)")
        axis.set_ylabel(f"{candidate_label} $n_e$ ($10^{{19}}$ m$^{{-3}}$)")
        axis.set_title("Pooled density-profile comparison")
        axis.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        _save_figure(fig, output_path)
    return fig


def save_evaluation_plots(
    result,
    output_dir,
    *,
    reference_label="Reference",
    candidate_label="Candidate",
    event_times=(),
    max_scatter_points=20000,
    show=False,
):
    import matplotlib.pyplot as plt

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_accuracy_time_series(
            result,
            output_path / "accuracy_time_series.png",
            candidate_label=candidate_label,
            event_times=event_times,
        ),
        plot_metric_distributions(
            result,
            output_path / "accuracy_distributions.png",
            candidate_label=candidate_label,
        ),
        plot_relative_error_cdf(
            result,
            output_path / "accuracy_cdf.png",
            candidate_label=candidate_label,
        ),
        plot_density_scatter(
            result,
            output_path / "accuracy_scatter.png",
            reference_label=reference_label,
            candidate_label=candidate_label,
            max_points=max_scatter_points,
        ),
    ]
    if show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)
    return [
        output_path / "accuracy_time_series.png",
        output_path / "accuracy_distributions.png",
        output_path / "accuracy_cdf.png",
        output_path / "accuracy_scatter.png",
    ]


def print_evaluation_summary(result):
    summary = result.summary()
    metrics = summary["metrics"]
    print(
        f"Frames: total={summary['total_frames']}, "
        f"selected={summary['selected_frames']}, "
        f"processed={summary['processed_frames']}"
    )
    print("========== Primary metrics (mean ± sample std) ==========")
    for name in ("mean_density_accuracy", "r2", "rmse"):
        values = metrics[name]
        if values["mean"] is None:
            print(f"{name}: no valid values")
        else:
            print(f"{name}: {values['mean']:.6f} ± {values['std']:.6f}")
    print("========== Additional metrics ==========")
    for name in (
        "mean_relative_error",
        "normalized_integrated_absolute_error",
        "max_relative_error",
        "mean_pointwise_accuracy",
    ):
        values = metrics[name]
        if values["mean"] is not None:
            print(f"{name}: {values['mean']:.6f}")
    p90 = metrics["mean_relative_error"].get("p90")
    if p90 is not None:
        print(f"mean_relative_error p90: {p90:.6f}")
    print(f"Skipped frames: {summary['skipped_frames']}")


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate density-profile MAT results with reference MAT "
            "results on their common fixed radial grid."
        )
    )
    parser.add_argument("reference", type=Path, help="Reference MAT file")
    parser.add_argument("candidate", type=Path, help="Candidate MAT file")
    parser.add_argument("--reference-variable", default=None)
    parser.add_argument("--candidate-variable", default=None)
    parser.add_argument("--max-profiles", type=int, default=2000)
    parser.add_argument("--min-valid-points", type=int, default=10)
    parser.add_argument("--min-density", type=float, default=1e-9)
    parser.add_argument("--radius-atol", type=float, default=1e-8)
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-interval", type=float, default=1.0)
    parser.add_argument("--event-time", type=float, action="append", default=[])
    parser.add_argument("--reference-label", default="Reference")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--max-scatter-points", type=int, default=20000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_outputs"),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    result, metadata = evaluate_mat_files(
        args.reference,
        args.candidate,
        reference_variable=args.reference_variable,
        candidate_variable=args.candidate_variable,
        max_profiles=args.max_profiles,
        min_valid_points=args.min_valid_points,
        min_density=args.min_density,
        radius_atol=args.radius_atol,
        time_start=args.time_start,
        time_interval=args.time_interval,
    )
    print_evaluation_summary(result)
    output_dir = save_evaluation_data(result, args.output_dir, metadata)
    if not args.no_plots:
        save_evaluation_plots(
            result,
            output_dir,
            reference_label=args.reference_label,
            candidate_label=args.candidate_label,
            event_times=args.event_time,
            max_scatter_points=args.max_scatter_points,
            show=args.show,
        )
    print(f"Evaluation outputs saved to: {output_dir}")
    return result


if __name__ == "__main__":
    main()
