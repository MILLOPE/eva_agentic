"""Aggregate frozen runs into a per-case summary table and render SVG charts.

This module keeps the visualization path read-only and dependency-free: the
summary CSV/JSON is the primary auditable artifact, and the charts are plain
SVG strings (no matplotlib required) so the evaluator venv stays lightweight.
Pure analysis logic lives here so the CLI stays a thin single entry.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from eva_agentic.schema import AttemptStatus, OutcomeStatus
from eva_agentic.store import load_run
from eva_agentic.summary import terminal_attempt

SUMMARY_FIELDS = [
    "run_id",
    "job_id",
    "case_id",
    "task_id",
    "task_index",
    "seed",
    "participant",
    "attempt",
    "status",
    "task_success",
    "duration_s",
    "termination_reason",
    "error_source",
    "success_source",
]

# Preserve a stable display order and color contract across charts.
STATUS_ORDER = [
    OutcomeStatus.SUCCESS,
    OutcomeStatus.TASK_FAILURE,
    OutcomeStatus.TIMEOUT,
    OutcomeStatus.INFRASTRUCTURE_FAILURE,
    OutcomeStatus.INVALID,
]
STATUS_COLORS = {
    "success": "#4c78a8",
    "task_failure": "#c44e52",
    "timeout": "#d8a047",
    "infrastructure_failure": "#7f8794",
    "invalid": "#d4d4d4",
    "unstarted": "#eeeeee",
}
STATUS_NAMES = {
    "success": "Success",
    "task_failure": "Task failure",
    "timeout": "Timeout",
    "infrastructure_failure": "Infrastructure",
    "invalid": "Invalid",
    "unstarted": "Unstarted",
}
GLYPHS = {
    "success": "S",
    "task_failure": "F",
    "timeout": "T",
    "infrastructure_failure": "I",
    "invalid": "V",
    "unstarted": "U",
}
# A planned trial only has a "valid" (decisive) outcome for these statuses.
VALID_STATUSES = ("success", "task_failure", "timeout")
PENDING_STATUSES = ("infrastructure_failure", "invalid", "unstarted")
STATUS_LEGEND_ORDER = ("success", "task_failure", "timeout",
                       "infrastructure_failure", "invalid", "unstarted")
DEFAULT_PAGE_SIZE = 24

# Extended evidence fields. The 14-field summary contract remains stable.
CASE_METRIC_FIELDS = [
    *SUMMARY_FIELDS,
    "exit_code", "model_name", "planner_runtime", "planner_rounds", "llm_calls",
    "tool_calls", "tool_errors", "loop_feedbacks", "proposal_repairs",
    "planner_loop_stopped", "vla_calls", "sam3_calls", "input_tokens",
    "output_tokens", "planner_elapsed_s", "state_records", "finish_status",
    "failure_reason",
]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def discover_runs(runs_root: str | Path) -> list[Path]:
    """Return run directories directly under ``runs_root`` that carry a frozen plan."""
    root = Path(runs_root)
    found: list[Path] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        if (directory / "inputs" / "plan.json").is_file():
            found.append(directory)
    return found


def collect_rows(run_dirs: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Expand frozen runs into one summary row per planned case."""
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        inputs = load_run(run_dir)
        run_id = run_dir.name
        for job in inputs.plan.jobs:
            case = job.cases[0]
            attempt = terminal_attempt(inputs.run_dir, job.participant, job.job_id)
            rows.append(_case_row(Path(run_dir), run_id, case.to_dict(), job, attempt))
    return rows


def write_case_metrics(rows: Sequence[Mapping[str, Any]], out_csv: str | Path, out_json: str | Path) -> None:
    """Write the extended per-case process table without changing summary schema."""
    csv_path = Path(out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CASE_METRIC_FIELDS})
    Path(out_json).write_text(json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary(rows: Sequence[Mapping[str, Any]], out_csv: str | Path, out_json: str | Path) -> None:
    """Write the per-case summary as CSV (primary) and JSON (structured)."""
    csv_path = Path(out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SUMMARY_FIELDS})
    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps([{key: row.get(key) for key in SUMMARY_FIELDS} for row in rows], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Metrics (pure analysis, dependency-free)
# --------------------------------------------------------------------------- #
def suite_of(row: Mapping[str, Any]) -> str:
    """Derive the benchmark suite for a row (task_id carries a `suite:index` form)."""
    task_id = str(row.get("task_id") or "")
    if ":" in task_id:
        return task_id.split(":", 1)[0]
    return task_id or "suite"


def _status_counter(rows: Iterable[Mapping[str, Any]]) -> "Counter[str]":
    counter: "Counter[str]" = Counter()
    for row in rows:
        counter[str(row.get("status") or "unstarted")] += 1
    return counter


def _yield_fields(counter: "Counter[str]", planned: int) -> dict[str, Any]:
    """Coverage semantics: valid = decisive outcomes, unknown = unresolved planning."""
    valid = sum(counter.get(s, 0) for s in VALID_STATUSES)
    unknown = sum(counter.get(s, 0) for s in PENDING_STATUSES)
    good = counter.get("success", 0)
    return {
        "planned": planned,
        "successes": good,
        "valid": valid,
        "unknown": unknown,
        "valid_pct": 100.0 * valid / planned if planned else 0.0,
        "yield_pct": 100.0 * good / planned if planned else 0.0,
        "upper_pct": 100.0 * (good + unknown) / planned if planned else 0.0,
        "conditional_success_pct": 100.0 * good / valid if valid else None,
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Compute task-level and suite-level aggregate metrics over a set of rows."""
    task_groups: dict[tuple, list[dict[str, Any]]] = {}
    suite_groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        suite = suite_of(row)
        participant = str(row.get("participant") or "unknown")
        tkey = (suite, participant, str(row.get("task_id") or ""))
        skey = (suite, participant)
        task_groups.setdefault(tkey, []).append(row)
        suite_groups.setdefault(skey, []).append(row)

    tasks: list[dict[str, Any]] = []
    for (suite, participant, task_id), group in sorted(task_groups.items()):
        counter = _status_counter(group)
        base = _yield_fields(counter, len(group))
        tasks.append({
            "suite_id": suite,
            "participant": participant,
            "task_id": task_id,
            "task_index": group[0].get("task_index"),
            "status_counts": dict(counter),
            **base,
        })

    suites: list[dict[str, Any]] = []
    for (suite, participant), group in sorted(suite_groups.items()):
        counter = _status_counter(group)
        durations = sorted(d for d in (r.get("duration_s") for r in group)
                           if isinstance(d, (int, float)) and d > 0)
        base = _yield_fields(counter, len(group))
        suites.append({
            "suite_id": suite,
            "participant": participant,
            "task_count": len({str(r.get("task_id")) for r in group}),
            "duration_n": len(durations),
            "duration_quantiles": _quantiles(durations),
            "status_counts": dict(counter),
            **base,
        })
    return {"tasks": tasks, "suites": suites}


def _quantiles(values: list[float], sizes: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> list[float | None]:
    if not values:
        return [None] * len(sizes)
    n = len(values)
    return [_quantile(values, size, n) for size in sizes]


def _quantile(values: list[float], size: float, n: int) -> float:
    h = (n - 1) * size
    lo = int(h)
    hi = min(lo + 1, n - 1)
    return values[lo] + (values[hi] - values[lo]) * (h - lo)


def pages_for(task_ids: Sequence[str], page_size: int = DEFAULT_PAGE_SIZE) -> list[list[str]]:
    """Split a task list into fixed-size pages so each page keeps the same scale."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    return [task_ids[i:i + page_size] for i in range(0, len(task_ids), page_size)]


def group_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (suite_of(row), str(row.get("participant") or "unknown"))




def visualize_runs(
    run_dirs: Iterable[str | Path],
    aggregate_out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Visualize frozen runs in place and optionally emit an aggregate report.

    Each run gets its own evidence bundle under ``<run_dir>/viz/`` containing the
    per-case summary CSV/JSON, the charts, an HTML report and a provenance file.
    When ``aggregate_out_dir`` is given, a combined report across all runs is also
    written there (kept separate so per-run evidence travels with its run).
    """
    viz_subdir = "viz"
    mode = rendering_mode()
    per_run: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        rows = collect_rows([run_dir])
        combined_rows.extend(rows)
        out = run_dir / viz_subdir
        write_summary(rows, out / "summary.csv", out / "summary.json")
        write_case_metrics(rows, out / "case_metrics.csv", out / "case_metrics.json")
        report = render_report(rows, out)
        write_provenance(out, mode)
        per_run.append({
            "run_id": run_dir.name,
            "out_dir": str(out),
            "cases": len(rows),
            "report_html": str(report),
            "charts": sorted(p.name for p in out.glob("*.png")) + sorted(p.name for p in out.glob("*.svg")),
        })
    result: dict[str, Any] = {"renderer": mode, "per_run": per_run}
    if aggregate_out_dir:
        out = Path(aggregate_out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_summary(combined_rows, out / "summary.csv", out / "summary.json")
        write_case_metrics(combined_rows, out / "case_metrics.csv", out / "case_metrics.json")
        report = render_report(combined_rows, out)
        write_provenance(out, mode)
        result["aggregate"] = {
            "out_dir": str(out),
            "runs": sorted({row["run_id"] for row in combined_rows}),
            "cases": len(combined_rows),
            "report_html": str(report),
            "charts": sorted(p.name for p in out.glob("*.png")) + sorted(p.name for p in out.glob("*.svg")),
        }
    return result


def write_provenance(out_dir: str | Path, mode: str) -> Path:
    """Record how the viz evidence was produced (desensitized, no secrets)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "eva-agentic visualize",
        "renderer": mode,
        "plot_env": "eva-viz" if mode == "seaborn" else "eva-agentic-eval",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if mode == "seaborn":
        import matplotlib
        import seaborn
        import numpy
        payload["software"] = {
            "matplotlib": matplotlib.__version__,
            "seaborn": seaborn.__version__,
            "numpy": numpy.__version__,
        }
    path = out / "provenance.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _case_row(run_dir: Path, run_id: str, case: Mapping[str, Any], job: Any, attempt: Mapping[str, Any] | None) -> dict[str, Any]:
    task_index = _task_index(case)
    base = {
        "run_id": run_id,
        "job_id": job.job_id,
        "case_id": case["case_id"],
        "task_id": case["task_id"],
        "task_index": task_index,
        "seed": case["seed"],
        "participant": job.participant,
    }
    if attempt is None:
        return {**base, "attempt": None, "status": "unstarted", "task_success": None,
                "duration_s": None, "termination_reason": None, "error_source": None, "success_source": None,
                **_empty_process_metrics()}
    attempt_status = attempt.get("status")
    if attempt_status != AttemptStatus.COMPLETED.value:
        return {**base, "attempt": attempt.get("attempt_id"),
                "status": OutcomeStatus.INFRASTRUCTURE_FAILURE.value, "task_success": None,
                "duration_s": _duration_s(attempt), "termination_reason": attempt_status,
                "error_source": None, "success_source": None, **_process_metrics(run_dir, job, attempt)}
    result = _matching_result(attempt, case["case_id"])
    if result is None:
        return {**base, "attempt": attempt.get("attempt_id"), "status": OutcomeStatus.INVALID.value,
                "task_success": None, "duration_s": _duration_s(attempt),
                "termination_reason": "missing_native_result", "error_source": None, "success_source": None,
                **_process_metrics(run_dir, job, attempt)}
    return {**base, "attempt": attempt.get("attempt_id"), "status": result.get("status"),
            "task_success": result.get("task_success"), "duration_s": _duration_s(attempt),
            "termination_reason": result.get("termination_reason"),
            "error_source": result.get("error_source"), "success_source": result.get("success_source"),
            **_process_metrics(run_dir, job, attempt, result)}


def _task_index(case: Mapping[str, Any]) -> int | None:
    value = (case.get("initialization") or {}).get("task_id")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _matching_result(attempt: Mapping[str, Any], case_id: str) -> Mapping[str, Any] | None:
    results = attempt.get("results")
    if not isinstance(results, list):
        return None
    for result in results:
        if isinstance(result, dict) and result.get("case_id") == case_id:
            return result
    return None


def _duration_s(attempt: Mapping[str, Any]) -> float | None:
    value = (attempt.get("process") or {}).get("duration_s")
    return float(value) if isinstance(value, (int, float)) else None


def _empty_process_metrics() -> dict[str, Any]:
    keys = CASE_METRIC_FIELDS[len(SUMMARY_FIELDS):]
    return {key: None for key in keys}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _process_metrics(
    run_dir: Path,
    job: Any,
    attempt: Mapping[str, Any],
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract behavioural evidence without changing the 14-field summary contract."""
    metrics = _empty_process_metrics()
    process = attempt.get("process") if isinstance(attempt.get("process"), dict) else {}
    attempt_id = attempt.get("attempt_id")
    attempt_dir = run_dir / "participants" / job.participant / "jobs" / job.job_id / "attempts" / f"{int(attempt_id or 0):04d}"
    native = attempt_dir / "native"
    transcripts = sorted(native.glob("transcript_*.json"))
    transcript = _read_json(transcripts[0]) if transcripts else {}
    stats = transcript.get("stats") if isinstance(transcript.get("stats"), dict) else {}
    finish = transcript.get("finish") if isinstance(transcript.get("finish"), dict) else {}

    metrics.update({
        "exit_code": process.get("exit_code"),
        "model_name": transcript.get("model"),
        "planner_runtime": stats.get("planner_runtime"),
        "planner_rounds": stats.get("model_requests", stats.get("turns_used")),
        "llm_calls": stats.get("model_requests", stats.get("turns_used")),
        "tool_calls": stats.get("tool_calls"),
        "tool_errors": stats.get("tool_execution_errors"),
        "loop_feedbacks": stats.get("loop_feedbacks"),
        "proposal_repairs": stats.get("proposal_repairs"),
        "planner_loop_stopped": stats.get("planner_loop_stopped"),
        "input_tokens": stats.get("total_input_tokens"),
        "output_tokens": stats.get("total_output_tokens"),
        "planner_elapsed_s": stats.get("model_elapsed_s"),
        "finish_status": finish.get("status"),
    })

    tool_counts: Counter[str] = Counter()
    messages = transcript.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("tool_calls"), list):
                continue
            for call in message["tool_calls"]:
                name = (((call or {}).get("function") or {}).get("name"))
                if name:
                    tool_counts[name] += 1
    metrics["vla_calls"] = sum(count for name, count in tool_counts.items() if name.startswith("pi0_"))
    metrics["sam3_calls"] = sum(count for name, count in tool_counts.items() if name in {"segment", "back_project"})

    states = _read_json(native / "states.json")
    metrics["state_records"] = len(states["steps"]) if isinstance(states.get("steps"), list) else None

    if result:
        native_metrics = result.get("metrics")
        if isinstance(native_metrics, dict) and metrics.get("exit_code") is None:
            metrics["exit_code"] = native_metrics.get("native_exit_code")

    if finish.get("summary") and (result is None or result.get("task_success") is not True):
        metrics["failure_reason"] = finish.get("summary")
    elif result and result.get("termination_reason"):
        metrics["failure_reason"] = result.get("termination_reason")
    elif result and result.get("error_source"):
        metrics["failure_reason"] = result.get("error_source")
    elif stats.get("planner_loop_stopped"):
        metrics["failure_reason"] = (
            f"planner loop guard stopped ({stats.get('tool_execution_errors', 0)} tool errors, "
            f"{stats.get('loop_feedbacks', 0)} loop feedbacks)"
        )
    return metrics


# --------------------------------------------------------------------------- #
# Chart rendering: seaborn when available, otherwise dependency-free SVG
# --------------------------------------------------------------------------- #
def rendering_mode() -> str:
    """Return the chart renderer in effect for this interpreter ("seaborn"|"svg")."""
    return "seaborn" if _seaborn_available() else "svg"


def _seaborn_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import seaborn  # noqa: F401
        return True
    except Exception:
        return False


def render_charts(rows: Sequence[Mapping[str, Any]], out_dir: str | Path) -> dict[str, Path]:
    """Render one chart per metric into ``out_dir``; returns a {chart_name: path} map.

    Uses matplotlib + seaborn when present (dedicated ``eva-viz`` env), falling back to
    pure-Python SVG so the dependency-free evaluator venv keeps working.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if _seaborn_available():
        return _render_charts_seaborn(rows, directory)
    charts = {
        "success_rate.svg": _success_rate_svg(rows),
        "status_heatmap.svg": _heatmap_svg(rows),
        "duration.svg": _duration_svg(rows),
    }
    paths: dict[str, Path] = {}
    for name, (width, height, body) in charts.items():
        path = directory / name
        path.write_text(_svg_document(width, height, body), encoding="utf-8")
        paths[name] = path
    return paths


def _render_charts_seaborn(rows: Sequence[Mapping[str, Any]], directory: Path) -> dict[str, Path]:
    """Render high-quality charts via matplotlib + seaborn (reference-styled)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=["#4c78a8", "#c44e52", "#d8a047", "#7f8794", "#8172b3", "#6b9e78"],
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.18,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "lines.linewidth": 0.9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        },
    )
    paths, records = _render_stresskit(rows, directory)
    _save_figure_manifest(directory, records)
    plt.close("all")
    return paths


def _drawing_defaults() -> dict[str, Any]:
    """Automatic palette taken from the active matplotlib cycle (no hardcoded theme)."""
    import matplotlib as mpl
    from matplotlib.colors import to_rgb, LinearSegmentedColormap
    cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    paper = to_rgb(mpl.rcParams["axes.facecolor"])
    ink = to_rgb(mpl.rcParams["text.color"])
    def mix(a, b, w):
        return tuple((1 - w) * x + w * y for x, y in zip(a, b))
    base = to_rgb(cycle[0])
    cmap = LinearSegmentedColormap.from_list(
        "academic_rate", [mix(paper, base, .045), mix(paper, base, .48), base]
    )
    status_colors = [mix(paper, to_rgb(cycle[i]), .62) for i in range(len(STATUS_LEGEND_ORDER))]
    return {"paper": paper, "ink": ink, "soft": mix(paper, ink, .045), "line": mix(paper, ink, .18),
            "cmap": cmap, "status_colors": status_colors, "method_colors": cycle}


def _save_fig(fig, stem: Path, pdf, records: list, name: str, kind: str, **extra: Any) -> Path:
    """Export one figure to PNG+SVG+PDF (all from the same Figure, no recompute) and to the PDF bundle."""
    import matplotlib.pyplot as plt
    fig.canvas.draw()
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix("." + ext), dpi=300)
    pdf.savefig(fig)
    records.append({"name": name, "kind": kind, "axes": len(fig.axes),
                    "size_inches": list(fig.get_size_inches()), **extra})
    plt.close(fig)
    return stem.with_suffix(".png")


def _header_footer(fig, title: str, subtitle: str, footer: str, synthetic: bool = False) -> None:
    fig.text(.045, .968, title, fontsize=9.8, weight="bold", va="top")
    fig.text(.965, .965, "SYNTHETIC DATA" if synthetic else "EVA-AGENTIC DATA", fontsize=6.4,
             ha="right", va="top", alpha=.62)
    fig.text(.045, .922, subtitle, fontsize=6.8, va="top", alpha=.68)
    fig.text(.045, .024, footer, fontsize=6.0, va="bottom", linespacing=1.35, alpha=.72)


def _render_stresskit(rows: Sequence[Mapping[str, Any]], directory: Path) -> tuple[dict[str, Path], list[dict]]:
    """Reference-styled plotting pipeline over real rows; returns (path map, manifest records)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    style = _drawing_defaults()
    metrics = aggregate_metrics(rows)
    paths: dict[str, Path] = {}
    records: list[dict] = []
    pdf = PdfPages(directory / "all_figures.pdf", metadata={"Title": "eva-agentic visualization", "Author": "eva-agentic"})
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)

    # --- 01 suite overview (planned-trial success yield per suite x participant) ---
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(left=.20, right=.97, bottom=.16, top=.86)
    suites = sorted({m["suite_id"] for m in metrics["suites"]})
    participants = sorted({m["participant"] for m in metrics["suites"]})
    sm = {(m["suite_id"], m["participant"]): m for m in metrics["suites"]}
    _header_footer(fig, "Suite-level planned success",
                   f"{len(suites)} suite(s)  /  {len(participants)} participant(s)  /  task-balanced within each cell",
                   "Cell: task-wise confirmed successes / planned trials. Smaller text: valid-outcome coverage. "
                   "P = pending (no valid outcome, not measured 0%).", synthetic=False)
    cmap, norm = style["cmap"], plt.Normalize(0, 100)
    for j, p in enumerate(participants):
        ax.text(j + .5, len(suites) + .35, p, ha="center", va="center", weight="bold", fontsize=8)
    for i, s in enumerate(suites):
        ax.text(-.08, len(suites) - i - .5, s, ha="right", va="center", fontsize=8)
        for j, p in enumerate(participants):
            m = sm.get((s, p))
            if m is None:
                ax.add_patch(Rectangle((j, len(suites) - i - 1), .9, .88, facecolor=style["soft"], edgecolor=style["line"], lw=.35))
                ax.text(j + .45, len(suites) - i - .56, "N/S", ha="center", va="center", fontsize=7.5)
                continue
            pending = m["valid"] == 0
            if pending:
                ax.add_patch(Rectangle((j, len(suites) - i - 1), .9, .88, facecolor=style["soft"], edgecolor=style["line"],
                                       lw=.35, hatch=".."))
                ax.text(j + .45, len(suites) - i - .56, "P", ha="center", va="center", fontsize=9)
                continue
            val = m["yield_pct"]
            fg = style["paper"] if val >= 68 else style["ink"]
            ax.add_patch(Rectangle((j, len(suites) - i - 1), .9, .88, facecolor=cmap(norm(val)), linewidth=0))
            ax.text(j + .45, len(suites) - i - .60, f"{val:.1f}", ha="center", va="center", fontsize=10, color=fg)
            ax.text(j + .45, len(suites) - i - .18, f"valid {m['valid_pct']:.0f}%", ha="center", va="center", fontsize=6.6, color=fg)
    ax.set_xlim(-.5, len(participants)); ax.set_ylim(-.5, len(suites) + .8); ax.axis("off")
    paths["00_suite_overview"] = _save_fig(fig, directory / "00_suite_overview", pdf, records, "00_suite_overview", "overview",
                                           suites=len(suites), participants=len(participants))

    # --- 02 task x seed diagnostic matrix per (suite, participant) ---
    for key, group in sorted(groups.items()):
        suite, participant = key
        idx: dict[str, dict[int, dict[str, Any]]] = {}
        for r in group:
            tid = str(r.get("task_id") or "")
            idx.setdefault(tid, {})
            idx[tid][int(r.get("seed") or 0)] = r
        task_ids = sorted(idx, key=lambda t: _task_sort_key(t))
        seeds = sorted({int(r.get("seed") or 0) for r in group})
        codes = {s: i for i, s in enumerate(STATUS_LEGEND_ORDER)}
        matrix = np.array([[codes.get(idx.get(t, {}).get(sd, {}).get("status"), codes["unstarted"]) for sd in seeds] for t in task_ids])
        fig, ax = plt.subplots(figsize=(7.2, max(3.2, .55 * max(len(task_ids), 1) + 2)))
        fig.subplots_adjust(left=.22, right=.96, bottom=.18, top=.86)
        _header_footer(fig, "Run-level outcome audit",
                       f"{suite}  /  participant {participant}  /  selected terminal attempts",
                       "Columns are seed IDs. Each glyph has a fixed discrete color mapping; unresolved outcomes (I/V) "
                       "stay distinct from task failure (F).", synthetic=False)
        cmap_l = _fixed_cmap(style["status_colors"])
        ax.imshow(matrix, cmap=cmap_l, vmin=-.5, vmax=len(STATUS_LEGEND_ORDER) - .5, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(seeds)), [f"s{d}" for d in seeds])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(task_ids)), [_short_task(t) for t in task_ids])
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, len(seeds), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(task_ids), 1), minor=True)
        ax.grid(which="minor", lw=1.1, color=style["paper"])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                status = STATUS_LEGEND_ORDER[int(matrix[i, j])]
                ax.text(j, i, GLYPHS[status], fontsize=7.5, ha="center", va="center")
        for sp in ax.spines.values():
            sp.set_visible(False)
        # legend
        from matplotlib.patches import Patch
        present_patch_count = {STATUS_LEGEND_ORDER[int(v)]: 0 for v in np.unique(matrix)}
        handles = [Patch(facecolor=style["status_colors"][i], label=f"{GLYPHS[s]}  {STATUS_NAMES[s]}")
                   for i, s in enumerate(STATUS_LEGEND_ORDER) if s in present_patch_count]
        fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(.5, .05),
                   frameon=False, fontsize=7.1, handlelength=1.2, columnspacing=2.5)
        paths[f"01_matrix_{suite}__{participant}"] = _save_fig(
            fig, directory / f"01_matrix_{suite}__{participant}", pdf, records, "01_matrix",
            "diagnostic", suite_id=suite, participant=participant, shape=list(matrix.shape))

    # --- 03 duration stratified per (suite, participant), never a cross-suite ranking ---
    if metrics["suites"]:
        fig, ax = plt.subplots(figsize=(7.2, max(2.6, .6 * len(metrics["suites"]) + 1.2)))
        fig.subplots_adjust(left=.24, right=.72, bottom=.20, top=.80)
        _header_footer(fig, "Recorded process duration",
                       "one protocol and resource profile per cell",
                       "Dot = median; thick interval = 25th-75th percentile; thin = 10th-90th (not confidence intervals). "
                       "Includes all final-attempt durations regardless of outcome; excludes unstarted.",
                       synthetic=False)
        for i, m in enumerate(sorted(metrics["suites"], key=lambda x: (x["suite_id"], x["participant"]))):
            q = m["duration_quantiles"]
            if not any(q):
                ax.text(.52, i, "Not scheduled", transform=ax.get_yaxis_transform(), va="center", fontsize=7.4)
                continue
            p10, q1, med, q3, p90 = q
            col = style["method_colors"][i % len(style["method_colors"])]
            ax.plot([p10, p90], [i, i], lw=.85, color=col)
            ax.plot([q1, q3], [i, i], lw=4.7, solid_capstyle="butt", alpha=.44, color=col)
            ax.scatter([med], [i], s=22, color=col, zorder=4)
            ax.text(1.035, i, f"{med:.1f} s   (n={m['duration_n']:,})", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.3)
        ax.set_yticks(range(len(metrics["suites"])), [f"{m['suite_id']}/{m['participant']}" for m in
                    sorted(metrics["suites"], key=lambda x: (x["suite_id"], x["participant"]))])
        ax.set_ylim(len(metrics["suites"]) - .65, -.65)
        ax.set_xlabel("Selected-attempt process duration (s)"); ax.set_xlim(left=0, right=max(1, ax.get_xlim()[1]))
        ax.grid(axis="x", alpha=.18, lw=.4); ax.set_axisbelow(True); ax.tick_params(length=2.2)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        paths["02_duration"] = _save_fig(fig, directory / "02_duration", pdf, records, "02_duration", "duration")

    # --- 04 planned-trial outcome composition (all statuses, complete denominator) ---
    if metrics["suites"]:
        cells = sorted(metrics["suites"], key=lambda x: (x["suite_id"], x["participant"]), reverse=True)
        fig, ax = plt.subplots(figsize=(7.2, max(3.0, .62 * len(cells) + 1.4)))
        fig.subplots_adjust(left=.30, right=.97, bottom=.20, top=.86)
        _header_footer(fig, "Planned-trial outcome composition",
                       "each bar = all planned trials for one suite-participant cell",
                       "A complete denominator is retained: success, task failure, timeout, infra, invalid, unstarted.",
                       synthetic=False)
        for i, m in enumerate(cells):
            ax.text(-1, i + .5, f"{m['suite_id']}/{m['participant']}", ha="right", va="center", fontsize=7)
            if not m["planned"]:
                ax.add_patch(Rectangle((0, i), 100, .68, facecolor=style["soft"], edgecolor=style["line"], lw=.3, hatch="//"))
                ax.text(50, i + .34, "No planned trials", ha="center", va="center", fontsize=7)
                continue
            start = 0.0
            for status, color in zip(STATUS_LEGEND_ORDER, style["status_colors"]):
                v = 100.0 * m["status_counts"].get(status, 0) / m["planned"]
                ax.add_patch(Rectangle((start, i), v, .68, facecolor=color, linewidth=0))
                if v >= 12:
                    ax.text(start + v / 2, i + .34, f"{v:.0f}", ha="center", va="center", fontsize=6.9)
                start += v
        ax.set_xlim(-1, 102); ax.set_ylim(len(cells) + .4, -.8); ax.set_yticks([])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Share of planned trials (%)"); ax.tick_params(axis="x", length=2.5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=style["status_colors"][i], label=STATUS_NAMES[s])
                   for i, s in enumerate(STATUS_LEGEND_ORDER)]
        fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(.5, .06),
                   frameon=False, fontsize=7, handlelength=1.2, columnspacing=2)
        paths["03_outcomes"] = _save_fig(fig, directory / "03_outcomes", pdf, records, "03_outcomes", "outcomes", cells=len(cells))

    # --- 05 process evidence: planner rounds and native tool-call load per case ---
    process_rows = sorted(
        [row for row in rows if isinstance(row.get("planner_rounds"), (int, float))],
        key=lambda r: (suite_of(r), str(r.get("participant")), _task_sort_key(r.get("task_id")), int(r.get("seed") or 0)),
    )
    if process_rows:
        n = len(process_rows)
        fig, ax = plt.subplots(figsize=(7.2, max(3.0, .43 * n + 1.6)))
        fig.subplots_adjust(left=.26, right=.90, bottom=.18, top=.86)
        _header_footer(fig, "Planner rounds and tool-call load",
                       "one selected terminal attempt per row; native planner metrics",
                       "Planner rounds are model requests reported by RPent. Tool calls are planner-requested tools; "
                       "the right-hand label also exposes tool errors. Missing metrics are not imputed as zero.",
                       synthetic=False)
        planner_color = style["method_colors"][0]
        tool_color = style["method_colors"][1]
        ys = list(range(n))
        for i, row in enumerate(process_rows):
            rounds = float(row.get("planner_rounds") or 0)
            tools = float(row.get("tool_calls") or 0)
            ax.barh(i - .18, rounds, height=.30, color=planner_color, alpha=.92, linewidth=0)
            ax.barh(i + .18, tools, height=.30, color=tool_color, alpha=.82, linewidth=0)
            ax.text(rounds + max(rounds, tools) * .025 + .25, i - .18, f"{rounds:.0f}",
                    va="center", fontsize=6.4, color=style["ink"])
            label = (f"L{rounds:.0f} / T{tools:.0f} / E{row.get('tool_errors') or 0}")
            ax.text(1.015, i, label, transform=ax.get_yaxis_transform(), va="center",
                    fontsize=6.5, color=style["ink"], alpha=.82)
        max_value = max(max(float(r.get("planner_rounds") or 0), float(r.get("tool_calls") or 0)) for r in process_rows)
        ax.set_yticks(ys, [f"{_short_task(r.get('task_id'))} s{r.get('seed')}" for r in process_rows], fontsize=6.8)
        ax.set_ylim(n - .45, -.55)
        ax.set_xlim(0, max(4, max_value * 1.18))
        ax.set_xlabel("Count per selected attempt")
        ax.grid(axis="x", alpha=.14, lw=.4)
        ax.tick_params(length=2.0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=planner_color, label="Planner rounds / LLM calls"),
                   Patch(facecolor=tool_color, label="Tool calls")]
        fig.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(.55, .055),
                   frameon=False, fontsize=6.8, handlelength=1.1, columnspacing=1.8)
        paths["05_process_metrics"] = _save_fig(fig, directory / "05_process_metrics", pdf, records,
                                                "05_process_metrics", "process", cases=n)

    # --- 05 task pages (fixed 0-100 scale, paginated) when a suite has many tasks ---
    task_metrics = sorted(metrics["tasks"], key=lambda t: (t["suite_id"], t["participant"], _task_sort_key(t["task_id"] or "")))
    page_groups: dict[str, list[dict[str, Any]]] = {}
    for t in task_metrics:
        page_groups.setdefault((t["suite_id"], t["participant"]), []).append(t)
    for gkey, items in sorted(page_groups.items()):
        ids = [t["task_id"] for t in items]
        pages = pages_for(ids)
        for pi, page in enumerate(pages):
            page_items = {str(i.get("task_id")): i for i in items}
            n = len(page)
            fig, ax = plt.subplots(figsize=(7.2, 2.12 + .151 * n))
            fig.subplots_adjust(left=.12, right=.92, bottom=.22 if n <= 12 else .16, top=.78 if n <= 12 else .84)
            _header_footer(fig, "Task-level planned success",
                           f"{gkey[0]}  /  {gkey[1]}  /  page {pi + 1}/{len(pages)}",
                           "0-100% scale identical on every page. * = unresolved trials; P = no valid outcome (not measured 0%).",
                           synthetic=False)
            for i, tid in enumerate(page):
                m = page_items.get(str(tid))
                if m is None:
                    val, valid_pct, unknown = 0.0, 0.0, 0
                else:
                    val, valid_pct, unknown = m["yield_pct"], m["valid_pct"], m["unknown"]
                if m is not None and m["valid"] == 0:
                    ax.add_patch(Rectangle((0, i), .94, .87, facecolor=style["soft"], edgecolor=style["line"], lw=.35, hatch=".."))
                    ax.text(.47, i + .43, "P", ha="center", va="center", fontsize=9)
                    continue
                if m is None:
                    ax.add_patch(Rectangle((0, i), .94, .87, facecolor=style["soft"], edgecolor=style["line"], lw=.35, hatch="//"))
                    ax.text(.47, i + .43, "N/S", ha="center", va="center", fontsize=8)
                    continue
                fg = style["paper"] if val >= 68 else style["ink"]
                ax.add_patch(Rectangle((0, i), .94, .87, facecolor=cmap(norm(val)), linewidth=0))
                ax.text(.47, i + .43, f"{val:.0f}{'*' if unknown else ''}", ha="center", va="center",
                        fontsize=8.2, color=fg)
            ax.set_ylim(n + .9, -.8); ax.set_xlim(-.05, 1.0); ax.axis("off")
            paths[f"04_tasks_{gkey[0]}__{gkey[1]}__p{pi + 1}"] = _save_fig(
                fig, directory / f"04_tasks_{gkey[0]}__{gkey[1]}__p{pi + 1}", pdf, records,
                "04_tasks", "task_page", suite_id=gkey[0], participant=gkey[1], page=pi + 1, task_count=n)
    pdf.close()
    return paths, records


def _fixed_cmap(colors) -> "Any":
    from matplotlib.colors import ListedColormap
    return ListedColormap(colors)


def _task_sort_key(task_id: Any) -> tuple:
    text = str(task_id)
    if ":" in text:
        index = text.split(":", 1)[1]
        if index.isdigit():
            return (0, int(index))
    return (1, text)


def _save_figure_manifest(directory: Path, records: list[dict]) -> None:
    write_json_obj = {"figures": records}
    (directory / "figure_manifest.json").write_text(
        json.dumps(write_json_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _blank_png(message: str, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def render_report(rows: Sequence[Mapping[str, Any]], out_dir: str | Path) -> Path:
    """Write a standalone interactive dashboard with charts and one master table."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    charts = render_charts(rows, directory)
    metrics = aggregate_metrics(rows)

    master_fields = [
        ("suite", "Suite"), ("task_id", "Task"), ("seed", "Seed"), ("participant", "Participant"),
        ("status", "Status"), ("task_success", "Success"), ("duration_s", "Duration (s)"),
        ("planner_rounds", "Planner rounds"), ("llm_calls", "LLM calls"), ("tool_calls", "Tool calls"),
        ("tool_errors", "Tool errors"), ("loop_feedbacks", "Loop feedbacks"), ("finish_status", "Finish"),
        ("failure_reason", "Failure reason"), ("exit_code", "Exit"), ("model_name", "Model"),
        ("attempt", "Attempt"), ("success_source", "Success source"),
    ]
    head = "".join(f'<th data-key="{key}">{label}</th>' for key, label in master_fields)
    body_table = "".join(
        "<tr>" + "".join(f'<td data-key="{key}">{_escape(_cell(suite_value(row, key)))}</td>' for key, _ in master_fields) + "</tr>"
        for row in rows
    )
    chart_blocks = "".join(
        f'<section><h3>{_escape(name)}</h3><img src="{path.name}" alt="{_escape(name)}"/></section>'
        for name, path in charts.items()
    )
    legend = "".join(
        f'<span class="legend-item"><span class="dot" style="background:{_escape(STATUS_COLORS[s])}"></span>'
        f'{_escape(GLYPHS[s])} {_escape(STATUS_NAMES[s])}</span>'
        for s in STATUS_LEGEND_ORDER
    )
    suite_rows = "".join(
        "<tr>" + "".join(f"<td>{_escape(_cell(m.get(col)))}</td>" for col in
                         ("suite_id", "participant", "planned", "successes", "valid", "unknown",
                          "yield_pct", "valid_pct", "upper_pct", "duration_n")) + "</tr>"
        for m in metrics["suites"]
    )
    suite_options = "".join(f'<option value="{_escape(s)}">{_escape(s)}</option>' for s in sorted({suite_of(row) for row in rows}))
    payload = {
        "rows": [{field: r.get(field) for field, _ in master_fields} for r in rows],
        "tasks": metrics["tasks"], "suites": metrics["suites"],
        "status_order": list(STATUS_LEGEND_ORDER), "colors": [STATUS_COLORS[s] for s in STATUS_LEGEND_ORDER],
        "glyphs": [GLYPHS[s] for s in STATUS_LEGEND_ORDER],
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>eva-agentic viz</title>
<style>
:root{{font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;color:#20272e;background:#f7f8fa}}
*{{box-sizing:border-box}}body{{max-width:1280px;margin:32px auto;padding:0 28px 72px}}
h1{{font-size:25px;letter-spacing:-.65px;margin:0}}h2{{font-size:17px;margin:34px 0 10px}}h3{{font-size:14px}}
p{{line-height:1.65}}.eyebrow{{font-size:11px;font-weight:750;letter-spacing:1.35px;color:#68777f;margin-bottom:8px}}
.muted{{color:#636f78;font-size:12.5px}}header{{padding-bottom:20px;border-bottom:1px solid #dce0e4}}
section{{margin-top:34px}}img{{max-width:100%;height:auto;background:white;border:1px solid #e2e5e8;border-radius:4px}}
table{{border-collapse:separate;border-spacing:0;width:100%;font-size:12px;background:white;border:1px solid #e0e4e8}}
th{{position:sticky;top:0;background:#eef1f4;color:#39444c;font-weight:650;text-align:center;padding:8px 7px;border-bottom:1px solid #d5dade;cursor:pointer;white-space:nowrap}}
td{{text-align:center;padding:7px 7px;border-bottom:1px solid #eef0f2;vertical-align:top;max-width:420px}}
tbody tr:hover td{{background:#f6f8fa}}.scroll{{overflow:auto;max-height:680px;border:1px solid #e0e4e8;border-radius:5px;background:white}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;margin:14px 0}}.legend-item{{display:inline-flex;align-items:center;gap:6px}}
.dot{{width:14px;height:14px;border-radius:3px;display:inline-block}}
.controls{{display:flex;gap:10px;align-items:center;margin:12px 0;flex-wrap:wrap}}select,input{{padding:6px 8px;border:1px solid #cfd5da;border-radius:4px;background:white}}
.note{{padding:13px 16px;background:#eef1f4;border-left:3px solid #68777f;line-height:1.65}}.footer{{margin-top:35px;color:#6a757d;font-size:12px}}
</style></head><body>
<header><div class="eyebrow">EVA-AGENTIC / EVALUATION EVIDENCE</div><h1>Run visualization</h1>
<p class="muted">{len(rows)} planned case(s). The CSV/JSON artifacts in this directory are authoritative; the table is view-only.</p></header>
<section><h2>Outcome legend and coverage</h2><div class="legend">{legend}</div>
<div class="note">Coverage: <b>valid</b> = success + task_failure + timeout (decisive native outcome); <b>unknown</b> =
infrastructure / invalid / unstarted (does not establish success or failure). A planned task with zero valid outcomes is
shown as <b>P</b>, not as a measured 0%. Different suites are never merged into a single score.</div></section>
<section><h2>Master case table</h2>
<div class="controls"><label>Suite <select id="suite"><option value="">All</option>{suite_options}</select></label>
<label>Status <select id="status"><option value="">All</option>{"".join(f'<option value="{s}">{STATUS_NAMES[s]}</option>' for s in STATUS_LEGEND_ORDER)}</select></label>
<input id="search" placeholder="Search task / reason / participant"></div>
<div class="scroll"><table id="master"><thead><tr>{head}</tr></thead><tbody>{body_table}</tbody></table></div></section>
{chart_blocks}
<section><h2>Suite metrics</h2><div class="scroll"><table><thead><tr>
<th>suite</th><th>participant</th><th>planned</th><th>success</th><th>valid</th><th>unknown</th>
<th>yield%</th><th>valid%</th><th>upper%</th><th>dur n</th></tr></thead>
<tbody>{suite_rows or "<tr><td colspan='10'>no suites</td></tr>"}</tbody></table></div></section>
<script id="payload" type="application/json">{payload_json}</script>
""" + r"""<script>
const table=document.getElementById('master'),tbody=table.tBodies[0];
const rows=[...tbody.rows];let sortKey=null,asc=true;
document.getElementById('suite').addEventListener('change',filter);
document.getElementById('status').addEventListener('change',filter);
document.getElementById('search').addEventListener('input',filter);
table.querySelectorAll('th').forEach((th,index)=>th.addEventListener('click',()=>{const key=th.dataset.key;if(sortKey===key)asc=!asc;else{sortKey=key;asc=true}rows.sort((a,b)=>compare(a.children[index],b.children[index]));if(!asc)rows.reverse();tbody.append(...rows);table.querySelectorAll('th').forEach(x=>x.classList.remove('sorted'));th.classList.add('sorted')}));
function filter(){const suite=document.getElementById('suite').value,status=document.getElementById('status').value,q=document.getElementById('search').value.toLowerCase();for(const row of rows){const cells=Object.fromEntries([...row.children].map(c=>[c.dataset.key,c.textContent.toLowerCase()]));row.hidden=Boolean((suite&&cells.suite!==suite)||(status&&cells.status!==status)||(q&&!Object.values(cells).join(' ').includes(q)))}}
function compare(a,b){const av=a.textContent.trim(),bv=b.textContent.trim(),an=parseFloat(av),bn=parseFloat(bv);if(!isNaN(an)&&!isNaN(bn))return an-bn;return av.localeCompare(bv)}
</script>
<div class="footer">Duration is selected-attempt process duration, not pure inference time. Missing process metrics display “None”; they are not imputed as zero.</div>
</body></html>
"""
    path = directory / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def suite_value(row: Mapping[str, Any], field: str) -> Any:
    return suite_of(row) if field == "suite" else row.get(field)

def _svg_document(width: int, height: int, body: str) -> str:
    marks = body.split("<chartbox ")
    if len(marks) == 2 and marks[1].endswith("/>"):
        width = int(marks[1][:-2])
        body = marks[0]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">{body}</svg>\n'
    )


def _axes(left: float, bottom: float, chart_height: float, log: bool = False) -> str:
    y_ticks = []
    if log:
        for power in range(0, 4):
            value = 10.0 ** power * 0.1
            y_ticks.append(value)
        labels = ["0.1", "1", "10", "100"]
    else:
        y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        labels = ["0", "25%", "50%", "75%", "100%"]
    parts: list[str] = []
    for frac, label in zip(y_ticks, labels):
        y = bottom - frac * chart_height
        parts.append(f'<line x1="{left - 6}" y1="{y:.0f}" x2="{left}" y2="{y:.0f}" stroke="#999"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 3:.0f}" text-anchor="end" font-size="10">{label}</text>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{left}" y2="{bottom - chart_height}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{left + 560}" y2="{bottom}" stroke="#333"/>')
    return "".join(parts)


def _success_rate_svg(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, str]:
    by_task: dict[Any, list[bool | None]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row["task_success"])
    task_order = sorted(by_task, key=lambda t: (t is not None, str(t)))
    bar_width = 90
    gap = 70
    left, bottom, chart_height = 100, 300, 230
    width = left + max(len(task_order), 1) * (bar_width + gap) - gap + 30
    parts = [_axes(left, bottom, chart_height)]
    x = left
    for task in task_order:
        values = by_task[task]
        total = len(values)
        rate = sum(1 for v in values if v is True) / total if total else 0.0
        h = rate * chart_height
        color = "#2e8b57" if total else STATUS_COLORS["invalid"]
        parts.append(f'<rect x="{x}" y="{bottom - h:.0f}" width="{bar_width}" height="{h:.0f}" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{x + bar_width / 2:.0f}" y="{bottom - h - 7:.0f}" text-anchor="middle" font-size="13">{rate:.0%}</text>')
        parts.append(f'<text x="{x + bar_width / 2:.0f}" y="{bottom + 20}" text-anchor="middle" font-size="11">{_escape(_short_task(task))}</text>')
        parts.append(f'<text x="{x + bar_width / 2:.0f}" y="{bottom + 36}" text-anchor="middle" font-size="10" fill="#666">{total} case(s)</text>')
        x += bar_width + gap
    return width, 330, '<text x="12" y="24" font-size="16" font-weight="bold">per-task success rate</text>' + "".join(parts)


def _heatmap_svg(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, str]:
    by_task: dict[Any, dict[int, str]] = {}
    for row in rows:
        key = row["task_index"] if row["task_index"] is not None else row["task_id"]
        by_task.setdefault(key, {})[row["seed"]] = row["status"] or "unstarted"
    task_order = sorted(by_task, key=lambda t: (isinstance(t, int), t))
    seed_order = sorted({seed for seeds in by_task.values() for seed in seeds})
    cell = 70
    x0, y0 = 120, 60
    width = x0 + max(len(seed_order), 1) * cell + 40
    height = y0 + max(len(task_order), 1) * cell + 70
    parts: list[str] = []
    for j, seed in enumerate(seed_order):
        parts.append(f'<text x="{x0 + j * cell + cell / 2:.0f}" y="{y0 - 16}" text-anchor="middle" font-size="12">seed {seed}</text>')
    legend_y = y0 + len(task_order) * cell + 40
    lx = x0
    legend_statuses = ["success", "task_failure", "timeout", "infrastructure_failure", "invalid", "unstarted"]
    for status in legend_statuses:
        color = STATUS_COLORS[status]
        parts.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{legend_y + 12}" font-size="11">{_escape(status)}</text>')
        lx += 120
    for i, task in enumerate(task_order):
        y = y0 + i * cell
        parts.append(f'<text x="{x0 - 10}" y="{y + cell / 2 + 4}" text-anchor="end" font-size="12">{_escape(str(task))}</text>')
        for j, seed in enumerate(seed_order):
            status = by_task[task].get(seed, "unstarted")
            color = STATUS_COLORS.get(status, STATUS_COLORS["unstarted"])
            x = x0 + j * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 6}" height="{cell - 6}" fill="{color}" rx="5"/>')
            parts.append(f'<text x="{x + (cell - 6) / 2:.0f}" y="{y + (cell - 6) / 2 + 4}" text-anchor="middle" font-size="10">{_escape(status)}</text>')
    title = '<text x="12" y="24" font-size="16" font-weight="bold">task \u00d7 seed status</text>'
    return width, height, title + "".join(parts)


def _duration_svg(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, str]:
    durations = [row["duration_s"] for row in rows if isinstance(row["duration_s"], (int, float))]
    left, bottom, chart_height = 100, 300, 230
    parts = [_axes(left, bottom, chart_height, log=True)]
    if not durations:
        parts.append(f'<text x="{left + 200}" y="{bottom - 60}" text-anchor="middle" font-size="13">no durations recorded</text>')
        return 500, 330, '<text x="12" y="24" font-size="16" font-weight="bold">per-case duration</text>' + "".join(parts)
    max_dur = max(durations)
    width = 22
    gap = 16
    x = left
    for value in sorted(durations):
        frac = math.log(value + 1) / math.log(max_dur + 1) if max_dur else 0
        h = frac * chart_height
        parts.append(f'<rect x="{x}" y="{bottom - h:.0f}" width="{width}" height="{h:.0f}" fill="#4a7bb5" rx="2"/>')
        parts.append(f'<text x="{x + width / 2:.0f}" y="{bottom - h - 6:.0f}" font-size="9" text-anchor="middle">{value:.1f}s</text>')
        x += width + gap
    return int(x + 30), 330, '<text x="12" y="24" font-size="16" font-weight="bold">per-case duration (log scale)</text>' + "".join(parts)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _short_task(task: Any) -> str:
    text = str(task)
    return text.rsplit(":", 1)[-1] if ":" in text else text


def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _cell(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
