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
    "success": "#2e8b57",
    "task_failure": "#cd5c5c",
    "timeout": "#d9a404",
    "infrastructure_failure": "#7a869b",
    "invalid": "#e0e0e0",
    "unstarted": "#f4f4f4",
}


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
            rows.append(_case_row(run_id, case.to_dict(), job, attempt))
    return rows


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
    json_path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")





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
    path = out / "provenance.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _case_row(run_id: str, case: Mapping[str, Any], job: Any, attempt: Mapping[str, Any] | None) -> dict[str, Any]:
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
                "duration_s": None, "termination_reason": None, "error_source": None, "success_source": None}
    attempt_status = attempt.get("status")
    if attempt_status != AttemptStatus.COMPLETED.value:
        return {**base, "attempt": attempt.get("attempt_id"),
                "status": OutcomeStatus.INFRASTRUCTURE_FAILURE.value, "task_success": None,
                "duration_s": _duration_s(attempt), "termination_reason": attempt_status,
                "error_source": None, "success_source": None}
    result = _matching_result(attempt, case["case_id"])
    if result is None:
        return {**base, "attempt": attempt.get("attempt_id"), "status": OutcomeStatus.INVALID.value,
                "task_success": None, "duration_s": _duration_s(attempt),
                "termination_reason": "missing_native_result", "error_source": None, "success_source": None}
    return {**base, "attempt": attempt.get("attempt_id"), "status": result.get("status"),
            "task_success": result.get("task_success"), "duration_s": _duration_s(attempt),
            "termination_reason": result.get("termination_reason"),
            "error_source": result.get("error_source"), "success_source": result.get("success_source")}


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
    """Render high-quality PNG charts via matplotlib + seaborn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pandas import DataFrame

    sns.set_theme(style="whitegrid", palette="deep")
    frame = _rows_frame(rows)
    paths = {
        "success_rate.png": _sn_success_rate(frame, directory),
        "status_heatmap.png": _sn_heatmap(frame, directory),
        "duration.png": _sn_duration(frame, directory),
    }
    plt.close("all")
    return {name: path for name, path in paths.items() if path is not None}


def _rows_frame(rows: Sequence[Mapping[str, Any]]):
    from pandas import DataFrame
    records = []
    for row in rows:
        key = row.get("task_index")
        if key is None:
            key = row.get("task_id")
        records.append({
            "task": str(key),
            "seed": row.get("seed"),
            "status": row.get("status") or "unstarted",
            "success": row.get("task_success"),
            "duration_s": row.get("duration_s"),
        })
    return DataFrame(records)


def _sn_success_rate(frame, directory: Path) -> Path | None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    if frame.empty:
        return _blank_png("no cases to plot", directory / "success_rate.png")
    grouped = frame.groupby("task", as_index=False)["success"].agg(
        total="count", ok=lambda s: int(s.eq(True).sum()))
    grouped["rate"] = grouped["ok"] / grouped["total"]
    ax = sns.barplot(data=grouped, x="task", y="rate", color="#2e8b57")
    ax.set_ylim(0, 1)
    ax.set_ylabel("success rate")
    ax.set_title("per-task success rate")
    for bar, rate in zip(ax.patches, grouped["rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.0%}", ha="center", fontsize=9)
    fig = ax.get_figure()
    fig.tight_layout()
    path = directory / "success_rate.png"
    fig.savefig(path, dpi=140)
    return path


def _sn_heatmap(frame, directory: Path) -> Path | None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    if frame.empty:
        return _blank_png("no cases to plot", directory / "status_heatmap.png")
    pivot = frame.pivot_table(index="task", columns="seed", values="status", aggfunc="first")
    present = sorted({str(v) for v in pivot.stack().unique().tolist() if not _isna(v)})
    statuses = present if present else ["unstarted"]
    if "unstarted" not in statuses:
        statuses = statuses + ["unstarted"]
    usable = {s: STATUS_COLORS.get(s, STATUS_COLORS["unstarted"]) for s in statuses}
    row_labels = list(pivot.index)
    seed_labels = list(pivot.columns)
    codes = {s: i for i, s in enumerate(usable)}
    values = [
        [str(pivot.loc[t, s]) if s in pivot.columns and t in pivot.index and not _isna(pivot.loc[t, s]) else "unstarted"
         for s in seed_labels]
        for t in row_labels
    ]
    matrix = [[codes[v] for v in row] for row in values]
    cmap = _listed_cmap(list(usable.values()))
    ax = sns.heatmap(
        matrix, annot=values, fmt="", cmap=cmap, cbar=False,
        xticklabels=seed_labels, yticklabels=row_labels,
        annot_kws={"fontsize": 8},
    )
    ax.set_title("task × seed status")
    ax.set_xlabel("seed")
    ax.set_ylabel("task")
    fig = ax.get_figure()
    fig.tight_layout()
    path = directory / "status_heatmap.png"
    fig.savefig(path, dpi=140)
    return path


def _listed_cmap(colors):
    from matplotlib.colors import ListedColormap
    return ListedColormap(colors)


def _isna(value) -> bool:
    try:
        import math
        return value is None or (isinstance(value, float) and math.isnan(value))
    except Exception:  # pragma: no cover
        return value is None


def _sn_duration(frame, directory: Path) -> Path | None:
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    frame = frame.dropna(subset=["duration_s"])
    frame = frame[frame["duration_s"] > 0]
    if frame.empty:
        return _blank_png("no positive durations recorded", directory / "duration.png")
    ordered = frame.sort_values("duration_s", ascending=False).reset_index(drop=True)
    ordered["idx"] = ordered.index
    # Plot log10(duration) on a linear axis: a bar baseline of 0 would be
    # -inf on a true log y-axis, so we avoid the log axis entirely.
    ordered["log10"] = ordered["duration_s"].map(lambda v: math.log10(v))
    ax = sns.barplot(data=ordered, x="idx", y="log10", color="#4a7bb5")
    tick_values = [10, 30, 100, 300, 1000]
    tick_positions = [math.log10(v) for v in tick_values]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(v) for v in tick_values])
    ax.set_ylabel("duration (s, log scale)")
    ax.set_xlabel("case")
    ax.set_title("per-case duration")
    ax.tick_params(axis="x", labelbottom=False)
    fig = ax.get_figure()
    fig.tight_layout()
    path = directory / "duration.png"
    fig.savefig(path, dpi=140)
    # free the helper array to avoid carrying __array_function__ state
    del ordered
    return path


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
    """Write a standalone HTML dashboard embedding the charts + summary table."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    charts = render_charts(rows, directory)
    head = "".join(f"<th>{field}</th>" for field in SUMMARY_FIELDS)
    body_table = "".join(
        "<tr>" + "".join(f"<td>{_escape(_cell(row.get(field)))}</td>" for field in SUMMARY_FIELDS) + "</tr>"
        for row in rows
    )
    svgs = "".join(f'<section><h3>{name}</h3><img src="{path.name}" alt="{name}"/></section>' for name, path in charts.items())
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>eva-agentic viz</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:3px 8px;font-size:12px}}h2,h3{{margin-top:2rem}}
img{{max-width:100%;border:1px solid #eee}}</style></head><body>
<h1>eva-agentic visualization</h1>
<p>{len(rows)} planned case(s). The summary CSV/JSON alongside is the authoritative artifact.</p>
{svgs}
<h3>summary table</h3>
<table><thead><tr>{head}</tr></thead><tbody>{body_table}</tbody></table>
</body></html>
"""
    path = directory / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


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
