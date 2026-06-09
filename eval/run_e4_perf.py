"""Experiment E4 — sidecar overhead (→ Figure 1).

Figure 1 has two panels and two CSVs:

  Panel (a) — POST /api/v1/segment/{type} 202 latency under concurrency.
  Driven by `eval/locustfile.py` against a running sidecar (uvicorn must
  actually serve HTTP for these numbers to mean anything). Reproduction:

      locust -f eval/locustfile.py --headless -u 50 -r 50 -t 60s \\
          --host http://127.0.0.1:8000 \\
          --csv=eval/results/e4_latency

  Panel (b) — Augmentation cost vs input triple count. Measured
  in-process here. For each padded variant of the rich_full synthetic
  input, run `augment_ai_ready` N times and record per-run wall time.

This runner produces panel (b) data and (when matplotlib + the locust
CSV are both present) draws the combined Figure 1. The stub
SegmentationService backend is what makes the 202 latency a
sidecar-overhead measurement rather than an AI-compute one — state that
in the figure caption.

Outputs:
    eval/results/e4_augment_cost.csv      panel (b) — (triples, p50_ms, p95_ms)
    eval/results/e4_latency.csv           panel (a) — (concurrency, p50, p95, p99)
                                          (produced by locust; this runner
                                          only reads it if present)
    eval/results/fig1_broker_overhead.png panels (a) + (b) combined PNG —
                                          quick inspection only
    eval/results/fig1a_latency.pdf        panel (a) standalone PDF — the
                                          artefact \includegraphics'd into
                                          the paper's subfigure (a)
    eval/results/fig1b_augment_cost.pdf   panel (b) standalone PDF — the
                                          artefact \includegraphics'd into
                                          the paper's subfigure (b)

Run from the root folder:

    python -m eval.run_e4_perf

Prereqs: Task 1 corpus built (uses rich_full as the augmentation base).
Dev deps optional — runner reports panel (b) without matplotlib; figure
generation skips cleanly if either CSV or matplotlib is missing.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from src.metadata.augmenter import augment_ai_ready
from src.metadata.types import SegmentationType

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = REPO_ROOT / "eval" / "corpus" / "synthetic"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
AUGMENT_COST_CSV = RESULTS_DIR / "e4_augment_cost.csv"
LATENCY_STATS_CSV = RESULTS_DIR / "e4_latency_stats.csv"  # locust default name
LATENCY_NORMALISED_CSV = RESULTS_DIR / "e4_latency.csv"
# Combined PNG retained for quick inspection; the LaTeX-facing artefacts
# are the two single-panel PDFs below, included as separate \subfigure
# entries in the paper.
FIGURE_PATH = RESULTS_DIR / "fig1_broker_overhead.png"
FIGURE_A_PDF = RESULTS_DIR / "fig1a_latency.pdf"
FIGURE_B_PDF = RESULTS_DIR / "fig1b_augment_cost.pdf"

_RICH_INPUT = SYNTHETIC_DIR / "rich_full.jsonld"
_COCO_URL = "https://example.org/coco/perf.json"
_SEGMENTATION_TYPE = SegmentationType.INSTANCE

_PAD_SIZES = (0, 25, 50, 100, 200, 400, 800, 1600)
_DEFAULT_REPEATS = 30


@dataclass
class CostRow:
    pad_predicates: int
    input_triples_approx: int
    n_runs: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass
class LatencyRow:
    concurrency: str
    n_requests: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    failures: int


# --- Panel (b): augmentation cost vs input size --------------------------


def _padded_input(base: dict, n_predicates: int) -> dict:
    """Return a deep-cloned base graph with `n_predicates` extra leaves
    on the dataset node. Each pad is a string-valued predicate under a
    fresh namespace — the augmenter must clone+preserve all of these,
    so the per-run wall time scales with input size.
    """
    cloned = copy.deepcopy(base)
    ctx = cloned.setdefault("@context", {})
    if isinstance(ctx, dict):
        ctx.setdefault("evpad", "https://eval.example/pad#")
    dataset = cloned["@graph"][0]
    for i in range(n_predicates):
        dataset[f"evpad:pad{i:05d}"] = f"value-{i}"
    return cloned


def _approx_triples(doc: dict) -> int:
    """Cheap proxy: leaf-count on the dataset node. The augmenter's
    work is proportional to this (deepcopy + dict walks scale with
    total leaves), so reporting it makes the CSV interpretable without
    rdflib being installed.
    """
    dataset = doc["@graph"][0]
    n = 0
    for v in dataset.values():
        if isinstance(v, list):
            n += len(v)
        else:
            n += 1
    return n


def _time_one(base: dict, n_pad: int, repeats: int) -> CostRow:
    timings_ms: list[float] = []
    padded = _padded_input(base, n_pad)
    approx = _approx_triples(padded)
    # Warm up once so the first-call JIT/import cost doesn't skew p50.
    augment_ai_ready(
        padded,
        job_id="warmup",
        segmentation_type=_SEGMENTATION_TYPE,
        coco_access_url=_COCO_URL,
    )
    for i in range(repeats):
        # Fresh deepcopy per run — the augmenter mutates its cloned copy
        # internally, but we hand it a fresh dict so this measures the
        # augmenter's clone, not our reuse of a pre-warmed copy.
        body = _padded_input(base, n_pad)
        t0 = time.perf_counter()
        augment_ai_ready(
            body,
            job_id=f"perf-{n_pad}-{i:03d}",
            segmentation_type=_SEGMENTATION_TYPE,
            coco_access_url=_COCO_URL,
        )
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    timings_ms.sort()
    return CostRow(
        pad_predicates=n_pad,
        input_triples_approx=approx,
        n_runs=repeats,
        p50_ms=round(_percentile(timings_ms, 50), 3),
        p95_ms=round(_percentile(timings_ms, 95), 3),
        p99_ms=round(_percentile(timings_ms, 99), 3),
        min_ms=round(timings_ms[0], 3),
        max_ms=round(timings_ms[-1], 3),
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    # Linear interpolation (matches statistics.quantiles default).
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def measure_augment_cost(repeats: int = _DEFAULT_REPEATS) -> list[CostRow]:
    if not _RICH_INPUT.exists():
        raise FileNotFoundError(
            f"{_RICH_INPUT} missing — run `python -m eval.build_corpus` first."
        )
    base = json.loads(_RICH_INPUT.read_text(encoding="utf-8"))
    rows: list[CostRow] = []
    for n in _PAD_SIZES:
        rows.append(_time_one(base, n, repeats))
    return rows


# --- Panel (a): normalise locust output (when present) -------------------


def _read_locust_stats() -> list[LatencyRow]:
    """locust writes `<prefix>_stats.csv` with columns including Type,
    Name, Request Count, Failure Count, plus percentile columns. We
    take the row whose Name is the POST endpoint and report p50/p95/p99
    in ms. Concurrency is unknown from this CSV alone — the runner is
    invoked once per concurrency level externally, so we look for
    multiple files and tag each by suffix.
    """
    out: list[LatencyRow] = []
    if not RESULTS_DIR.exists():
        return out
    for stats_path in sorted(RESULTS_DIR.glob("e4_latency*_stats.csv")):
        # Tag: e4_latency_c1_stats.csv -> "c1", e4_latency_stats.csv -> "all"
        stem = stats_path.stem.replace("_stats", "")
        tag = stem.replace("e4_latency", "").strip("_") or "all"
        with stats_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name", "")
                if "Aggregated" in name or "/api/v1/segment" not in name:
                    continue
                try:
                    p50 = float(row.get("50%", 0) or row.get("Median Response Time", 0))
                    p95 = float(row.get("95%", 0))
                    p99 = float(row.get("99%", 0))
                    n_req = int(row.get("Request Count", 0))
                    n_fail = int(row.get("Failure Count", 0))
                except (TypeError, ValueError):
                    continue
                out.append(LatencyRow(
                    concurrency=tag,
                    n_requests=n_req,
                    p50_ms=p50,
                    p95_ms=p95,
                    p99_ms=p99,
                    failures=n_fail,
                ))
    # Natural-sort by concurrency integer (c1, c10, c50 — not lexicographic
    # c10, c1, c50 from the glob). Non-numeric tags ("all") sink to the end.
    def _key(r: LatencyRow) -> tuple[int, int, str]:
        m = re.match(r"c(\d+)$", r.concurrency)
        if m:
            return (0, int(m.group(1)), r.concurrency)
        return (1, 0, r.concurrency)
    out.sort(key=_key)
    return out


# --- CSV writers ----------------------------------------------------------


def _write_cost(rows: list[CostRow]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "pad_predicates", "input_triples_approx", "n_runs",
        "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms",
    ]
    with AUGMENT_COST_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _write_latency(rows: list[LatencyRow]) -> None:
    if not rows:
        return
    fields = ["concurrency", "n_requests", "p50_ms", "p95_ms", "p99_ms", "failures"]
    with LATENCY_NORMALISED_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


# --- Figure 1 ------------------------------------------------------------


def _draw_panel_a(ax, latency_rows: list[LatencyRow], *, show_title: bool = True) -> None:
    if latency_rows:
        labels = [r.concurrency for r in latency_rows]
        p50 = [r.p50_ms for r in latency_rows]
        p95 = [r.p95_ms for r in latency_rows]
        p99 = [r.p99_ms for r in latency_rows]
        x = list(range(len(labels)))
        width = 0.25
        ax.bar([i - width for i in x], p50, width=width, label="p50")
        ax.bar(x, p95, width=width, label="p95")
        ax.bar([i + width for i in x], p99, width=width, label="p99")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("time to 202 (ms)")
        ax.set_xlabel("concurrency")
        if show_title:
            ax.set_title("(a) POST /api/v1/segment/{type} 202 latency")
        ax.legend()
    else:
        ax.text(
            0.5, 0.5,
            "no locust stats — run\n`locust -f eval/locustfile.py …`",
            ha="center", va="center", transform=ax.transAxes,
        )
        if show_title:
            ax.set_title("(a) 202 latency — data missing")
        ax.set_xticks([])
        ax.set_yticks([])


def _draw_panel_b(ax, cost_rows: list[CostRow], *, show_title: bool = True) -> None:
    xs = [r.input_triples_approx for r in cost_rows]
    p50s = [r.p50_ms for r in cost_rows]
    p95s = [r.p95_ms for r in cost_rows]
    ax.plot(xs, p50s, marker="o", label="p50")
    ax.plot(xs, p95s, marker="x", linestyle="--", label="p95")
    ax.set_xlabel("input dataset-node leaves (≈ triples)")
    ax.set_ylabel("augment wall time (ms)")
    if show_title:
        ax.set_title("(b) Augmentation cost vs input size")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _try_draw_figure(cost_rows: list[CostRow], latency_rows: list[LatencyRow]) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — never pop a window in CI
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not installed — skipped figure"
    if not cost_rows:
        return "no augment-cost rows — skipped figure"

    # Combined PNG — convenient for at-a-glance inspection in the repo.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _draw_panel_a(axes[0], latency_rows)
    _draw_panel_b(axes[1], cost_rows)
    fig.suptitle(
        "Figure 1 — Sidecar overhead "
        "(stub segmentation backend isolates sidecar wall time)"
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=144)
    plt.close(fig)

    # Single-panel PDFs — what the LaTeX paper actually \includegraphics.
    # Vector output, sized to fit a two-column subfigure pair. Titles are
    # suppressed because the LaTeX \caption / \subcaption supplies them.
    fig_a, ax_a = plt.subplots(figsize=(5.5, 4))
    _draw_panel_a(ax_a, latency_rows, show_title=False)
    fig_a.tight_layout()
    fig_a.savefig(FIGURE_A_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(5.5, 4))
    _draw_panel_b(ax_b, cost_rows, show_title=False)
    fig_b.tight_layout()
    fig_b.savefig(FIGURE_B_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig_b)

    return None


# --- Main ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repeats", type=int, default=_DEFAULT_REPEATS,
        help=f"per-size augment iterations (default {_DEFAULT_REPEATS})",
    )
    args = parser.parse_args()

    cost_rows = measure_augment_cost(repeats=args.repeats)
    _write_cost(cost_rows)
    latency_rows = _read_locust_stats()
    if latency_rows:
        _write_latency(latency_rows)
    fig_note = _try_draw_figure(cost_rows, latency_rows)

    print(f"E4 panel (b): {len(cost_rows)} input sizes timed.")
    p50_at_baseline = cost_rows[0].p50_ms
    p50_at_max = cost_rows[-1].p50_ms
    print(
        f"  augment p50: {p50_at_baseline:.2f} ms @ {cost_rows[0].input_triples_approx} leaves "
        f"→ {p50_at_max:.2f} ms @ {cost_rows[-1].input_triples_approx} leaves"
    )
    print(f"  → {AUGMENT_COST_CSV.relative_to(REPO_ROOT)}")
    if latency_rows:
        print(f"E4 panel (a): {len(latency_rows)} locust runs normalised")
        print(f"  → {LATENCY_NORMALISED_CSV.relative_to(REPO_ROOT)}")
    else:
        print(
            "E4 panel (a): no locust stats found — run\n"
            "  locust -f eval/locustfile.py --headless -u 50 -r 50 -t 60s "
            "--host http://127.0.0.1:8000 --csv=eval/results/e4_latency",
            file=sys.stderr,
        )
    if fig_note:
        print(f"figure: {fig_note}", file=sys.stderr)
    else:
        print(f"  → {FIGURE_PATH.relative_to(REPO_ROOT)}")
        print(f"  → {FIGURE_A_PDF.relative_to(REPO_ROOT)}")
        print(f"  → {FIGURE_B_PDF.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
