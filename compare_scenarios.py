"""
compare_scenarios.py -- for comparing FIXED-PARAMETER BehaviorSpace runs
(the combined-intervention-scenarios REDO: baseline, OTP-expansion,
naloxone-expansion, SSP-expansion, test-strips, combined) where each CSV has
no varied parameter to group by -- every row in a given file used the SAME
settings, and the 100 repetitions are just sampling variability at that one
setting.

analyze_behaviorspace.py expects a swept parameter and will correctly warn
"no varied parameter detected" on a file like this -- that's not a bug, it's
the wrong tool for this shape of comparison. This script is the right one:
it takes several such CSVs (one per scenario/arm) and produces:

  1. scenario_comparison_survival_curve.png -- mean survival-fraction vs
     tick, one line per FILE (scenario), instead of per parameter value
  2. scenario_comparison_endpoint_summary.csv -- one row per scenario with
     the same 7 endpoint metrics used elsewhere in this project

Usage (command line):
    python3 compare_scenarios.py <output-dir> <name1>=<csv1> <name2>=<csv2> ...

Usage (VS Code Run button):
    Edit SCENARIOS and OUTPUT_DIR below, then click Run.
"""
import sys
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Poster-print styling ─────────────────────────────────────────────────
# This figure gets embedded at ~6"x4" on a 36x24 printed poster, so
# default matplotlib font sizes (designed for a ~6x4 inch on-screen figure
# at reading distance) come out too small to read from a few feet away.
# Bump everything up and export at higher DPI so the poster panel is legible.
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "legend.title_fontsize": 16,
    "lines.linewidth": 3.0,
    "axes.linewidth": 1.3,
    "grid.linewidth": 0.8,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})
POSTER_FIGSIZE = (12, 8)
POSTER_DPI = 300

try:
    from lifelines.statistics import logrank_test
    from lifelines import CoxPHFitter
    _HAVE_LIFELINES = True
except ImportError:
    _HAVE_LIFELINES = False

# ── EDIT THESE, then click Run ─────────────────────────────────────────────
SCENARIOS = {
    "baseline":     'data/scenario-baseline-results.csv',
    "otp":          'data/scenario-otp-results.csv',
    "naloxone":     'data/scenario-naloxone-results.csv',
    "ssp":          'data/scenario-ssp-results.csv',
    "test-strips":  'data/scenario-test-strips-results.csv',
    "combined":     'data/scenario-combined-results.csv',
}
# Optional: one risk-intervals folder per scenario (from export-run-risk-
# intervals in each experiment's Post run commands). Leave a scenario's value
# as "" to skip log-rank/Cox for it. Leave the whole dict empty ({}) to skip
# survival tests entirely and only get the curve + endpoint summary.
RISK_INTERVALS_DIRS = {
    "baseline":    'data/scenario-baseline',
    "otp":         'data/scenario-otp',
    "naloxone":    'data/scenario-naloxone',
    "ssp":         'data/scenario-ssp',
    "test-strips": 'data/scenario-test-strips',
    "combined":    'data/scenario-combined',
}
BASELINE_NAME = "baseline"   # which key in SCENARIOS/RISK_INTERVALS_DIRS is the reference arm
OUTPUT_DIR = "results"
# ────────────────────────────────────────────────────────────────────────────

REPORTERS = [
    "fatal-od-rate", "cumulative-fatal-od-count", "fatal-fraction",
    "survival-fraction", "n-migrated", "n-at-risk", "person-time-inflation",
]


def load(path):
    return pd.read_csv(path, skiprows=6)


def survival_curve_plot(dfs, out_path):
    fig, ax = plt.subplots(figsize=POSTER_FIGSIZE)
    all_mins = []
    for name, df in dfs.items():
        curve = df.groupby("[step]")["survival-fraction"].mean().reset_index()
        ax.plot(curve["[step]"], curve["survival-fraction"], label=name, linewidth=3.4)
        all_mins.append(curve["survival-fraction"].min())

    y_min = max(0.0, min(all_mins) - 0.03)
    y_min = int(y_min * 20) / 20
    ax.set_ylim(y_min, 1.0)
    ax.set_xlabel("Tick (day)", fontsize=19)
    ax.set_ylabel("Kaplan-Meier survival fraction", fontsize=19)
    ax.set_title("Survival curves — scenario comparison\n(mean across repetitions)",
                 fontsize=21, pad=14)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=POSTER_DPI)
    plt.close(fig)


def load_scenario_intervals(risk_dir, scenario_name):
    """
    Loads risk-intervals-run-*.csv for one scenario and tags every row with
    a composite ID scenario_name + run_number, since each scenario's
    BehaviorSpace experiment numbers its own runs starting at 1 -- without
    this, run #7 from 'baseline' and run #7 from 'otp' would collide and
    be treated as the same cluster in the Cox model.
    """
    if not risk_dir or not os.path.isdir(risk_dir):
        return None
    files = sorted(glob.glob(os.path.join(risk_dir, "risk-intervals-run-*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["scenario"] = scenario_name
    df["cluster_id"] = scenario_name + "_" + df["run_number"].astype(str)
    return df


def scenario_survival_tests(risk_dirs, baseline_name, out_path):
    """
    For each non-baseline scenario with a risk-intervals folder given, runs:
      - a pairwise log-rank test against the baseline arm (same
        pseudoreplication caveat as analyze_behaviorspace.py -- pools agents
        across repetitions, directional/exploratory only)
      - a Cox model with scenario as a 0/1 predictor (this scenario vs
        baseline) and run-clustered standard errors (cluster_id = scenario +
        run number), giving a hazard ratio + 95% CI -- the citable number.
    Skips cleanly (with a note) if lifelines isn't installed or fewer than
    two scenarios have risk-interval data.
    """
    if not _HAVE_LIFELINES:
        print("  (skipping survival tests: lifelines not installed -- run `pip install lifelines`)")
        return None

    baseline_dir = risk_dirs.get(baseline_name, "")
    baseline_df = load_scenario_intervals(baseline_dir, baseline_name)
    if baseline_df is None:
        print(f"  (skipping survival tests: no risk-intervals data for baseline arm '{baseline_name}')")
        return None

    lines = [f"Scenario comparison survival tests (reference arm: {baseline_name})", ""]
    lines.append("CAUTION: log-rank p-values pool agents across repetitions and treat them "
                 "as independent, which inflates apparent significance. The Cox hazard ratio "
                 "(run-clustered standard errors) is the number defensible enough to cite.")
    lines.append("")

    any_comparison = False
    for name, risk_dir in risk_dirs.items():
        if name == baseline_name:
            continue
        arm_df = load_scenario_intervals(risk_dir, name)
        if arm_df is None:
            lines.append(f"[{name}] SKIPPED -- no risk-intervals folder given or found.")
            lines.append("")
            continue

        any_comparison = True
        combined = pd.concat([baseline_df, arm_df], ignore_index=True)

        # ── Log-rank ──────────────────────────────────────────────────────
        r = logrank_test(
            baseline_df["time_at_risk"], arm_df["time_at_risk"],
            event_observed_A=baseline_df["event"], event_observed_B=arm_df["event"],
        )
        lines.append(f"[{name}] vs [{baseline_name}]")
        lines.append(f"  N agents: {len(arm_df)} vs {len(baseline_df)}")
        lines.append(f"  Log-rank: statistic = {r.test_statistic:.4f}   p = {r.p_value:.3g}")

        # ── Cox with run-clustered SEs ───────────────────────────────────
        combined["is_arm"] = (combined["scenario"] == name).astype(int)
        cox_data = combined[["time_at_risk", "event", "is_arm", "cluster_id"]].dropna()
        try:
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col="time_at_risk", event_col="event",
                    cluster_col="cluster_id")
            import math
            coef = cph.summary.loc["is_arm", "coef"]
            se = cph.summary.loc["is_arm", "se(coef)"]
            p = cph.summary.loc["is_arm", "p"]
            hr = math.exp(coef)
            hr_lo = math.exp(coef - 1.96 * se)
            hr_hi = math.exp(coef + 1.96 * se)
            lines.append(f"  Cox (cluster-robust): HR = {hr:.4f}   95% CI [{hr_lo:.4f}, {hr_hi:.4f}]   p = {p:.3g}")
            if hr < 1:
                lines.append(f"    -> {name} associated with a {(1-hr)*100:.1f}% lower hazard of death vs {baseline_name}")
            elif hr > 1:
                lines.append(f"    -> {name} associated with a {(hr-1)*100:.1f}% higher hazard of death vs {baseline_name}")
        except Exception as e:
            lines.append(f"  Cox model failed: {e}")
        lines.append("")

    if not any_comparison:
        print("  (skipping survival tests: no non-baseline scenario had risk-intervals data)")
        return None

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)
    return report


def endpoint_summary(dfs, out_path):
    rows = []
    for name, df in dfs.items():
        final_tick = df["[step]"].max()
        endpoint = df[df["[step]"] == final_tick]
        present = [r for r in REPORTERS if r in endpoint.columns]
        row = {"scenario": name, "final_tick": final_tick, "n_runs": endpoint["[run number]"].nunique()}
        for r in present:
            row[f"{r}_mean"] = endpoint[r].mean()
            row[f"{r}_std"] = endpoint[r].std()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    return summary


def main():
    if len(sys.argv) >= 2:
        out_dir = sys.argv[1]
        scenarios = {}
        risk_dirs = {}
        for arg in sys.argv[2:]:
            name, path = arg.split("=", 1)
            if path.startswith("risk:"):
                risk_dirs[name] = path[len("risk:"):]
            else:
                scenarios[name] = path
    else:
        out_dir = OUTPUT_DIR
        scenarios = SCENARIOS
        risk_dirs = RISK_INTERVALS_DIRS

    os.makedirs(out_dir, exist_ok=True)

    dfs = {}
    for name, path in scenarios.items():
        if not os.path.exists(path):
            print(f"ERROR: '{path}' not found for scenario '{name}'. Skipping.")
            continue
        dfs[name] = load(path)
        print(f"{name}: {dfs[name]['[run number]'].nunique()} runs, "
              f"{dfs[name]['[step]'].max()} ticks -- loaded from {path}")

    if not dfs:
        print("No scenario files loaded -- nothing to do.")
        sys.exit(1)

    curve_path = os.path.join(out_dir, "scenario_comparison_survival_curve.png")
    summary_path = os.path.join(out_dir, "scenario_comparison_endpoint_summary.csv")
    tests_path = os.path.join(out_dir, "scenario_comparison_survival_tests.txt")

    survival_curve_plot(dfs, curve_path)
    summary = endpoint_summary(dfs, summary_path)

    print(f"\n-> wrote {curve_path}")
    print(f"-> wrote {summary_path}\n")
    print(summary.to_string(index=False))

    if risk_dirs:
        report = scenario_survival_tests(risk_dirs, BASELINE_NAME, tests_path)
        if report:
            print(f"\n-> wrote {tests_path}\n")
            print(report)


if __name__ == "__main__":
    main()
