"""
Generalized survival-curve + log-rank/Cox analysis for Kensington ABM
BehaviorSpace Table-format CSVs, covering ALL THREE of the model's
censored time-to-event exports:

    export type              interval filename pattern           duration column
    ------------------------ ----------------------------------- --------------------
    death (mortality)        risk-intervals-run-*.csv             time_at_risk
    OTP enrollment           otp-enrollment-intervals-run-*.csv   time_to_enrollment
    SSP visit                ssp-visit-intervals-run-*.csv        time_to_visit

REWRITE NOTE (2026-08-11): the previous version of this script built its
survival curve from the Table CSV's live `survival-fraction` reporter,
averaged across repetitions at each tick. That reporter only exists for
the mortality export -- OTP-enrollment and SSP-visit runs have no
per-tick survival-probability reporter in the Table CSV at all (they log
`ticks-to-otp-enrollment` etc., which is a running MEAN wait time among
agents enrolled so far, not a survival probability), so the old
survival_curve() crashed outright on those files with `KeyError:
'survival-fraction'`. It also misclassified new metric columns
(ticks-to-otp-enrollment, n-ever-enrolled, ...) as swept PARAMETERS,
since NON_PARAM_COLS only knew about the original 7 mortality reporters.

This version fixes both problems by building the curve directly from the
per-agent interval CSVs (risk-intervals-run-*.csv / etc.) instead of a
Table-CSV reporter: a real Kaplan-Meier estimate, grouped by the swept
BehaviorSpace parameter (looked up per run_number from the Table CSV and
merged onto the interval data). This is correct and uniform across all
three export types -- the Table CSV is now used only to (a) detect which
column was actually swept and (b) produce the endpoint summary of the
live reporters; the survival curve and every statistical test come from
the interval files, which the RISK_INTERVALS_DIR / INTERVAL_FILE_PATTERN /
DURATION_COL settings below must be pointed at.

Produces, for a SINGLE varied parameter:
  1. <n>_survival_curve.png    -- real KM curve, one line per parameter
                                    value (or continuous colormap if >10
                                    values), from the interval files.
  2. <n>_endpoint_summary.csv  -- tick-90 (or whatever final [step] is)
                                    summary of the Table CSV's live
                                    reporters, mean +/- std, per
                                    parameter value.
  3. <n>_logrank_results.txt   -- multivariate + pairwise log-rank tests
                                    on the interval data. Flagged as
                                    directional/exploratory (pools agents
                                    across repetitions).
  4. <n>_cox_results.txt       -- Cox model with run-clustered SEs --
                                    the citable hazard ratio + 95% CI.
                                    Skipped (with a note) for non-numeric
                                    (boolean) parameters.

For TWO+ varied parameters (a combination experiment): the curve
faceted into small multiples, same idea as before, now built from real
per-agent KM curves in each panel instead of averaged live reporters.
Log-rank/Cox use only the first varied parameter (same limitation as
before) with a printed note.

RISK_INTERVALS_DIR is now REQUIRED for the curve itself, not just an
optional add-on for the statistical tests -- there is no other source
of survival data for the OTP/SSP export types.

Usage (command line):
    python3 Survival_Curve_BehaviorSpace_Analysis.py <table-csv> [output-dir] [risk-intervals-dir]

Usage (VS Code Run button):
    Edit the constants below, then click Run.
"""
import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Poster-print styling ─────────────────────────────────────────────────
# These figures get embedded at ~6"x4" on a 36x24 printed poster, so
# default matplotlib font sizes (design for a ~6x4 inch on-screen figure at
# reading distance) come out too small to read from a few feet away. Bump
# everything up and export at higher DPI so the poster panel is legible.
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
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    _HAVE_LIFELINES = True
except ImportError:
    _HAVE_LIFELINES = False

# ── EDIT THESE, then click Run ─────────────────────────────────────────────
CSV_PATH = 'data/crowding-weight-results.csv'
OUTPUT_DIR = 'results'
RISK_INTERVALS_DIR = 'data/crowding-weight'

# Which export type is this? Pick the matching pair below and uncomment it
# (or set your own). This controls both which files get loaded from
# RISK_INTERVALS_DIR and which column the survival curve/log-rank/Cox
# treat as the duration.
#
# DEATH (mortality):
INTERVAL_FILE_PATTERN = "risk-intervals-run-*.csv"
DURATION_COL = "time_at_risk"
#
# OTP ENROLLMENT:
# INTERVAL_FILE_PATTERN = "otp-enrollment-intervals-run-*.csv"
# DURATION_COL = "time_to_enrollment"
#
# SSP VISIT:
# INTERVAL_FILE_PATTERN = "ssp-visit-intervals-run-*.csv"
# DURATION_COL = "time_to_visit"

EVENT_COL = "event"   # same column name in all three export types

# "hour" if hourly-ticks? was ON for this run, "day" if OFF. Only changes
# the x-axis label -- [step] in the CSV is already the raw tick count.
TICK_UNIT = "day"

# "smoothed": plots the MEAN of each run's own KM curve, one line per
#   parameter value -- visually clean, matching the look of the old
#   mortality-only script. This works for ALL THREE export types (unlike
#   the old script) because it's built from the interval files' own
#   run_number column, not a live Table-CSV reporter that only mortality
#   has.
# "pooled": one real KM curve fit on every agent from every run combined
#   -- jagged, but this is the exact same pooled dataset the log-rank and
#   Cox tests below use, so the plot and the statistics are built from
#   identical data.
# Either way, log-rank/Cox ALWAYS use the pooled dataset -- CURVE_STYLE
# only changes what gets drawn, never what gets statistically tested.
CURVE_STYLE = "smoothed"
# ────────────────────────────────────────────────────────────────────────────

# Every reporter name this model has EVER exposed as a BehaviorSpace metric,
# across all three export types plus the network/passerby gap-fill sweeps --
# used only to decide which Table-CSV columns are NOT swept parameters.
# A column that varies row-to-row but is in this set is an OUTCOME being
# measured, not a condition being tested.
METRIC_NAMES = {
    # mortality
    "fatal-od-rate", "cumulative-fatal-od-count", "cumulative-nonfatal-od-count",
    "fatal-fraction", "fatal-fraction-low", "fatal-fraction-mid", "fatal-fraction-high",
    "survival-fraction", "pct-surviving", "n-migrated", "n-at-risk",
    "person-time-inflation", "person-days-at-risk", "cumulative-od-rate",
    "n-events-death", "total-agent-days-at-risk",
    # OTP enrollment
    "ticks-to-otp-enrollment", "days-to-otp-enrollment", "n-ever-enrolled",
    "treatment-coverage",
    # SSP visit
    "ticks-to-ssp-visit", "days-to-ssp-visit", "n-ever-visited-ssp",
    "ssp-coverage", "n-pp-visits",
    # network / passerby / contagion gap-fill sweeps
    "net-conversion-flow", "observed-homophily", "mean-peer-use-fraction",
    "mean-peer-nonuse-fraction", "mean-degree-all", "mean-degree-passersby",
    "n-isolated-people", "n-isolated-agents", "passerby-bystander-coverage",
    "mean-tie-degree", "mean-clustering-coefficient", "n-social-ties",
    "n-passersby", "n-people", "pct-non-user", "n-initiations", "n-desistances",
    # test-strip / adulterant
    "test-strip-coverage", "n-substance-conversions", "pct-pure-fentanyl",
    "pct-adulterant", "pct-xylazine", "pct-medetomidine",
    # other population/outreach monitors that could show up as metrics
    "n-active", "n-in-treatment", "n-dead", "n-present", "n-inmigrated",
    "naloxone-coverage", "outreach-contact-rate", "n-outreach-contacts",
}
NON_PARAM_COLS = {"[run number]", "[step]"} | METRIC_NAMES


def load(path):
    return pd.read_csv(path, skiprows=6)


def detect_varied_params(df):
    candidate_cols = [c for c in df.columns if c not in NON_PARAM_COLS]
    return [c for c in candidate_cols if df[c].nunique() > 1]


def load_intervals(risk_dir, pattern):
    files = sorted(glob.glob(os.path.join(risk_dir, pattern)))
    if not files:
        return None, []
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df, files


def merge_param(intervals, table_df, param):
    """Attaches each interval row's swept-parameter value by looking up its
    run_number in the Table CSV (one value per run, since BehaviorSpace
    holds a run's parameters fixed for its whole duration)."""
    run_param = table_df.groupby("[run number]")[param].first().reset_index()
    run_param = run_param.rename(columns={"[run number]": "run_number"})
    merged = intervals.merge(run_param, on="run_number", how="left")
    if merged[param].isna().any():
        print(f"  WARNING: some run_numbers in the interval files have no "
              f"matching '{param}' value -- check the Table CSV and interval "
              f"files came from the same experiment.")
    return merged


def make_group_label(row, varied):
    return ", ".join(f"{p}={row[p]}" for p in varied)


def mean_of_run_km_curves(sub, timeline):
    """
    Fits one KM curve per run_number within `sub`, evaluates every curve
    on the same tick grid (`timeline`), and returns the elementwise mean.
    This generalizes the old mortality-only script's "average ~100
    already-smooth per-run curves" visual style to ALL THREE export
    types: it only needs the per-agent interval data (which every export
    type has), not a live per-tick Table-CSV reporter (which only
    mortality had, which is why the old script couldn't run on
    OTP/SSP data at all).
    """
    kmf = KaplanMeierFitter()
    curves = []
    for _, run_df in sub.groupby("run_number"):
        if len(run_df) == 0:
            continue
        kmf.fit(run_df[DURATION_COL], event_observed=run_df[EVENT_COL], timeline=timeline)
        curves.append(kmf.survival_function_.iloc[:, 0].values)
    if not curves:
        return np.ones_like(timeline, dtype=float)
    return np.mean(curves, axis=0)


def km_survival_curve(intervals, varied, out_path, exp_name):
    """
    Single varied parameter: one curve per parameter value, built from
    the per-agent interval data (not a Table-CSV reporter). See
    CURVE_STYLE at the top of the file for the pooled-vs-smoothed choice.
    """
    if len(varied) >= 2:
        faceted_km_curves(intervals, varied[0], varied[1], out_path, exp_name)
        return

    param = varied[0]
    groups = sorted(intervals[param].dropna().unique())
    fig, ax = plt.subplots(figsize=POSTER_FIGSIZE)

    cmap = plt.get_cmap("viridis")
    use_colorbar = len(groups) > 10
    if use_colorbar:
        norm = plt.Normalize(vmin=min(groups), vmax=max(groups))

    if CURVE_STYLE == "smoothed":
        timeline = np.arange(0, int(intervals[DURATION_COL].max()) + 1)
        for g in groups:
            sub = intervals[intervals[param] == g]
            mean_curve = mean_of_run_km_curves(sub, timeline)
            color = cmap(norm(g)) if use_colorbar else None
            ax.plot(timeline, mean_curve, label=str(g),
                    linewidth=2.6 if use_colorbar else 3.4, color=color)
    else:
        kmf = KaplanMeierFitter()
        for g in groups:
            sub = intervals[intervals[param] == g]
            kmf.fit(sub[DURATION_COL], event_observed=sub[EVENT_COL], label=str(g))
            color = cmap(norm(g)) if use_colorbar else None
            kmf.plot_survival_function(ax=ax, ci_show=False,
                                        linewidth=2.6 if use_colorbar else 3.4,
                                        color=color)

    if use_colorbar:
        ax.get_legend().remove()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(param, fontsize=17)
        cbar.ax.tick_params(labelsize=14)
    else:
        ax.legend(fontsize=16, loc="lower left", title=param,
                   title_fontsize=17, framealpha=0.9)

    style_note = ("mean of per-run KM curves" if CURVE_STYLE == "smoothed"
                  else "pooled KM estimate")
    ax.set_xlabel(f"Time ({TICK_UNIT}s), {DURATION_COL}", fontsize=19)
    ax.set_ylabel("Survival probability", fontsize=19)
    ax.set_title(f"Survival curve — {exp_name}\n(grouped by {param}, {style_note})",
                 fontsize=21, pad=14)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=POSTER_DPI)
    plt.close(fig)


def faceted_km_curves(intervals, param_a, param_b, out_path, exp_name):
    """
    Two-parameter combination experiment: one panel per value of whichever
    parameter has fewer distinct values, real KM curves colored by the
    other parameter within each panel via a shared colormap.
    """
    n_a, n_b = intervals[param_a].nunique(), intervals[param_b].nunique()
    facet_param, color_param = (param_a, param_b) if n_a <= n_b else (param_b, param_a)

    facet_vals = sorted(intervals[facet_param].unique())
    color_vals = sorted(intervals[color_param].unique())
    n_facets = len(facet_vals)
    ncols = min(n_facets, 4)
    nrows = int(np.ceil(n_facets / ncols))

    norm = plt.Normalize(vmin=min(color_vals), vmax=max(color_vals))
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    axes_flat = axes.flatten()
    kmf = KaplanMeierFitter()

    timeline = np.arange(0, int(intervals[DURATION_COL].max()) + 1) if CURVE_STYLE == "smoothed" else None

    for ax, fv in zip(axes_flat, facet_vals):
        sub_facet = intervals[intervals[facet_param] == fv]
        for cv in color_vals:
            sub = sub_facet[sub_facet[color_param] == cv]
            if len(sub) == 0:
                continue
            if CURVE_STYLE == "smoothed":
                mean_curve = mean_of_run_km_curves(sub, timeline)
                ax.plot(timeline, mean_curve, color=cmap(norm(cv)), linewidth=2.4)
            else:
                kmf.fit(sub[DURATION_COL], event_observed=sub[EVENT_COL])
                ax.plot(kmf.survival_function_.index,
                         kmf.survival_function_.iloc[:, 0],
                         color=cmap(norm(cv)), linewidth=2.4)
        ax.set_title(f"{facet_param} = {fv}", fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.02)

    for ax in axes_flat[n_facets:]:
        ax.axis("off")

    fig.text(0.5, 0.02, f"Time ({TICK_UNIT}s), {DURATION_COL}", ha="center", fontsize=18)
    fig.text(0.01, 0.5, "Survival probability", va="center", rotation="vertical", fontsize=18)

    fig.tight_layout(rect=(0.03, 0.05, 1, 0.86))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat[:n_facets].tolist(), pad=0.02, fraction=0.035)
    cbar.set_label(color_param, fontsize=17)
    cbar.ax.tick_params(labelsize=14)

    style_note = ("mean of per-run KM curves" if CURVE_STYLE == "smoothed"
                  else "pooled KM estimate")
    fig.suptitle(
        f"Survival curves — {exp_name}\n"
        f"(panels = {facet_param}, color = {color_param}, {style_note})",
        y=0.98, fontsize=22,
    )
    fig.savefig(out_path, dpi=POSTER_DPI, bbox_inches="tight")
    plt.close(fig)


def endpoint_summary(df, varied, out_path):
    """Summarizes the Table CSV's own live reporters at the final tick,
    grouped by the varied parameter(s). Auto-detects which reporter
    columns are present rather than assuming a fixed list, so this works
    unmodified across mortality / OTP / SSP / network experiments."""
    final_tick = df["[step]"].max()
    endpoint = df[df["[step]"] == final_tick].copy()
    present_reporters = [c for c in df.columns if c in METRIC_NAMES]
    if not present_reporters:
        print("  WARNING: no known reporter columns found for endpoint summary.")
        return pd.DataFrame()
    agg = endpoint.groupby(varied)[present_reporters].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.insert(0, "final_tick", final_tick)
    agg.to_csv(out_path, index=False)
    return agg


def logrank_tests(merged, param, out_path, exp_name):
    groups = sorted(merged[param].dropna().unique())
    lines = [f"Log-rank tests — {exp_name}", f"Grouped by: {param}",
             f"Duration column: {DURATION_COL}", ""]
    lines.append("CAUTION: pools all agents across repetitions and treats them as "
                 "independent, which inflates apparent significance (agents within a "
                 "run share a world, network, and random seed). Read as directional "
                 "evidence -- see the accompanying _cox_results.txt for the "
                 "clustering-corrected, citable hazard ratio.")
    lines.append("")

    result_all = multivariate_logrank_test(
        merged[DURATION_COL], merged[param], merged[EVENT_COL]
    )
    lines.append(f"1. Multivariate log-rank test across all {len(groups)} groups:")
    lines.append(f"   test statistic = {result_all.test_statistic:.4f}")
    lines.append(f"   p-value = {result_all.p_value:.3g}")
    sig = "SIGNIFICANT (p < 0.05)" if result_all.p_value < 0.05 else "not significant (p >= 0.05)"
    lines.append(f"   -> survival curves differ across {param}: {sig}")
    lines.append("")

    low, high = groups[0], groups[-1]
    low_data = merged[merged[param] == low]
    high_data = merged[merged[param] == high]
    result_pair = logrank_test(
        low_data[DURATION_COL], high_data[DURATION_COL],
        event_observed_A=low_data[EVENT_COL], event_observed_B=high_data[EVENT_COL],
    )
    lines.append(f"2. Pairwise log-rank test: {param}={low} vs {param}={high}")
    lines.append(f"   n agents: {len(low_data)} vs {len(high_data)}")
    lines.append(f"   test statistic = {result_pair.test_statistic:.4f}")
    lines.append(f"   p-value = {result_pair.p_value:.3g}")
    sig = "SIGNIFICANT (p < 0.05)" if result_pair.p_value < 0.05 else "not significant (p >= 0.05)"
    lines.append(f"   -> the low and high extremes' curves differ: {sig}")
    lines.append("")

    lines.append("3. Adjacent-group pairwise log-rank tests:")
    for g1, g2 in zip(groups[:-1], groups[1:]):
        d1 = merged[merged[param] == g1]
        d2 = merged[merged[param] == g2]
        r = logrank_test(
            d1[DURATION_COL], d2[DURATION_COL],
            event_observed_A=d1[EVENT_COL], event_observed_B=d2[EVENT_COL],
        )
        s = "SIG" if r.p_value < 0.05 else "ns"
        lines.append(f"   {param}={g1} vs {param}={g2}: "
                      f"stat={r.test_statistic:.3f}  p={r.p_value:.3g}  [{s}]")
    lines.append("")

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)
    return report


def cox_model(merged, param, out_path, exp_name):
    is_numeric = (
        pd.api.types.is_numeric_dtype(merged[param])
        and not pd.api.types.is_bool_dtype(merged[param])
    )
    if not is_numeric:
        lines = [
            f"Cox model — {exp_name}", f"Parameter: {param}", "",
            f"SKIPPED: '{param}' is not numeric (likely a boolean switch). A "
            f"hazard ratio 'per unit increase' isn't meaningful for a two-level "
            f"switch; see the accompanying _logrank_results.txt instead.",
        ]
        report = "\n".join(lines)
        with open(out_path, "w") as f:
            f.write(report)
        return report

    cph = CoxPHFitter()
    cox_data = merged[[DURATION_COL, EVENT_COL, param, "run_number"]].dropna()
    cph.fit(cox_data, duration_col=DURATION_COL, event_col=EVENT_COL,
            cluster_col="run_number")

    tested_vals = sorted(cox_data[param].unique())
    step = float(np.median(np.diff(tested_vals))) if len(tested_vals) > 1 else 1.0

    coef = cph.summary.loc[param, "coef"]
    se = cph.summary.loc[param, "se(coef)"]
    p_val = cph.summary.loc[param, "p"]

    hr = np.exp(coef * step)
    hr_lo = np.exp((coef - 1.96 * se) * step)
    hr_hi = np.exp((coef + 1.96 * se) * step)

    lines = [
        f"Cox proportional hazards model (run-clustered SEs) — {exp_name}",
        f"Parameter: {param}   Duration column: {DURATION_COL}",
        f"N agents: {len(cox_data)}   N runs (clusters): {cox_data['run_number'].nunique()}",
        "",
        "Standard errors are clustered by run_number -- this is the number "
        "defensible enough to cite, not the pooled log-rank p-value.",
        "",
        f"Hazard ratio per step of {step:.6g} in {param}: {hr:.4f}",
        f"95% CI: [{hr_lo:.4f}, {hr_hi:.4f}]",
        f"p-value (cluster-robust): {p_val:.3g}",
        "",
    ]
    if hr < 1:
        pct = (1 - hr) * 100
        lines.append(f"Interpretation: each step of {step:.6g} in {param} is "
                      f"associated with a {pct:.1f}% reduction in the hazard "
                      f"(95% CI {(1-hr_hi)*100:.1f}% to {(1-hr_lo)*100:.1f}% reduction).")
    elif hr > 1:
        pct = (hr - 1) * 100
        lines.append(f"Interpretation: each step of {step:.6g} in {param} is "
                      f"associated with a {pct:.1f}% increase in the hazard "
                      f"(95% CI {(hr_lo-1)*100:.1f}% to {(hr_hi-1)*100:.1f}% increase).")
    else:
        lines.append(f"Interpretation: no detectable association between {param} and hazard.")
    sig = "SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "not significant (p >= 0.05)"
    lines.append(f"-> {sig}, with run-level correlation accounted for.")
    lines.append("")

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)
    return report


def main():
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
        risk_dir = sys.argv[3] if len(sys.argv) > 3 else RISK_INTERVALS_DIR
    else:
        path = CSV_PATH
        out_dir = OUTPUT_DIR
        risk_dir = RISK_INTERVALS_DIR

    if not _HAVE_LIFELINES:
        print("ERROR: lifelines is not installed. Run `pip install lifelines`.")
        sys.exit(1)

    if not os.path.exists(path):
        print(f"ERROR: '{path}' not found.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(path))[0]
    df = load(path)
    varied = detect_varied_params(df)
    if not varied:
        print(f"WARNING: no varied parameter detected in {path} -- check the file.")
        return

    print(f"{base}: varied parameter(s) = {varied}  "
          f"({df['[run number]'].nunique()} runs, {df['[step]'].max()} ticks)")

    curve_path = os.path.join(out_dir, f"{base}_survival_curve.png")
    summary_path = os.path.join(out_dir, f"{base}_endpoint_summary.csv")
    logrank_path = os.path.join(out_dir, f"{base}_logrank_results.txt")
    cox_path = os.path.join(out_dir, f"{base}_cox_results.txt")

    agg = endpoint_summary(df, varied, summary_path)
    print(f"  -> wrote {summary_path}")
    if not agg.empty:
        print(agg.to_string(index=False))
    print()

    intervals, files = load_intervals(risk_dir, INTERVAL_FILE_PATTERN)
    if intervals is None:
        print(f"ERROR: no files matching '{INTERVAL_FILE_PATTERN}' found in "
              f"'{risk_dir}'. Cannot build a survival curve without per-agent "
              f"interval data -- check RISK_INTERVALS_DIR and "
              f"INTERVAL_FILE_PATTERN at the top of this script.")
        sys.exit(1)
    print(f"Loaded {len(files)} interval files, {len(intervals)} agent-rows.")

    param = varied[0]
    if len(varied) > 1:
        print(f"  NOTE: log-rank/Cox use '{param}' only; ignoring {varied[1:]}.")
        merged_all = intervals.copy()
        for p in varied:
            merged_all = merge_param(merged_all, df, p) if p not in merged_all.columns else merged_all
        km_survival_curve(merged_all, varied, curve_path, base)
    else:
        merged_all = merge_param(intervals, df, param)
        km_survival_curve(merged_all, varied, curve_path, base)
    print(f"  -> wrote {curve_path}")

    merged = merge_param(intervals, df, param) if param not in intervals.columns else intervals

    report = logrank_tests(merged, param, logrank_path, base)
    print(f"  -> wrote {logrank_path}\n")
    print(report)

    cox_report = cox_model(merged, param, cox_path, base)
    print(f"  -> wrote {cox_path}\n")
    print(cox_report)


if __name__ == "__main__":
    main()
