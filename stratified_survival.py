"""
stratified_survival.py -- stratifies any of the model's censored time-to-
event exports (risk-intervals, otp-enrollment-intervals, ssp-visit-
intervals) by a PER-AGENT covariate baked into the export itself
(housing_status, social_network_size), rather than by a BehaviorSpace-swept
parameter or by comparing separate scenario files.

This is a different comparison axis from the other two scripts in this
project:
  - Survival_Curve_BehaviorSpace_Analysis.py groups by a parameter that
    varies ACROSS BehaviorSpace runs (e.g. base-od-probability), read from
    the Table-format results CSV.
  - compare_scenarios.py compares different FILES entirely (baseline vs.
    otp vs. combined, etc.).
  - THIS script takes ONE risk-interval-style folder from a SINGLE fixed
    arm (no varied parameter needed -- a plain "Post run commands" export
    across repetitions is enough) and splits the AGENTS WITHIN it by a
    covariate that's already a column in every row: housing_status,
    social_network_size, or any other column the .nlogox exports now
    include per-agent.

Works on any of the three interval export shapes as long as you point
DURATION_COL/EVENT_COL at the right columns for that shape:
  risk-intervals-run-*.csv          -> time_at_risk,       event
  otp-enrollment-intervals-run-*.csv -> time_to_enrollment, event
  ssp-visit-intervals-run-*.csv      -> time_to_visit,      event

Produces:
  1. <exp_name>_survival_curve.png   -- KM curve per stratum of
                                          PRIMARY_STRATA_COL, with a
                                          combined-group log-rank p-value
                                          annotated on the plot.
  2. <exp_name>_group_summary.csv    -- n agents, n events, event rate per
                                          stratum.
  3. <exp_name>_logrank_results.txt  -- multivariate log-rank across all
                                          strata, plus every pairwise
                                          comparison.
  4. <exp_name>_cox_results.txt      -- Cox proportional-hazards model with
                                          run-clustered standard errors,
                                          covariates = every entry in
                                          COX_COVARIATES fit JOINTLY (so a
                                          housing_status effect is reported
                                          net of social_network_size, and
                                          vice versa, not as two separate
                                          marginal models). Categorical
                                          covariates are dummy-encoded
                                          automatically; numeric ones are
                                          used as-is with the hazard ratio
                                          reported per one real unit.

CAUTION baked into the output, same as the other two scripts: the log-rank
test pools every agent as if independent, which inflates significance
since agents within the same BehaviorSpace run share a world, network, and
random seed. The Cox model's run-clustered standard errors correct for
this -- that HR + 95% CI is the number defensible enough to cite, not the
log-rank p-value.

Usage (command line):
    python3 stratified_survival.py <risk-intervals-dir> <exp-name> [output-dir]

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
RISK_DIR = 'data/stratified-mortality-results'  # folder with *-run-*.csv files (any of the three export types)
EXP_NAME = "stratified-mortality"         # prefix for every output file
OUTPUT_DIR = 'results'

# Column names for THIS export type -- see the shape table in the docstring.
# risk-intervals:          DURATION_COL = "time_at_risk"
# otp-enrollment-intervals: DURATION_COL = "time_to_enrollment"
# ssp-visit-intervals:      DURATION_COL = "time_to_visit"
DURATION_COL = "time_at_risk"
EVENT_COL = "event"
CLUSTER_COL = "run_number"       # for cluster-robust Cox SEs

# The covariate to split the KM curve / log-rank test by. Must be
# CATEGORICAL (or will be auto-binned into NUMERIC_BINS quantile groups if
# numeric -- see below) since a survival curve needs discrete groups to
# draw separate lines for.
PRIMARY_STRATA_COL = "housing_status"

# If PRIMARY_STRATA_COL is numeric (e.g. social_network_size), it gets cut
# into this many roughly-equal-sized quantile groups FOR THE PLOT ONLY --
# the Cox model below always uses the raw numeric value, never the binned
# version, so the citable hazard ratio is per-unit, not per-bin.
NUMERIC_BINS = 3

# Every covariate to include in the JOINT Cox model, fit together in one
# model rather than as separate marginal models. Include PRIMARY_STRATA_COL
# here too if you want its Cox-adjusted estimate alongside the raw KM plot.
COX_COVARIATES = ["housing_status", "social_network_size"]

# Number of points on the common time grid used to average per-run KM curves
# together for the "smoothed" plot style (see smoothed_group_curve() below).
# Higher = finer resolution; doesn't change the underlying data, just the
# density of the averaging grid.
TIME_GRID_POINTS = 200

# Light post-smoothing applied to the averaged curve, to remove residual
# run-to-run jitter that survives averaging when the run count is limited
# (e.g. ~10-20 runs per arm). This is a centered moving average over this
# many grid points -- it does NOT change the underlying KM estimate or any
# statistic; log-rank/Cox are computed on the raw pooled data exactly as
# before. Set to 1 to disable (no smoothing).
SMOOTHING_WINDOW = 9
# ────────────────────────────────────────────────────────────────────────────


def load_intervals(risk_dir):
    files = sorted(glob.glob(os.path.join(risk_dir, "*-run-*.csv")))
    if not files:
        print(f"ERROR: no *-run-*.csv files found in '{risk_dir}'.")
        sys.exit(1)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(files)} files, {len(df)} agent-rows from {risk_dir}")
    for col in (DURATION_COL, EVENT_COL):
        if col not in df.columns:
            print(f"ERROR: column '{col}' not found. Columns present: "
                  f"{list(df.columns)}")
            print("Check DURATION_COL/EVENT_COL match the export type "
                  "(see the shape table at the top of this script).")
            sys.exit(1)
    return df


def resolve_strata_col(df, col):
    """
    Returns (label_series, is_numeric_binned). If `col` is already
    categorical/object, returns it unchanged. If numeric, cuts it into
    NUMERIC_BINS quantile groups with readable labels (e.g. "2-5") --
    for the PLOT only, never for the Cox model.
    """
    if col not in df.columns:
        print(f"ERROR: strata column '{col}' not found. Columns present: "
              f"{list(df.columns)}")
        sys.exit(1)
    series = df[col]
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        binned = pd.qcut(series, q=NUMERIC_BINS, duplicates="drop")
        labels = binned.apply(lambda iv: f"{iv.left:.1f}-{iv.right:.1f}")
        print(f"NOTE: '{col}' is numeric -- binned into {labels.nunique()} "
              f"quantile groups for the survival-curve plot only. The Cox "
              f"model below uses the raw numeric value, not these bins.")
        return labels, True
    return series.astype(str), False


def apply_smoothing(curve, window):
    """Centered moving average with edge handling (shrinking window near
    the boundaries so the curve still starts at/near 1.0 and ends at the
    true final value, rather than being pulled by zero-padding)."""
    if window <= 1:
        return curve
    s = pd.Series(curve)
    return s.rolling(window=window, center=True, min_periods=1).mean().values


def smoothed_group_curve(df, mask, time_grid):
    """
    Fits a separate KM curve per run_number (within this stratum), evaluates
    each on the common time_grid, then averages across runs at each grid
    point. This is the same idea as CURVE_STYLE="smoothed" in
    Survival_Curve_BehaviorSpace_Analysis.py: it removes the run-to-run
    jaggedness you get from pooling every agent into one KM fit, since each
    run's own curve is a clean step function and only the AVERAGE across
    runs is plotted, not the pooled event times.

    Runs with zero agents in this stratum are skipped. A run's KM curve is
    evaluated at each grid point using its step function (carrying the last
    known value forward, standard KM convention). A light moving-average
    (SMOOTHING_WINDOW) is applied afterward to remove residual jitter left
    over when the run count is limited -- this only smooths the drawn line,
    it never touches the log-rank/Cox statistics.
    """
    sub = df.loc[mask]
    kmf = KaplanMeierFitter()
    curves = []
    for run_id, run_df in sub.groupby(CLUSTER_COL):
        if run_df[EVENT_COL].sum() == 0 and len(run_df) == 0:
            continue
        kmf.fit(run_df[DURATION_COL], event_observed=run_df[EVENT_COL])
        # step-function value at each grid point
        vals = kmf.survival_function_at_times(time_grid).values
        curves.append(vals)
    if not curves:
        return None
    avg_curve = np.mean(np.vstack(curves), axis=0)
    return apply_smoothing(avg_curve, SMOOTHING_WINDOW)


def survival_curve_plot(df, strata_labels, out_path, exp_name):
    fig, ax = plt.subplots(figsize=POSTER_FIGSIZE)

    groups = sorted(strata_labels.dropna().unique())
    time_grid = np.linspace(0, df[DURATION_COL].max(), TIME_GRID_POINTS)

    for g in groups:
        mask = (strata_labels == g).values
        n = mask.sum()
        n_events = df.loc[mask, EVENT_COL].sum()
        avg_curve = smoothed_group_curve(df, mask, time_grid)
        if avg_curve is None:
            continue
        ax.plot(
            time_grid, avg_curve, linewidth=3.4,
            label=f"{g}  (n={n}, events={n_events})",
        )

    # Multivariate log-rank across all strata, annotated directly on the plot
    # so the figure is self-contained even if someone never opens the .txt.
    if _HAVE_LIFELINES and len(groups) > 1:
        result = multivariate_logrank_test(
            df[DURATION_COL], strata_labels, df[EVENT_COL]
        )
        sig = "p < 0.05" if result.p_value < 0.05 else "n.s."
        # Bottom-right: curves in this model always decline from 1.0, so the
        # bottom-right corner stays empty regardless of how many strata or
        # how steep the decline is -- unlike bottom-left, which the legend
        # also wants and would otherwise collide with.
        ax.text(
            0.98, 0.02,
            f"Multivariate log-rank: p = {result.p_value:.3g} ({sig})",
            transform=ax.transAxes, fontsize=15, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    ax.set_xlabel("Ticks", fontsize=19)
    ax.set_ylabel("Survival probability", fontsize=19)
    ax.set_title(f"Stratified survival curve — {exp_name}\n(mean of per-run KM curves)",
                 fontsize=21, pad=14)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=POSTER_DPI)
    plt.close(fig)


def group_summary(df, strata_labels, out_path):
    tmp = df.copy()
    tmp["_stratum"] = strata_labels.astype(str)  # plain string, not a pandas
                                                  # Categorical -- avoids a
                                                  # groupby FutureWarning
                                                  # about observed=False on
                                                  # categorical dtypes.
    summary = tmp.groupby("_stratum").agg(
        n_agents=(EVENT_COL, "count"),
        n_events=(EVENT_COL, "sum"),
        mean_duration=(DURATION_COL, "mean"),
        median_duration=(DURATION_COL, "median"),
    ).reset_index()
    summary["event_rate"] = summary["n_events"] / summary["n_agents"]
    summary.to_csv(out_path, index=False)
    return summary


def logrank_tests(df, strata_labels, out_path, exp_name):
    groups = sorted(strata_labels.dropna().unique())
    lines = [f"Log-rank tests — {exp_name}", f"Stratified by: {PRIMARY_STRATA_COL}", ""]
    lines.append("CAUTION: pools all agents across repetitions/runs and treats them as "
                 "independent, which inflates apparent significance (agents within a "
                 "run share a world, network, and random seed). Read as directional "
                 "evidence -- see the accompanying _cox_results.txt for the "
                 "clustering-corrected, citable hazard ratio.")
    lines.append("")

    result_all = multivariate_logrank_test(df[DURATION_COL], strata_labels, df[EVENT_COL])
    lines.append(f"1. Multivariate log-rank test across all {len(groups)} groups:")
    lines.append(f"   test statistic = {result_all.test_statistic:.4f}")
    lines.append(f"   p-value = {result_all.p_value:.3g}")
    sig = "SIGNIFICANT (p < 0.05)" if result_all.p_value < 0.05 else "not significant (p >= 0.05)"
    lines.append(f"   -> survival curves differ across {PRIMARY_STRATA_COL}: {sig}")
    lines.append("")

    lines.append("2. Pairwise log-rank tests (every group vs. every other):")
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            m1 = strata_labels == g1
            m2 = strata_labels == g2
            r = logrank_test(
                df.loc[m1, DURATION_COL], df.loc[m2, DURATION_COL],
                event_observed_A=df.loc[m1, EVENT_COL], event_observed_B=df.loc[m2, EVENT_COL],
            )
            s = "SIG" if r.p_value < 0.05 else "ns"
            lines.append(f"   {g1} vs {g2}: stat={r.test_statistic:.3f}  "
                          f"p={r.p_value:.3g}  [{s}]")
    lines.append("")

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)
    return report


def cox_model(df, covariates, out_path, exp_name):
    """
    Joint Cox model: every column in `covariates` fit together in one
    model, with run-clustered standard errors. Categorical covariates are
    dummy-encoded (reference = alphabetically-first level); numeric
    covariates are used as-is, with the hazard ratio reported per one real
    unit of that covariate (e.g. per one additional social-tie).
    """
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        print(f"ERROR: covariate(s) not found in data: {missing}. "
              f"Columns present: {list(df.columns)}")
        sys.exit(1)

    cols_needed = [DURATION_COL, EVENT_COL, CLUSTER_COL] + covariates
    cox_data = df[cols_needed].dropna().copy()

    categorical = [c for c in covariates
                   if not pd.api.types.is_numeric_dtype(cox_data[c])
                   or pd.api.types.is_bool_dtype(cox_data[c])]
    numeric = [c for c in covariates if c not in categorical]

    dummies = pd.get_dummies(cox_data[categorical], drop_first=True) if categorical else pd.DataFrame(index=cox_data.index)
    model_df = pd.concat(
        [cox_data[[DURATION_COL, EVENT_COL, CLUSTER_COL] + numeric], dummies],
        axis=1,
    )

    cph = CoxPHFitter()
    cph.fit(model_df, duration_col=DURATION_COL, event_col=EVENT_COL,
            cluster_col=CLUSTER_COL)

    lines = [
        f"Cox proportional hazards model (run-clustered SEs) — {exp_name}",
        f"Covariates (joint model): {covariates}",
        f"N agents: {len(model_df)}   N runs (clusters): {model_df[CLUSTER_COL].nunique()}",
        "",
        "Standard errors are clustered by run_number, accounting for agents "
        "within the same run being correlated (shared world, network, random "
        "seed). This is the number defensible enough to cite, not a pooled "
        "log-rank p-value.",
        "",
    ]

    for term in cph.summary.index:
        coef = cph.summary.loc[term, "coef"]
        se = cph.summary.loc[term, "se(coef)"]
        p_val = cph.summary.loc[term, "p"]
        hr = np.exp(coef)
        hr_lo = np.exp(coef - 1.96 * se)
        hr_hi = np.exp(coef + 1.96 * se)

        is_dummy = term in dummies.columns
        if is_dummy:
            unit_desc = f"'{term}' vs. reference level"
        else:
            unit_desc = f"per one additional unit of '{term}'"

        lines.append(f"{term}:")
        lines.append(f"  HR = {hr:.4f}   95% CI [{hr_lo:.4f}, {hr_hi:.4f}]   "
                      f"p = {p_val:.3g}")
        if hr < 1:
            lines.append(f"  -> {unit_desc}: {(1 - hr) * 100:.1f}% lower hazard")
        elif hr > 1:
            lines.append(f"  -> {unit_desc}: {(hr - 1) * 100:.1f}% higher hazard")
        sig = "SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "not significant (p >= 0.05)"
        lines.append(f"  -> {sig}")
        lines.append("")

    if categorical:
        for c in categorical:
            ref = sorted(cox_data[c].astype(str).unique())[0]
            lines.append(f"Reference level for '{c}': {ref}")
        lines.append("")

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)
    return report


def main():
    if len(sys.argv) >= 3:
        risk_dir = sys.argv[1]
        exp_name = sys.argv[2]
        out_dir = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DIR
    else:
        risk_dir = RISK_DIR
        exp_name = EXP_NAME
        out_dir = OUTPUT_DIR

    if not _HAVE_LIFELINES:
        print("ERROR: lifelines is not installed. Run `pip install lifelines` "
              "and try again -- this script cannot run without it.")
        sys.exit(1)

    if not os.path.isdir(risk_dir):
        print(f"ERROR: '{risk_dir}' is not a directory.")
        print("Edit RISK_DIR at the top of this script, then click Run again.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    df = load_intervals(risk_dir)
    strata_labels, was_binned = resolve_strata_col(df, PRIMARY_STRATA_COL)

    curve_path = os.path.join(out_dir, f"{exp_name}_survival_curve.png")
    summary_path = os.path.join(out_dir, f"{exp_name}_group_summary.csv")
    logrank_path = os.path.join(out_dir, f"{exp_name}_logrank_results.txt")
    cox_path = os.path.join(out_dir, f"{exp_name}_cox_results.txt")

    survival_curve_plot(df, strata_labels, curve_path, exp_name)
    summary = group_summary(df, strata_labels, summary_path)
    print(f"\n-> wrote {curve_path}")
    print(f"-> wrote {summary_path}")
    print(summary.to_string(index=False))

    logrank_report = logrank_tests(df, strata_labels, logrank_path, exp_name)
    print(f"\n-> wrote {logrank_path}\n")
    print(logrank_report)

    cox_report = cox_model(df, COX_COVARIATES, cox_path, exp_name)
    print(f"-> wrote {cox_path}\n")
    print(cox_report)


if __name__ == "__main__":
    main()