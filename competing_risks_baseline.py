"""
competing_risks_baseline.py -- combines the THREE separate censored
time-to-event exports from a single BehaviorSpace fixed arm
(risk-intervals, otp-enrollment-intervals, ssp-visit-intervals) into ONE
per-agent competing-risks table, instead of three independent KM curves
that each pretend the other two events don't exist.

WHY THIS IS A DIFFERENT ANALYSIS FROM THREE SEPARATE KM CURVES:
A plain Kaplan-Meier curve for "time to death" treats every agent who
enrolls in OTP or desists as ordinary right-censoring -- as if they just
vanished from observation for an unrelated reason. That's wrong: enrolling
in OTP or desisting are not random dropout, they are themselves outcomes
that PRECLUDE death from being the first thing that happens to that
agent. Doing this for all three events separately silently overestimates
each event's own probability, because each analysis is quietly ignoring
that the other two events are competing for the same "first thing that
happens" slot. The correct tool is a competing-risks framework: for each
agent, find whichever of {death, otp_enrollment, desistance} happened
FIRST (chronologically), and estimate a proper Cumulative Incidence
Function (CIF) per cause -- which is what AalenJohansenFitter below does.
Migration is included as a competing risk too (an agent who migrates out
is no longer at risk of any of the other three events either), and
"censored" covers agents who reached none of the four by the end of the
run.

Per-agent event assignment logic:
  1. From risk-intervals-run-*.csv: death/migration/desistance candidates,
     each with duration = time_at_risk, IF that was the agent's
     exit_reason. "censored" (still active at run end) is the fallback if
     none of the other candidates fire.
  2. From otp-enrollment-intervals-run-*.csv: an otp_enrollment candidate
     with duration = time_to_enrollment, IF event == 1 (they enrolled by
     their own censor point).
  3. Whichever candidate has the SMALLEST duration is the agent's actual
     first event -- e.g. an agent who enrolls in OTP on day 30 and later
     dies on day 50 correctly counts as an "otp_enrollment" case at
     duration 30, not a "death" case at duration 50, because enrollment
     was chronologically first.

SSP-visit data is loaded and reported in the per-agent table and the
summary counts, but is NOT treated as a competing risk for the CIF/Cox
analysis below -- visiting the SSP doesn't remove an agent from the risk
set for death/OTP/desistance the way those three do (an agent can visit
the SSP and then still die, enroll, or desist later), so folding it into
the same "first absorbing event" framework as the other three would be
statistically wrong. If you want time-to-SSP-visit stratified/modeled,
use stratified_survival.py on the SSP export directly instead.

Produces:
  1. <exp_name>_competing_risks_table.csv -- one row per agent: who,
       run_number, duration, event_type, housing_status,
       social_network_size, plus the raw per-source durations for
       reference.
  2. <exp_name>_event_counts.csv         -- N and % of agents per event
       type (death / otp_enrollment / desistance / migration / censored),
       plus SSP-visit incidence reported separately.
  3. <exp_name>_cif_curve.png            -- Aalen-Johansen cumulative
       incidence functions, one line per competing event type (NOT a KM
       curve -- CIFs for competing risks don't sum to a survival curve
       the way KM's complement does).
  4. <exp_name>_cause_specific_cox_<event>.txt -- one file per event
       type, a cause-specific Cox model (agents who had a DIFFERENT event
       type are censored at their own event time) with run-clustered SEs,
       covariates = COX_COVARIATES. This is the standard, most widely
       available alternative to a Fine-Gray subdistribution-hazard model
       (lifelines does not implement Fine-Gray as of this version) --
       note the interpretation difference in the CAUTION printed in each
       file.

Usage (command line):
    python3 competing_risks_baseline.py <risk-intervals-dir> <otp-enrollment-dir> <ssp-visit-dir> <exp-name> [output-dir]

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
    from lifelines import AalenJohansenFitter, CoxPHFitter
    _HAVE_LIFELINES = True
except ImportError:
    _HAVE_LIFELINES = False

# ── EDIT THESE, then click Run ─────────────────────────────────────────────
RISK_DIR = 'data/competing-risks-baseline-results'
OTP_DIR = 'data/competing-risks-baseline-results'
SSP_DIR = 'data/competing-risks-baseline-results'
EXP_NAME = 'competing-risks-baseline'
OUTPUT_DIR = 'results'

# Joint cause-specific Cox covariates, same convention as
# stratified_survival.py -- fit together, so a housing_status effect on
# (say) death is reported net of social_network_size, not as a separate
# marginal model.
COX_COVARIATES = ["housing_status", "social_network_size"]

# Same idea as stratified_survival.py: the CIF plot is built by fitting a
# separate Aalen-Johansen curve PER RUN, evaluating each on a common time
# grid, and averaging across runs -- instead of one pooled fit across every
# agent from every run (which produces a jagged line since it mixes
# run-to-run seed noise into a single curve). A light moving-average is
# applied afterward to remove residual jitter from averaging a limited
# number of runs. Neither of these changes the event-counts table or the
# cause-specific Cox models below -- those still use the full pooled data.
TIME_GRID_POINTS = 200
SMOOTHING_WINDOW = 9
# ────────────────────────────────────────────────────────────────────────────


def load_export(risk_dir, pattern):
    files = sorted(glob.glob(os.path.join(risk_dir, pattern)))
    if not files:
        return None, []
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df, files


def build_competing_risks_table(risk_df, otp_df):
    """
    One row per (who, run_number): the agent's first event among
    {death, migration, desistance, otp_enrollment}, or "censored" if none
    fired. housing_status/social_network_size come from risk_df, which
    every PWUD appears in regardless of what else happened to them.
    """
    key = ["who", "run_number"]
    base = risk_df[key + ["spawn_tick", "time_at_risk", "exit_reason",
                           "housing_status", "social_network_size"]].copy()
    base = base.rename(columns={"time_at_risk": "risk_duration"})

    otp = otp_df[key + ["time_to_enrollment", "event"]].copy() if otp_df is not None else None
    if otp is not None:
        otp = otp.rename(columns={"time_to_enrollment": "otp_duration",
                                   "event": "otp_event"})
        merged = base.merge(otp, on=key, how="left")
    else:
        merged = base.copy()
        merged["otp_duration"] = np.nan
        merged["otp_event"] = 0

    def pick_first_event(row):
        candidates = []
        if row["exit_reason"] == "death":
            candidates.append((row["risk_duration"], "death"))
        elif row["exit_reason"] == "migration":
            candidates.append((row["risk_duration"], "migration"))
        elif row["exit_reason"] == "desistance":
            candidates.append((row["risk_duration"], "desistance"))
        if row.get("otp_event", 0) == 1 and pd.notna(row.get("otp_duration")):
            candidates.append((row["otp_duration"], "otp_enrollment"))
        if not candidates:
            return pd.Series({"duration": row["risk_duration"], "event_type": "censored"})
        candidates.sort(key=lambda c: c[0])
        return pd.Series({"duration": candidates[0][0], "event_type": candidates[0][1]})

    picked = merged.apply(pick_first_event, axis=1)
    merged["duration"] = picked["duration"]
    merged["event_type"] = picked["event_type"]
    return merged


def event_counts(table, ssp_df, out_path):
    counts = table["event_type"].value_counts()
    total = len(table)
    lines_df = counts.reset_index()
    lines_df.columns = ["event_type", "n_agents"]
    lines_df["pct"] = (lines_df["n_agents"] / total * 100).round(2)

    ssp_note = pd.DataFrame()
    if ssp_df is not None:
        n_visited = (ssp_df["event"] == 1).sum()
        ssp_note = pd.DataFrame([{
            "event_type": "ssp_visit (reported separately, not a competing risk)",
            "n_agents": n_visited,
            "pct": round(n_visited / len(ssp_df) * 100, 2) if len(ssp_df) else 0,
        }])
    out = pd.concat([lines_df, ssp_note], ignore_index=True)
    out.to_csv(out_path, index=False)
    return out


def apply_smoothing(curve, window):
    """Centered moving average with edge handling (shrinking window near
    the boundaries so the curve still starts at/near 0 and ends at the
    true final value, rather than being pulled by zero-padding)."""
    if window <= 1:
        return curve
    s = pd.Series(curve)
    return s.rolling(window=window, center=True, min_periods=1).mean().values


def smoothed_cif_curve(table, code_map, event_of_interest, time_grid):
    """
    Fits a separate Aalen-Johansen CIF per run_number (using the full set of
    competing-event codes within that run, so the competing-risks logic is
    still respected within each run), evaluates each on the common
    time_grid, then averages across runs at each grid point. Mirrors
    smoothed_group_curve() in stratified_survival.py. A light moving-average
    is applied afterward (SMOOTHING_WINDOW) to remove leftover jitter from
    averaging a limited number of runs -- this only affects the drawn line,
    never the event_counts table or the cause-specific Cox models.
    """
    code = code_map[event_of_interest]
    ajf = AalenJohansenFitter()
    curves = []
    for run_id, run_df in table.groupby("run_number"):
        if len(run_df) == 0:
            continue
        try:
            ajf.fit(run_df["duration"], run_df["event_code"], event_of_interest=code)
        except Exception:
            continue
        vals = ajf.predict(time_grid).values
        curves.append(vals)
    if not curves:
        return None
    avg_curve = np.mean(np.vstack(curves), axis=0)
    return apply_smoothing(avg_curve, SMOOTHING_WINDOW)


def cif_plot(table, out_path, exp_name):
    fig, ax = plt.subplots(figsize=POSTER_FIGSIZE)
    event_types = [e for e in table["event_type"].unique() if e != "censored"]

    # AalenJohansenFitter needs a single integer-coded event column where
    # 0 = censored and each competing cause gets its own positive integer.
    code_map = {"censored": 0}
    for i, e in enumerate(sorted(event_types), start=1):
        code_map[e] = i
    table = table.copy()
    table["event_code"] = table["event_type"].map(code_map)

    time_grid = np.linspace(0, table["duration"].max(), TIME_GRID_POINTS)

    for e in sorted(event_types):
        n = int((table["event_type"] == e).sum())
        avg_curve = smoothed_cif_curve(table, code_map, e, time_grid)
        if avg_curve is None:
            continue
        ax.plot(time_grid, avg_curve, linewidth=3.4, label=f"{e} (n={n})")

    ax.set_xlabel("Ticks", fontsize=19)
    ax.set_ylabel("Cumulative incidence", fontsize=19)
    ax.set_title(f"Cumulative incidence functions — {exp_name}\n"
                 f"(competing risks: death, migration, desistance, otp_enrollment;\n"
                 f"mean of per-run curves)", fontsize=18, pad=14)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=POSTER_DPI)
    plt.close(fig)


def cause_specific_cox(table, event_of_interest, covariates, out_path, exp_name):
    """
    Cause-specific Cox model for ONE event type: agents who experienced a
    DIFFERENT event type are treated as censored at their own event's
    duration (standard cause-specific-hazard convention). This estimates
    the hazard of event_of_interest among agents still at risk of it --
    NOT the same quantity as a Fine-Gray subdistribution hazard, which
    keeps agents who had a competing event in the risk set indefinitely.
    lifelines does not implement Fine-Gray as of this version; a
    cause-specific Cox is the standard, most widely available substitute,
    but answers a subtly different question ("hazard among those still
    at risk" vs. "hazard in the whole original population including
    those removed by a competing event") -- state that distinction if
    this goes in a write-up.
    """
    data = table.copy()
    data["event"] = (data["event_type"] == event_of_interest).astype(int)

    missing = [c for c in covariates if c not in data.columns]
    if missing:
        print(f"  SKIPPED cause-specific Cox for '{event_of_interest}': "
              f"missing covariate(s) {missing}.")
        return None

    categorical = [c for c in covariates
                   if not pd.api.types.is_numeric_dtype(data[c])
                   or pd.api.types.is_bool_dtype(data[c])]
    numeric = [c for c in covariates if c not in categorical]
    dummies = pd.get_dummies(data[categorical], drop_first=True) if categorical else pd.DataFrame(index=data.index)
    cox_data = pd.concat(
        [data[["duration", "event", "run_number"] + numeric], dummies], axis=1
    ).dropna()

    n_events = cox_data["event"].sum()
    if n_events < 10:
        lines = [
            f"Cause-specific Cox model — {exp_name} — event type: {event_of_interest}", "",
            f"SKIPPED: only {n_events} events of this type -- too few for a stable "
            f"Cox fit. Consider more repetitions or a larger n-pwud if this event "
            f"type is one you specifically care about.",
        ]
        report = "\n".join(lines)
        with open(out_path, "w") as f:
            f.write(report)
        return report

    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="duration", event_col="event",
            cluster_col="run_number")

    lines = [
        f"Cause-specific Cox model (run-clustered SEs) — {exp_name}",
        f"Event of interest: {event_of_interest}",
        f"N agents: {len(cox_data)}   N events: {n_events}   "
        f"N runs (clusters): {cox_data['run_number'].nunique()}",
        "",
        "CAUTION: this is a CAUSE-SPECIFIC hazard model -- agents who experienced "
        "a DIFFERENT event type are censored at their own event's duration. This "
        "estimates the hazard among agents still at risk of THIS event, which is "
        "not the same quantity as a Fine-Gray subdistribution hazard (which would "
        "keep competing-event agents in the risk set). Standard errors are "
        "clustered by run_number.",
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
        unit_desc = f"'{term}' vs. reference level" if is_dummy else f"per one additional unit of '{term}'"
        lines.append(f"{term}:")
        lines.append(f"  HR = {hr:.4f}   95% CI [{hr_lo:.4f}, {hr_hi:.4f}]   p = {p_val:.3g}")
        if hr < 1:
            lines.append(f"  -> {unit_desc}: {(1-hr)*100:.1f}% lower cause-specific hazard")
        elif hr > 1:
            lines.append(f"  -> {unit_desc}: {(hr-1)*100:.1f}% higher cause-specific hazard")
        sig = "SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "not significant (p >= 0.05)"
        lines.append(f"  -> {sig}")
        lines.append("")

    if categorical:
        for c in categorical:
            ref = sorted(data[c].astype(str).unique())[0]
            lines.append(f"Reference level for '{c}': {ref}")
        lines.append("")

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)
    return report


def main():
    if len(sys.argv) >= 5:
        risk_dir, otp_dir, ssp_dir, exp_name = sys.argv[1:5]
        out_dir = sys.argv[5] if len(sys.argv) > 5 else OUTPUT_DIR
    else:
        risk_dir, otp_dir, ssp_dir, exp_name, out_dir = RISK_DIR, OTP_DIR, SSP_DIR, EXP_NAME, OUTPUT_DIR

    if not _HAVE_LIFELINES:
        print("ERROR: lifelines is not installed. Run `pip install lifelines`.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    risk_df, risk_files = load_export(risk_dir, "risk-intervals-run-*.csv")
    if risk_df is None:
        print(f"ERROR: no risk-intervals-run-*.csv files found in '{risk_dir}'. "
              f"This export is required (it's the base population and provides "
              f"death/migration/desistance).")
        sys.exit(1)
    print(f"Loaded {len(risk_files)} risk-interval files, {len(risk_df)} agent-rows.")

    otp_df, otp_files = load_export(otp_dir, "otp-enrollment-intervals-run-*.csv")
    if otp_df is None:
        print(f"WARNING: no otp-enrollment-intervals-run-*.csv files found in "
              f"'{otp_dir}' -- otp_enrollment will not be included as a competing risk.")
    else:
        print(f"Loaded {len(otp_files)} OTP-enrollment files, {len(otp_df)} agent-rows.")

    ssp_df, ssp_files = load_export(ssp_dir, "ssp-visit-intervals-run-*.csv")
    if ssp_df is None:
        print(f"NOTE: no ssp-visit-intervals-run-*.csv files found in '{ssp_dir}' "
              f"-- SSP-visit incidence will be omitted from the event-counts file.")
    else:
        print(f"Loaded {len(ssp_files)} SSP-visit files, {len(ssp_df)} agent-rows.")

    table = build_competing_risks_table(risk_df, otp_df)

    table_path = os.path.join(out_dir, f"{exp_name}_competing_risks_table.csv")
    table.to_csv(table_path, index=False)
    print(f"\n-> wrote {table_path} ({len(table)} agents)")

    counts_path = os.path.join(out_dir, f"{exp_name}_event_counts.csv")
    counts = event_counts(table, ssp_df, counts_path)
    print(f"-> wrote {counts_path}")
    print(counts.to_string(index=False))

    cif_path = os.path.join(out_dir, f"{exp_name}_cif_curve.png")
    cif_plot(table, cif_path, exp_name)
    print(f"\n-> wrote {cif_path}")

    for event_type in ["death", "otp_enrollment", "desistance", "migration"]:
        if event_type not in table["event_type"].unique():
            continue
        cox_path = os.path.join(out_dir, f"{exp_name}_cause_specific_cox_{event_type}.txt")
        report = cause_specific_cox(table, event_type, COX_COVARIATES, cox_path, exp_name)
        if report:
            print(f"\n-> wrote {cox_path}")
            print(report)


if __name__ == "__main__":
    main()