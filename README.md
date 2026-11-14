# Kensington overdose simulation analysis

Python analysis code by Ansh Gosain.

These scripts analyze NetLogo BehaviorSpace exports from an agent-based study of harm reduction interventions in Kensington, Philadelphia. The scenario analysis compares simulated survival across intervention arms. The interval analyses examine how survival varies with model parameters, housing status, and competing first events.

## Run the scenario example

```sh
python -m pip install -r requirements.txt
python compare_scenarios.py results baseline=examples/baseline.csv combined=examples/combined.csv
```

The two small synthetic tables illustrate the input format. The script writes a survival plot and a CSV summary to `results/`. At the final tick the example’s mean survival fractions are 0.85 for baseline and 0.95 for combined. Research estimates require the original simulation exports.

## Analyses

| Poster panel | Script | Inputs |
|---|---|---|
| 2A · Intervention scenarios | `compare_scenarios.py` | One BehaviorSpace table per scenario, with optional interval files for survival tests |
| 2B · Crowding weight | `Survival_Curve_BehaviorSpace_Analysis.py` | Parameter sweep table and corresponding per-agent intervals |
| 2C · Housing status | `stratified_survival.py` | Per-agent intervals with housing or network covariates |
| 2D · Competing risks | `competing_risks_baseline.py` | Mortality and treatment enrollment exports from the same runs, with optional syringe-service visit counts |

The competing-risks analysis matches agents across exports by run number and agent ID. Use exports from the same simulation realizations. The utility `concatenate_risk_intervals.py` combines per-run interval CSVs with matching headers.
