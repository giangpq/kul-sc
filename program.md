# program.md

This is an autonomous pricing research experiment for campsite accommodations.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may4`). The branch
   `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `data/processed/` — confirm `train.csv`, `val.csv`, `test.csv` exist.
   - `src/prepare.py` — **fixed, do not modify**. Contains data splits, the locked
     `simulate_revenue()` function, and the **locked price grid** used for all
     revenue evaluations.
   - `src/features.py` — **you may modify**. Feature engineering logic. Contains
     baseline features (do not remove) and an open experimental section.
   - `src/train.py` — **you may modify**. Model architecture and hyperparameters.
   - `src/price.py` — **you may modify**. Pricing strategy logic (how to pick the
     best price from the locked grid, future-week weighting). Does NOT control
     the price grid itself — that lives in `prepare.py`.
4. **Verify data exists**: Check that `data/processed/` contains the three split files.
   If not, tell the human to run `uv run src/prepare.py`.
5. **Initialize results files**:
   - `results_model.tsv` — for Phase 1 experiments (model quality).
   - `results_pricing.tsv` — for Phase 2 experiments (revenue lift).
   Create each with just the header row (see formats below). Do NOT commit either file.
6. **Confirm and go**: Confirm setup looks good, then kick off Phase 1.

---

## What you CAN do

- Modify `src/features.py` — engineer new features freely. All statistics (e.g. mean
  encodings, group aggregations) must be fitted on **training data only**, then applied
  to val/test. No leakage.
- Modify `src/train.py` — model type, hyperparameters, training logic.
  Use a Tweedie objective for whichever model family is active
  (e.g. LightGBM `objective="tweedie"`, XGBoost `objective="reg:tweedie"`,
  scikit-learn Tweedie-compatible loss/objective where applicable).
- Modify `src/price.py` — pricing selection logic only.
- Use `notebooks/experiment.py` interactively via marimo-pair skill for exploration.

## What you CANNOT do

- Modify `src/prepare.py`. It is read-only.
- Change the train/val/test splits.
- Evaluate on the test set during the experiment loop.
- Install new packages. Only use what's in `pyproject.toml` (optuna, mlflow is included).

---

## GPU check

Before the first run, confirm GPU acceleration is available:
- LightGBM: set `device="gpu"` — confirm no errors.
- XGBoost: set `device="cuda"` — confirm no errors.
- scikit-learn: CPU only, expected.

If GPU is unavailable, fall back to CPU silently.

---

## MLflow tracking

All experiments are tracked via MLflow:
- **Phase 1** logs to experiment `campsite-demand`.
- **Phase 2** logs to experiment `campsite-pricing`.

Each Phase 2 run logs `source_model_run_id` — the Phase 1 run it was built on —
so the link between model and pricing strategy is always traceable.

Phase 2 always loads the best Phase 1 model automatically (lowest `val_rmse`),
via `load_best_model_artifacts()` in `price.py`. No manual copying of model files.

---

## PHASE 1 — Model quality loop (up to 25 experiments)

**Goal**: Minimise val RMSE.
**Run command**: `uv run src/train.py > run.log 2>&1`
**Do NOT run `price.py` during this phase** — it takes ~6 minutes per run.

### Experiment axes (rotate — no more than 3 consecutive on the same axis)

1. **Feature engineering** (`features.py`): booking momentum, occupancy rate, price
   elasticity signals, target encodings, interaction terms; DO NOT use lag features - columns ending with LastYear since they are all 0s.
2. **Model architecture** (`train.py`): LightGBM, XGBoost, Random Forest; depth,
   estimators. Baseline is LightGBM — treat as a starting point, not a constraint.
3. **Hyperparameter tuning** (`train.py`): learning rate, leaf count, subsampling.
   Optuna is allowed here.

### The first run (baseline)

Run the existing scripts without any modifications and record the result.

### Output format

```
---
val_rmse:              12.450000
val_mae:               8.310000
val_wape:              0.184000
```

Extract with:
```
grep "^val_rmse:\|^val_mae:\|^val_wape:" run.log
```

### results_model.tsv format

```
commit	val_rmse	val_mae	val_wape	status	description
```

- `status`: `keep`, `discard`, or `crash`

Example:
```
commit	val_rmse	val_mae	val_wape	status	description
a1b2c3d	12.450000	8.310000	0.184000	keep	baseline
b2c3d4e	11.200000	7.800000	0.171000	keep	add momentum + occupancy features
c3d4e5f	13.100000	9.200000	0.199000	discard	switch to random forest
```

### Keep/discard rule

- **Keep** if `val_rmse` improves (lower). The MLflow run for this experiment is retained.
- **Discard** (`git reset --hard`) otherwise. The MLflow run remains for reference but the code is rolled back.
- Fix trivial crashes (typos, imports) and re-run. Discard fundamentally broken ideas after 2–3 attempts.
- **Simplicity criterion**: a marginal RMSE improvement that adds significant complexity is not worth it.

### Transition to Phase 2

After 25 Phase 1 experiments, or if RMSE has not improved for 5 consecutive runs, stop and wait for explicit human approval before starting Phase 2.

---

## PHASE 2 — Pricing strategy loop (up to 5 experiments)

**Goal**: Maximise simulated revenue lift on the val set.
**Run command**: `uv run src/price.py > run.log 2>&1`

Note: `train.py` does **not** need to re-run in Phase 2 unless the model changes.
`simulate_revenue()` is slow by design (~6 min) — think carefully before each experiment.

### Experiment axis

- **Pricing selection logic** (`price.py`): future-week discounting, occupancy-conditional
  strategies, smoothing, per-segment adjustments.
- Do NOT re-run Phase 1 model experiments here unless there is a strong reason.

### Output format

```
---
revenue_lift_mean_pct: 4.230000
revenue_lift_std_pct:  1.870000
```

Extract with:
```
grep "^revenue_lift_mean_pct:" run.log
```

### results_pricing.tsv format

```
commit	revenue_lift_mean_pct	revenue_lift_std_pct	source_model_run_id	status	description
```

Example:
```
commit	revenue_lift_mean_pct	revenue_lift_std_pct	source_model_run_id	status	description
a1b2c3d	0.000000	0.000000	abc1234	keep	baseline pricing
b2c3d4e	4.230000	1.870000	abc1234	keep	future-week discounting
c3d4e5f	3.100000	2.500000	abc1234	discard	occupancy threshold gating
```

### Keep/discard rule

- **Keep** if `revenue_lift_mean_pct` improves (higher).
- **Discard** (`git reset --hard`) otherwise.

---

## NEVER STOP

After logging each experiment, immediately proceed to the next one without ending
your response or waiting for human input. Run all experiments autonomously within
each phase. Stop only when both phases are complete, then write a summary.

---

## Final evaluation

After all experiments, evaluate the best model on the **test set once** (never repeated).
Temporarily point `price.py` at `test.csv`, run it, then restore `val.csv`.
Record the final test revenue lift in `logs/final_results.md`.
