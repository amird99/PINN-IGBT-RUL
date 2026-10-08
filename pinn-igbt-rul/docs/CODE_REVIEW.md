# Code review of `PINN_in_RUL_Estimation_IGBT.ipynb`

Reference paper: Z. Lu, C. Guo, M. Liu, R. Shi, *Remaining useful lifetime estimation for discrete power electronic devices
using physics-informed neural network*, Scientific Reports 13, 10167 (2023), doi:10.1038/s41598-023-37154-5.

**Scope.** I read all 65 cells, the saved outputs (shapes, training logs, figures) and then the paper. **Cell numbers** are 0-based
positions in the `.ipynb` JSON (Colab shows them offset by one). ID prefixes: **C** = changes the results, **M** = evaluation /
reproducibility hygiene, **m** = minor.

**Limits.**
* PyTorch was not installed where I worked, so the torch code in this repo (`models`, `losses`, `engine`, `runner`, `utils`) was
  **written and re-read but not executed**. The NumPy side (preprocessing, windows, splits, scaler, metrics, config) is covered by 17
  passing tests, which also reproduce numbers printed by your notebook (cycle counts 217/191/207/204; 155 test / 628 train in-sample
  windows). Run `pytest` once before publishing.
* Findings that use your logs are read off the printed values, not re-run.
* The paper does **not** specify: epochs, learning rate, batch size, framework/seed, how cycles are segmented, or whether
  standardisation is per device. Author-confirmed: `n_single_blocks` was determined from the dataset, standardisation is per device,
  and 2000 epochs / lr 1e-3 / batch 32 were chosen arbitrarily (not tuned). Those settings come from the author, not from the paper.

---

## 0. Notebook vs. paper

**Matches the paper (verified line by line).**

| Item | Paper | Notebook |
|---|---|---|
| Loss (Eq. 6/12), incl. `1/(n−1)` in MDC, `1/n` in BCC, `(1−α)·MSE + αγ·MDC + β·BCC` | ✔ | ✔ identical |
| Target `RUL(t) = 1 − t/N_f` (Eq. 1) | linear 1→0 | `np.linspace(1, 0, n)` ✔ |
| Window `s = 10`, many-to-one RNN, 80 recurrent units, 10 FC units, 1 linear output | ✔ | ✔ |
| Adam, MSE | ✔ | ✔ |
| Failure point = abrupt `V_CE` drop; data after it cut off | ✔ | ✔ (`find_failure`) |
| In-sample test set = "for every 5 samples, the last one" | ✔ | ✔ (`% 5 == 4`) |
| Out-of-sample = 4-fold leave-one-device-out | ✔ | ✔ |

**Deviates from the paper.**

| ID | Paper | Notebook |
|---|---|---|
| C4 | Preprocessing = average per cycle → **standardise** → **EMA (span 15)** | Standardise and EMA are computed, then **never used** (cell 29 uses `CE_cycle`) |
| C5 | 10-unit layer uses **tanh** (Eq. 3) | `nn.ReLU()` |
| C6 | Out-of-sample PI-RNN: α = 0.1, **β = 100** (Table 2) | β = 1 |
| C7 | PI-RNN is also trained **in-sample**, per device and on all four devices (Table 1) | Only an unshuffled in-sample *baseline* on the pooled data |
| C1 | Baseline RNN = same loss with α = β = 0, i.e. **same training loop** as the PI-RNN | Baseline: shuffled mini-batches of 32; PINN: one full-sequence step per device |

**Numbers.** Test MSE ×1e-3, out-of-sample (paper Table 2 vs. your logs; notebook PINN = composite at epoch 1900 divided by 0.9,
an approximation):

| Test device | Paper RNN | Paper PI-RNN | Notebook baseline | Notebook PINN |
|---|---|---|---|---|
| 5 (Case 1) | 14.99 | 2.47 | 7.0 | ≈ 4.3 |
| 4 (Case 2) | 1.66 | 1.48 | **12.1** | **≈ 17.4** |
| 3 (Case 3) | 6.63 | 5.77 | 8.4 | ≈ 8.7 |
| 2 (Case 4) | 8.73 | 5.88 | 10.3 | ≈ 7.5 |
| **Mean** | **8.01** | **3.90** | **9.5** | **≈ 9.5** |

Your runs are far from the paper on Device 4 (about 8-12× worse) and show no average PINN gain, while the paper reports −51 %.
A plausible mechanism, consistent with your cell-21 plot but **not tested**: raw `V_CE` plateaus sit at ≈ 5.0 V for Devices 2/4 and
≈ 5.5 V for Devices 3/5, and Device 4 spends cycles 70-148 at 4.5 V where the other devices pass through it earlier. A model trained on
raw voltage learns the wrong voltage↔life mapping for Device 4 (its prediction sits flat at ≈ 0.55 for that whole stretch, cell 55).
Per-device standardisation (C4) is exactly what removes this offset. Re-run with the paper-faithful defaults before drawing conclusions.

**Parameter count.** The paper reports 7,381 parameters (RNN) and 27,061 (LSTM). Those are exactly the Keras-style single-bias counts
(`80·81+80 + 810 + 11`). PyTorch's `nn.RNN` has two bias vectors, so this repo's model has **7,461**. The function class is identical.

### What does *not* apply

* **Physics / residual.** There is **no ODE/PDE residual**; the paper says so explicitly ("regularization, i.e., formulating soft
  constraints into the loss function"). MDC penalises `relu(ŷ_i − ŷ_{i−1})²` (sign correct for a target falling from 1 to 0, one-cycle
  finite difference), BCC penalises leaving `[0, 1]`. Both match Eq. 9 and 11.
* **Autograd.** No `autograd.grad`, no input derivatives, so `create_graph` / `retain_graph` / `requires_grad` on inputs are not
  needed. The loop is standard (`zero_grad → forward → backward → step`), `parts` are detached, `loss.item()` is used for logging.
  The one subtlety: `relu` hinge terms have **exactly zero gradient whenever the constraint is satisfied** (see C1).

### Checked and correct

Window/target alignment (tested); scalers fitted on **training windows only**, LODO test device never touches them; no silent shape
broadcasting (baseline `(B,1)/(B,1)`, PINN `(n,)/(n,)`); batch-size-weighted epoch losses; `eval()` + `no_grad` at evaluation;
failure detector checked visually (cell 16).

---

## 1. Summary

| ID | Severity | Where | One-line problem |
|---|---|---|---|
| C1 | **Critical** | cells 53, 58, 61 | Baseline and PINN use different training regimes; physics terms are numerically inactive in your logs |
| C3 | **Critical** | cells 47-48, 50 | "Load model" cell runs before training: warm start, and a clean run fails |
| C4 | **Critical** | cells 23, 26, 29 | Standardisation + EMA (part of the paper's preprocessing) are computed but unused |
| C5 | Moderate | cell 39 | ReLU instead of the paper's tanh |
| C6 | Moderate | cell 61 | β = 1 instead of the paper's β = 100 for out-of-sample |
| C7 | Moderate | notebook scope | In-sample PI-RNN and per-device in-sample runs (Table 1) are missing |
| C2 | Moderate | cell 31 | In-sample split is leaky (the paper's protocol, but optimistic) |
| M1 | Moderate | cell 34 | In-sample training loader uses `shuffle=False` |
| M2 | Moderate | cells 42, 53, 61 | Baseline logs MSE, PINN logs the composite loss |
| M3 | Moderate | cells 59, 61 | Logged `ols/mdc/bcc` are from the last batch only |
| M4 | Moderate | cells 54, 56, 62, 64 | Scaler not saved with the weights |
| M5 | Moderate | cell 20 | Cycle segmentation hard-codes record lengths and `n_single`; comment is a chat leftover |
| M6 | Moderate | cells 53, 61 | No validation split; last-epoch score is noisy |
| M7 | Moderate | all | One seed |
| M8 | Moderate | cell 58 | `PINNLoss` silently needs time-ordered batches |
| m1-m13 | Minor | various | New-API use, seeding, dead code, Colab magics, personal e-mail, notebook size… |

---

## 2. Critical issues

### C1 – Different regimes for baseline and PINN; inactive physics terms
**Location.** Baseline LODO loop (cell 53) vs `PINNLoss` / PINN loop (cells 58, 61).

**Why.** In the paper the baseline *is* the PINN loss with α = β = 0 ("if both α and β are equal to 0, the loss function represents the
case for the baseline"), so both models share one training loop. In the notebook they do not:

1. *Different budgets.* The baseline takes ~18-19 shuffled steps per epoch (575-601 windows / 32) ⇒ ≈ 37 000 steps; the PINN takes one
   full-sequence step per training device ⇒ **6 000 steps**, and its loss is still decreasing late in training (Device 2 fold: 8.4e-3 at
   epoch 400 → 4.1e-3 at 1900). A difference between the two could be the optimiser regime rather than the physics.
2. *Inactive constraints.* With `α=0.1, β=1, γ=0.1` the loss is `0.9·MSE + 0.01·MDC + 1·BCC`. Your log shows `mdc` ≲ 1e-4 and `bcc = 0.0000`
   on every line: weighted physics terms of ~1e-6 versus a data term of ~3e-3. An upward bump of +0.2 over 3 cycles in a 208-window
   sequence gives `mdc ≈ 6e-5`; to make it 5 % of the data term `αγ ≈ 3.5` would be needed (≈ 350× the current 0.01). The paper says γ
   puts OLS and MDC "on the same scale"; with your magnitudes it does not. Your test plots agree: PINN and baseline curves are nearly
   identical and the PINN still shows upward bumps on Devices 2/3/5 (cell 63). Re-check this after fixing C4-C6, since the paper
   reports a visible MDC effect at α = 0.1.

**Fix.** `configs/baseline_lodo.yaml` (α = β = 0, same `device_sequence` loop) is now the baseline; the old shuffled-mini-batch setup
survives only as `legacy_notebook_baseline_lodo.yaml`. Every loss term is logged per epoch (`engine.train_epoch`), and test-time `mdc`
and `violation_rate` are reported (`metrics.py`). Reproduce the paper's studies with `make sweep-alpha` (Fig. 7) and `make bcc-only` (Fig. 8).

### C3 – Stale-state "load model" cell before training
**Location.** Cell 47 (`load_state_dict(torch.load("VanillaRNN_InSample.pth"))`) sits between model creation (cell 45) and `run_training`
(cell 48); the weights are only saved afterwards (cell 50).

**Why.** Run top-to-bottom on a clean machine it raises `FileNotFoundError`. After one earlier session it silently **warm-starts**, so
cell 48 trains 2000 *more* epochs. Your log is consistent with that (train loss 45e-4 at epoch 100 and flat at 200, a level the LODO
baseline reaches only after ~900 epochs), though the execution counts were stripped so I cannot prove it. Also: `torch.load` there has no
`map_location` (fails on CPU for CUDA-saved weights) and no `weights_only` (FutureWarning on torch ≥ 2.4; default changed in 2.6).

**Fix.** Training always starts from a seeded fresh model; loading is a separate step:
```python
set_seed(cfg.seed)
model = RNNRegressor(**asdict(cfg.model)).to(device)             # runner.run_experiment
torch.load(path, map_location=device, weights_only=True)         # utils.load_checkpoint
```

### C4 – Standardisation and EMA smoothing computed but unused (confirmed by the paper)
**Location.** `CE_standard` (cell 23) and `CE_EMA` (cell 26) are plotted; `data` (cell 29) is built from **`CE_cycle`**.

**Why.** The paper's preprocessing is average downsampling → standardisation → EMA (span 15). Both models in the notebook are trained on raw
cycle means, so the results do not correspond to the paper's pipeline, and likely explain much of the gap in §0.

**Fix.** The defaults now follow the paper (`per_device_standardize: true`, `smoothing: ema`, `ema_span: 15`). The paper does not state whether
standardisation is per device or global; per device is what the author used (confirmed). The NumPy EMA is verified identical to
`pandas.Series.ewm(span, adjust=False)`, and `test_preprocessing_chain_matches_paper_order` pins the order. Note: per-device standardisation
uses whole-life statistics, so it is a look-ahead at deployment time (fine for reproducing the paper, worth stating in your write-up).
To reproduce the notebook: `configs/legacy_notebook_*.yaml`.

---

## 3. Moderate issues

### C5 – ReLU instead of tanh
**Location.** `self.relu = nn.ReLU()` (cell 39). Paper Eq. 3: `f = tanh(W_y h + b_y)`.
**Fix.** `model.activation: tanh` is the default (`ReLU` via `activation: relu`). No parameters change, so seeds/initialisation are identical.

### C6 – β = 1 for the out-of-sample PINN
**Location.** `beta=1` in cell 61. Paper Table 2 / Fig. 9 use **β = 100** (β = 1 is only for the in-sample Table 1).
**Why it matters.** The paper's biggest gain (Case 1, test MSE 14.99e-3 → 2.47e-3) comes from BCC fixing negative predictions at the end of life.
**Fix.** `configs/pinn_lodo.yaml` uses β = 100; `pinn_in_sample.yaml` uses β = 1.

### C7 – Table 1 experiments missing
**Location.** Notebook scope. The paper trains RNN and PI-RNN in-sample on each device individually and on all four together.
**Fix.** `pinn_in_sample.yaml` / `baseline_in_sample.yaml` (all four devices); single device via `--set data.devices=[Device3]`. The training windows
of each device keep their time order, so the MDC term compares consecutive training samples (a removed test window just makes it compare
samples two steps apart, which must still decrease). *(An earlier draft of this review wrongly said in-sample PINN was infeasible.)*

### C2 – In-sample split leaks (paper's protocol, optimistic)
**Location.** `in_sample_window_split`, cell 31.
**Why.** Windows have stride 1 and length 10, so window *k* shares 9 of 10 cycles with *k±1* and their labels differ by 1/(N−1). The paper uses this
split "to evaluate the model's learning performance", not generalisation, so it is acceptable, but in-sample scores must not be read as
generalisation. `test_default_split_leaks_and_purged_split_does_not` demonstrates it.
**Fix** (opt-in, default = paper): `--set data.insample_block_size=20 --set data.insample_purge=9`.

### M1 – `shuffle=False` for the in-sample training loader
**Location.** Cell 34. `train_ds` is ordered by device and time, so every batch is 32 consecutive windows of one device with near-identical
labels, in the same order each epoch. Your log shows spikes (202.7e-4 at epoch 300, 73.3e-4 at 1900 against ~45-55e-4 around them); the LODO
baseline does shuffle.
**Fix.** `shuffle=True` with a seeded generator (**behaviour change, flagged**); `legacy_notebook_baseline_in_sample.yaml` keeps `shuffle: false`.

### M2 – Non-comparable loss numbers
Cell 42 prints MSE ×1e-4 / ×1e-3; cell 61 prints the composite PINN loss (`0.9·MSE + …`). **Fix.** One `regression_metrics` (MSE, RMSE, MAE, R² as in paper
Eq. 13) plus `monotonicity_metrics` and `boundary_metrics` for every model; the per-epoch monitor is always `test_mse`.

### M3 – Loss parts of the last batch only
`last_parts = parts` (cell 59): printed `ols/mdc/bcc` belong to the last training device only. **Fix.** `engine.train_epoch` returns sample-weighted means.

### M4 – Scaler not stored with the model
Cells 54, 62 save `state_dict` only; `mu/sd` live in `folds` (cells 36, 60), which the PINN cell overwrites. **Fix.** `utils.save_checkpoint` stores weights +
config + scaler + history; `runner.evaluate_run` rebuilds the scaler and **raises** if it differs from the stored one.

### M5 – Cycle segmentation assumptions
Cell 20: `CHUNK = 125_000`, `CYCLE = CHUNK // 2`, `n_single = {...}`, comment "my reading of your description". The paper only says the signal was "downsampled to one
sample in one square wave cycle"; the segmentation is the author's. The author confirms `n_single` was **determined from the dataset**, so it is data-derived, not guessed; the
chat-leftover comment is gone. **Fix.** Values live in `configs/default.yaml` with the provenance noted. Optional safeguard:
`np.unique([len(t.timeDomain.collectorEmitterVoltage) for t in transients], return_counts=True)` per device; derive edges from those lengths if they vary.

### M6 – No validation split; noisy last-epoch score
Baseline Device 2 test loss ranges 75-103e-4 over the last six logged epochs, so the score depends on where you stop; with four devices LODO test = validation.
The paper reports no epoch count either; the author confirms epochs/lr/batch size were arbitrary and not tuned on the test devices, which
removes the tuning-leakage concern and leaves epoch-to-epoch noise as the risk (documented in the README). **Fix.** Report last epoch only (the code never selects on test), several seeds, or nested LODO if you tune.

### M7 – Single seed
Run `--set seed=0..4` and report mean ± std.

### M8 – `PINNLoss` silently needs time-ordered batches
With cell 53's shuffled `DataLoader` it would compute MDC across unrelated windows with no error. **Fix.** `Config.validate()` rejects `loss.name=pinn` unless
`train.batching=device_sequence` (tested).

---

## 4. Minor issues

| ID | Location | Problem → fix |
|---|---|---|
| m1 | cell 43 | `torch.accelerator` needs torch ≥ 2.6 → `utils.resolve_device` |
| m2 | cells 47, 56, 64 | `torch.load` without `weights_only` → `torch.load(path, map_location=device, weights_only=True)` |
| m3 | cells 34, 45, 52, 61 | Only `torch.manual_seed`; no NumPy/`random` seed or cuDNN determinism; global-RNG shuffling → `utils.set_seed` + `DataLoader(generator=…)` (shuffle order therefore differs from the notebook) |
| m4 | various | Dead/duplicate code: `coarse` (15), `names_te` + `stack_with_names` (32), `h_n` (39), `to_tensor` twice (34, 53), `predict` twice with different signatures (49, 55), `load_results_from_checkpoints` twice (56, 64), trivial `make_windows_device` (36) → removed |
| m5 | cells 2-6 | Colab-only magics and hard-coded Drive paths → `data.raw_dir` + `scripts/download_data.py` |
| m5b | cell 4 | `git config --global user.email …` puts a **personal e-mail** in the notebook (and its history) → removed; if the notebook was ever pushed, scrub history (`git filter-repo`) or treat it as public |
| m6 | cell 61 | `range(2000)` and `"/2000"` hard-coded; logs at epochs 0…1900 vs 100…2000 elsewhere → `train.epochs`, logs at 1 and every `log_every` |
| m7 | cells 21-27 | x-label "Sample" on per-cycle plots → "Cycle" |
| m8 | cell 15 | `find_failure` has no sanity checks → raises if no healthy block; keep the visual check |
| m9 | cells 40-42 | Constant LR, no clipping: not a bug; `train.grad_clip` added (default `null`) |
| m10 | cell 58 | `sum/(n−1)` ≡ `mean` → simplified (verified identical) |
| m11 | cell 59 | Devices always visited in the same order → optional `train.shuffle_devices` |
| m12 | file | 1.1 MB from embedded figures → `nbstripout` before committing |
| m13 | model | Unbounded output; BCC is soft → `boundary_metrics` reports violations; clip at inference if a guaranteed range is needed |

---

## 5. What changed vs. the notebook

**Now follows the paper (defaults; flagged as bug fixes):** C4 standardise + EMA; C5 tanh; C6 β = 100 (out-of-sample); C1 baseline = α=β=0 in the same loop; C7 in-sample PINN.

**Other flagged fixes:** C3 no warm start; M1 shuffled in-sample training; M2 common metrics; M3 correct loss-part logging; M4 scaler in the checkpoint; m3 seeding
(batch order differs from the notebook, so numbers will not match bit-for-bit).

**Unchanged:** window, split sizes (tested: 155/628), LODO folds and scalers, Adam, lr 1e-3, 2000 epochs, batch size 32 (mini-batch variants), architecture sizes
and initialisation, no early stopping.

**Reproduce the notebook exactly:** `configs/legacy_notebook_*.yaml` (raw `V_CE`, ReLU, β = 1, shuffled mini-batch baseline, unshuffled in-sample baseline).

| Notebook | Repo |
|---|---|
| cells 2-6, 8-11 | `data.load_vce`, `scripts/download_data.py` |
| cells 15-21 | `data.find_failure_index`, `data.cycle_means` |
| cells 23-27 | `data.build_cycle_series` (now used) |
| cells 29-36 | `windows.py` (windows, splits, `Scaler`) |
| cell 39 | `models.RNNRegressor` |
| cells 40-43, 45-48, 53 | `engine.py`, `runner.py` |
| cells 58-61 | `losses.PINNLoss`, `runner._train_batches` (`device_sequence`) |
| cells 49, 55, 63 | `plotting.py`, `metrics.py` |
| cells 50, 54, 56, 62, 64 | `utils.save_checkpoint` / `load_checkpoint`, `runner.evaluate_run` |

---

## Resolved with the author

* `n_single_blocks` was determined from the dataset.
* Standardisation is per device.
* Epochs / learning rate / batch size were chosen arbitrarily (not tuned); α, β, γ are the paper's.
* Dataset handling: not redistributed; cited as "J. Celaya, Phil Wysocki, and K. Goebel (2009) 'IGBT Accelerated Aging Data Set', NASA Prognostics
  Data Repository, NASA Ames Research Center, Moffett Field, CA".
* Environment: torch 2.11.0+cu130, numpy 2.1.3, scipy 1.16.3 (pinned in `requirements.txt`).

## Still open

1. Author name and GitHub username for `LICENSE`, `CITATION.cff` and the clone URL (placeholders `TODO` for now).
2. (Optional) matplotlib and PyYAML versions, if you want them pinned exactly.
3. The torch code has still not been executed by the reviewer (torch 2.11 postdates the reviewer's knowledge): run `pytest` and one short
   run (`--set train.epochs=5`) on your machine and report any error.
