# pinn-igbt-rul

**Physics-informed RNN for Remaining Useful Life (RUL) estimation of IGBTs** on the NASA IGBT accelerated-aging data.

An unofficial PyTorch re-implementation of Z. Lu, C. Guo, M. Liu and R. Shi, *"Remaining useful lifetime estimation for
discrete power electronic devices using physics-informed neural network"*, **Scientific Reports 13, 10167 (2023)**
([doi:10.1038/s41598-023-37154-5](https://doi.org/10.1038/s41598-023-37154-5)). A vanilla RNN is trained with two extra loss terms that encode
physical facts about the RUL curve, so predictions stop "going upwards" and stop "ending in the wrong place". This repository is not
affiliated with the paper's authors and does not claim to reproduce its numbers (see [Expected results](#expected-results)).

> **Status:** portfolio project; a cleaned-up, tested refactor of a Colab notebook. An audit of that notebook against the paper is in
> [`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md). Cells marked `TBD` below must be filled from a fresh run.

---

## The physics problem

An IGBT degrades under thermal overstress; its collector-emitter voltage `V_CE` rises as it ages until the device latches up and fails.
The paper uses `V_CE` as the precursor signal and the linear target (paper Eq. 1)

$$\mathrm{RUL}(t) = 1 - \frac{t}{N_f}, \qquad t \in [0, N_f] \quad (1 = \text{new},\ 0 = \text{failure}),$$

from which two physical rules follow: RUL **decreases monotonically**, and it **starts at 1 and ends at 0**. They are imposed as **soft constraints**
in the loss (no differential equation, no autograd derivative; the "derivative" is a finite difference between consecutive cycles):

| Constraint | Meaning | Penalty (paper Eq. 9 / 11) |
|---|---|---|
| **MDC** – monotonic decrease | RUL never increases with time | $\frac{1}{n-1}\sum_{i=1}^{n-1}\max(0,\hat y_i-\hat y_{i-1})^2$ |
| **BCC** – boundary condition | RUL stays in $[0,1]$ | $\frac1n\sum_i\left[\max(0,-\hat y_i)^2+\max(0,\hat y_i-1)^2\right]$ |

Total loss (paper Eq. 6 / 12), implemented in [`src/pinn_igbt/losses.py`](src/pinn_igbt/losses.py):

$$\mathcal{L} = (1-\alpha)\,\underbrace{\tfrac1n\sum_i (y_i-\hat y_i)^2}_{\text{OLS (MSE)}} \;+\; \alpha\gamma\,\mathrm{MDC} \;+\; \beta\,\mathrm{BCC}$$

With $\alpha=\beta=0$ this is the baseline RNN. Because MDC compares *consecutive cycles*, training uses one **time-ordered batch per device**.

## Model and preprocessing (as in the paper)

* **Network:** many-to-one RNN, window $s=10$ of one feature → 80 tanh recurrent units → 10 tanh units → 1 linear output. (PyTorch's `nn.RNN` has an extra
  bias vector: 7,461 parameters vs. the paper's 7,381. The function class is the same.)
* **Preprocessing:** failure point = abrupt `V_CE` drop, later data discarded → mean `V_CE` per switching cycle → standardisation → EMA (span 15).
  The paper does not say whether standardisation is per device or global; this implementation standardises **each device separately**.
  Cycle boundaries come from the dataset's record structure (`data.chunk_size`, `data.n_single_blocks`).
* **Optimiser:** Adam. The paper gives no epochs, learning rate or batch size; this repo uses 2000 epochs, lr 1e-3 and (for mini-batch variants) batch size 32,
  chosen arbitrarily and **not tuned**. α, β, γ are the paper's values.

## Evaluation protocols

* **Out-of-sample (leave-one-device-out, 4-fold)** – train on three devices, test on the fourth (paper Table 2). The meaningful generalisation number.
  Paper settings: α = 0.1, β = 100, γ = 0.1.
* **In-sample** – 80 / 20 split in which the last of every 5 windows is test (paper Table 1; α = 0.1, β = 1), on each device alone or on all four.
  Windows overlap by 9 of 10 cycles, so scores are **optimistic**; the paper uses it to assess *learning*, not generalisation. A leakage-free variant:
  `--set data.insample_block_size=20 --set data.insample_purge=9`.

## Repository layout

```
pinn-igbt-rul/
├── configs/            # YAML experiments; default.yaml lists every hyper-parameter
├── data/               # raw data goes in data/raw (not committed); see data/README.md
├── docs/CODE_REVIEW.md # audit of the original notebook vs. the paper
├── notebooks/          # thin walkthrough notebook (no logic)
├── results/            # run outputs (git-ignored); curated ones in results/pretrained, results/figures
├── scripts/            # train.py, evaluate.py, download_data.py
├── src/pinn_igbt/      # data, windows, models, losses, engine, runner, metrics, plotting, config, utils
└── tests/              # pytest suite (preprocessing, splits, losses, end-to-end smoke tests)
```

## Data

The dataset is **not** included in this repository and must not be committed (`data/raw/` is git-ignored). Download it from the NASA Prognostics
Data Repository and cite it as:

> J. Celaya, Phil Wysocki, and K. Goebel (2009) "IGBT Accelerated Aging Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

See [`data/README.md`](data/README.md) for the link and the expected file layout.

## Installation

```bash
git clone https://github.com/TODO-user/pinn-igbt-rul.git
cd pinn-igbt-rul
python -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install -e ".[dev]"                                 # or: pip install -r requirements.txt
# get the NASA data yourself (it is not redistributed here): see data/README.md
```

## Usage

```bash
make baseline      # out-of-sample, alpha = beta = 0            (configs/baseline_lodo.yaml)
make pinn          # out-of-sample, alpha = 0.1, beta = 100     (configs/pinn_lodo.yaml)
make in-sample     # in-sample baseline and PI-RNN, all four devices
make sweep-alpha   # paper Fig. 7: MDC only, alpha in {0, .1, .3, .5, .7, .9}
make bcc-only      # paper Fig. 8: BCC only (alpha = 0, beta = 100)
make legacy        # the ORIGINAL notebook's setup (raw V_CE, ReLU, beta = 1), for comparison
make test
```

Equivalent explicit commands, with overrides:

```bash
python scripts/train.py --config configs/pinn_lodo.yaml
python scripts/train.py --config configs/pinn_lodo.yaml --set train.epochs=300 --set loss.alpha=0.5 --set seed=1
python scripts/train.py --config configs/baseline_in_sample.yaml --set "data.devices=[Device3]"   # one device
python scripts/evaluate.py --run-dir results/pinn_lodo      # re-score saved checkpoints, no retraining
```

Each run writes `results/<name>/`: `config.yaml`, `metrics.json`, `predictions.png`, `history.png`, `predictions.npz` and one checkpoint per fold. A checkpoint
bundles the weights **with** the input scaler and config needed to reuse them.

## Reproducibility

* Seeds for Python, NumPy and PyTorch are fixed (`seed: 42`) right before each model is built; deterministic kernels are requested.
* Bit-exact equality is **not** guaranteed across hardware/CUDA versions. Run-to-run variation is large on this data (test MSE can swing by tens of
  percent between epochs), so report **several seeds** (mean ± std).
* Scores are taken at the **last epoch**; the test device is monitored but never used for model selection.
* `requirements.txt` pins the author's environment: torch 2.11.0 (CUDA 13 build), numpy 2.1.3, scipy 1.16.3.

## Expected results

![LODO predictions](results/figures/lodo_predictions.png)
<!-- TODO: copy results/pinn_lodo/predictions.png to results/figures/lodo_predictions.png after a fresh run. -->

Out-of-sample test MSE (×10⁻³). *Paper* columns are copied from Table 2 of the paper (reported by the authors, **not reproduced here**);
fill *This repo* from `results/*/metrics.json`, ideally as mean ± std over ≥ 5 seeds.

| Test device | Paper RNN | Paper PI-RNN | This repo: baseline | This repo: PINN |
|---|---|---|---|---|
| 5 (Case 1) | 14.99 | 2.47 | TBD | TBD |
| 4 (Case 2) | 1.66 | 1.48 | TBD | TBD |
| 3 (Case 3) | 6.63 | 5.77 | TBD | TBD |
| 2 (Case 4) | 8.73 | 5.88 | TBD | TBD |
| **Mean** | **8.01** | **3.90** (−51.3 %) | TBD | TBD |

Also report R² and the MDC-violation rate (`metrics.json`).

What to expect qualitatively:

* Per-cycle `V_CE` rises in steps and then **plateaus for roughly the second half of life**, so late-life RUL is weakly identifiable; the paper notes the
  predicted RUL "tends to be constant at the end".
* The constraints act on training data only; on an unseen device small upward bumps can remain (see the MDC-violation metric).
* The MDC term is small next to the data term at the default weights; sweep `loss.alpha` / `loss.gamma` (`make sweep-alpha`) to see its effect.

## Known limitations

* One input feature (cycle-mean `V_CE`), four devices, one test device per fold.
* Cycle segmentation assumes fixed-length 125 000-sample records (`data.chunk_size`, `data.n_single_blocks`, derived from the dataset by the author); the paper does not describe it.
* Per-device standardisation uses whole-life statistics (a look-ahead at deployment time).
* The model predicts *normalised* RUL; converting to cycles needs the end of life.
* The paper's LSTM / PI-LSTM experiments are not implemented.

## Citation

If you use this code, cite the original paper and the dataset:

```bibtex
@article{lu2023remaining,
  title   = {Remaining useful lifetime estimation for discrete power electronic devices using physics-informed neural network},
  author  = {Lu, Zhonghai and Guo, Chao and Liu, Mingrui and Shi, Rui},
  journal = {Scientific Reports},
  volume  = {13},
  pages   = {10167},
  year    = {2023},
  doi     = {10.1038/s41598-023-37154-5}
}
```

Dataset (not redistributed here): J. Celaya, Phil Wysocki, and K. Goebel (2009) "IGBT Accelerated Aging Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

```bibtex
@misc{celaya2009igbt,
  author       = {Celaya, J. and Wysocki, Phil and Goebel, K.},
  title        = {{IGBT Accelerated Aging Data Set}},
  howpublished = {NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA},
  year         = {2009}
}
```

Software: see [`CITATION.cff`](CITATION.cff).

## License

Code: [MIT](LICENSE). The NASA dataset is **not** included and is subject to its own terms.
