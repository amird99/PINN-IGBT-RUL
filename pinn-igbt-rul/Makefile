.PHONY: install data test baseline pinn in-sample sweep-alpha bcc-only legacy all

install:
	pip install -e ".[dev]"

data:
	python scripts/download_data.py --out data/raw

test:
	pytest -q

# --- paper experiments -------------------------------------------------------------------
baseline:      # out-of-sample, alpha = beta = 0
	python scripts/train.py --config configs/baseline_lodo.yaml

pinn:          # out-of-sample, alpha = 0.1, beta = 100
	python scripts/train.py --config configs/pinn_lodo.yaml

in-sample:     # Table 1 (all four devices; use --set data.devices=[DeviceN] for one device)
	python scripts/train.py --config configs/baseline_in_sample.yaml
	python scripts/train.py --config configs/pinn_in_sample.yaml

sweep-alpha:   # Fig. 7: MDC only (beta = 0), alpha in {0, .1, .3, .5, .7, .9}
	for a in 0 0.1 0.3 0.5 0.7 0.9; do \
	  python scripts/train.py --config configs/pinn_lodo.yaml --set loss.alpha=$$a --set loss.beta=0 --set name=pinn_alpha_$$a; \
	done

bcc-only:      # Fig. 8: BCC only (alpha = 0, beta = 100)
	python scripts/train.py --config configs/pinn_lodo.yaml --set loss.alpha=0 --set loss.beta=100 --set name=pinn_bcc_only

# --- the original notebook's (non-paper) setup, for comparison -----------------------------
legacy:
	python scripts/train.py --config configs/legacy_notebook_baseline_lodo.yaml
	python scripts/train.py --config configs/legacy_notebook_pinn_lodo.yaml

all: baseline pinn in-sample
