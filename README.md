# Stochastic SINDy (SSISDE)

Recover drift, diffusion, and Brownian noise from simulated stochastic processes,
then fit a Stochastic SINDy model. Each experiment compares three drift-recovery
methods: **Method 1, Method 2, and ODE2**.

## Experiments

| Experiment | Notebook |
|---|---|
| Black-Scholes | [Open notebook](experiments/black_scholes/Stochastic_SINDy_Black_Scholes.ipynb) |
| Constant coefficients | [Open notebook](experiments/constant/Stochastic_SINDy_constant.ipynb) |
| Quadratic coefficients | [Open notebook](experiments/quadratic/Stochastic_SINDy_Quadratic.ipynb) |
| Ornstein-Uhlenbeck | [Open notebook](experiments/ou/Stochastic_SINDy_OU.ipynb) |

## Setup

Use **Python 3.11**. From the repository folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

## Run an experiment

In VS Code, open a notebook above, select `.venv/bin/python` as the kernel, and
choose **Run All**.

Or run from the terminal:

```bash
python scripts/run_experiment.py experiments/black_scholes/Stochastic_SINDy_Black_Scholes.ipynb
```

Replace the notebook path to run another experiment. Add `--in-place` to save
fresh outputs in the source notebook. Default runs use 100,000 samples and
cross-validation, so they can take time.

To refresh all four experiment notebooks:

```bash
for notebook in experiments/*/*.ipynb; do
  python scripts/run_experiment.py "$notebook" --in-place || break
done
```

- **Plots:** saved beside each notebook in `experiments/<case>/`.
- **Results and executed notebooks:** saved in `outputs/<case>/`.
- **Method comparison:** `ssisde_comparison.json` reports noise, drift, and
  diffusion RMSE for all three methods.

Current figures and numerical results are tracked in Git with the source
notebooks. Duplicate executed notebooks, caches, logs, and retired tuning
artifacts remain local.

## Code structure

```text
data_generation/
  processes.py        Process coefficients and experiment generators
  simulation.py       Brownian noise and Euler-Maruyama simulation
  coarse_graining.py  Sampling onto the recovery grid
models/
  diffusion.py        Quadratic variation, diffusion fitting, and derivatives
  drift.py            Signature regression and drift recovery
  noise.py            Q-noise and physical Brownian-motion recovery
  stochastic_sindy.py Model fitting, tuning, and plots
```

`experiments/` contains the four experiment folders; `original/` contains the
input notebooks with retired process options removed. See the
[function reference](output/pdf/function_reference/SSISDE_function_reference.pdf)
for function inputs and outputs.

## Verification

```bash
python scripts/verify_preservation.py
```

This checks numerical preservation against Git commit `d8152df` and tests ODE2
against known solutions. The checkout must include that commit's history.
All four experiments passed a full local rerun on **2026-10-20**, using 100,000
samples and cross-validation. The 53 saved figures come from those runs; 12
obsolete figures from earlier workflows were removed.

## Colab

Use a Python 3.10–3.12 runtime. The Black-Scholes and OU setup cells can download
the private repository using a GitHub token with read-only Contents access.
For constant or quadratic experiments, first obtain the checkout using one of
those setup cells. Push local changes before starting a fresh remote runtime.
Live Colab execution has not been verified.

## Deferred example

The double-well example is removed for now. Its notebooks, studies, and reports
remain in Git commit `ae83792619c348eb7cac72976117fdddaf84aef7`.
Local backups are `.archive/double_well-ae83792.tar.gz` and
`.archive/original-notebooks-ae83792.tar.gz` (ignored by Git).
Restore into a separate directory first, then adapt imports to the modules above.
