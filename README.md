# Stochastic SINDy (SSISDE)

Recover drift, diffusion, and Brownian noise from a single simulated path, then
identify a sparse Stochastic SINDy model. Each experiment compares **Method 1,
Method 2, and ODE2** using the same observed path and final polynomial libraries.

The repository contains four active experiments, their comparison figures,
numerical results, and reproducible report sources. Start with the
[recovered-model summary PDF](output/pdf/recovered_models/SSISDE_recovered_models.pdf)
for all 12 experiment/method combinations, including the exact and selected
functions, final increment MSE, and selected hyperparameters.

## Experiments

The diffusion below is the amplitude multiplying Brownian increments, not the
variance. Every experiment uses 100,000 fine-grid samples, duration `T = 1`, and
initial state `X0 = 1`. The recovery grid contains 1,000 sampled states.

| Experiment | Exact drift | Exact diffusion | Seed | Notebook |
|---|---|---|---|---|
| Black-Scholes | `0.5*x` | `0.3*x` | 3 | [Open](experiments/black_scholes/Stochastic_SINDy_Black_Scholes.ipynb) |
| Constant coefficients | `0.5` | `1.0` | 31 | [Open](experiments/constant/Stochastic_SINDy_constant.ipynb) |
| Quadratic coefficients | `0.7*x + 0.3*x^2` | `0.5*x + 0.1*x^2` | 2 | [Open](experiments/quadratic/Stochastic_SINDy_Quadratic.ipynb) |
| Ornstein-Uhlenbeck (OU) | `-0.5*x` | `0.3` | 3 | [Open](experiments/ou/Stochastic_SINDy_OU.ipynb) |

These are controlled simulation experiments: the true coefficients and driving
noise are available for error comparisons. Drift recovery also uses supplied
initial information; ODE2 additionally uses the initial spatial drift slope.
The notebooks document the inputs used for each process.

## Setup

Use **Python 3.11**. From the repository folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell. `requirements.txt` contains the numerical dependencies;
`requirements-dev.txt` also installs the notebook kernel and execution tools.
The pinned versions keep the existing PySINDy workflow reproducible.

In VS Code, select `.venv/bin/python` as the notebook kernel, or the equivalent
Python executable inside `.venv` on Windows. Run cells in order from a fresh
kernel so that each stage uses the current experiment's arrays.

## Run an experiment

Open one of the notebooks above and choose **Run All**, or use the terminal:

```bash
python scripts/run_experiment.py experiments/black_scholes/Stochastic_SINDy_Black_Scholes.ipynb
python scripts/run_experiment.py experiments/constant/Stochastic_SINDy_constant.ipynb
python scripts/run_experiment.py experiments/quadratic/Stochastic_SINDy_Quadratic.ipynb
python scripts/run_experiment.py experiments/ou/Stochastic_SINDy_OU.ipynb
```

Each command runs the notebook in a fresh kernel. It saves figures beside that
notebook and numerical outputs under `outputs/<case>/`. It also writes a local
`*.executed.ipynb` execution record, including partial outputs if execution fails.
Add `--in-place` to update the source notebook's saved outputs after a successful
run. Runs use the full sample count and cross-validation, so they can take time.

To refresh all four source notebooks:

```bash
for notebook in experiments/*/*.ipynb; do
  python scripts/run_experiment.py "$notebook" --in-place || break
done
```

## Recovery and model selection

1. Simulate the process and estimate diffusion from quadratic variation.
2. Recover the Q-noise and compute the quantities needed for drift recovery.
3. Compare Method 1, Method 2, and ODE2 on the sampled path.
4. Use each drift estimate to recover its own physical Brownian-noise path.
5. Fit a joint drift-diffusion model using that recovered noise, then compare
   the recovered coefficients, noise, and reconstructed path with the simulation.

Both final polynomial libraries contain `1, x, x^2, ..., x^5`. The active
experiments use ElasticNet with seven-fold time-series cross-validation.
The alpha grid contains 35 logarithmically spaced values from `1e-5` to `1e1`.
The rho grid is `[0.7, 0.85, 0.95, 1.0]` for constant coefficients and
`[0.3, 0.4, 0.5, 0.7, 0.85, 0.95, 1.0]` for the other three experiments.

Within each fold, normalization uses the training rows. ElasticNet selects
active terms, and restricted least squares refits their coefficients before
validation error is evaluated. The one-standard-error rule selects a candidate
within one standard error of the lowest mean validation error, preferring larger
alpha and then larger rho. This favors stronger regularization rather than
explicitly minimizing the number of terms.

The selected model is fitted again on all data and de-biased on its active
terms. Final polynomial coefficients are then rounded to five decimal places.
Predictions, plots, and final errors use these rounded coefficients; unrounded
coefficients and increment MSE are retained in the comparison JSON files.

## Results and figures

Each `outputs/<case>/ssisde_comparison.json` contains entries for `Method 1`,
`Method 2`, and `ODE2`:

| Field | Meaning |
|---|---|
| `mu`, `sigma` | Final recovered polynomial functions |
| `mu_coefficients`, `sigma_coefficients` | Coefficients in ascending polynomial order, including the constant term |
| `alpha`, `rho` | Selected ElasticNet strength and L1 mixing ratio |
| `input_drift_rmse` | Error of the drift supplied to noise recovery |
| `noise_rmse` | Error of the recovered physical noise against the simulated noise |
| `drift_rmse`, `diffusion_rmse` | Errors of the final functions along the sampled path |
| `increment_mse` | Mean squared increment prediction error using the final rounded model and recovered noise |

`model_truth.json` records the exact coefficients. `ssisde_comparison_paths.npz`
contains the sampled state, noise, and fitted arrays for all three methods.
Additional `results.json` and diffusion-tuning files are retained where the
notebook exports them.

**Interpret the errors together.** Final increment MSE is a fit error, not the
cross-validation score or an independent forecast error. A small increment MSE
and a close reconstructed path do not by themselves establish accurate drift
or noise recovery. Compare the drift, diffusion, and noise errors as well.

Each experiment folder keeps nine method figures: CV curves, support-size
curves, and a four-panel results plot for each of the three methods. It also
contains one combined drift-comparison figure, for **40 saved figures** overall.
Other diagnostic plots remain visible inside the notebooks.

| Experiment | Method-figure prefix | Combined drift figure |
|---|---|---|
| Black-Scholes | `Psi_approx_BS` | `drift_comparison_BS.png` |
| Constant coefficients | `constant` | `drift_comparison.png` |
| Quadratic coefficients | `Psi_true_Quad` | `drift_comparison_Quad.png` |
| OU | `Psi_approx_OU` | `drift_comparison_OU.png` |

For example, the OU ODE2 results figure is
`experiments/ou/Psi_approx_OU_ode2_results.png`. Other suffixes are
`_CV_curves.png` and `_delta_vs_n.png`, with method identifiers `method_1`,
`method_2`, and `ode2`. Support-size diagnostics and final rounded-model support
can differ; the plotted markers distinguish them.

## PDF reports

- [Recovered models, hyperparameters, and MSE](output/pdf/recovered_models/SSISDE_recovered_models.pdf):
  all four processes and all three methods, with exact and recovered functions.
- [Function reference](output/pdf/function_reference/SSISDE_function_reference.pdf):
  inputs and outputs of the data-generation and estimation modules.

To refresh the model-summary PDF after changing experiment results, install a
LaTeX distribution with `pdflatex`, then run from the repository root:

```bash
python output/pdf/recovered_models/build_tables.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf/recovered_models output/pdf/recovered_models/SSISDE_recovered_models.tex
```

The generator reads the current comparison and truth JSON files. It displays
final MSE to six significant figures in scientific notation and coefficients,
alpha, and rho to five decimal places. Hyperparameter selection retains full
precision. The editable [LaTeX source](output/pdf/recovered_models/SSISDE_recovered_models.tex)
is tracked with the PDF.

The function-reference generator requires the optional `reportlab` package:

```bash
python -m pip install reportlab
python output/pdf/function_reference/build_reference.py
```

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
  stochastic_sindy.py Model fitting, cross-validation, and plots
experiments/           Four current notebooks and their comparison figures
outputs/               Saved numerical results and comparison arrays
output/pdf/            PDF reports and their generators
scripts/               Notebook execution and numerical verification
```

## Verification

```bash
python scripts/verify_preservation.py
```

This checks module and notebook imports, verifies ODE2 against known solutions,
compares preserved numerical stages against Git commit `d8152df`, and checks
that predictions and MSE use the rounded final coefficients. The checkout must
include that commit's history; a shallow clone or downloaded ZIP may not.
The verification is a regression check, not a substitute for rerunning a changed
experiment at its full sample count.

## Colab and generated files

Use a Python 3.10–3.12 runtime. The Black-Scholes and OU setup cells can download
the private repository using a GitHub token with read-only Contents access.
For constant or quadratic experiments, first obtain the checkout using one of
those setup cells. Push local changes before starting a fresh remote runtime.
Live Colab execution has not been verified.

Current notebooks, comparison figures, numerical results, and PDFs are tracked
in Git. Virtual environments, execution copies, caches, logs, and LaTeX build
intermediates are ignored. Rebuild reports after rerunning experiments so that
the published tables match the saved results.
