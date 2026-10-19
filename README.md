# Stochastic SINDy (SSISDE)

Private development repository for the Black-Scholes, constant, quadratic, and
Ornstein-Uhlenbeck (OU) SSISDE experiments.

## Layout

```text
data_generation/
  processes.py       Process coefficients and named experiment generators
  simulation.py      Brownian noise and Euler-Maruyama simulation
  coarse_graining.py Sampling the fine paths onto the recovery grid
models/
  diffusion.py       Quadratic variation, variance fitting/tuning, derivatives
  drift.py           Signature regression, drift coefficients, Methods 1/2/ODE2
  noise.py           Q-noise and physical Brownian-motion recovery
  stochastic_sindy.py SSISDE fitting, cross-validation, plotting, optional tuning
experiments/
  black_scholes/      Black-Scholes notebook and generated plots
  constant/          Constant-coefficient notebook and generated plots
  quadratic/         Quadratic-coefficient notebook and generated plots
  ou/                Ornstein-Uhlenbeck notebook and generated plots
original/            Input notebooks and outputs; retired process options removed
scripts/             Numerical-preservation check and full notebook runner
output/pdf/          Function reference and its build script
requirements.txt     Scientific dependencies
requirements-dev.txt Scientific dependencies plus Jupyter execution tools
```

The seven Python modules group related stages without changing their numerical
formulas. Import functions from the modules above; the previous one-function
module paths have been removed. The four active notebooks use the new imports.
The [function reference](output/pdf/function_reference/SSISDE_function_reference.pdf)
describes the consolidated API.

## Experiments

- [Black-Scholes](experiments/black_scholes/Stochastic_SINDy_Black_Scholes.ipynb)
- [Time-shifted Brownian motion](experiments/constant/Stochastic_SINDy_constant.ipynb)
- [Quadratic coefficients](experiments/quadratic/Stochastic_SINDy_Quadratic.ipynb)
- [Ornstein-Uhlenbeck](experiments/ou/Stochastic_SINDy_OU.ipynb)

Each of the four experiment notebooks fits SSISDE three times. Each fit uses
noise recovered from its own drift estimate: Method 1, Method 2 or ODE2. All three use degree-5 libraries,
seven folds, 35 alpha values and `rhos=[0.3, 0.4, 0.5, 0.7, 0.85, 0.95, 1.0]`.
The drift panels are titled `Drift from Method 1`, `Drift from Method 2` and
`Drift from ODE2`; each fit has separate CV, model-size and results plots.

The fitted models and noise paths are available as `identifiers["Method 1"]`,
`noise_paths["Method 1"]`, and the corresponding Method 2 and ODE2 entries.
The final comparison table reports noise, fitted drift and fitted diffusion
RMSE. Each notebook saves `ssisde_comparison.json` and
`ssisde_comparison_paths.npz` in its output directory. Earlier single-method
result fields retain their original noise source; `ssisde_comparison` contains
all three fits.

All experiment notebooks compare Method 1, Method 2, and ODE2 against the actual
drift in a single three-panel figure with a shared vertical scale and per-method
RMSE, matching the constant-case layout. Each saves `drift_comparison.png` beside
the notebook.
Simulations retain the supplied seeds, coefficients, and sample
sizes. ODE2 uses initial slope zero for the constant case and `0.7 + 0.6*S0` for
the quadratic case. Black-Scholes keeps its default slope `r`.

In the constant notebook, change `thresholds` and `nus` to adjust the SR3 search. Selection
uses the last 20% of the trajectory, with a three-sample gap after the training
data. Actual diffusion values are used only to report error. The search uses
`max_iter=10000` and `tol=1e-8`, and rejects nonconverged fits or nonpositive
variance. The original settings are retained as a comparison. The polynomial
library for diffusion variance has degree 4. A small pathwise error does not
establish that the recovered polynomial is accurate outside the observed range.

Terminology follows [Noise estimation of SDE from a single data trajectory](https://arxiv.org/abs/2509.25484v2).
The quadratic notebook uses the supplied `Stochastic_SINDy_Quadratic.ipynb`:
`mu(X) = 0.7*X + 0.3*X**2`, `sigma(X) = 0.5*X + 0.1*X**2`, and seed 2.
Its diffusion fit keeps `threshold=0.04`, `nu=0.05`, `max_iter=500`, `tol=1e-14`,
and a degree-4 library without a constant term. Noise recovery uses Method 2,
as in that source notebook. These settings are in `models/diffusion.py`.

Each new notebook saves figures in its own folder and numerical results in
`outputs/<case>/`. The completed notebook also retains its displayed results.
To run either experiment from the terminal:

```bash
.venv/bin/python scripts/run_experiment.py experiments/constant/Stochastic_SINDy_constant.ipynb --in-place
.venv/bin/python scripts/run_experiment.py experiments/quadratic/Stochastic_SINDy_Quadratic.ipynb --in-place
```

For Colab, first obtain a runtime copy of the repository using the existing
Black-Scholes setup cell, then open the desired experiment with the same runtime.
New local files must be pushed before that remote copy can include them.

The constant experiment completed with `N=100000`. Its diffusion RMSE is
`0.0001964`, and ODE2 drift RMSE is about `3e-10`. Results for the quadratic
experiment are displayed in its notebook and saved to `outputs/quadratic/`.

## Ornstein-Uhlenbeck experiment

The OU notebook repeats the Black-Scholes workflow with `N=100000`, `T=1`,
`X0=1`, and seed 3. It uses `mu(X) = -0.5*X`, `sigma(X) = 0.3`, and the known
initial slope `-0.5` for ODE2. Each drift method supplies its own recovered noise
for a separate final SSISDE fit, with the same degree-5 libraries, seven folds,
35 alpha values, and seven elastic-net mixing values as Black-Scholes.

Only diffusion regression parameters are tuned. The degree-4 variance library
and SR3 selection on the last 20% of the observed path are retained. True diffusion
is used only to report error. Numerical results are saved under `outputs/ou/`.

```bash
.venv/bin/python scripts/run_experiment.py experiments/ou/Stochastic_SINDy_OU.ipynb --in-place
```

## Run locally in VS Code

Use Python 3.11 (the scientific pins support Python 3.10–3.12). From this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Open `experiments/black_scholes/Stochastic_SINDy_Black_Scholes.ipynb`, select `.venv/bin/python`
as its kernel, and choose **Run All**. The first cells locate the checkout and
install scientific dependencies into that kernel. If dependencies were already
imported before their versions changed, restart the kernel and run all cells.

The original run uses `N = 100000`. Quadratic variation recomputes a growing
prefix in each iteration, and the final fit performs time-series cross-validation;
these original operations can take time. Their settings are unchanged.

## Colab compute from VS Code

1. Install the recommended Python, Jupyter, and **Google Colab** extensions.
2. Open the experiment notebook and select **Select Kernel → Colab → Auto Connect**,
   then sign in to Google when prompted.
3. Use a runtime with Python 3.10–3.12 to retain the notebook's original package
   pins. The setup cell checks the runtime version before installing packages.
4. Run all cells. If the runtime has no SSISDE checkout, the setup cell downloads
   the private repository's `main` branch into `/content/SSISDE`. Enter a GitHub
   fine-grained token with read-only **Contents** access to this repository in the
   hidden prompt. The token is not written to a file or stored in the notebook.
   Alternatively, put an authenticated checkout at `/content/SSISDE` beforehand.

Colab runs on a separate machine: the local scripts must be pushed to GitHub
before a fresh runtime can download them. The bootstrap reuses an existing runtime
copy; to load newer scripts, use a fresh runtime or remove that runtime copy and
rerun setup. This download is a source snapshot, not a Git checkout. Keep edits and
Git operations in the local VS Code workspace.

Reference: [Google's Colab VS Code extension instructions](https://github.com/googlecolab/colab-vscode).
Local execution is verified separately from a live Colab session.

## Functions and code preservation

The notebooks import functions from `data_generation/` and `models/`. Each
stage receives its inputs and returns its outputs; there is no `%run -i` or
shared notebook namespace. For example:

```python
from data_generation.processes import generate_ou
from models.drift import recover_drift, recover_drift_ode2

t, dt, X, W, mu_true, sigma_true = generate_ou(
    N=100000, T=1.0, S0=1.0, seed=3,
)
# After computing the diffusion, signature, and drift coefficients:
mu_pred1, mu_pred2 = recover_drift(S, a, b, b_, mu_0)
mu_ode2, slope = recover_drift_ode2(
    S, sigma_pred, sigma_prime, sigma_2prime, mu_0, lambda_0,
)
```

The experiment notebooks show the complete sequence. Their parameters, formulas,
seeds, array shapes, indexing, rounding, and model-selection settings are retained.
The ODE2 coefficient has one correction: deriving `(mu/sigma)'' = 0` gives
`B = sigma_2prime/sigma_pred - 2*sigma_prime**2/sigma_pred**2`.
The squared-derivative term previously had the wrong sign. The retained experiments
have previously been run with this correction; their other settings are unchanged.
`StochSINDy` remains a class with `fit`, `mu`, and `si` methods. Its plotting
methods accept `output_prefix` explicitly instead of reading a notebook variable.

| Function | Returns |
|---|---|
| `generate_noise(N, T, seed)` | Time grid, time step, Brownian increments and path |
| `generate_constant(...)`, `generate_quadratic(...)`, `generate_ou(...)` | Time, step, state, Brownian path, actual drift and diffusion |
| `estimate_quadratic_variation(S, t)` | Quadratic variation, its derivative, difference method |
| `diffusion_derivatives(sigma, S, method)` | First, second and third derivatives |
| `fit_signature(S, BQ, t, keep_cols, M)` | Signature coefficient for each window |
| `coarse_grain(...)` | The six coarse arrays used by the drift and noise stages |
| `drift_coefficients(...)` | Initial drift and coefficients `a`, `b`, `b_` |
| `recover_drift(...)` | Method 1 and Method 2 drift estimates |
| `recover_drift_ode2(...)` | Drift and slope estimates |
| `recover_p_noise(...)` | P-noise path, coarse time step and time grid |

The input notebooks remain in `original/`, with retired process options removed;
their original versions remain in Git history and the local recovery archive. The verification script
compares the function outputs with the scripts at commit `d8152df`, before this
refactor. It checks all four active simulations and their unchanged recovery stages, diffusion
fitting, and the three SSISDE penalties. ODE2 is checked separately against exact
Black-Scholes and OU solutions and convergence to a nonlinear exact solution,
since its corrected coefficient intentionally differs from that commit.
This check needs a Git checkout containing
that commit. Full notebook execution is a separate check.

```bash
.venv/bin/python scripts/verify_preservation.py
.venv/bin/python scripts/run_experiment.py --in-place
```

The runner writes the Black-Scholes execution record to `outputs/black_scholes/`.
Its notebook and figures are under `experiments/black_scholes/`; figure filenames
are unchanged. Completed notebook outputs are included in Git; separate generated
figures, execution logs, and `.venv` are excluded. The repository remains private during development; making it
public is a separate future action.

## Verified local run

After module consolidation, all four retained notebooks completed in fresh local
kernels with their full sample sizes and CV grids. The numerical-preservation
check, notebook imports, and unchanged saved notebook outputs were also verified.
Fresh execution records are under `outputs/<case>/`.

The earlier Black-Scholes notebook completed using Python 3.11.13 and the listed dependencies,
with all original settings (`N = 100000` and the full CV grid). It generated
13 inline figures and five PNG files. The final reported fit was:

```text
MSE: 5.244e-09
μ(X) = 0.503773*X
σ(X) = 0.00323797 + 0.295761*X
[CV best] alpha=0.000387, l1_ratio (rho)=0.85
```

The run emitted a convergence warning and dependency deprecation warnings, but
no execution errors. These are recorded in the local executed notebook; the
scientific code and solver settings were left unchanged. The numerical-preservation
check and full notebook executions also passed. A live Colab run has not yet been verified.

## Restoring deferred work

The double-well example is removed from the active code, experiments, outputs,
and reports. Its complete tracked version remains at Git commit
`ae83792619c348eb7cac72976117fdddaf84aef7`, including the baseline notebook,
coefficient studies, tuning study, comparison script, and LaTeX/PDF note.
To inspect that version without overwriting the current checkout:

```bash
git archive ae83792619c348eb7cac72976117fdddaf84aef7 -o /tmp/ssisde-before-consolidation.tar
```

This checkout also has a verified local archive at
`.archive/double_well-ae83792.tar.gz`, containing the removed example, generated
plots/results, and its original supporting modules. It is ignored by Git.
The original input notebooks are also preserved in
`.archive/original-notebooks-ae83792.tar.gz`. Git history is the portable recovery
source. Extract either archive into a separate directory for review.
When restoring the example here, migrate its imports to
the seven modules above and restore its process coefficients explicitly; do not
overwrite the consolidated modules with the archived copies.
