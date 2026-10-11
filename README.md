# Stochastic SINDy (SSISDE)

Private development repository for the Black-Scholes, constant, quadratic, OU,
and double-well SSISDE experiments.

## Layout

```text
data_generation/    Functions for process coefficients, simulation, and coarse graining (.py)
models/             Functions for variation, diffusion, signatures, drift, noise, and SINDy (.py)
experiments/        Separate folders with notebooks and plots for each process
original/           Untouched input notebook, including its original outputs
scripts/            Numerical-preservation check and full notebook runner
requirements.txt    Scientific dependencies
requirements-dev.txt Scientific dependencies plus Jupyter execution tools
```

## Experiments

- [Black-Scholes](experiments/black_scholes/Stochastic_SINDy_Black_Scholes.ipynb)
- [Time-shifted Brownian motion](experiments/constant/Stochastic_SINDy_constant.ipynb)
- [Quadratic coefficients](experiments/quadratic/Stochastic_SINDy_Quadratic.ipynb)
- [Ornstein-Uhlenbeck](experiments/ou/Stochastic_SINDy_OU.ipynb)
- [Double-well potential](experiments/double_well/Stochastic_SINDy_Double_Well.ipynb)

The seven main experiment notebooks, including the two double-well coefficient-study
notebooks, fits SSISDE three times. Each fit uses noise recovered from its own
drift estimate: Method 1, Method 2 or ODE2. All three use degree-5 libraries,
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

The separate [double-well SSISDE tuning notebook](experiments/double_well/ssisde_tuning/Stochastic_SINDy_Double_Well_SSISDE_Tuning.ipynb)
compares the existing fits with 30 configurations per recovered-noise method.
It tests polynomial degrees, weaker regularization, Elastic Net, Lasso, Ridge
thresholds and both CV selection rules. The last 20% of coarse increments selects
the configuration; inner CV uses only the first 80%. The selected configuration
is refitted on the full path. True drift and diffusion are used only afterward
to report error. The data and recovered noise paths are unchanged.

For the original double well on seed 3, tuning leaves Method 1 and Method 2
drift RMSE unchanged at `0.579673` and `0.417465`. ODE2's validation-selected
cubic fit reduces increment validation error but increases actual drift RMSE
from `0.997516` to `1.040098`. These settings do not improve drift recovery and
have not replaced the main experiments' settings. All candidate settings and
validation errors are saved beside the tuning notebook in `tuning_results.json`.
Validation is conditional on the recovered noise, not an independent check of
the whole recovery procedure. The tuning helper is
`models.stochastic_sindy_tuning.tune_stochastic_sindy(X, BP, dt)`.

All experiment notebooks compare Method 1, Method 2, and ODE2 against the actual
drift in a single three-panel figure with a shared vertical scale and per-method
RMSE, matching the constant-case layout. This includes Black-Scholes, constant,
quadratic, OU, double well, both coefficient studies, and the tuning notebook.
Each saves `drift_comparison.png` beside the notebook; the two coefficient studies
use `drift_comparison_small_curvature.png` and `drift_comparison_linear_ratio.png`
to keep their figures separate in the shared folder.
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
as in that source notebook. These settings are in `models/diffusion_quadratic.py`.

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

## OU and double-well experiments

The [double-well coefficient study](experiments/double_well/coefficient_study/Stochastic_SINDy_Double_Well_Small_Curvature.ipynb)
repeats the same workflow with `mu(x) = 0.5*x - 0.01*x**3` and
`sigma(x) = 1 + 0.01*x**2`. These coefficients gave the smallest RMS of
`(mu/sigma)''` along the seed-3 path among six tested sets with positive quadratic
diffusion coefficients. Two decreasing-diffusion cases were also tested. The notebook contains
the complete executed experiment and a comparison table. The original
double-well notebook below remains the baseline.

`data_generation.double_well.generate_data` accepts the coefficients `a, b, c, d`
for `mu(x) = a*x - b*x**3` and `sigma(x) = c + d*x**2`. Its defaults preserve the
original problem. In the new notebook, edit the `coefficients` dictionary.
ODE2's initial slope is computed from those coefficients. Other numerical and
regression settings are unchanged.

With seed 3, the selected problem reduces curvature RMS from `1.792701` to
`0.043851`, and ODE2 drift RMSE from `1.038716` to `0.010529` (relative RMSE
`4.29%`). The potential is `V(x) = 0.0025*x**4 - 0.25*x**2`, with wells at
`+/-sqrt(50)`. The observed path lies between approximately `-0.1523` and
`1.1005`, so this tests the central region, not transitions between both wells.
The condition is approximate, not an identity for all states. A lower-noise
variant (`c=0.5`) has higher curvature RMS but lower ODE2 error; curvature alone
does not determine numerical recovery accuracy. The coefficient-screening
results use Method 1 noise for the final SSISDE fit; the experiment notebooks
now additionally fit using Method 2 and ODE2 noise.

The [exact-ratio experiment](experiments/double_well/coefficient_study/Stochastic_SINDy_Double_Well_Linear_Ratio.ipynb)
uses `mu(x) = 0.5*x - 0.125*x**3` and `sigma(x) = 1 - 0.25*x**2`.
Here `mu/sigma = 0.5*x` on `(-2, 2)`, and the diffusion vanishes at the wells
`+/-2`. Despite zero ratio curvature, its seed-3 ODE2 RMSE is `0.164352`:
satisfying the assumption exactly does not remove errors in the estimation and
numerical recovery stages.

Trial results, including separate seed-7 and seed-11 checks for the original,
small-curvature, lower-noise and exact-ratio cases, are stored as JSON next to the notebooks.
For the small-curvature case, ODE2 RMSE is `0.020724` on seed 7 and `0.437719`
on seed 11, so the seed-3 gain is not uniform across trajectories. The original
problem's seed-11 diffusion fit failed the existing convergence/positivity checks;
its JSON records the failure. All other 15 trials completed with the same settings.
To repeat the coefficient study:

```bash
.venv/bin/python scripts/compare_double_well_coefficients.py
.venv/bin/python scripts/compare_double_well_coefficients.py --cases original cubic_001 lower_noise matched_factor --seeds 7 11
```

The baseline OU and double-well notebooks repeat the Black-Scholes workflow with `N=100000`, `T=1`,
`X0=1`, and seed 3. OU uses `mu(X) = -0.5*X`, `sigma(X) = 0.3`.
The double well uses `mu(X) = 0.5*X - X**3`, `sigma(X) = 1 + 0.25*X**2`.

They use the same quadratic-variation calculation, finite-difference derivatives,
signature regression, coarse graining, and drift equations as Black-Scholes.
Method 1, Method 2, and ODE2 are compared for drift. ODE2 uses the known initial
slope (`-0.5` for OU and `-2.5` for the double well). Each drift method supplies
its own noise for a separate final SSISDE fit, with the same degree-5 libraries,
seven folds, 35 alpha values, and seven elastic-net mixing values as Black-Scholes.

Only diffusion regression parameters are tuned. The existing degree-4 variance
library and SR3 selection on the last 20% of the observed path are retained.
The true diffusion is used only to report error. There are no increment-based
drift fits, odd-symmetry fits, or separate OU libraries. Poor recovery results
are retained for inspection.

```bash
.venv/bin/python scripts/run_experiment.py experiments/ou/Stochastic_SINDy_OU.ipynb --in-place
.venv/bin/python scripts/run_experiment.py experiments/double_well/Stochastic_SINDy_Double_Well.ipynb --in-place
```

The executed notebooks contain the plots and results. Numerical results are
saved under `outputs/ou/` and `outputs/double_well/`. The existing double-well
[LaTeX note](output/pdf/double_well/double_well_changes.tex) and
[PDF](output/pdf/double_well/double_well_changes.pdf) describe this revision.
The Black-Scholes, constant, and quadratic notebooks are unchanged.

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
from data_generation.ou import generate_data
from models.drift_recovery import recover_drift
from models.drift_recovery_from_ODE2 import recover_drift_ode2

t, dt, X, W, mu_true, sigma_true = generate_data(
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
The squared-derivative term previously had the wrong sign. All five experiments
have been rerun with this correction; their other settings are unchanged.
`StochSINDy` remains a class with `fit`, `mu`, and `si` methods. Its plotting
methods accept `output_prefix` explicitly instead of reading a notebook variable.

| Function | Returns |
|---|---|
| `generate_noise(N, T, seed)` | Time grid, time step, Brownian increments and path |
| Each process module's `generate_data(...)` | Time, step, state, Brownian path, actual drift and diffusion |
| `estimate_quadratic_variation(S, t)` | Quadratic variation, its derivative, difference method |
| `diffusion_derivatives(sigma, S, method)` | First, second and third derivatives |
| `fit_signature(S, BQ, t, keep_cols, M)` | Signature coefficient for each window |
| `coarse_grain(...)` | The six coarse arrays used by the drift and noise stages |
| `drift_coefficients(...)` | Initial drift and coefficients `a`, `b`, `b_` |
| `recover_drift(...)` | Method 1 and Method 2 drift estimates |
| `recover_drift_ode2(...)` | Drift and slope estimates |
| `recover_p_noise(...)` | P-noise path, coarse time step and time grid |

The untouched source notebooks remain in `original/`. The verification script
compares the function outputs with the scripts at commit `d8152df`, before this
refactor. It checks all five simulations and their unchanged recovery stages, diffusion
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

The full notebook completed using Python 3.11.13 and the listed dependencies,
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
