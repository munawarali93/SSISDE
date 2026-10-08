# Stochastic SINDy (SSISDE)

Private development repository for the Black-Scholes, constant, quadratic, OU,
and double-well SSISDE experiments.

## Layout

```text
data_generation/    Original parameters, process simulation, and coarse graining (.py)
models/             Original variation, diffusion, signature, drift, noise, and SINDy code (.py)
experiments/        Notebook that runs the scripts and displays the original plots
original/           Untouched input notebook, including its original outputs
scripts/            Source-preservation check and full notebook runner
requirements.txt    Scientific dependencies
requirements-dev.txt Scientific dependencies plus Jupyter execution tools
```

## Experiments

- [Black-Scholes](experiments/Stochastic_SINDy_Black_Scholes.ipynb)
- [Time-shifted Brownian motion](experiments/constant/Stochastic_SINDy_constant.ipynb)
- [Quadratic coefficients](experiments/quadratic/Stochastic_SINDy_Quadratic.ipynb)
- [Ornstein-Uhlenbeck](experiments/ou/Stochastic_SINDy_OU.ipynb)
- [Double-well potential](experiments/double_well/Stochastic_SINDy_Double_Well.ipynb)

The constant and quadratic notebooks compare Method 1, Method 2, and ODE2 against
the actual drift. Simulations retain the supplied seeds, coefficients, and sample
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

The OU experiment uses `kappa=0.5`, `theta=0`, `sigma=0.3`, `X0=1`, `T=1`,
`N=100000`, and seed 3. It compares Methods 1 and 2 with ODE2 using the known
initial slope. Every method uses the known initial drift. ODE2 is conditional
on the supplied slope. Both the general
degree-5 SSISDE library and a separate linear-drift, constant-diffusion library
are reported. A small increment error alone does not establish drift recovery.

For OU, the Brownian and state samples share the same time indices, quadratic
variation includes every preceding increment, and diffusion derivatives are
evaluated from the fitted polynomial. Other experiments retain their existing
computations. OU figures are in `experiments/ou/`, with numerical results in
`outputs/ou/results.json`.

On the OU path, estimated diffusion is `0.299025` versus `0.3`. ODE2 with the
known slope recovers drift `-0.5*X`, with zero drift RMSE and Brownian path RMSE
`0.002057`. This result uses the actual initial slope and does not independently
identify `kappa`.

```bash
.venv/bin/python scripts/run_experiment.py experiments/ou/Stochastic_SINDy_OU.ipynb --in-place
```

## Double-well experiment

The new experiment uses `mu(X) = 0.5*X - X**3` and
`sigma(X) = 1 + 0.25*X**2`. The first differential in the supplied image is
interpreted as `dt`. Its generator is `data_generation/double_well.py`.
The fixed seed is 42, with `X0=1`, `T=200`, and one million samples
(`dt=0.0002`). The longer observation period covers both wells.

SR3 searches 12 parameter pairs using the last 20% of the observed trajectory.
The selected diffusion variance is
`1.00198797 + 0.50032259*X**2 + 0.05995041*X**4`, with diffusion RMSE `0.00170021`.
The true coefficients and Brownian path are not used to choose the regression settings.

Method 1, Method 2, and ODE2 are reused without modification. All three receive
the initial drift, and ODE2 receives the known initial slope. Their drift
equations do not recover this cubic drift accurately. The notebook includes
an explicit check of the ODE2 equation using the exact coefficients.

For comparison, `models/increment_drift.py` fits the drift from observed
increments using five time-series folds. The notebook reports both an
unrestricted polynomial fit and a fit supplying the known odd symmetry of the
double-well drift. These are separate comparisons, not changes to the paper's
methods. All recovered noise paths are compared with the simulated Brownian
path, and the SSISDE stage reports errors against the actual coefficients.

With odd symmetry supplied, the increment fit gives
`mu(X) = 0.782957*X - 1.17707*X**3`, drift RMSE `0.149874`, and Brownian-path
RMSE `0.390244`. The unrestricted increment fit gives drift RMSE `0.201668`
but Brownian-path RMSE `12.8687`: a small drift bias can accumulate over the
long observation period. These are results for one fixed trajectory.

Open the notebook and choose **Run All**, or run:

```bash
.venv/bin/python scripts/run_experiment.py experiments/double_well/Stochastic_SINDy_Double_Well.ipynb --in-place
```

Completed figures are embedded in the notebook. Numerical results and tuning
tables are saved under `outputs/double_well/`. The earlier experiments are unchanged.

## Run locally in VS Code

Use Python 3.11 (the scientific pins support Python 3.10–3.12). From this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Open `experiments/Stochastic_SINDy_Black_Scholes.ipynb`, select `.venv/bin/python`
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

## Code preservation

The scientific code is extracted verbatim. `%run -i` executes each script in the
notebook namespace, preserving the original globals and order; these files are
execution stages, not independent importable APIs. Run them through the experiment.
In particular, `StochSINDy` still uses the notebook's global `model` value for plot
filenames. No formulas, random seeds, sample sizes, estimators, CV grids, thresholds,
or plotting commands were changed.

The added code handles project discovery, remote download, and environment setup.
The original `!pip` cell is replaced by an installation targeting the selected
Python kernel. Its four version pins are unchanged. `iisignature`, `matplotlib`,
and `tqdm` are also pinned, and `setuptools<81` retains `pkg_resources` for PySINDy.
An untouched copy of the supplied notebook is retained in `original/`.

```bash
.venv/bin/python scripts/verify_preservation.py
.venv/bin/python scripts/run_experiment.py
```

The runner writes an executed notebook to `outputs/`. Figures retain the original
filenames and are saved under `experiments/`. Generated results and `.venv` are
excluded from Git. The repository remains private during development; making it
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
scientific code and solver settings were left unchanged. The source-preservation
check and `pip check` also passed. A live Colab run has not yet been verified.
