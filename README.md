# Stochastic SINDy (SSISDE)

Private development repository for the Black–Scholes experiment supplied as
`Stochastic_SINDy_Black_Scholes.ipynb`.

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
