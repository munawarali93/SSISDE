"""Generate the LaTeX summary from the current experiment result files.

Run from any directory with Python 3:
    python output/pdf/recovered_models/build_tables.py
Then compile SSISDE_recovered_models.tex with pdflatex.
"""

import json
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = Path(__file__).resolve().parent / "SSISDE_recovered_models.tex"
CASES = [
    ("black_scholes", "Black--Scholes"),
    ("constant", "Constant coefficients"),
    ("quadratic", "Quadratic coefficients"),
    ("ou", "Ornstein--Uhlenbeck"),
]
METHODS = ("Method 1", "Method 2", "ODE2")


def mse_tex(value):
    """Keep small final errors visible with six significant figures."""
    if not value.is_finite() or value < 0:
        raise ValueError(f"Invalid final MSE: {value}")
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.5E}".split("E")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def polynomial_tex(coefficients):
    """Format the final or exact polynomial with five decimal places."""
    terms = []
    for power, coefficient in enumerate(coefficients):
        rounded = coefficient.quantize(Decimal("0.00001"))
        if rounded == 0:
            continue
        negative = rounded < 0
        body = f"{abs(rounded):.5f}"
        if power:
            body += r"\,x" + (rf"^{{{power}}}" if power > 1 else "")
        prefix = (" - " if negative else " + ") if terms else ("-" if negative else "")
        terms.append(prefix + body)
    return "".join(terms) if terms else "0.00000"


def main():
    rows = []
    for case_index, (case, label) in enumerate(CASES):
        source = ROOT / "outputs" / case / "ssisde_comparison.json"
        results = json.loads(source.read_text(), parse_float=Decimal, parse_int=Decimal)
        truth = json.loads(source.with_name("model_truth.json").read_text(),
                           parse_float=Decimal, parse_int=Decimal)
        if set(results) != set(METHODS):
            raise ValueError(f"Unexpected methods in {source}")
        if case_index:
            rows.append(r"\addlinespace[5pt]")
        rows.append(r"\rowcolor{groupbg}")
        exact_mu = polynomial_tex(truth['mu_coefficients'])
        exact_sigma = polynomial_tex(truth['sigma_coefficients'])
        rows.append(rf"\multicolumn{{6}}{{l}}{{\strut\textbf{{{label}}}"
                    rf"\quad Exact: $\mu(x)={exact_mu}$;\quad $\sigma(x)={exact_sigma}$}} \\[2pt]")
        for method in METHODS:
            fit = results[method]
            if fit.get('coefficient_decimals') != 5:
                raise ValueError(f"Rerun {case}: final model must use five-decimal coefficients.")
            for field in ('mu_coefficients', 'sigma_coefficients'):
                if any(c != c.quantize(Decimal('0.00001')) for c in fit[field]):
                    raise ValueError(f"Unrounded final coefficients in {case}/{method}/{field}")
            rows.append(f"% {case} / {method}: alpha={fit['alpha']}, rho={fit['rho']}")
            fields = [
                method,
                f"${polynomial_tex(fit['mu_coefficients'])}$",
                f"${polynomial_tex(fit['sigma_coefficients'])}$",
                f"${fit['alpha']:.5f}$",
                f"${fit['rho']:.5f}$",
                f"${mse_tex(fit['increment_mse'])}$",
            ]
            rows.append(" & ".join(fields) + r" \\[4pt]")
    document = r"""\documentclass[10pt,a4paper,landscape]{article}
\usepackage[margin=14mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{booktabs,array}
\usepackage[table]{xcolor}
\usepackage{hyperref}
\hypersetup{pdftitle={SSISDE: exact and recovered drift and diffusion},
  pdfauthor={SSISDE},hidelinks}
\definecolor{ink}{HTML}{16334A}
\definecolor{groupbg}{HTML}{EDF2F6}
\definecolor{muted}{HTML}{465565}
\setlength{\parindent}{0pt}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}
\pagestyle{empty}
\begin{document}
{\color{ink}\LARGE\bfseries Exact and recovered drift and diffusion}\par
\vspace{3pt}
{\color{muted}SSISDE experiment results \hfill REPORTDATE}\par
\vspace{8pt}
Final Stochastic SINDy fits using noise recovered from each drift method:
\quad $dX_t=\widehat{\mu}(X_t)\,dt+\widehat{\sigma}(X_t)\,dW_t$.
\par\vspace{8pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l l c c r @{}}
\toprule
\textbf{Method} & \textbf{Recovered drift $\widehat{\mu}(x)$}
& \textbf{Recovered diffusion $\widehat{\sigma}(x)$}
& \textbf{$\alpha$} & \textbf{$\rho$} & \textbf{Final MSE} \\
\midrule
"""
    document += "\n".join(rows)
    document += r"""
\bottomrule
\end{tabular*}
\par\vspace{9pt}
{\small
\textbf{Final MSE.} Mean squared increment prediction error on the fitted data:
$\frac{1}{N}\sum_{i=0}^{N-1}[\Delta X_i-\widehat{\mu}(X_i)\Delta t
-\widehat{\sigma}(X_i)\Delta\widehat{W}_i]^2$, using each method's recovered
noise and the final rounded coefficients. Values are the saved
\texttt{increment\_mse}, shown to six significant figures; these are not
cross-validation errors.\par
\vspace{4pt}
\textbf{Parameters.} $\alpha$ is the ElasticNet regularization strength;
$\rho$ is its L1 mixing ratio. Each pair applies to the joint drift--diffusion
fit and was selected by seven-fold cross-validation with the one-standard-error
rule. All four experiments used 100,000 simulated samples.\par
\vspace{4pt}
\textbf{Selection.} Both polynomial libraries contain powers zero through five.
The constant experiment uses the
$\rho$ grid: $0.70000,\,0.85000,\,0.95000,\,1.00000$.\par
\vspace{4pt}
\textbf{Precision.} Final recovered coefficients are rounded in code to five
decimal places after model selection and refitting. Predictions, plots, and
errors use these rounded models. Unrounded coefficients are saved for comparison.
$\alpha$ and $\rho$ are displayed to five decimal places; selection uses full
precision. The exact functions define the simulated processes, and $\sigma$
denotes diffusion amplitude, not variance.\par
\vspace{4pt}
\textbf{Source.} \texttt{outputs/<case>/ssisde\_comparison.json} and
\texttt{model\_truth.json},
where \texttt{<case>} is \texttt{black\_scholes}, \texttt{constant},
\texttt{quadratic}, or \texttt{ou}.
}
\end{document}
"""
    report_date = datetime.now(ZoneInfo('America/New_York')).strftime('%d %B %Y')
    DESTINATION.write_text(document.replace('REPORTDATE', report_date))
    print(f"Wrote {DESTINATION} (12 fitted models)")


if __name__ == "__main__":
    main()
