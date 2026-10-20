"""Generate the LaTeX summary from the current experiment result files.

Run from any directory with Python 3:
    python output/pdf/recovered_models/build_tables.py
Then compile SSISDE_recovered_models.tex with pdflatex.
"""

import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = Path(__file__).resolve().parent / "SSISDE_recovered_models.tex"
CASES = [
    ("black_scholes", "Black--Scholes"),
    ("constant", "Constant coefficients"),
    ("quadratic", "Quadratic coefficients"),
    ("ou", "Ornstein--Uhlenbeck"),
]
METHODS = ("Method 1", "Method 2", "ODE2")


def polynomial_tex(coefficients):
    """Round full-precision coefficients once, omitting display-zero terms."""
    terms = []
    for power, coefficient in enumerate(coefficients):
        rounded = coefficient.quantize(Decimal("0.0001"))
        if rounded == 0:
            continue
        negative = rounded < 0
        body = f"{abs(rounded):.4f}"
        if power:
            body += r"\,x" + (rf"^{{{power}}}" if power > 1 else "")
        prefix = (" - " if negative else " + ") if terms else ("-" if negative else "")
        terms.append(prefix + body)
    return "".join(terms) if terms else "0.0000"


def main():
    rows = []
    for case_index, (case, label) in enumerate(CASES):
        source = ROOT / "outputs" / case / "ssisde_comparison.json"
        results = json.loads(source.read_text(), parse_float=Decimal, parse_int=Decimal)
        if set(results) != set(METHODS):
            raise ValueError(f"Unexpected methods in {source}")
        if case_index:
            rows.append(r"\addlinespace[5pt]")
        rows.append(r"\rowcolor{groupbg}")
        rows.append(rf"\multicolumn{{5}}{{l}}{{\strut\textbf{{{label}}}}} \\[2pt]")
        for method in METHODS:
            fit = results[method]
            rows.append(f"% {case} / {method}: alpha={fit['alpha']}, rho={fit['rho']}")
            fields = [
                method,
                f"${polynomial_tex(fit['mu_coefficients'])}$",
                f"${polynomial_tex(fit['sigma_coefficients'])}$",
                f"${fit['alpha']:.4f}$",
                f"${fit['rho']:.4f}$",
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
\hypersetup{pdftitle={SSISDE: recovered drift and diffusion models},
  pdfauthor={SSISDE},hidelinks}
\definecolor{ink}{HTML}{16334A}
\definecolor{groupbg}{HTML}{EDF2F6}
\definecolor{muted}{HTML}{465565}
\setlength{\parindent}{0pt}
\setlength{\tabcolsep}{7pt}
\renewcommand{\arraystretch}{1.15}
\pagestyle{empty}
\begin{document}
{\color{ink}\LARGE\bfseries Recovered drift and diffusion models}\par
\vspace{3pt}
{\color{muted}SSISDE experiment results \hfill 20 October 2026}\par
\vspace{8pt}
Final Stochastic SINDy fits using noise recovered from each drift method:
\quad $dX_t=\widehat{\mu}(X_t)\,dt+\widehat{\sigma}(X_t)\,dW_t$.
\par\vspace{8pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l l c c @{}}
\toprule
\textbf{Method} & \textbf{Recovered drift $\widehat{\mu}(x)$}
& \textbf{Recovered diffusion $\widehat{\sigma}(x)$}
& \textbf{$\alpha$} & \textbf{$\rho$} \\
\midrule
"""
    document += "\n".join(rows)
    document += r"""
\bottomrule
\end{tabular*}
\par\vspace{9pt}
{\small
\textbf{Parameters.} $\alpha$ is the ElasticNet regularization strength;
$\rho$ is its L1 mixing ratio. Each pair applies to the joint drift--diffusion
fit and was selected by seven-fold cross-validation with the one-standard-error
rule. All four experiments used 100,000 simulated samples.\par
\vspace{4pt}
\textbf{Selection.} The double-well outer search over polynomial degrees and
regression types has been removed. The constant experiment uses its original
$\rho$ grid: $0.7000,\,0.8500,\,0.9500,\,1.0000$.\par
\vspace{4pt}
\textbf{Precision.} All displayed coefficients, $\alpha$, and $\rho$ are rounded
to four decimal places. Terms that round to zero are omitted from the displayed
polynomials. Full-precision coefficients and parameters remain in the result
files; rounding does not change the fitted models.
$\widehat{\sigma}$ is the diffusion amplitude, not its square.\par
\vspace{4pt}
\textbf{Source.} \texttt{outputs/<case>/ssisde\_comparison.json},
where \texttt{<case>} is \texttt{black\_scholes}, \texttt{constant},
\texttt{quadratic}, or \texttt{ou}.
}
\end{document}
"""
    DESTINATION.write_text(document)
    print(f"Wrote {DESTINATION} (12 fitted models)")


if __name__ == "__main__":
    main()
