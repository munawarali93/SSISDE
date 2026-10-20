"""Generate the LaTeX summary from the current experiment result files.

Run from any directory with Python 3:
    python output/pdf/recovered_models/build_tables.py
Then compile SSISDE_recovered_models.tex with pdflatex.
"""

import json
from decimal import Decimal
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = Path(__file__).resolve().parent / "SSISDE_recovered_models.tex"
CASES = [
    ("black_scholes", "Black--Scholes"),
    ("constant", "Constant coefficients"),
    ("quadratic", "Quadratic coefficients"),
    ("ou", "Ornstein--Uhlenbeck"),
]
METHODS = ("Method 1", "Method 2", "ODE2")


def number_tex(value):
    """Keep the reported digits, converting scientific notation to LaTeX."""
    if "e" in value.lower():
        mantissa, exponent = value.lower().split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return value


def polynomial_tex(polynomial):
    terms = []
    for term in polynomial.split(" + "):
        match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?:\*X(?:\^(\d+))?)?", term)
        if match is None:
            raise ValueError(f"Unexpected polynomial term: {term}")
        coefficient, power = match.groups()
        negative = coefficient.startswith("-")
        body = number_tex(coefficient.lstrip("+-"))
        if "*X" in term:
            body += r"\,x" + (rf"^{{{power}}}" if power else "")
        prefix = (" - " if negative else " + ") if terms else ("-" if negative else "")
        terms.append(prefix + body)
    return "".join(terms)


def alpha_tex(alpha):
    # Eight significant digits are enough to identify the selected grid point.
    mantissa, exponent = f"{alpha:.7e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def main():
    rows = []
    for case_index, (case, label) in enumerate(CASES):
        source = ROOT / "outputs" / case / "ssisde_comparison.json"
        results = json.loads(source.read_text(), parse_float=Decimal)
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
                f"${polynomial_tex(fit['mu'])}$",
                f"${polynomial_tex(fit['sigma'])}$",
                f"${alpha_tex(fit['alpha'])}$",
                f"${fit['rho']}$",
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
\textbf{Precision.} Polynomial coefficients retain the precision stored in the
experiment results, including small nonzero terms. $\alpha$ is displayed to
eight significant digits; its full stored value appears in the LaTeX source
comments. $\widehat{\sigma}$ is the diffusion amplitude, not its square.\par
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
