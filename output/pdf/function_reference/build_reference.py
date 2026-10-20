"""Build the two-page function reference with ReportLab."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "SSISDE_function_reference.pdf"
INK = colors.HexColor("#16334A")
styles = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=INK, spaceAfter=7),
    "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=INK, spaceBefore=9, spaceAfter=7),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.3, leading=12.1, spaceAfter=4),
    "file": ParagraphStyle("file", fontName="Helvetica-Bold", fontSize=9.6, leading=12, textColor=INK, spaceAfter=2),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.5, leading=11, textColor=colors.HexColor("#465565"), spaceAfter=4),
}
story = []

def p(text, style="body"):
    return Paragraph(text, styles[style])

def entry(name, text):
    story.append(KeepTogether([p(name, "file"), p(text), Spacer(1, 3)]))

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#CBD5DF"))
    canvas.line(36, 35, A4[0] - 36, 35)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#465565"))
    canvas.drawString(36, 23, "SSISDE  |  Function inputs and outputs")
    canvas.drawRightString(A4[0] - 36, 23, f"{doc.page} / 2")

story += [p("SSISDE function reference", "title"),
          p("Seven modules: three for data generation and four for estimation and recovery.", "small"),
          p("<b>Notation.</b> S or X = state path; t = time grid; dt = time step; N = sample count; "
            "T = duration; S0 = initial state. W is simulated Brownian motion, BQ is recovered Q-noise, "
            "and BP is recovered physical Brownian motion. Trailing underscores mark fine-grid arrays. "
            "Tuples below follow the code's return order."),
          p("Data generation", "section")]

entry("data_generation/simulation.py",
      "<b>generate_noise(N, T, seed)</b> returns <b>(t, dt, dW, W)</b>: time grid, step, Brownian increments and path. "
      "<b>simulate(mu, sigma, N, T, S0, seed)</b> takes callable coefficients and returns "
      "<b>(t, dt, S, W, mu(S), sigma(S))</b> using Euler-Maruyama.")
entry("data_generation/processes.py",
      "<b>mu(x, drift, ...)</b> and <b>sigma(x, diff, ...)</b> evaluate a named process's coefficients. "
      "<b>simulate_process(dW, dt, S0, ...)</b> accepts existing Brownian increments and returns "
      "<b>(S_, mu_exact_, sigma_exact_)</b>. Black-Scholes uses drift='BS', diff='BS', r and sigm.")
story.append(p("<b>generate_constant(N, T, S0, seed, r, sigm)</b>, "
               "<b>generate_quadratic(N, T, S0, seed)</b>, and "
               "<b>generate_ou(N, T, S0, seed, kappa, theta, sigm)</b> each return the six outputs of simulate. "
               "Their defaults preserve the experiment seeds (31, 2 and 3 respectively)."))
story.append(p("<b>constant_drift / constant_diffusion</b>: r and sigm. "
               "<b>quadratic_drift / quadratic_diffusion</b>: 0.3x<super>2</super> + 0.7x and "
               "0.1x<super>2</super> + 0.5x. <b>ou_drift / ou_diffusion</b>: kappa(theta - x) and sigm. "
               "Each takes x, with optional coefficients where applicable."))
entry("data_generation/coarse_graining.py",
      "<b>coarse_grain(S_, BQ_, W_, mu_exact_, sigma_exact_, sigma_pred_, M, K)</b> returns "
      "<b>(sigma_pred, mu_exact, S, BQ, W, sigma_exact)</b>, taking every M-th entry for K samples.")

story.append(p("Diffusion estimation", "section"))
entry("models/diffusion.py",
      "<b>compute_quadratic_variation(W)</b> returns the sum of squared increments. "
      "<b>estimate_quadratic_variation(S_, t)</b> returns <b>(QV, dQV, diff_method)</b>: cumulative "
      "variation, its time derivative and the finite-difference object.")
story.append(p("<b>fit_diffusion_model(S_, t, dQV, ...)</b> and <b>fit_quadratic_diffusion(S_, t, dQV, ...)</b> "
               "return fitted PySINDy variance models. Both use degree 4; only the first includes a constant term. "
               "Optional settings are threshold, nu, max_iter and tol."))
story.append(p("<b>diffusion_model(threshold, nu, max_iter, tol)</b> returns an unfitted variance model. "
               "<b>fit_diffusion(S, t, dQV, thresholds, nus, max_iter, tol)</b> selects settings on the last 20% "
               "of the path and returns <b>(model, best, results)</b>: fitted model, selected settings and trial records."))
story.append(p("<b>diffusion_derivatives(sigma_pred_, S_, diff_method)</b> and "
               "<b>coarse_diffusion_derivatives(sigma_pred, S, diff_method)</b> return the first, second and third "
               "diffusion derivatives with respect to the state. Regression fits sigma<super>2</super>; "
               "take its positive square root before computing these derivatives."))

story.append(PageBreak())
story += [p("Drift, noise and identification", "title"),
          p("The remaining three model modules", "small")]
entry("models/drift.py",
      "<b>fit_signature(S_, BQ_, t, keep_cols, M)</b> returns <b>l21</b>, one fitted signature coefficient per window. "
      "keep_cols selects signature columns and M sets the window length.")
story.append(p("<b>drift_coefficients(mu_exact, sigma_pred, sigma_prime, sigma_2prime, l21)</b> returns "
               "<b>(mu_0, a, b, b_)</b>: initial drift and recovery coefficients. Only mu_exact[0] is used."))
story.append(p("<b>recover_drift(S, a, b, b_, mu_0)</b> returns <b>(mu_pred1, mu_pred2)</b> for Methods 1 and 2. "
               "<b>recover_drift_ode2(S, sigma_pred, sigma_prime, sigma_2prime, mu_0, lambda_0)</b> returns "
               "<b>(mu_pred11, lamda_pred)</b>: drift and spatial slope. Supply the initial drift and slope explicitly."))
entry("models/noise.py",
      "<b>recover_q_noise(S_, sigma_pred_)</b> returns the fine-grid Q-noise path <b>BQ_</b>. "
      "<b>recover_p_noise(BQ, mu_pred1, sigma_pred, T)</b> returns <b>(BP, dt, t)</b>: physical noise, coarse time step and grid.")
story.append(p("<b>recover_p_noise_method2(BQ, mu_pred2, sigma_pred, T)</b> returns the same tuple using Method 2 drift. "
               "<b>recover_noise(BQ, drift, diffusion, dt)</b> accepts a time step directly and returns only the physical noise path."))
entry("models/stochastic_sindy.py",
      "<b>StochSINDy(...)</b> creates an identifier. Inputs set polynomial degrees (mu_deg, si_deg), "
      "penalty (pen), cross-validation (use_cv, k, alphas, rhos, use_1se), and thresholds (thr1, thr2).")
story.append(p("<b>fit(X, BP, dt, alpha_mu, alpha_si)</b> returns the fitted model. "
               "<b>mu(x)</b> and <b>si(x)</b> predict drift and diffusion; <b>show()</b> prints equations and errors. "
               "<b>plot_cv(output_prefix)</b>, <b>plot_size(count_bias, output_prefix)</b> and "
               "<b>plot_results(X, BP, BQ, W, mu_true, si_true, output_prefix, drift_title)</b> display diagnostics "
               "and save plots where implemented."))
story.append(p("The double-well outer search over model configurations has been removed. "
               "StochSINDy retains its original within-model cross-validation for alpha and rho. "
               "<b>show(decimals=4)</b> rounds printed results only; fitted coefficients retain full precision."))
story += [Spacer(1, 12), p("Example imports", "section"),
          p('<font name="Courier" size="8.6">from data_generation.processes import generate_ou<br/>'
            'from models.drift import recover_drift, recover_drift_ode2<br/>'
            'from models.noise import recover_p_noise<br/>'
            'from models.stochastic_sindy import StochSINDy</font>'),
          p("Optional arguments retain their source defaults. The consolidation changes module locations; "
            "the four experiment notebooks retain their numerical settings and formulas.", "small")]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=36, leftMargin=36,
                        topMargin=34, bottomMargin=46,
                        title="SSISDE: Function inputs and outputs", author="SSISDE")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
