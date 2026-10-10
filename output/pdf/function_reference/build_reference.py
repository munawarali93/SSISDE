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
          p("Inputs and outputs of all 23 Python files in data_generation/ and models/.", "small"),
          p("<b>Notation.</b> S or X = state path; t = time grid; dt = time step; N = sample count; "
            "T = duration; S0 = initial state; seed = random seed. W is simulated Brownian motion, "
            "BQ is recovered Q-noise, and BP is recovered physical Brownian motion. "
            "A trailing underscore marks fine-grid arrays in the experiments. Tuples below follow the code's return order."),
          p("data_generation/", "section")]

entry("parameters.py", "<b>generate_noise</b>: input N, T, seed. Returns <b>(t, dt, dW, W)</b>: time grid, step, Brownian increments and cumulative Brownian path.")
entry("processes.py", "<b>mu</b> and <b>sigma</b>: input state x, drift/diffusion label and coefficients; return the corresponding coefficient values. "
      "<b>simulate_process</b>: input dW, dt, S0, drift, diff and coefficients; returns <b>(S_, mu_exact_, sigma_exact_)</b>. "
      "Coefficients: r, sigm, kappa, theta, w1-w4, K, beta; K here is a population-model coefficient.")
entry("simulation.py", "<b>simulate</b>: input callable functions mu and sigma, plus N, T, S0, seed. Returns "
      "<b>(t, dt, S, W, mu(S), sigma(S))</b>: time grid, step, state, Brownian path, actual drift and actual diffusion.")
entry("constant.py", "<b>mu(x, r)</b> returns constant drift r; <b>sigma(x, sigm)</b> returns constant diffusion sigm. "
      "<b>generate_data</b>: input N, T, S0, seed, r, sigm; returns the same six outputs as simulate.")
entry("quadratic.py", "<b>mu(x)</b> returns 0.3x<super>2</super> + 0.7x; <b>sigma(x)</b> returns 0.1x<super>2</super> + 0.5x. "
      "<b>generate_data</b>: input N, T, S0, seed; returns the same six outputs as simulate.")
entry("ou.py", "<b>mu(x, kappa, theta)</b> returns kappa(theta - x); <b>sigma(x, sigm)</b> returns sigm. "
      "<b>generate_data</b>: input N, T, S0, seed, kappa, theta, sigm; returns the same six outputs as simulate.")
entry("double_well.py", "<b>mu(x)</b> returns 0.5x - x<super>3</super>; <b>sigma(x)</b> returns 1 + 0.25x<super>2</super>. "
      "<b>generate_data</b>: input N, T, S0, seed; returns the same six outputs as simulate.")
entry("coarse_graining.py", "<b>coarse_grain</b>: input S_, BQ_, W_, mu_exact_, sigma_exact_, sigma_pred_, stride M and coarse count K. "
      "Returns <b>(sigma_pred, mu_exact, S, BQ, W, sigma_exact)</b>, sampling every M-th entry.")

story.append(p("models/stochastic_sindy.py", "section"))
story.append(p("<b>StochSINDy(...)</b> creates a model. Inputs set polynomial degrees (mu_deg, si_deg), penalty (pen), "
               "cross-validation (use_cv, k, alphas, rhos, use_1se), and coefficient thresholds (thr1, thr2)."))
story.append(p("<b>fit(X, BP, dt, alpha_mu, alpha_si)</b> returns the fitted model itself. "
               "<b>mu(x)</b> and <b>si(x)</b> return drift and diffusion predictions at x. "
               "<b>show()</b> prints fitted equations and errors."))
story.append(p("<b>plot_cv(output_prefix)</b> shows cross-validation plots; <b>plot_size(count_bias, output_prefix)</b> shows model-size plots. "
               "<b>plot_results(X, BP, BQ, W, mu_true, si_true, output_prefix)</b> compares fitted functions and noise; "
               "true functions are optional. Plot methods return None and save PNGs where implemented (plot_cv saves for elastic net)."))

story.append(PageBreak())
story += [p("Estimation and recovery", "title"),
          p("models/  |  Remaining 14 files", "small"),
          p("<b>Key distinction.</b> Diffusion regression fits sigma<super>2</super>; its positive square root gives sigma_pred. "
            "sigma_prime, sigma_2prime and sigma_3prime are derivatives with respect to the state, not time."), Spacer(1, 6)]

entry("quadratic_variation.py", "<b>compute_quadratic_variation(W)</b> returns the scalar sum of squared path increments. "
      "<b>estimate_quadratic_variation(S_, t)</b> returns <b>(QV, dQV, diff_method)</b>: cumulative quadratic variation, its time derivative and the differentiation object.")
entry("diffusion.py", "<b>fit_diffusion_model</b>: input S_, t, dQV and settings threshold, nu, max_iter, tol. "
      "Returns a <b>fitted PySINDy model of sigma<super>2</super></b>, using a degree-4 polynomial library with a constant term.")
entry("diffusion_quadratic.py", "<b>fit_quadratic_diffusion</b>: input S_, t, dQV and threshold, nu, max_iter, tol. "
      "Returns a <b>fitted model of sigma<super>2</super></b>, using a degree-4 library without a constant term.")
entry("diffusion_tuning.py", "<b>diffusion_model(threshold, nu, max_iter, tol)</b> returns an <b>unfitted</b> variance model. "
      "<b>fit_diffusion(S, t, dQV, thresholds, nus, max_iter, tol)</b> returns <b>(model, best, results)</b>: fitted model, selected settings record and all search records.")
entry("diffusion_derivatives.py / coarse_diffusion_derivatives.py", "<b>diffusion_derivatives(sigma_pred_, S_, diff_method)</b> returns "
      "<b>(sigma_prime_, sigma_2prime_, sigma_3prime_)</b>. "
      "<b>coarse_diffusion_derivatives(sigma_pred, S, diff_method)</b> returns the same three derivatives on the coarse grid, without trailing underscores.")
entry("q_brownian_motion.py", "<b>recover_q_noise(S_, sigma_pred_)</b> returns <b>BQ_</b>, the fine-grid recovered Q-noise path.")
entry("signature_regression.py", "<b>fit_signature(S_, BQ_, t, keep_cols, M)</b>: input state, Q-noise, time grid, selected signature columns and window length. "
      "Returns <b>l21</b>, one fitted signature coefficient per window.")
entry("drift_coefficients.py", "<b>drift_coefficients(mu_exact, sigma_pred, sigma_prime, sigma_2prime, l21)</b> returns "
      "<b>(mu_0, a, b, b_)</b>: initial drift and coefficient arrays for Methods 1 and 2. Only mu_exact[0] is used from the actual drift.")
entry("drift_recovery.py", "<b>recover_drift(S, a, b, b_, mu_0)</b> returns <b>(mu_pred1, mu_pred2)</b>, the drift arrays recovered by Methods 1 and 2.")
entry("drift_recovery_from_ODE2.py", "<b>recover_drift_ode2(S, sigma_pred, sigma_prime, sigma_2prime, mu_0, lambda_0)</b> "
      "returns <b>(mu_pred11, lamda_pred)</b>, the recovered drift and its spatial slope. mu_0 and lambda_0 supply the initial drift and initial slope.")
entry("p_brownian_motion.py / p_brownian_motion_method2.py", "<b>recover_p_noise(BQ, mu_pred1, sigma_pred, T)</b> and "
      "<b>recover_p_noise_method2(BQ, mu_pred2, sigma_pred, T)</b> each return <b>(BP, dt, t)</b>: recovered physical noise, coarse time step and time grid, using the respective drift estimate.")
entry("noise_recovery.py", "<b>recover_noise(BQ, drift, diffusion, dt)</b> returns the <b>recovered physical noise path</b>. "
      "This helper accepts the time step directly and returns only the path.")

story += [Spacer(1, 5), p("<b>Using a function in an experiment</b>", "small"),
          p('<font name="Courier" size="8.6">from models.drift_recovery import recover_drift<br/>'
            'mu_pred1, mu_pred2 = recover_drift(S, a, b, b_, mu_0)</font>'),
          p("Optional arguments keep the defaults defined in the source files. This reference documents the current code; it changes no computations.", "small")]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=36, leftMargin=36,
                        topMargin=34, bottomMargin=46,
                        title="SSISDE: Function inputs and outputs", author="SSISDE")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
