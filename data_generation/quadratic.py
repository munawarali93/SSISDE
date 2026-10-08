from data_generation.simulation import simulate

N, T, S0, seed = 100000, 1, 1.0, 2
r = 0.5


def mu(x):
    return 0.3 * x**2 + 0.7 * x


def sigma(x):
    return 0.1 * x**2 + 0.5 * x


t, dt, S_, W_, mu_exact_, sigma_exact_ = simulate(mu, sigma, N, T, S0, seed)
