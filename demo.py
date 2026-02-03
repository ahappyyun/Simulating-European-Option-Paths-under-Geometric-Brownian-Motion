import numpy as np

def simulate_gbm_paths(S0, r, q, sigma, T, M, N, seed=None):
    """
    Simulate GBM paths under risk-neutral measure:
        dS = (r-q) S dt + sigma S dW

    Returns
    -------
    t : (M+1,) time grid
    S : (N, M+1) simulated paths
    """
    rng = np.random.default_rng(seed)
    dt = T / M
    t = np.linspace(0.0, T, M + 1)

    # Z shape: (N, M)
    Z = rng.standard_normal((N, M))
    drift = (r - q - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt) * Z

    # log-increments and cumulative sum
    log_increments = drift + diff                         # (N, M)
    log_S = np.log(S0) + np.cumsum(log_increments, axis=1) # (N, M)

    # build full path array including S0 at t0
    S = np.empty((N, M + 1), dtype=float)
    S[:, 0] = S0
    S[:, 1:] = np.exp(log_S)

    return t, S

def price_european_option_mc(S0, K, r, q, sigma, T, M, N, option_type="call", seed=None):
    """
    Monte Carlo pricing of a European call/put using simulated GBM paths.
    Returns price estimate and a 95% confidence interval.
    """
    _, S = simulate_gbm_paths(S0, r, q, sigma, T, M, N, seed=seed)
    ST = S[:, -1]

    if option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type.lower() == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    disc_payoff = np.exp(-r * T) * payoff

    price = disc_payoff.mean()
    # sample std dev with ddof=1 for an unbiased estimator
    sd = disc_payoff.std(ddof=1)
    se = sd / np.sqrt(N)
    ci95 = (price - 1.96 * se, price + 1.96 * se)

    return price, ci95

if __name__ == "__main__":
    # Example parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    q = 0.00
    sigma = 0.20
    T = 1.0
    M = 252         # daily steps (approx)
    N = 200_000     # number of paths

    call_price, call_ci = price_european_option_mc(
        S0, K, r, q, sigma, T, M, N, option_type="call", seed=123
    )
    put_price, put_ci = price_european_option_mc(
        S0, K, r, q, sigma, T, M, N, option_type="put", seed=123
    )

    print(f"European Call MC: {call_price:.4f}, 95% CI: [{call_ci[0]:.4f}, {call_ci[1]:.4f}]")
    print(f"European Put  MC: {put_price:.4f}, 95% CI: [{put_ci[0]:.4f}, {put_ci[1]:.4f}]")
