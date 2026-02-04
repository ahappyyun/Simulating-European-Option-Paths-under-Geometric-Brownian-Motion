import numpy as np
from typing import Callable, Optional, Tuple

Array = np.ndarray


def simulate_logprice_sde(
    S0: float,
    T: float,
    M: int,
    N: int,
    mu_y: Callable[[float, Array], Array],
    sigma_y: Callable[[float, Array], Array],
    method: str = "euler",
    sigma_y_dy: Optional[Callable[[float, Array], Array]] = None,
    seed: Optional[int] = None,
) -> Tuple[Array, Array, Array]:
    """
    Simulate log-price Y_t and price S_t = exp(Y_t):

        dY_t = mu(t, Y_t) dt + sigma(t, Y_t) dW_t

    Parameters
    ----------
    S0 : initial stock price
    T  : time horizon
    M  : number of time steps
    N  : number of paths
    mu_y(t, y)       : drift of log-price
    sigma_y(t, y)    : volatility of log-price
    method           : 'euler' or 'milstein'
    sigma_y_dy(t, y) : derivative ∂sigma/∂y (required for Milstein)
    seed             : random seed

    Returns
    -------
    t : (M+1,) time grid
    Y : (N, M+1) log-price paths
    S : (N, M+1) price paths (positive)
    """
    method = method.lower()
    if method not in ("euler", "milstein"):
        raise ValueError("method must be 'euler' or 'milstein'")
    if method == "milstein" and sigma_y_dy is None:
        raise ValueError("Milstein requires sigma_y_dy(t, y)")

    rng = np.random.default_rng(seed)
    dt = T / M
    t = np.linspace(0.0, T, M + 1)

    # Brownian increments
    dW = rng.standard_normal((N, M)) * np.sqrt(dt)

    Y = np.empty((N, M + 1), dtype=float)
    Y[:, 0] = np.log(S0)
    y = Y[:, 0].copy()

    for k in range(M):
        tk = t[k]
        mu = mu_y(tk, y)
        sig = sigma_y(tk, y)

        y_next = y + mu * dt + sig * dW[:, k]

        if method == "milstein":
            sig_y = sigma_y_dy(tk, y)
            y_next = y_next + 0.5 * sig * sig_y * (dW[:, k] ** 2 - dt)

        Y[:, k + 1] = y_next
        y = y_next

    S = np.exp(Y)   # positivity guaranteed
    return t, Y, S


# --------------------------------------------------
# Example: log-price SDE with state-dependent vol
# --------------------------------------------------
if __name__ == "__main__":
    # Model parameters
    S0 = 100.0
    r, q = 0.05, 0.00
    sigma0 = 0.20
    alpha = 0.50   # controls state-dependence
    T = 1.0
    M = 2000
    N = 20000

    # sigma(Y) and its derivative
    def sigma_y(t: float, y: Array) -> Array:
        return sigma0 * (1.0 + alpha * np.tanh(y))

    def sigma_y_dy(t: float, y: Array) -> Array:
        return sigma0 * alpha / (np.cosh(y) ** 2)

    # risk-neutral drift for log-price
    def mu_y(t: float, y: Array) -> Array:
        sig = sigma_y(t, y)
        return (r - q) - 0.5 * sig**2

    # Simulate
    t, Y, S = simulate_logprice_sde(
        S0=S0, T=T, M=M, N=N,
        mu_y=mu_y, sigma_y=sigma_y,
        method="milstein",
        sigma_y_dy=sigma_y_dy,
        seed=42
    )

    print("E[S_T] =", S[:, -1].mean())
    print("min(S_T) =", S[:, -1].min())  # always positive
