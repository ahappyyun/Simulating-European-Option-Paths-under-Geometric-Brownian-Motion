import numpy as np
from typing import Callable, Optional, Tuple

Array = np.ndarray

def simulate_sde_paths(
    x0: float,
    T: float,
    M: int,
    N: int,
    drift: Callable[[float, Array], Array],
    diffusion: Callable[[float, Array], Array],
    method: str = "euler",
    diffusion_dx: Optional[Callable[[float, Array], Array]] = None,
    seed: Optional[int] = None,
    return_increments: bool = False,
) -> Tuple[Array, Array, Optional[Array]]:
    """
    Simulate paths for a 1D Ito SDE:
        dX_t = a(t, X_t) dt + b(t, X_t) dW_t

    Parameters
    ----------
    x0 : initial value X_0
    T  : terminal time
    M  : number of time steps
    N  : number of paths
    drift(t, x) : returns a(t,x) for vector x (shape (N,))
    diffusion(t, x) : returns b(t,x) for vector x (shape (N,))
    method : "euler" or "milstein"
    diffusion_dx(t, x) : required for Milstein if diffusion depends on x
    seed : random seed
    return_increments : if True, also return dW array (N, M)

    Returns
    -------
    t : (M+1,) time grid
    X : (N, M+1) simulated paths
    dW : (N, M) Brownian increments if return_increments else None
    """
    method = method.lower()
    if method not in ("euler", "milstein"):
        raise ValueError("method must be 'euler' or 'milstein'.")

    if method == "milstein" and diffusion_dx is None:
        raise ValueError("Milstein method requires diffusion_dx(t, x).")

    rng = np.random.default_rng(seed)
    dt = T / M
    t = np.linspace(0.0, T, M + 1)

    # Brownian increments: dW ~ N(0, dt)
    dW = rng.standard_normal((N, M)) * np.sqrt(dt)

    X = np.empty((N, M + 1), dtype=float)
    X[:, 0] = x0

    x = X[:, 0].copy()
    for k in range(M):
        tk = t[k]
        a = drift(tk, x)
        b = diffusion(tk, x)

        if a.shape != x.shape or b.shape != x.shape:
            raise ValueError("drift/diffusion must return arrays of shape (N,)")

        x_next = x + a * dt + b * dW[:, k]

        if method == "milstein":
            bx = diffusion_dx(tk, x)
            if bx.shape != x.shape:
                raise ValueError("diffusion_dx must return arrays of shape (N,)")
            x_next = x_next + 0.5 * b * bx * (dW[:, k] ** 2 - dt)

        X[:, k + 1] = x_next
        x = x_next

    return t, X, (dW if return_increments else None)

import numpy as np

# Parameters
kappa, theta, sigma = 2.0, 0.5, 0.7

def drift(t, x):
    return kappa * (theta - x)

def diffusion(t, x):
    return sigma * np.sqrt(1.0 + x**2)

# Milstein needs derivative wrt x:
# b(x)=sigma*sqrt(1+x^2) => b'(x)=sigma * x / sqrt(1+x^2)
def diffusion_dx(t, x):
    return sigma * (x / np.sqrt(1.0 + x**2))

t, X_euler, _ = simulate_sde_paths(
    x0=0.2, T=1.0, M=2000, N=5000,
    drift=drift, diffusion=diffusion,
    method="euler", seed=0
)

t, X_milst, _ = simulate_sde_paths(
    x0=0.2, T=1.0, M=2000, N=5000,
    drift=drift, diffusion=diffusion,
    method="milstein", diffusion_dx=diffusion_dx, seed=0
)

print("Euler mean at T:", X_euler[:, -1].mean())
print("Milstein mean at T:", X_milst[:, -1].mean())
