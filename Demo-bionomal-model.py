import numpy as np

def crr_binomial_european(S0, K, r, q, sigma, T, M, option_type="call"):
    dt = T / M
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    a = np.exp((r - q) * dt)  # growth under Q per step

    p = (a - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Risk-neutral probability out of [0,1]: p={p}. "
                         "Try smaller dt (larger M) or check parameters.")

    # Terminal stock prices S_T(j) with j up-moves: j=0..M
    j = np.arange(M + 1)
    ST = S0 * (u ** j) * (d ** (M - j))

    if option_type.lower() == "call":
        V = np.maximum(ST - K, 0.0)
    elif option_type.lower() == "put":
        V = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    # Backward induction
    for _ in range(M):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])

    return float(V[0]), p, (u, d)

S0, K, r, q, sigma, T = 100, 100, 0.05, 0.0, 0.2, 1.0


# Binomial CRR
crr_price, p, (u, d) = crr_binomial_european(S0, K, r, q, sigma, T, M=500, option_type="call")
print("CRR call:", crr_price, "p:", p, "u,d:", (u, d))
