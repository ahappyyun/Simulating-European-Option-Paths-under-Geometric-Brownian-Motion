# Simulating European Option Paths under Geometric Brownian Motion

## Problem Formulation: European Option under GBM

### Stock price model
The stock price \( S_t \) follows a Geometric Brownian Motion (GBM):
\[
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,
\]
where \( \mu \) is the drift and \( \sigma \) is the volatility.

---

### Risk-neutral dynamics
For option pricing, we work under the risk-neutral measure \( \mathbb{Q} \), where the drift becomes \( r - q \):
\[
dS_t = (r - q) S_t\,dt + \sigma S_t\,dW_t^{\mathbb{Q}}.
\]

The solution is
\[
S_t = S_0 \exp\!\left(\left(r - q - \tfrac12 \sigma^2\right)t + \sigma W_t^{\mathbb{Q}}\right).
\]

---

### Time discretization
Let \( T \) be the maturity and divide \([0,T]\) into \( M \) steps with \( \Delta t = T/M \).
The exact discretization gives
\[
S_{t_{k+1}}
= S_{t_k}
\exp\!\left(\left(r - q - \tfrac12 \sigma^2\right)\Delta t
+ \sigma \sqrt{\Delta t}\, Z_k \right),
\quad Z_k \sim \mathcal{N}(0,1).
\]

---

### European option payoff
For strike \( K \):
- Call: \( (S_T - K)^+ \)
- Put: \( (K - S_T)^+ \)

---

### Monte Carlo pricing
The option price is the discounted risk-neutral expectation:
\[
V_0 = e^{-rT}\mathbb{E}^{\mathbb{Q}}[\text{Payoff}].
\]

Using \( N \) simulated paths:
\[
\widehat{V}_0
= e^{-rT}\frac{1}{N}\sum_{i=1}^N \text{Payoff}^{(i)}.
\]
