# Simulating European Option Paths under Different apporaches

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

---

### Binomial Tree

Divide \([0,T]\) into \(M\) steps with \(\Delta t = T/M\).
The stock moves up/down each step:
\[
S_{n+1} =
\begin{cases}
uS_n & \text{(up)} \\
dS_n & \text{(down)}
\end{cases}
\]

CRR parameters:
\[
u = e^{\sigma\sqrt{\Delta t}}, \quad d = e^{-\sigma\sqrt{\Delta t}} = \frac{1}{u}.
\]

Risk-neutral probability:
\[
p = \frac{e^{(r-q)\Delta t} - d}{u-d}.
\]

European payoff at maturity:
- Call: \( (S_T-K)^+ \)
- Put: \( (K-S_T)^+ \)

Price via backward induction:
\[
V_n(j) = e^{-r\Delta t}\Big(p\,V_{n+1}(j+1) + (1-p)\,V_{n+1}(j)\Big),
\]
starting from terminal payoffs at \(n=M\).

---

### General stochastic differential equation

Let \( X_t \) denote the stock price (or state variable), assumed to follow a general Itô SDE:
\[
dX_t = a(t, X_t)\,dt + b(t, X_t)\,dW_t,
\qquad X_0 = x_0,
\]
where  
- \( a(t,x) \) is the drift function,  
- \( b(t,x) \) is the diffusion (volatility) function,  
- \( W_t \) is a standard Brownian motion.

In general, this SDE does **not admit a closed-form solution**.

---

### Time discretization

Let the time horizon be \( T \), and divide \([0,T]\) into \( M \) steps:
\[
0 = t_0 < t_1 < \cdots < t_M = T,
\qquad \Delta t = \frac{T}{M}.
\]

Brownian increments satisfy
\[
\Delta W_k = W_{t_{k+1}} - W_{t_k} \sim \mathcal{N}(0, \Delta t),
\quad k = 0,\dots,M-1.
\]

---

### Euler–Maruyama scheme

The Euler–Maruyama method approximates the SDE by
\[
X_{k+1}
= X_k
+ a(t_k, X_k)\,\Delta t
+ b(t_k, X_k)\,\Delta W_k.
\]

---

### Milstein scheme (state-dependent diffusion)

If \( b(t,x) \) is differentiable with respect to \( x \), the Milstein scheme is
\[
X_{k+1}
= X_k
+ a(t_k, X_k)\,\Delta t
+ b(t_k, X_k)\,\Delta W_k
+ \tfrac12 b(t_k, X_k)\,b_x(t_k, X_k)
\big((\Delta W_k)^2 - \Delta t\big),
\]
where \( b_x = \partial b / \partial x \).

---

### Example: nonlinear stochastic volatility SDE

As a concrete example, consider the following SDE:
\[
dX_t
= \kappa(\theta - X_t)\,dt
+ \sigma \sqrt{1 + X_t^2}\,dW_t,
\qquad X_0 = x_0,
\]
where \( \kappa > 0 \), \( \theta \in \mathbb{R} \), and \( \sigma > 0 \).

This model has **state-dependent diffusion** and does not admit a closed-form solution,
so numerical schemes such as Euler–Maruyama or Milstein are required to simulate its paths.

---

### Key idea

By iterating the chosen numerical scheme and generating independent Gaussian increments,
one can simulate many sample paths \( \{X_t^{(i)}\} \), which can then be used for
Monte Carlo estimation or option pricing.
