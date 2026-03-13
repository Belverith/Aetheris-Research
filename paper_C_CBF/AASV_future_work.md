# AASV — Saved for Future Paper

Removed from Paper_C.md on 2026-03-09 to tighten scope.
Experimental backing exists in core_safety/ (aasv_proof_final.py, robust_black_swan_proof.py, generate_figure_10b.py, generate_figure_11.py).

---

## Original Section (LaTeX)

```latex
% ============================================================================
\section{Active Adversarial Safety Verification}
\label{sec:aasv}
% ============================================================================

While MCBC certifies $P_{\text{fail}} < \epsilon$ via passive sampling, the Concentration of Measure phenomenon in high dimensions \cite{vershynin2018} implies that narrow failure regions (``Black Swan'' singularities) with diameter larger than $\varepsilon_s / L_h$ but negligible volume may evade uniform sampling. We augment MCBC with three active mechanisms.

\subsection{The Hunter: Momentum PGD on the Barrier Landscape}

Instead of relying on random samples, the Hunter actively seeks to \textit{minimize} $h(x)$ along the system's trajectory using Momentum-Accelerated Projected Gradient Descent with stochastic restarts \cite{madry2018, nesterov2004}:
\begin{align}
    v_{t+1} &= \mu v_t - \alpha \nabla_x h(\tilde{x}_{\text{plan}} + \xi) \\
    x_{\text{adv}} &= \text{Proj}_{\mathcal{S}}(\tilde{x}_{\text{plan}} + v_{t+1})
\end{align}
where $\mu \in [0.8, 0.95]$ is the momentum coefficient and $\xi \sim \mathcal{N}(0, \sigma^2 I)$ provides stochastic perturbation to escape saddle points. We execute $k$ parallel restarts, each running for $T$ iterations.

\begin{theorem}[AASV Detection Bound]
\label{thm:aasv}
Let $h$ be $L_h$-Lipschitz and let $x^* \in \partial\mathcal{S}$ be a failure mode. If each of $k$ independent PGD restarts has detection probability $\geq p_{\emph{hit}}$, then:
\begin{equation}
    P(\text{missed spike}) \leq (1 - p_{\emph{hit}})^k
\end{equation}
For $M$ independent failure modes, the probability of missing \emph{any} is bounded by $M(1 - p_{\emph{hit}})^k$ via union bound. Setting $k \geq \frac{\ln(M/\alpha)}{\ln(1/(1-p_{\emph{hit}}))}$ ensures total missed-detection probability $\leq \alpha$.
\end{theorem}

\begin{remark}[Status of $p_{\text{hit}}$]
The bound in Theorem~\ref{thm:aasv} is conditional: $p_{\text{hit}}$ must be calibrated empirically on representative barrier instances. For our $\mathbb{R}^{128}$ experiments with Gaussian spike barriers, empirical measurement yields $p_{\text{hit}} \geq 0.05$ per restart; with $k = 60$ restarts, the missed-detection probability is $\leq 0.046$ per spike. Practitioners should treat $p_{\text{hit}}$ as a system-specific parameter requiring empirical calibration.
\end{remark}

\subsection{The Buffer: Adaptive Spectral Safety Margins}

Drawing on tube-based robust MPC \cite{mayne2005}, we strengthen the barrier condition to:
\begin{equation}
    h(x) \geq \rho(x) + \epsilon_{\text{model}} + \Delta_{\text{noise}}
    \label{eq:robust_barrier}
\end{equation}
where $\rho(x) = \tilde{\sigma}_{\max}(J_f(x)) \cdot d_{\text{step}}$ is the local volatility margin, proportional to the estimated spectral radius of the dynamics Jacobian. This creates a dynamic safety tube that thickens in volatile regions and thins in smooth regions, avoiding the ``Frozen Robot'' problem \cite{trautman2010}.

\textbf{Matrix-free spectral estimation.} Computing $\tilde{\sigma}_{\max}(J_f)$ via full SVD scales as $O(n^3)$. We use Hutchinson's stochastic trace estimator \cite{hutchinson1990}: draw $k$ random probes $z_i \sim \mathcal{N}(0, I)$, compute Jacobian-vector products $J_f z_i$ via forward-mode automatic differentiation in $O(n)$ per probe, and estimate $\tilde{\sigma}_{\max}$ from the Rayleigh quotients $\|J_f z_i\| / \|z_i\|$. With $k = 5$ probes, this achieves $O(n)$ total cost with $< 10\%$ relative error in our experiments.

\subsection{Anti-Memory: Orthogonal Prototype Retention}

When the barrier $h$ is a learned function (e.g., a neural network trained on streaming data), catastrophic forgetting \cite{kirkpatrick2017} may cause the network to lose previously safe state representations. The Anti-Memory module retains an orthonormal set of ``prototype'' safe states $\{p_1, \ldots, p_m\}$ via Gram-Schmidt, and periodically verifies that $h(p_i) > 0$ for all prototypes. If any prototype's barrier value drops below a threshold, the network is retrained with the prototype included as a hard constraint. This provides a lightweight integrity check against barrier drift.
```

## Related bibliography entries to include if this becomes its own paper:

- vershynin2018
- madry2018
- nesterov2004
- mayne2005
- hutchinson1990
- trautman2010
- kirkpatrick2017
