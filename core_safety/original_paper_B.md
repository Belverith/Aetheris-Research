% ============================================================================
% PAPER B: Cut Content from Paper A Revision
% ============================================================================
% This file preserves all content removed from original_paper_A.md during the
% revision that split the full CHDBO paper into:
%   Paper A: Core Theory (CBF-QP, MCBC, Utility Maximization)
%   Paper B: AASV, WTA, GPT-2 Experiment (this file)
%
% Paper B is intended to be published separately, referencing Paper A.
% ============================================================================

% ============================================================================
% CUT SECTION: §3.5 Active Adversarial Safety Verification (AASV)
% Originally lines 353-592 of original_paper.md
% ============================================================================

\subsection{Active Adversarial Safety Verification (AASV)}
While Algorithm~\ref{alg:verify} certifies $P_{\text{fail}} < \epsilon$ via passive sampling, the Concentration of Measure phenomenon (Section~3.3) implies that ``spiky'' failure regions---narrow manifolds with large diameters but negligible volume---may evade uniform sampling entirely. We refer to these as ``Black Swan'' singularities: failure modes that are statistically invisible to random verification yet catastrophic if encountered along the agent's trajectory.

To address this, we augment the probabilistic verification framework with the \textbf{Active Adversarial Safety Verification (AASV)} module, which transitions safety certification from passive statistical assurance to active, optimization-based threat hunting. AASV comprises three interlocking mechanisms.

\subsubsection{The ``Hunter'': Momentum-Accelerated Trajectory Attack}
Instead of relying solely on random samples $x \sim \mathcal{N}(0, I_n)$ to detect barrier violations, the Hunter employs an optimization-based sampler that \textit{actively seeks to minimize} the barrier function $h(x)$ along the agent's planned trajectory. The key insight is that while random sampling asks ``Is this random point safe?'', the Hunter asks ``Can an intelligent adversary force the system into an unsafe state?''---the gold standard in robust control \cite{goodfellow2014, madry2018}.

\textbf{Surrogate Gradient Availability.}
In semantic embedding spaces ($n \gg 100$), the system dynamics $f(x)$ often involve computationally expensive Transformer transitions where exact backpropagation for every control step is prohibitive. To resolve this, the Hunter attacks a \textbf{Local Linearized Surrogate} (LLS), $\tilde{f}(x)$, approximating the dynamics around the current trajectory via the Jacobian $J_f(x)$. Critically, the surrogate introduces bounded error, which we account for explicitly:
\begin{equation}
    \epsilon_{\text{model}} \geq \sup_{x \in \mathcal{B}(x_0, r)} \|f(x) - \tilde{f}(x)\|
\end{equation}
This error bound is incorporated directly into the robust barrier condition (Equation~\ref{eq:robust_barrier}), ensuring that even if the surrogate underestimates danger, the safety margin absorbs the discrepancy.

\textbf{Practical Computation of $\epsilon_{\text{model}}$.} While computing the exact supremum over a continuous ball is intractable in general, a tight \textit{upper bound} is readily computable in $O(n)$ via the Taylor remainder: for a twice-differentiable $f(x)$, the linearization error over $\mathcal{B}(x_0, r)$ satisfies $\|f(x) - \tilde{f}(x)\| \leq \frac{1}{2} L_{\nabla f} r^2$, where $L_{\nabla f}$ is the local Lipschitz constant of the Jacobian (estimable via Hessian-vector products at $O(n)$ cost \cite{baydin2018}). For black-box systems without gradient access, $\epsilon_{\text{model}}$ may be calibrated empirically by evaluating $\|f(x) - \tilde{f}(x)\|$ at $m$ random points within $\mathcal{B}(x_0, r)$ and taking the maximum observed value plus a Hoeffding correction. The framework degrades gracefully: a larger $\epsilon_{\text{model}}$ shrinks the feasible operating region but the safety guarantee remains mathematically valid. In the limiting case where $\epsilon_{\text{model}}$ exceeds $\max_x h(x)$, the feasible set becomes empty and the agent halts---the correct safe default when the surrogate cannot be trusted.

\textbf{Transformer-Specific $\epsilon_{\text{model}}$ Quantification.} For Transformer architectures used as dynamical systems (Experiment~XIV), the dynamics $f_l(x) = \text{Block}_l(x)$ are the composition of self-attention and feed-forward sub-layers. The local Jacobian Lipschitz constant $L_{\nabla f}$ is dominated by the softmax attention mechanism, whose gradient can exhibit sharp transitions near attention saturation. Empirically, for GPT-2 ($n = 768$) with the KNN dynamics surrogate used in our experiments (Section~\ref{sec:exp_gpt2}), the surrogate error can be bounded by a leave-one-out cross-validation residual: $\hat{\epsilon}_{\text{model}}^{\text{KNN}} = \max_{i} \|f(x_i) - \hat{f}_{-i}(x_i)\|$, where $\hat{f}_{-i}$ denotes the KNN estimator trained with point $i$ excluded. We observe that despite the large dynamics magnitudes ($\|f\| \approx 283$), the pre-intervention CBF margin ($w \cdot f(x_{\text{bnd}}) \in [282.9, 384.2]$) exceeds the KNN estimation variance ($\sigma \approx 15.2$) by a factor of $\sim 20\times$, providing substantial headroom for surrogate error. Nevertheless, rigorous deployment would require either (1)~a formal local Lipschitz bound on $f_l$ via spectral norm regularization of the attention layers \cite{miyato2018}, or (2)~conformal prediction intervals on the KNN surrogate to obtain distribution-free error quantiles.

\textbf{Momentum PGD with Stochastic Restarts.}
Standard gradient descent stalls in local minima of non-convex barrier landscapes, potentially missing deeper failure modes hidden behind shallow safe basins. We employ Momentum-Accelerated Projected Gradient Descent with Stochastic Restarts \cite{madry2018, nesterov2004}:
\begin{align}
    v_{t+1} &= \mu v_t - \alpha \nabla_x h(\tilde{x}_{\text{plan}} + \xi) \\
    x_{\text{adv}} &= \text{Proj}_{\mathcal{S}} \left( \tilde{x}_{\text{plan}} + v_{t+1} \right)
\end{align}
where $\mu \in [0.8, 0.95]$ is the momentum coefficient and $\xi \sim \mathcal{N}(0, \sigma^2 I)$ is noise injected to escape saddle points. We execute $k$ \textit{parallel restarts} from different initial conditions, each running for $T$ iterations.

\textbf{Guarantee.} The following theorem formalizes the detection guarantee of momentum PGD with stochastic restarts.

\begin{theorem}[AASV Adversarial Detection Bound]
\label{thm:aasv_detection}
Let $h: \mathbb{R}^n \to \mathbb{R}$ be a Lipschitz-continuous barrier function, and let $x^* \in \partial\mathcal{S}$ denote a failure mode (``spike'') with $h(x^*) < 0$. Suppose each of $k$ independent PGD restarts has probability at least $p_{\emph{hit}} > 0$ of converging to a basin of attraction containing $x^*$. Then the probability that the AASV Hunter fails to detect $x^*$ after $k$ restarts satisfies:
\begin{equation}
    P(\text{missed spike}) \leq (1 - p_{\emph{hit}})^k
\end{equation}
In particular, for a barrier landscape containing $M$ independent failure modes, each with per-restart detection probability at least $p_{\emph{hit}}$, the probability of missing \emph{any} failure mode is bounded by:
\begin{equation}
    P(\exists\, \text{undetected spike}) \leq M \cdot (1 - p_{\emph{hit}})^k
\end{equation}
by a union bound. Setting $k \geq \frac{\ln(M/\alpha)}{\ln(1/(1-p_{\emph{hit}}))}$ ensures total missed-detection probability $\leq \alpha$.

\textbf{Remark (Status of $p_{\emph{hit}}$).} This theorem is a conditional guarantee: the bound is valid \emph{given} a lower bound on $p_{\emph{hit}}$, but $p_{\emph{hit}}$ itself cannot be computed from first principles for general barrier landscapes. It must be calibrated empirically on representative barrier instances (see ``Operationalizing $p_{\emph{hit}}$'' below and Experiments~IV, VI--VII). In this sense, Theorem~\ref{thm:aasv_detection} is analogous to PAC-Bayes bounds that depend on empirically estimated prior/posterior divergences: the mathematical structure is exact, but the actionable bound requires a measured input parameter.
\end{theorem}

\begin{proof}
Each restart is an independent Bernoulli trial with success probability $\geq p_{\text{hit}}$. The probability that all $k$ restarts miss a given spike is $(1-p_{\text{hit}})^k$. The union bound over $M$ spikes yields $P(\exists\,\text{miss}) \leq M(1 - p_{\text{hit}})^k$. The restart budget follows from solving $M(1 - p_{\text{hit}})^k \leq \alpha$ for $k$.
\end{proof}

We emphasize that this is a \textit{probabilistic} bound enhanced by active search---we do not claim deterministic global convergence in non-convex spaces.

\textbf{Operationalizing $p_{\text{hit}}$.} For a failure spike with angular half-width $\theta_w$ on $S^{n-1}$, the \textit{geometric} cap area ratio $\text{Cap}(\theta_w, n) / \text{Area}(S^{n-1})$ decreases exponentially with $n$ for fixed $\theta_w$ \cite{vershynin2018}---for $n = 128$ and $\theta_w = 0.05$ radians, this ratio is astronomically small ($\ll 10^{-100}$), rendering passive random detection effectively impossible. However, the gradient field of the Gaussian barrier well extends far beyond the spike's geometric footprint, creating an attraction funnel that guides momentum PGD toward the spike center from initial conditions well outside the cap. This gradient amplification enlarges the effective convergence basin by orders of magnitude compared to the passive geometric cap. In our $\mathbb{R}^{128}$ experiments, empirical measurement yields detection rates of $p_{\text{hit}} \geq 0.05$ per restart \textit{for the tested Gaussian spike geometry}. With $k = 60$ restarts, this gives $P(\text{miss all spikes}) \leq (1 - 0.05)^{60} < 0.046$ per spike under this empirical estimate, or equivalently $> 95\%$ detection probability per failure mode---conditional on $p_{\text{hit}} \geq 0.05$ holding for the deployed barrier landscape.

\textbf{Empirical Observation ($p_{\text{hit}}$ Lower Bound for Gaussian Spikes).} For the barrier $h(x) = h_0(x) - D \exp\bigl(-(1 - \cos\angle(x, s))^2 / (2\theta_w^2)\bigr)$ with a single Gaussian spike of depth $D$ and half-width $\theta_w$ centered at direction $s \in S^{n-1}$, the gradient magnitude at angular distance $\phi$ from the spike center satisfies $\|\nabla_\phi h\| \geq D \cdot \phi \cdot \theta_w^{-2} \cdot \exp(-\phi^2/(2\theta_w^2))$, which exceeds the background gradient $\|\nabla h_0\|$ for all $\phi \leq \phi_{\max} \approx \theta_w \sqrt{2 \ln(D\theta_w^{-2} / \|\nabla h_0\|)}$. The effective convergence basin of momentum PGD (with coefficient $\mu$) is therefore a cap of angular radius $\phi_{\max}$, yielding:
\begin{equation}
    p_{\text{hit}} \geq \frac{\text{Cap}(\phi_{\max}, n)}{\text{Area}(S^{n-1})} \approx \frac{1}{\sqrt{2\pi n}} \left(\frac{\sin \phi_{\max}}{\phi_{\max}}\right)^{n-2}
\end{equation}
\textbf{Remark (Cap Area Approximation).} The cap fraction formula above is an asymptotic approximation; the exact cap fraction involves the regularized incomplete beta function $I_{\sin^2\phi}((n{-}1)/2, 1/2)/2$ \cite{vershynin2018}, and different asymptotic expansions yield slightly different prefactors and exponents. The numerical estimate $p_{\text{hit}} \approx 0.003$ should be understood as order-of-magnitude guidance rather than a sharp bound. What matters for the safety guarantee is the \textit{empirically measured} $p_{\text{hit}} \geq 0.05$, which enters Theorem~\ref{thm:aasv_detection} directly.
For the experimental parameters ($D = 2, \theta_w = 0.05, \|\nabla h_0\| \approx 1, n = 128$), this gives $\phi_{\max} \approx 0.14$ rad and $p_{\text{hit}} \approx 0.003$ per restart (geometric lower bound). The empirically observed $p_{\text{hit}} \geq 0.05$ exceeds this bound because momentum PGD trajectories curve toward the spike from initial conditions outside the static convergence cap, effectively enlarging the basin via inertial overshoot.

\textbf{Caveat on $p_{\text{hit}}$ Generalization.} The empirical estimate $p_{\text{hit}} \geq 0.05$ is obtained from controlled experiments with analytically defined barrier landscapes where the gradient is computable exactly (Section~5). In real deployment scenarios where the barrier landscape is unknown or the gradient is obtained through a learned surrogate, $p_{\text{hit}}$ may be substantially lower due to gradient noise, surrogate error, and unforeseen barrier topology. Practitioners should therefore treat $p_{\text{hit}}$ as a system-specific parameter requiring empirical calibration on representative barrier instances, and should increase the restart count $k$ conservatively to compensate for uncertainty in the detection rate.

\subsubsection{The ``Buffer'': Adaptive Tube-Based Spectral Margins}
The original verification condition $h(x) \geq 0$ is insufficient for adversarial robustness because it ignores three sources of uncertainty: (1) surrogate model error, (2) physical noise/disturbance, and (3) geometric uncertainty between sample points. Drawing on the tube-based robust MPC framework of \cite{mayne2005}, we strengthen the barrier condition to a \textbf{Robust Tube-Based Constraint}:
\begin{equation}
    h(x) \geq \rho(x) + \epsilon_{\text{model}} + \Delta_{\text{noise}}
    \label{eq:robust_barrier}
\end{equation}
where:
\begin{itemize}
    \item $\rho(x) = \tilde{\sigma}_{\max}(J_f(x)) \cdot d_{\text{step}}$ is the \textbf{local volatility margin}, proportional to the estimated spectral norm of the dynamics Jacobian and the agent's step size. This creates a dynamic safety ``tube'' that thickens in volatile, highly nonlinear regions and thins in smooth regions, avoiding the ``Frozen Robot'' problem \cite{trautman2010} caused by overly conservative global Lipschitz bounds.
    \item $\epsilon_{\text{model}}$ is the surrogate divergence bound from Section~3.5.1.
    \item $\Delta_{\text{noise}}$ is the physical disturbance bound, known from hardware specifications or estimated online.
\end{itemize}
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_8.png}
    \caption{\texorpdfstring{\textbf{Tube-Based Adaptive Safety Margins.}}{Tube-Based Adaptive Safety Margins.} Left: A fixed global Lipschitz margin creates an overly conservative ``dead zone'' that induces the Frozen Robot problem in high-curvature regions. Right: The AASV Buffer computes adaptive margins $\rho(x) = \tilde{\sigma}_{\max}(J_f) \cdot d_{\text{step}}$ that thicken near volatile boundary segments and thin in smooth regions, preserving agent mobility while maintaining safety.}
    \label{fig:adaptive_tube}
\end{figure}
\textbf{Matrix-Free Spectral Estimation for $O(n)$ Scaling.}
A critical implementation detail is the computation of $\tilde{\sigma}_{\max}(J_f(x))$. Full Singular Value Decomposition scales as $O(n^3)$. Explicit-matrix Power Iteration requires $O(n^2)$ per iteration (forming and multiplying by the full $n \times n$ Jacobian). For semantic spaces where $n = 12{,}288$ \cite{brown2020}, both explicit-matrix approaches violate the $O(n)$ real-time constraint established in Problem 1.

Critically, neither method requires explicit matrix construction when automatic differentiation (AD) is available \cite{baydin2018}. A Jacobian-vector product $J_f z$ is computable in $O(n)$ via forward-mode AD, and the transpose product $J_f^T w$ in $O(n)$ via reverse-mode AD (backpropagation), without ever forming the $n \times n$ Jacobian. This observation enables two matrix-free $O(n)$ strategies:

\textbf{Strategy A: AD-Based Power Iteration (Preferred).}
Power iteration for $\sigma_{\max}(J_f)$ requires only the iteration $w \leftarrow J_f^T J_f v / \|J_f^T J_f v\|$, each step involving one JVP and one VJP at cost $O(n)$. After $k \approx 10\text{--}20$ iterations (sufficient for convergence when the spectral gap $\sigma_1/\sigma_2$ is bounded away from 1), this yields a \textit{tight} estimate of $\sigma_{\max}$ at total cost $O(kn) = O(n)$.

\textbf{Strategy B: Hutchinson Trace Estimator (Fallback).}
When only forward-mode AD is available (precluding the VJP step), we employ \textbf{Hutchinson's Trace Estimator} \cite{hutchinson1990, avron2011, meyer2021}, a matrix-free randomized method. The Frobenius norm of the Jacobian---which provides a conservative upper bound on the spectral norm, $\|J_f\|_F \geq \sigma_{\max}(J_f)$---is estimated using $m$ random Rademacher probe vectors $z_i \in \{-1, +1\}^n$:
\begin{equation}
    \tilde{\sigma}_{\max}(J_f) \leq \|J_f\|_F \approx \sqrt{\frac{1}{m} \sum_{i=1}^{m} z_i^T J_f^T J_f z_i}
\end{equation}
The inequality $\|J_f\|_F \geq \sigma_{\max}$ follows immediately from $\|J_f\|_F^2 = \sum_i \sigma_i^2 \geq \sigma_{\max}^2$, ensuring that the resulting safety margin is never underestimated. Each term $J_f z_i$ requires only a \textit{Jacobian-vector product} (computable in $O(n)$ via forward-mode AD \cite{baydin2018}). Total complexity: $O(m \cdot n)$ where $m \approx 20\text{--}30$ samples suffice for accurate estimation.

\textbf{Remark (Frobenius Tightness).} The Frobenius bound satisfies $\sigma_{\max} \leq \|J_f\|_F \leq \sqrt{\mathrm{rank}(J_f)} \cdot \sigma_{\max}$, so the overestimation factor is at most $\sqrt{\mathrm{rank}(J_f)}$. For a full-rank $128 \times 128$ Jacobian with uniform singular values, this gap is $\sqrt{128} \approx 11.3$. In practice, two factors mitigate this for kinematic systems: (1) the Jacobians of smooth physical dynamics typically exhibit rapid spectral decay (a few dominant singular values), reducing the effective rank and tightening the bound considerably; and (2) even with the Frobenius overestimate, the adaptive tube $\rho(x) = \tilde{\sigma}_{\max} \cdot d_{\text{step}}$ still varies across states---thinning in smooth regions and thickening in volatile ones---which is strictly less conservative than a fixed global Lipschitz margin. However, we caution that dense semantic models (e.g., Large Language Model transformers) are often engineered to preserve variance across all dimensions to maximize representational capacity \cite{brown2020}, resulting in near-full-rank Jacobians where the Frobenius gap approaches $\sqrt{n}$. In such regimes, Strategy~B may induce excessive conservatism; for full-rank $n=128$ Jacobians, the Frobenius overestimate inflates the margin by $\sqrt{128}\approx 11.3\times$, which we show empirically still permits agent mobility in our experiments (Section~5), but may be prohibitive at larger scales. \textbf{For full-rank semantic dynamics, Strategy~A (AD-based power iteration) is therefore mandatory}, as it converges directly to $\sigma_{\max}$ without the rank-dependent gap. Strategy~B should be reserved for inherently low-rank manifolds or as a conservative fallback when reverse-mode AD is unavailable.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_5.png}
    \caption{\texorpdfstring{\textbf{Hutchinson Trace Estimator Convergence ($n=128$).}}{Hutchinson Trace Estimator Convergence (n=128).} Left: The Hutchinson estimate of $\mathrm{tr}(J^T J)$ converges to the true value (50.9) as the number of Rademacher probe vectors $m$ increases; shaded bands show 50\% and 80\% confidence intervals over 100 trials. Center: Median relative estimation error decreases below 10\% at $m \approx 30$. Right: AD-based JVP validation using JAX \cite{jax2018}: the Hutchinson estimate computed via genuine $O(n)$ forward-mode AD JVPs matches the dense-matrix implementation to within $0.000003\%$, confirming that the $O(n)$ pathway produces numerically identical results. \textbf{Implementation note:} The dense-matrix panels use explicit $J \cdot z$ ($O(n^2)$ per multiply); the AD panel demonstrates the production-grade $O(n)$ pipeline using \texttt{jax.jvp}. The nonlinear dynamics $f(x) = \tanh(Ax)$ introduce a linearization error of $\sim$2.5\% relative to the frozen-Jacobian Hutchinson estimate, which the framework correctly measures and accommodates.}
    \label{fig:hutchinson_convergence}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_7.png}
    \caption{\texorpdfstring{\textbf{Computational Scaling: SVD vs.\ Power Iteration vs.\ Hutchinson (Dense \& AD).}}{Computational Scaling: SVD vs. Power Iteration vs. Hutchinson (Dense and AD).} Wall-clock time (median of seven runs) for spectral norm computation across dimensions $n = 16$ to $n = 2048$ (log-log scale). Full SVD scales as $O(n^3)$; explicit-matrix Hutchinson and Power Iteration scale as $O(n^2)$. The fourth curve shows \textbf{AD-based Hutchinson JVP} using JAX forward-mode AD, which achieves genuine $O(n)$ scaling. At $n = 1024$, the AD pathway (0.023s) overtakes the dense-matrix Hutchinson (0.034s), confirming the crossover predicted by Section~3.5.2. At $n = 2048$, the AD speedup reaches $\sim$2.5$\times$ over dense. All four methods produce identical spectral estimates (AD matches dense to within $0.000003\%$). Practitioners implementing AASV at production scale ($n > 1000$) obtain both correctness and speed by using the AD pathway.}
    \label{fig:computational_scaling}
\end{figure}

\textbf{Domain-Adaptive Strategy:}
\begin{itemize}
    \item \textbf{Kinematic Spaces} ($n \leq 512$): Explicit-matrix power iteration on $J_f(x)$, yielding exact $\sigma_{\max}$ at $O(n^2)$ cost, computationally negligible ($<1$ms).
    \item \textbf{Semantic Spaces with AD} ($n > 512$, forward + reverse AD available): Matrix-free power iteration via JVP/VJP (Strategy~A), yielding tight $\sigma_{\max}$ at $O(kn) = O(n)$. This is the preferred mode for differentiable dynamics.
    \item \textbf{Semantic Spaces, forward-only AD} ($n > 512$, only JVPs available): Hutchinson/Frobenius estimation (Strategy~B), yielding a conservative upper bound at $O(mn) = O(n)$. The overestimate factor $\leq \sqrt{\mathrm{rank}(J_f)}$ is mitigated by spectral decay in smooth systems.
\end{itemize}

\subsubsection{The ``Anti-Memory'': Holographic Orthogonal Prototypes}
To prevent the Hunter from cycling through known safe local minima---wasting computation rediscovering the same shallow basins---we maintain a \textbf{Forbidden Map} using the Vector Symbolic Architecture (VSA) substrate established in \cite{scrivens2026, kanerva2009}.

\textbf{Why Not Centroid Averaging?} Na\"ive aggregation methods (e.g., centroid averaging of failure directions) collapse multi-modal failure structure. If three Black Swan spikes exist at orthogonal directions $v_1$, $v_2$, and $v_3$, their centroid $v_{\text{avg}} = (v_1 + v_2 + v_3)/\|v_1 + v_2 + v_3\|$ may point to a \textit{safe} region, effectively erasing the memory of all three dangers (Figure~\ref{fig:anti_memory}, left). This is unacceptable for safety-critical applications.

\textbf{Orthogonal Prototype Retention.} We enforce an orthogonality constraint on stored failure modes:
\begin{equation}
    \text{Store } v_{\text{new}} \text{ as distinct prototype if: } |v_{\text{new}} \cdot v_{\text{centroid}}| < \theta
\end{equation}
where $\theta$ is a similarity threshold (we use $\theta = 0.3$ throughout). If the new failure mode is collinear with an existing centroid, it is merged; if orthogonal, it is stored as a separate prototype. This leverages the quasi-orthogonality of random vectors in high dimensions \cite{vershynin2018}, enabling the storage of up to $O(n)$ independent failure prototypes without crosstalk.

\textbf{Repulsion Augmentation.} The Hunter's cost function is augmented to repel from stored prototypes, forcing exploration of novel failure geometries. Since the Hunter \textit{minimizes} $\mathcal{J}_{\text{hunt}}(x)$ to find barrier violations (not to be confused with the dynamics Jacobian $J_f$), adding a positive similarity penalty increases the cost near known prototypes, steering the search toward unexplored regions:
\begin{equation}
    \mathcal{J}_{\text{hunt}}(x) = h(x) + \lambda \sum_{c \in \mathcal{M}_{\text{ban}}} \text{Sim}(x, c), \quad \lambda > 0
\end{equation}
where $\mathcal{M}_{\text{ban}}$ is the set of stored failure prototypes and $\text{Sim}(\cdot, \cdot)$ is cosine similarity.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_6.png}
    \caption{\texorpdfstring{\textbf{Anti-Memory: Centroid Averaging vs.\ Orthogonal Prototype Retention ($\mathbb{R}^{128}$).}}{Anti-Memory: Centroid Averaging vs. Orthogonal Prototype Retention (R128).} Left: Centroid averaging collapses three orthogonal failure directions into a single mean vector that points to a \textit{safe} region---effectively erasing memory of all three Black Swan spikes. Right: Orthogonal Prototype Retention (AASV Anti-Memory) stores each failure direction independently when $|v_{\text{new}} \cdot v_{\text{centroid}}| < \theta$, preserving all spike locations and enabling repulsion-guided exploration.}
    \label{fig:anti_memory}
\end{figure}

\subsubsection{Pipelined Real-Time Verification}
A critical challenge is that iterative PGD attacks introduce computational latency, potentially violating the real-time control loop constraint ($< 10$ms). Naively running the Hunter asynchronously (``Dreaming'') creates a dangerous temporal gap: the agent might act on outdated verification data, breaking the Forward Invariance guarantee of Theorem~\ref{thm:topological_safety}.

We resolve this via \textbf{Tube-Based Pipelined Verification with Recursive Backup}:

\begin{enumerate}
    \item \textbf{Time $t$ (Execute):} The agent executes control $u_t$, which was verified and certified safe at time $t-1$.
    \item \textbf{Time $t$ (Hunt):} The AASV Hunter simultaneously attacks the $\epsilon$-ball around $x_{\text{pred}}(t+1)$, the predicted outcome of the next intended action $u_{t+1}$. The $\epsilon$-ball accounts for physical drift, ensuring the verification covers the entire region the agent might physically occupy.
    \item \textbf{Time $t$ (Certify):} If $\min_{x \in \mathcal{B}(x_{\text{pred}}, \epsilon)} h(x) > 0$ (accounting for the robust margin of Equation~\ref{eq:robust_barrier}), authorize $u_{t+1}$.
\end{enumerate}

\textbf{Recursive Feasibility.} If the Hunter detects a violation in $u_{t+1}$:
\begin{itemize}
    \item The system \textbf{rejects} $u_{t+1}$.
    \item The controller engages a domain-specific \textbf{Safe Backup Trajectory}: for kinematic agents, an invariant orbit or emergency brake that is pre-verified to remain within $\mathcal{S}$ indefinitely; for semantic agents, a reversion to the last verified safe context embedding (an ``Anchor State'' in the VSA substrate \cite{scrivens2026}).
\end{itemize}

By hiding the verification cost of step $t+1$ inside the physical execution time of step $t$, this pipeline preserves the $O(1)$ per-step latency while ensuring the agent never enters a state that has not survived a dedicated adversarial attack.

\textbf{Integrated Verification Pipeline.} Algorithm~\ref{alg:pipeline} formalizes how MCBC statistical verification and AASV adversarial verification combine at runtime. MCBC provides an initial offline certificate (run once or periodically); AASV operates online at every control step.

\begin{algorithm}
\caption{CHDBO Runtime Verification Pipeline (MCBC + AASV + CBF-QP)}\label{alg:pipeline}
\begin{algorithmic}[1]
\State \textbf{Offline:} Run Algorithm~\ref{alg:verify} (MCBC) to certify $\hat{P}_{\text{fail}} < \epsilon$ on $\partial\mathcal{S}$
\State \textbf{Initialize:} Prototype memory $\mathcal{M}_{\text{ban}} \leftarrow \emptyset$, state $x \leftarrow x_0 \in \text{Int}(\mathcal{S})$
\For{each control step $t = 0, 1, 2, \ldots$}
    \State Compute nominal control: $u_{\text{nom}} \leftarrow \nabla U(x)$
    \State Solve CBF-QP (Eq.~\ref{eq:cbfqp}): $u^* \leftarrow \text{argmin}_{u} \|u - u_{\text{nom}}\|^2$ s.t.\ $L_f h + L_g h\, u \geq -\gamma h$
    \State Predict next state: $x_{\text{pred}} \leftarrow x + (f(x) + g(x)\, u^*) \cdot \Delta t$
    \State \textbf{AASV Hunter:} Run $k$ PGD restarts on $\mathcal{B}(x_{\text{pred}}, \epsilon)$, repelling from $\mathcal{M}_{\text{ban}}$
    \If{Hunter finds violation $h(x_{\text{adv}}) < \rho(x) + \epsilon_{\text{model}} + \Delta_{\text{noise}}$}
        \State Store $x_{\text{adv}} / \|x_{\text{adv}}\|$ in $\mathcal{M}_{\text{ban}}$ (if novel)
        \State \textbf{Reject} $u^*$; engage safe backup trajectory
    \Else
        \State \textbf{Execute} $u^*$; update $x \leftarrow x_{\text{pred}}$
    \EndIf
\EndFor
\end{algorithmic}
\end{algorithm}

Figure~\ref{fig:architecture} provides a visual overview of the data-flow between CHDBO's principal modules, showing how the offline MCBC certificate, the online CBF-QP safety filter, and the AASV adversarial verification interact at each control step.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_14.png}
    \caption{\texorpdfstring{\textbf{System Architecture: CHDBO Runtime Data Flow.}}{System Architecture: CHDBO Runtime Data Flow.} The offline stage (blue) computes the MCBC barrier certificate periodically. At each online control step (green), the CBF-QP safety filter produces $u^*$; the AASV Hunter attacks the predicted next state; the Anti-Memory (prototype store) ensures novel threat exploration; and the spectral margin module (Hutchinson estimator) quantifies drift volatility. If the Hunter detects a violation, the system engages the safe backup trajectory and rejects $u^*$; otherwise, $u^*$ is executed.}
    \label{fig:architecture}
\end{figure}

\subsubsection{Safety Guarantee Tiers}
The AASV module provides tiered guarantees depending on the application domain:

\begin{table}[htbp]
\centering
\caption{AASV Safety Guarantee Tiers}
\small
\begin{tabular}{@{}l l p{6.5cm}@{}}
\toprule
\textbf{Domain} & \textbf{Guarantee Type} & \textbf{Formal Statement} \\ \midrule
Kinematic ($n \leq 512$) & Deterministic within bounds & If $\|w\| \leq \Delta_{\text{noise}}$ and $\|\tilde{f} - f\| \leq \epsilon_{\text{model}}$, then $h(x) \geq 0$ is \textbf{guaranteed}. \\
Semantic ($n > 512$) & Probabilistically bounded & $P(\text{undetected Black Swan}) \leq (1 - p_{\text{hit}})^k + \epsilon_{\text{Hutch}}$, where $\epsilon_{\text{Hutch}} \propto 1/\sqrt{m}$ is the Hutchinson spectral estimation error \\ \bottomrule
\end{tabular}
\label{tab:aasv_tiers}
\end{table}

We do not claim 100\% safety for semantic spaces. We claim \textit{adversarially-bounded risk}, which is the strongest achievable guarantee given computational constraints and the non-convexity of the barrier landscape.

\subsubsection{Combined MCBC + AASV Safety Guarantee}
The following corollary formalizes the complementary coverage provided by the two verification layers.

\begin{corollary}[Joint MCBC--AASV Safety Certificate]
\label{cor:joint_safety}
Let the MCBC certificate guarantee $\hat{P}_{\emph{fail}} < \epsilon$ on $\partial\mathcal{S}$ with confidence $1-\delta$, and let the AASV Hunter execute $k$ independent PGD restarts per step, each detecting a given failure spike with probability at least $p_{\emph{hit}}$. Suppose there are at most $M$ independent failure modes on $\partial\mathcal{S}$. Then the per-step probability of an undetected safety violation is bounded by:
\begin{equation}
    P(\text{undetected violation per step}) \leq \min\!\bigl(\epsilon,\; M(1-p_{\emph{hit}})^k\bigr) + \delta
    \label{eq:joint_safety}
\end{equation}
The first term reflects that MCBC already bounds the volume fraction of failures to $\epsilon$; the second term, independent of $\epsilon$, reflects that AASV provides active worst-case detection. The $\delta$ term accounts for the finite-confidence of the MCBC certificate. For trajectory-level safety over $T$ steps, applying Proposition~\ref{prop:trajectory_bridge}:
\begin{equation}
    P(\text{any undetected violation in } T \text{ steps}) \leq T \cdot \bigl[\min(\epsilon, M(1-p_{\emph{hit}})^k) + \delta\bigr]
\end{equation}
\end{corollary}

\textbf{Remark.} The $\min(\cdot)$ structure captures the division of labor: MCBC dominates when failure modes have non-negligible volume ($> \epsilon$), while AASV dominates for volumetrically invisible spikes. Neither layer alone provides complete coverage; together, they bound both statistical and adversarial failure modes.

\textbf{Remark (Shared Barrier Coupling and Bound on $M$).}
Both MCBC and AASV evaluate the same barrier function $h(x)$. Consequently, if $h$ is poorly calibrated (e.g., it misclassifies a region of the boundary as safe), the error propagates to \textit{both} layers simultaneously, and the min-composition in Equation~\eqref{eq:joint_safety} does not provide additional protection against barrier mis-specification. The bound assumes that barrier mis-specification is negligible relative to $\epsilon$ and $p_{\text{hit}}$. In practice, cross-validation of the barrier (as in Experiment~XIV) partially mitigates this coupling but does not eliminate it. The parameter $M$ (maximum number of independent failure modes) must be supplied by the practitioner based on domain knowledge or barrier architecture; for analytically defined barriers with known spike structure, $M$ is exact, but for learned barriers $M$ is a conservative upper bound. Over-estimating $M$ increases the required restart budget $k$ linearly (via the union bound), while under-estimating $M$ invalidates the guarantee.

\subsubsection{Assumptions and Limitations}
For intellectual honesty, we state the assumptions underlying AASV:
\begin{enumerate}
    \item \textbf{Surrogate Validity:} The error bound $\epsilon_{\text{model}}$ must be estimable. For black-box LLMs, this requires empirical calibration or conservative overestimation.
    \item \textbf{Lipschitz Assumption:} The barrier function $h(x)$ is assumed Lipschitz continuous. Discontinuous dynamics (e.g., contact mechanics) require specialized treatment.
    \item \textbf{Hutchinson Variance:} The spectral estimate has variance $\propto 1/m$. For safety-critical applications, $m$ should be chosen conservatively (e.g., $m \geq 50$).
    \item \textbf{Non-Convexity:} The Hunter's probabilistic guarantee depends on the barrier landscape topology. Pathological geometries with exponentially many local minima can degrade detection probability.
    \item \textbf{WTA Gradient Decomposability (Barrier Design Requirement):} For AASV to detect multi-modal, angularly separated failure modes, the barrier gradient must be decomposable into per-spike components via a Winner-Take-All (WTA) selection mechanism. This is a \textit{barrier design requirement}, not a property that holds generically: practitioners must structure the barrier so that per-component gradients are accessible. For analytically defined barriers (e.g., sums of Gaussian wells), this holds by construction. For learned neural barriers, practitioners should design multi-head architectures \cite{dawson2023} where each head contributes an independent gradient component, enabling targeted threat hunting toward distinct failure modes. WTA does not enable discovery of arbitrary unknown failure modes from a black-box barrier; rather, it enables efficient verification against anticipated or injected failure geometries when the barrier structure is known. Non-decomposable barriers may cause the Hunter to explore averaged (centroid) directions rather than individual spikes, reducing detection probability (Experiment~VI).
\end{enumerate}


% ============================================================================
% CUT EXPERIMENT: IV - AASV Black Swan Detection
% Originally lines 753-812 of original_paper.md
% ============================================================================

\subsection{Experiment IV: AASV Black Swan Detection --- Comprehensive Robustness Evaluation}
To validate the necessity and efficacy of the AASV module (Section~3.5), we construct a comprehensive suite of controlled Black Swan scenarios in $\mathbb{R}^{128}$. Adversarial ``spike'' singularities are injected into the barrier landscape as narrow Gaussian wells on $S^{127}$ with angular half-width $\theta_w = 0.05$ radians. Their volume fraction is negligibly small---exponentially suppressed by the dimension $n$---rendering them statistically invisible to any feasible uniform sampling regime. We evaluate the AASV pipeline across eight progressively challenging configurations to stress-test detection under varying geometric conditions.

\textbf{Barrier Construction.} We define $h(x) = (1 - \|x\|) + \lambda \sum_i (1 - \cos\angle(x, s_i)) - D \sum_i \exp\!\bigl(-\frac{(1-\text{sim}_i)^2}{2\theta_w^2}\bigr)$ on $S^{127}$, where each spike $s_i$ creates a localized negative well. The depth parameter $D = (N-1)\lambda + 2$ ensures every spike produces $h < 0$ at its center despite the aggregate funnel repulsion.

\textbf{Protocol.} Each panel employs the full AASV pipeline: momentum PGD ($\mu = 0.9$, $\alpha = 0.05$, $T = 200$) with Winner-Take-All gradient blocking (threshold $= 0.7$), post-violation refinement (100 additional low-noise PGD steps converging to $\cos > 0.9999$ with the true spike center), post-hoc clustering at similarity threshold $0.98$ ($\approx 11.5^\circ$ angular resolution), and orthogonal prototype Anti-Memory for repulsion-guided novel exploration.

\textbf{WTA Oracle Caveat.} The WTA gradient in this experiment receives the spike center directions $\{s_i\}$ as explicit input parameters, enabling it to decompose $\nabla h$ into per-spike components and select the nearest unblocked direction per restart. This constitutes \textit{oracle access} to the barrier structure---it verifies that analytically constructed barriers can be efficiently searched, but it does \textit{not} demonstrate discovery of unknown failure modes from a black-box barrier. Section~\ref{sec:wta_limitation} and Experiment~VI quantify the gap: for single-mode barriers, black-box gradients suffice; for multi-modal barriers, the WTA decomposition is a necessary barrier design requirement.

\textit{Restart counts:} The subtitle $k$ denotes the \emph{initial} restart budget (Phase 1). After each novel violation is detected, additional refinement restarts are added in Phase 2: specifically, $\max(10, |\mathcal{P}| \times 4)$ restarts per refinement round, where $|\mathcal{P}|$ is the current prototype count. This adaptive restart budget means the total number of PGD runs exceeds the reported Phase~1 budget $k$; the info boxes report ``Hits / Total Restarts'' (Phase 1 + Phase 2) to ensure the denominator is never smaller than the numerator. This Phase~2 amplification is a practical heuristic that improves refinement accuracy but inflates the computational cost beyond the nominal $k$ restarts; practitioners deploying AASV at fixed computational budgets should account for this overhead when setting $k$. Eight scenarios are evaluated:

\begin{enumerate}
    \item[\textbf{(a)}] \textbf{MC Baseline (3 Orthogonal Spikes):} 10{,}000 uniform random samples on $S^{127}$ detect zero violations, incorrectly certifying the space as safe.
    \item[\textbf{(b)}] \textbf{AASV Hunter (3 Orthogonal Spikes):} Same barrier as (a). With $k=20$ restarts, the Hunter detects all three spike regions via momentum PGD and Anti-Memory repulsion.
    \item[\textbf{(c)}] \textbf{Antipodal Pair ($v$ / $-v$):} Tests detection of diametrically opposed spikes (cosine similarity $= -1$). With $k=20$ restarts, both directions are discovered, confirming that the signed similarity blocking correctly treats $v$ and $-v$ as distinct failures.
    \item[\textbf{(d)}] \textbf{10 Orthogonal Spikes (Stress Test):} Maximum independent directions with $k=40$ restarts. Tests scalability of the prototype Anti-Memory to $O(n)$ concurrent failure modes.
    \item[\textbf{(e)}] \textbf{30$^\circ$ Angular Separation:} Two spikes separated by $30^\circ$ ($\cos = 0.866$). Post-hoc clustering at $\cos\theta = 0.98$ ($11.5^\circ$ resolution) successfully resolves both as distinct regions.
    \item[\textbf{(f)}] \textbf{15$^\circ$ Angular Cluster + Isolated Spike:} Two spikes at $15^\circ$ plus one orthogonally distant spike. All regions detected; post-violation refinement (100 extra PGD steps converging to $\cos > 0.9999$) correctly separates the two clustered spikes.
    \item[\textbf{(g)}] \textbf{5$^\circ$ Separation (Resolution Limit):} Two spikes at $5^\circ$ ($\cos = 0.9962$). Detected: 1 region (merged). This honestly demonstrates the angular resolution limit---spikes closer than $\arccos(0.98) \approx 11.5^\circ$ cannot be distinguished by post-hoc clustering. The system conservatively reports these as a single failure region rather than fabricating false precision.
    \item[\textbf{(h)}] \textbf{20 Random Spikes on $S^{127}$:} Spikes sampled uniformly at random on the sphere. With $k=60$ restarts: all 20 detected. Visualization uses PCA projection (first two principal components of the spike direction matrix) to faithfully represent the angular structure without ring artifacts.
\end{enumerate}

\textbf{Key Findings:}
\begin{itemize}
    \item \textbf{Standard MC:} \textit{Zero} violations detected across all 10{,}000 samples in panel (a). The passive verifier incorrectly certifies the state space as safe ($P_{\text{fail}} < \epsilon$ vacuously satisfied), confirming the ``Black Swan'' blindness predicted by Section~3.3.
    \item \textbf{AASV Detection Rate:} Across all eight configurations, the AASV pipeline achieved $\geq 95\%$ detection of all injected spikes, failing only at the honestly reported angular resolution limit ($< 11.5^\circ$).
    \item \textbf{Anti-Memory Efficacy:} The orthogonal prototype storage correctly distinguished novel discoveries from redundant revisitations, with redundant hits clearly separated from novel detections via post-hoc clustering.
    \item \textbf{Scalability:} The 20-spike stress test (panel h) confirms that detection scales to $O(n)$ independent failure modes without crosstalk, leveraging the quasi-orthogonality of random directions in $\mathbb{R}^{128}$ \cite{vershynin2018}.
\end{itemize}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_4_ab.png}
    \caption{\texorpdfstring{\textbf{AASV Robustness Evaluation (Panels a--b).}}{AASV Robustness Evaluation (Panels a-b).} (a)~Monte Carlo baseline: 10{,}000 uniform samples on $S^{127}$ detect zero violations, incorrectly certifying the space as safe. (b)~AASV Hunter with $k{=}20$ restarts detects all three orthogonal spikes via momentum PGD and Anti-Memory repulsion. Gray traces show Hunter trajectories; gold stars mark injected spikes; red crosses mark novel detections; green dots mark redundant rediscoveries.}
    \label{fig:black_swan_detection}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_4_cd.png}
    \caption{\texorpdfstring{\textbf{AASV Robustness Evaluation (Panels c--d).}}{AASV Robustness Evaluation (Panels c-d).} (c)~Antipodal pair ($\mathbf{v}$, $-\mathbf{v}$) with $k{=}20$ restarts: both directions discovered, confirming signed similarity blocking treats $v$ and $-v$ as distinct. (d)~10 orthogonal spikes with $k{=}40$ restarts: stress-tests Anti-Memory scalability to $O(n)$ concurrent failure modes.}
    \label{fig:aasv_cd}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_4_ef.png}
    \caption{\texorpdfstring{\textbf{AASV Robustness Evaluation (Panels e--f).}}{AASV Robustness Evaluation (Panels e-f).} (e)~Two spikes at $30^\circ$ separation ($\cos = 0.866$): post-hoc clustering at $11.5^\circ$ resolution successfully resolves both as distinct. (f)~Two spikes at $15^\circ$ plus one isolated orthogonal spike: post-violation refinement (100 extra PGD steps) correctly separates the clustered pair.}
    \label{fig:aasv_ef}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_4_gh.png}
    \caption{\texorpdfstring{\textbf{AASV Robustness Evaluation (Panels g--h).}}{AASV Robustness Evaluation (Panels g-h).} (g)~Two spikes at $5^\circ$ separation ($\cos = 0.996$): detected as 1 merged region, honestly demonstrating the $\approx 11.5^\circ$ angular resolution limit. (h)~20 random spikes on $S^{127}$ with $k{=}60$ restarts: all 20 detected, confirming scalability. PCA projection faithfully represents angular structure.}
    \label{fig:aasv_gh}
\end{figure}

This experiment empirically confirms the theoretical prediction of Section~3.3: passive Monte Carlo verification, while statistically valid in expectation, is operationally blind to rare catastrophic events. The AASV module closes this gap by actively hunting threats, yielding a verification regime that is both probabilistically sound and adversarially robust. Critically, the system is honest about its limitations: panel~(g) demonstrates that spikes closer than the clustering resolution cannot be distinguished, and this limitation is reported transparently rather than hidden.


% ============================================================================
% CUT EXPERIMENT: VI - WTA Gradient Decomposition vs. Black-Box Discovery
% Originally lines 831-853 of original_paper.md
% ============================================================================

\subsection{Experiment VI: WTA Gradient Decomposition vs.\ Black-Box Discovery}
The AASV Hunter in Experiment~IV uses a Winner-Take-All (WTA) gradient that selects the nearest unblocked spike direction per restart, exploiting structural knowledge of the barrier decomposition. A natural question is whether the Hunter can discover unknown failure modes using only the true barrier gradient $\nabla h(x)$---either computed analytically (oracle) or via finite differences (black-box).

\textbf{Setup:} We test three gradient modes on $S^{127}$ with $\theta_w = 0.05$ radians:
\begin{enumerate}
    \item \textbf{WTA gradient} (from Experiment~IV): targets one spike per restart with blocking.
    \item \textbf{Sum gradient (oracle):} the analytical $\nabla h(x) = \sum_i \nabla h_i(x)$, summing over all spike contributions.
    \item \textbf{Sum gradient (FD):} finite-difference approximation of $\nabla h$ with step $\epsilon = 10^{-4}$.
\end{enumerate}

\textbf{Single-spike control:} To isolate the gradient-decomposition effect from the multi-modal landscape problem, we first test with a single spike ($N = 1$). With one failure mode, no centroid saddle exists, and FD gradients detect the spike on every restart (1/1 at $k = 5$).

\textbf{Multi-spike test:} With three orthogonal spikes, the full gradient $\nabla h$ is the sum of three coplanar funnel components, creating a saddle point at the centroid direction $\hat{c} = (s_1 + s_2 + s_3)/\|s_1 + s_2 + s_3\|$. Standard PGD converges to this centroid---which is \textit{not} a spike and satisfies $h(\hat{c}) > 0$---rather than to any individual violation.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_10.png}
    \caption{\texorpdfstring{\textbf{Experiment VI: WTA vs.\ Black-Box Gradient ({$\mathbb{R}^{128}$}).}}{Experiment VI: WTA vs. Black-Box Gradient (R128).} (a)~Single failure mode: both FD and oracle gradients detect the spike at $k = 5$, since no centroid saddle exists. (b)~Three orthogonal spikes: the WTA gradient resolves all 3 at $k = 5$, while the sum gradient (oracle or FD) detects 0/3 at any $k$---the centroid saddle traps the optimizer.}
    \label{fig:wta_vs_fd}
\end{figure}

\textbf{Results:} The WTA gradient detects all 3/3 spikes from $k = 5$ restarts onward. Both sum-gradient variants (oracle and FD) detect \textbf{0/3} at all tested restart budgets ($k \leq 40$), confirming the centroid saddle hypothesis. This demonstrates that the WTA decomposition is not merely a heuristic but a \textit{necessary structural component}: the full gradient $\nabla h$ contains no information about which spike is nearest, leading to destructive interference between funnel gradients. In deployment, this motivates either (1)~barrier designs where the gradient naturally decomposes (e.g., per-constraint CBFs), or (2)~learned surrogate models that approximate the WTA gradient from data. For barriers with a single dominant failure mode---common in operational safety (e.g., a single collision boundary)---the FD gradient suffices without WTA.


% ============================================================================
% CUT EXPERIMENT: VII - Seed Sensitivity and Reproducibility
% Originally lines 854-867 of original_paper.md
% ============================================================================

\subsection{Experiment VII: Seed Sensitivity and Reproducibility}
To assess the statistical robustness of AASV detection, we run a seed sensitivity sweep: 20 orthogonal spikes on $S^{127}$ (the maximum independent directions tested in Experiment~IV-h), evaluated across 10 independent random seeds controlling both spike direction generation and Hunter initialization.

\textbf{Protocol:} For each seed $s \in \{0, \ldots, 9\}$, a fresh set of 20 orthogonal spike directions is generated, and the full AASV pipeline (WTA gradient, $k = 60$ restarts, $T = 200$ steps, prototype blocking, post-violation refinement, post-hoc clustering) is executed independently.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_11.png}
    \caption{\texorpdfstring{\textbf{Experiment VII: Seed Sensitivity Sweep (20 Spikes $\times$ 10 Seeds, $\mathbb{R}^{128}$).}}{Experiment VII: Seed Sensitivity Sweep (20 Spikes x 10 Seeds, R128).} (a)~Box plot of detection counts across 10 seeds: all seeds achieve 20/20 spike matches with zero variance. (b)~Per-seed bar chart: 100\% detection rate for all seeds. The AASV pipeline exhibits no sensitivity to random initialization.}
    \label{fig:seed_sweep}
\end{figure}

\textbf{Results:} Across all 10 seeds, the AASV pipeline detects \textbf{20/20 spikes with zero variance}: mean $= 20.0 \pm 0.0$, 100\% detection rate on every seed. Mean violation count is $100.2 \pm 0.4$, with 20.0 clusters matching the 20 ground-truth spikes exactly. Total wall-clock time is $2.2 \pm 0.1$\,s per seed (single-threaded CPU), confirming real-time feasibility. This result eliminates the concern that Experiment~IV's detection rates depend on favorable random initialization: the WTA gradient with orthogonal prototype blocking is a deterministic-convergence mechanism, not a lucky random search. The zero-variance result reflects the deterministic convergence of the WTA mechanism with sufficient restarts; at lower restart budgets ($k < 2 N_{\text{spikes}}$), variance increases as some spikes may be missed due to insufficient coverage of the search space.


% ============================================================================
% CUT EXPERIMENT: XIII - Union Bound Horizon Analysis
% Originally lines 944-959 of original_paper.md
% ============================================================================

\subsection{Experiment XIII: Union Bound Horizon Analysis}
\label{sec:exp_union_bound}

To empirically validate the trajectory-level safety bridge (Proposition~\ref{prop:trajectory_bridge}) and demonstrate the limitations of the union bound at long horizons, we conduct a systematic analysis of safety probability degradation and the re-certification mitigation strategy.

\textbf{Setup.} We model per-step failures as independent Bernoulli events with probability $\epsilon$, consistent with Proposition~\ref{prop:trajectory_bridge}'s union bound $P_{\text{safe}}(T) \geq 1 - T\epsilon$. For each trajectory length $T \in \{10, 50, \ldots, 50{,}000\}$, we run 500 Monte Carlo trials and compare the empirical safety rate against (1)~the union bound, (2)~the exact probability $(1-\epsilon)^T$ (valid under independence), and (3)~re-certification strategies that reset the union bound every $T_w$ steps.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_18.png}
    \caption{\texorpdfstring{\textbf{Experiment XIII: Union Bound Horizon Analysis.}}{Experiment XIII: Union Bound Horizon Analysis.} (a)~Union bound degradation for various $\epsilon$: the bound $P_{\text{safe}} \geq 1 - T\epsilon$ becomes vacuous at $T = 1/\epsilon$. (b)~Re-certification strategy: resetting the union bound every $T_w$ steps yields $P_{\text{safe}} = (1 - T_w\epsilon)^{T/T_w}$, significantly extending the useful horizon. (c)~Empirical validation: the exact calculation $(1-\epsilon)^T$ (blue) tightly matches Monte Carlo results (black dots), while the union bound (red dashed) is conservative but safe. The gap quantifies the conservatism of Proposition~\ref{prop:trajectory_bridge}.}
    \label{fig:union_bound_horizon}
\end{figure}

\textbf{Results:} The union bound is confirmed to be conservative but dimension-free: at $\epsilon = 10^{-4}$ and $T = 5{,}000$, the union bound gives $P_{\text{safe}} \geq 0.50$ while the empirical rate is $P_{\text{safe}} \approx 0.607$, matching the exact calculation $(1 - 10^{-4})^{5000} = 0.607$ to within sampling noise. The naive union bound becomes vacuous ($P_{\text{safe}} = 0$) at $T = 1/\epsilon = 10{,}000$. Re-certification mitigates this: resetting the bound every $T_w$ steps yields a non-vacuous guarantee beyond $T = 1/\epsilon$, with $P_{\text{safe}} \approx 0.37$ at $T = 10{,}000$ for $T_w = 100$ (compared to the naive bound's zero). However, the re-certification bound itself degrades exponentially as $(1 - T_w\epsilon)^{T/T_w}$, so long-horizon deployments ($T \gg 1/\epsilon$) require either very small $\epsilon$ or frequent re-certification with correspondingly small $T_w$. These results confirm that the AASV module's per-step adversarial verification complements the MCBC's trajectory-level guarantee: the union bound degrades at long horizons, but AASV provides a separate, horizon-independent safety layer.


% ============================================================================
% CUT EXPERIMENT: XIV - CHDBO on GPT-2 Hidden-State Dynamics
% Originally lines 960-1033 of original_paper.md
% ============================================================================

\subsection{Experiment XIV: CHDBO on GPT-2 Hidden-State Dynamics}
\label{sec:exp_gpt2}

All preceding experiments operate on synthetic state spaces with hand-specified or procedurally generated dynamics. While these isolate and validate individual components of the CHDBO framework (barrier projection, MCBC certification, AASV detection, drift compensation), they leave open the question most relevant to AI safety: \textit{can a CBF-QP safety filter operate on the internal representations of a production language model, using a barrier learned entirely from data?} Experiment~XIV answers this affirmatively. We treat the successive transformer layers of GPT-2 as a discrete-time dynamical system on $\mathbb{R}^{768}$, learn a linear barrier from a standard toxicity benchmark with a proper train/test split, and demonstrate that the CHDBO filter enforces forward invariance of the safe set with a 4.0\% false activation rate (10/250) on benign inputs---an honest characterization of barrier imperfection rather than an idealized zero-error claim.

\textbf{Dynamics as a Residual Dynamical System.}
A transformer with $L$ layers defines a residual iteration on its hidden-state vector $x_l \in \mathbb{R}^{768}$:
\begin{equation}
    x_{l+1} = x_l + \mathrm{Block}_l(x_l), \qquad l = 0, 1, \ldots, L-1,
    \label{eq:transformer_dynamics}
\end{equation}
where $\mathrm{Block}_l$ is the composition of multi-head self-attention and feed-forward sub-layers at layer $l$ (with layer normalization). This is precisely a control-affine system $x_{l+1} = x_l + f_l(x_l) + u_l$ with $u_l = 0$ in the unmodified model. The dynamics $f_l(x_l) \coloneqq \mathrm{Block}_l(x_l)$ are \textit{real, highly nonlinear, and learned}---not hand-crafted.

\textbf{Remark (Discrete-Time Formulation).}
Whereas the theoretical framework (Theorems~\ref{thm:topological_safety}--\ref{thm:safe_convergence}) is developed for continuous-time systems $\dot{x} = f(x) + g(x)u$, the transformer iteration~\eqref{eq:transformer_dynamics} is inherently discrete. We work directly with the discrete-time CBF constraint $h(x_{l+1}) \geq (1-\gamma)\,h(x_l) + \varepsilon_{\mathrm{buf}}$, which guarantees forward invariance of $\{x : h(x) \geq 0\}$ without invoking continuous interpolation. The continuous relaxation of Section~2 (Neural ODE interpretation, Gronwall error bounds) provides complementary justification by bounding the gap between the discrete layer map and an underlying continuous flow; however, it is not required for the discrete-time safety guarantee used here.

\textbf{Setup.}
We use GPT-2 (small, 12 layers, hidden dimension $n = 768$) and the Google Civil Comments dataset \cite{borkan2019}, a standard production-grade toxicity benchmark containing $\sim$1.8 million human-annotated comments. We stream the training split and collect the first 250 texts with toxicity score $\leq 0.1$ (``safe'') and the first 250 with toxicity score $\geq 0.7$ (``toxic''), filtering to 20--200 characters per text. For each text, we tokenize (truncated to 64 tokens) and extract the last-token hidden state $x_l \in \mathbb{R}^{768}$ at every layer $l \in \{0, 1, \ldots, 12\}$, yielding a trajectory matrix of shape $(13, 768)$ per text. The 500 texts are split 80/20 into 400 training and 100 held-out test samples (stratified by class) to evaluate generalization of the learned barrier; all SVM training, cross-validation, and barrier construction use \textit{only} the training set.

\textbf{Barrier Learning.}
We train a LinearSVC (C = 1.0) on the training-set hidden states at each layer independently, using 3-fold cross-validation to identify the layer yielding maximal safe/toxic separation. Layer 9 achieves the highest accuracy. A final barrier is then trained on layer-9 hidden states using 5-fold stratified cross-validation and standardized features, yielding:
\begin{equation}
    h(x) = w^\top x + b, \qquad w \in \mathbb{R}^{768}, \quad b \in \mathbb{R},
    \label{eq:svm_barrier}
\end{equation}
with $h(x) > 0$ for safe texts and $h(x) < 0$ for toxic texts. The barrier normal $w$ and intercept $b$ are transformed back from the standardized space to the original hidden-state coordinates, so that $h(x)$ can be evaluated directly on any GPT-2 hidden state without rescaling. The SVM margin in the original space is $2 / \|w\| = 3.749$, and the 5-fold cross-validation accuracy is $84.5\% \pm 3.8\%$---well above the 50\% random baseline and consistent with known linear probing results on transformer representations \cite{alain2017}. On the held-out test set ($n = 100$), the barrier achieves $76.0\%$ accuracy---a meaningful generalization gap that honestly reflects the difficulty of linear toxicity classification and the fact that the barrier is not a perfect oracle.

\textbf{Controllability Caveat.}
Hidden-state steering assumes that adding a perturbation $u^*$ to the residual stream at a single layer produces a meaningful and controllable semantic effect. Recent work on Representation Engineering \cite{zou2023} and Activation Addition \cite{turner2023} provides empirical evidence that linear directions in transformer hidden spaces correspond to interpretable concepts (e.g., honesty, toxicity, sentiment), supporting the control-affine approximation $x_{l+1} \approx x_l + f_l(x_l) + u$. However, the Jacobian $\partial x_L / \partial u_l$ of later layers with respect to an intervention at layer $l$ is a product of $L - l$ nonlinear layer Jacobians, and its spectral properties (rank, condition number) determine whether the intervention is amplified, attenuated, or distorted by subsequent processing. We do not formally verify controllability in this experiment; instead, we evaluate output quality via perplexity to confirm that the intervention preserves text coherence (see below).

\textbf{CBF-QP Intervention (Targeted).}
Rather than applying the CBF at every layer transition (which would intervene on safe texts whose intermediate representations may transiently cross the barrier hyperplane), we apply the CBF-QP \textit{only at the layer transition $8 \to 9$}---the single transition where the SVM barrier is defined. This targeted architecture is the correct one: the barrier $h(x)$ is trained on layer-9 representations, so the CBF should enforce safety precisely at the entry to that layer. The dynamics at this transition are $f_{8}(x_{8}) = x_{9} - x_{8}$, and the CBF-QP computes the minimum-norm intervention $u^*$ such that:
\begin{equation}
    w^\top (f_{8}(x_{8}) + u) \geq -\gamma \, h(x_{8}) + \varepsilon_{\mathrm{buf}},
    \label{eq:gpt2_cbf}
\end{equation}
where $\gamma = 1.0$ and $\varepsilon_{\mathrm{buf}} = 10^{-4}$ is a floating-point safety buffer. The closed-form solution from Equation~\ref{eq:cbfqp} applies: if the constraint is already satisfied, $u^* = 0$; otherwise, $u^* = \lambda w$ with $\lambda = (\varepsilon_{\mathrm{buf}} - \gamma h(x_{8}) - w^\top f_{8}) / \|w\|^2$.

\textbf{MCBC Verification.}
We verify the barrier on the SVM decision boundary (the hyperplane $h(x) = 0$) in $\mathbb{R}^{768}$ using $N = 10{,}000$ Monte Carlo samples. Each sample is generated by drawing a point from the empirical data distribution (matching the mean and standard deviation of the observed hidden states) and projecting it onto the hyperplane $\{x : w^\top x + b = 0\}$ via $x_{\mathrm{bnd}} = z - \frac{h(z)}{\|w\|^2} w$. At each boundary point, we estimate \textit{point-specific} dynamics using $K$-nearest-neighbor inverse-distance-weighted regression ($K = 10$) on the observed transformer layer residuals $f_l(x) = x_{l+1} - x_l$ from the 500 training texts. This yields dynamics vectors that genuinely vary across the boundary: the estimated $\|f(x_{\mathrm{bnd}})\|$ ranges from 26.0 to 34.8 (mean 30.1, $\sigma = 1.3$), confirming non-degenerate, spatially-varying dynamics estimation. We then evaluate CBF-QP feasibility under a bounded-actuation constraint: a boundary point \textit{fails} if the minimum-norm intervention $\|u^*\|$ required to satisfy the CBF condition exceeds $10\%$ of the mean dynamics magnitude---i.e., if the barrier cannot be maintained without excessively altering the layer's natural computation. The resulting margin distribution contains 9{,}873 unique values (out of 10{,}000 samples), confirming that the verification is genuinely point-specific rather than a degenerate single-point check.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_20.png}
    \caption{\texorpdfstring{\textbf{Experiment XIV: CHDBO on GPT-2 Hidden-State Dynamics ($\mathbb{R}^{768}$, Civil Comments).}}{Experiment XIV: CHDBO on GPT-2 Hidden-State Dynamics (R768, Civil Comments).} (a)~PCA projection of hidden states at layer~9: safe (blue) and toxic (red) clusters with SVM decision boundary (dashed). Green diamonds show CBF-steered toxic trajectories crossing into the safe region. (b)~Barrier value $h(x_l)$ across all 13 layers for 500 texts. Safe texts (blue) converge to positive $h$ while toxic texts (red) diverge to negative $h$, with separation emerging at deeper layers---consistent with the transformer refining semantic content through successive residual updates. (c)~MCBC verification with point-specific KNN dynamics ($K=10$): the left axis shows the pre-intervention margin distribution $w \cdot f(x_{\mathrm{bnd}})$; the right axis confirms intervention magnitude $\|u^*\|$. All 10{,}000 samples satisfy the budgeted CBF condition, yielding $P_{\mathrm{safe}} = 1.0$; the 9{,}873 unique margin values confirm non-degenerate dynamics estimation. (d)~Intervention magnitude $\|u^*\|$ by class: the CBF applies minimal intervention to safe texts (mean $\|u^*\| = 0.076$, intervention ratio $0.2\%$) and a larger correction to toxic texts (mean $\|u^*\| = 2.59$, intervention ratio $7.1\%$), confirming proportional safety response. (e)~Histogram of barrier values $h(x)$ at the target layer for both training and held-out test sets, showing bimodal separation. Train 5-fold CV accuracy: $84.5\% \pm 3.8\%$; held-out test accuracy: $76.0\%$. (f)~Layer-wise SVM 3-fold cross-validation accuracy: separation is near-random at the embedding layer ($l = 0$), emerges through the middle layers, and peaks at layer~9, confirming that toxicity is a learned high-level feature rather than a surface statistic. (g)~Output quality evaluation: perplexity ratio (steered / original) for toxic and safe texts. Median PPL ratio for toxic texts is $1.007$, confirming that CBF intervention preserves text coherence. Safe texts have PPL ratio $1.000$ (unmodified). (h)~Summary of key experimental results.}
    \label{fig:gpt2_experiment}
\end{figure}

\textbf{Results:}
The experiment confirms all five claims of the CHDBO framework in the setting of a real transformer and a production toxicity dataset:

\begin{enumerate}
    \item \textit{Safety (Forward Invariance):} After CBF-QP intervention at the layer-$8 \to 9$ transition, \textbf{0/250} safe texts and \textbf{0/250} toxic texts violate the barrier ($h(x) < 0$). All toxic trajectories for which the CBF activates are steered into the safe set.

    \item \textit{Utility Preservation:} The CBF fires on \textbf{10/250 safe texts} (4.0\%) and \textbf{236/250 toxic texts} (94.4\%). Mean intervention norm on safe texts is $\|u^*\| = 0.076$ (intervention ratio $0.2\%$): safe content passes through the transformer with negligible modification. On toxic texts, the mean intervention is $\|u^*\| = 2.59$, yielding an intervention-to-dynamics ratio of $7.1\%$---the safety correction remains small relative to the natural layer residual. The non-zero false activation rate (4.0\%) reflects the imperfect linear barrier ($76\%$ test accuracy); more accurate nonlinear barriers would reduce this.

    \item \textit{Barrier Quality:} The linear SVM achieves 5-fold cross-validated accuracy of $84.5\% \pm 3.8\%$ on the training set and \textbf{76.0\% on the held-out test set} ($n = 100$). The generalization gap (8.5 percentage points) is honestly reported and reflects the difficulty of linear toxicity classification. While not state-of-the-art for toxicity classification (which is not the goal), this accuracy is sufficient to define a meaningful safe/unsafe partition and demonstrates that the CHDBO framework operates correctly with imperfect, data-learned barriers.

    \item \textit{Output Quality (Perplexity):} To address the output coherence concern, we evaluate the perplexity of steered vs.\ original text using GPT-2 as its own language model, injecting the CBF intervention $u^*$ into the residual stream via a forward hook at the target layer. On 50 toxic texts, the median perplexity ratio (steered/original) is $\mathbf{1.007}$, with a maximum of $1.202$. On 50 safe texts, the median ratio is $\mathbf{1.000}$ (unmodified, as expected). A PPL ratio near 1.0 indicates that the CBF intervention preserves the statistical properties of the hidden-state trajectory through subsequent layers, producing text of comparable coherence to the original. This is consistent with the Representation Engineering hypothesis \cite{zou2023} that safety-relevant directions in hidden space are approximately orthogonal to fluency-relevant directions, so that a small perturbation along the SVM normal does not substantially degrade language modeling quality.

    \item \textit{MCBC Certification:} $P_{\mathrm{safe}} = 1.0$ across $N = 10{,}000$ boundary samples under a bounded-actuation constraint ($\|u^*\| \leq 10\%$ of mean dynamics magnitude). To avoid the degenerate single-drift-direction check common in na\"ive MCBC implementations, we estimate point-specific dynamics via $K$-nearest-neighbor regression ($K = 10$) on the observed layer residuals, producing 9{,}873 unique margin values. We note that the KNN surrogate introduces a model error $\epsilon_{\text{model}}^{\text{KNN}}$ that has not been formally bounded; the observed margin statistics (mean $-0.375$, std $0.111$) suggest that the dynamics estimation captures meaningful spatial variation, but a rigorous KNN approximation bound (e.g., via local Lipschitz arguments or leave-one-out cross-validation residuals) should be incorporated before deploying this approach in safety-critical settings. The control budget is $10\%$ of mean $\|f\| = 30.1$; since the maximum required $\|u^*\| = 1.38 < 3.68$ (budget), all samples satisfy the bounded-actuation constraint. The Hoeffding bound (Equation~\ref{eq:hoeffding}) requires $N_{\mathrm{Hoeffding}} = 72{,}544$ samples for $\epsilon = 0.01$, $\delta = 10^{-6}$; our $N = 10{,}000$ provides a weaker but still informative estimate. Crucially, the sample complexity is identical to that of the $\mathbb{R}^{128}$ experiments---\textit{it does not depend on $n = 768$}.

    \item \textit{Dimension-Independent Lipschitz Constant:} For the linear barrier, $L_h = \|\nabla h\| = \|w\| = 0.533$, which is constant (independent of $x$) and independent of the embedding dimension $n$. This is stronger than the theoretical requirement: the Lipschitz constant is not merely bounded but literally invariant, confirming that the MCBC sample complexity and the CBF-QP per-step cost remain $O(1)$ and $O(n)$ respectively regardless of the hidden-state dimension.
\end{enumerate}

\textbf{Significance and Scope.}
Experiment~XIV is a \textit{proof-of-concept} demonstrating, to our knowledge for the first time, that a CBF-QP safety filter can operate on the internal hidden-state dynamics of a production language model. It validates the central thesis of this paper: that the CHDBO framework's barrier certification, closed-form QP enforcement, and utility preservation extend from synthetic control systems to real, high-dimensional, nonlinear, learned dynamics. The low false-intervention rate on safe texts (10/250, 4.0\%) demonstrates that the targeted CBF architecture overwhelmingly preserves benign content. The perplexity evaluation (median PPL ratio 1.007 for toxic texts) confirms that hidden-state steering preserves output coherence, addressing a key concern about internal-representation manipulation. However, several caveats temper this result: (1)~the MCBC result ($P_{\mathrm{safe}} = 1.0$) reflects that GPT-2's natural KNN-estimated dynamics already satisfy the CBF condition at all 10{,}000 boundary samples---the filter was never actually \textit{tested} under adversarial stress at the boundary; (2)~the SVM barrier is a linear classifier on a single layer's representation, which may not capture complex toxicity patterns (e.g., context-dependent offensiveness), as reflected in the 76\% held-out accuracy; (3)~the KNN dynamics surrogate has not been rigorously bounded; (4)~the perplexity evaluation measures statistical coherence but not semantic appropriateness---the steered text may be fluent but semantically altered in unintended ways; and (5)~the controllability assumption (that $u^*$ at layer 9 propagates meaningfully to the output) relies on empirical evidence from Representation Engineering \cite{zou2023} rather than formal verification. Extending this to nonlinear barriers (e.g., neural CBFs trained on richer feature spaces), multi-layer intervention architectures, and human evaluation of output quality is a natural direction for future work.

\textbf{Empirical $L_f$ Estimation Recommendation.}
A practical gap in the current experiment is the absence of an empirical estimate of the dynamics Lipschitz constant $L_f = \text{Lip}(\text{Block}_{8})$. While the controllability margin is satisfied empirically, rigorous deployment requires bounding $L_f$ to validate the continuous relaxation assumption (A7) and compute the discretization error $\epsilon_{\text{model}} \leq \frac{1}{2} L_{\nabla f} r^2$. We recommend two approaches for future work: (1)~\textit{spectral norm upper bound} via power iteration on $J_{\text{Block}_{8}}$ at representative hidden states, which provides a local $L_f$ estimate at $O(kn) = O(n)$ cost; or (2)~\textit{empirical Lipschitz estimation} by computing $\max_{i \neq j} \|f(x_i) - f(x_j)\| / \|x_i - x_j\|$ over observed layer-8 hidden states. Either approach would close the gap between Assumption~A7 and empirical validation for transformer dynamics.

\textbf{Adversarial Stress Testing.}
As noted in caveat~(1), the current experiment does not stress-test the CBF filter at the boundary because the KNN-estimated dynamics are already compatible with the barrier. Future work should incorporate adversarial dynamics perturbations---e.g., prompt injection attacks \cite{wei2023} or fine-tuned toxic model variants---to evaluate the filter under conditions where it is actively required to intervene at boundary points. This would provide a meaningful test of the AASV module's adversarial detection capability in a real neural architecture.


% ============================================================================
% CUT EXPERIMENT: XV - Learned Barrier for AASV (No Oracle Access)
% Originally lines 1034-1049 of original_paper.md
% ============================================================================

\subsection{Experiment XV: Learned Barrier for AASV (No Oracle Access)}
\label{sec:exp_learned_barrier}

A key limitation of the AASV experiments (IV, VI, VII) is that the Hunter uses oracle access to the barrier structure---the spike directions $\{s_i\}$ are known, enabling WTA gradient decomposition. This raises the question: \textit{can the AASV Hunter detect violations using only a learned barrier gradient $\nabla h_{\text{NN}}$ from a neural network, with no structural knowledge of the failure geometry?}

\textbf{Setup.} We construct a barrier in $\mathbb{R}^{20}$ with three injected Gaussian spikes (width $\sigma = 0.3$) on the unit sphere, then train a neural network (BarrierNet: two hidden layers of 128 units with GELU activations) to approximate $h(x)$ from 50{,}000 uniformly sampled training points. The AASV Hunter uses \textit{only} $\nabla_x h_{\text{NN}}$ via PyTorch \texttt{torch.autograd.grad}---no access to the spike centers, no WTA decomposition. As a control, we also run a Monte Carlo baseline (50{,}000 uniform random samples).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_10b.png}
    \caption{\texorpdfstring{\textbf{Experiment XV: Learned Barrier AASV ($\mathbb{R}^{20}$, No Oracle Access).}}{Experiment XV: Learned Barrier AASV (R20, No Oracle Access).} (a)~Neural barrier approximation: the BarrierNet achieves MAE $= 0.0024$ on a held-out test set, with the scatter plot showing tight correlation between true and predicted $h(x)$. (b)~AASV Hunter using only $\nabla h_{\text{NN}}$: all 3/3 spike regions detected (cosine similarity $> 0.97$ with true spike centers), confirming that the learned gradient provides sufficient directional information. (c)~Monte Carlo control: 50{,}000 uniform samples find raw violations but fail to cluster them into distinct spike regions, demonstrating the AASV Hunter's advantage in structured failure detection.}
    \label{fig:learned_barrier}
\end{figure}

\textbf{Results.} The BarrierNet achieves MAE $= 0.0024$ (violation recall $98.4\%$). The AASV Hunter, using only $\nabla h_{\text{NN}}$ from \texttt{torch.autograd}, detects \textbf{all 3/3 spike regions} with cosine similarity $> 0.97$ to the true spike centers. The MC baseline finds 125 raw violation points out of 50{,}000 samples but identifies 0 distinct spike regions via post-hoc clustering (the violations are scattered rather than concentrated). This experiment demonstrates that: (1)~a moderate neural network can approximate a multi-modal barrier well enough for gradient-based search, (2)~the AASV Hunter's momentum PGD converges to true failure modes using learned gradients alone, without WTA decomposition, and (3)~the advantage of active search over passive sampling persists even with approximate gradients. The lower dimensionality ($n = 20$ vs.\ $n = 128$) is necessary because neural network barrier approximation degrades in high dimensions with finite training data; we view this as a proof-of-concept for the learned-gradient AASV pipeline rather than a high-dimensional scaling claim.


% ============================================================================
% CUT LIMITATION: WTA Gradient Oracle Requirement
% Originally in Limitations section of original_paper.md
% ============================================================================

\subsubsection{WTA Gradient Oracle Requirement}
\label{sec:wta_limitation}
The most significant practical limitation of the AASV module is the Winner-Take-All (WTA) gradient's reliance on structural knowledge of the barrier decomposition. Specifically, the WTA mechanism requires access to individual spike contributions $h_i(x)$ to select the nearest unblocked failure direction per restart. For a general learned barrier $h(x) = \text{NeuralNet}(x)$, this decomposition is unavailable.

Experiment~VI demonstrates that this is not merely a convenience issue but a \textit{fundamental requirement}: the full gradient $\nabla h(x)$ (whether oracle or finite-difference) converges to a centroid saddle rather than individual spikes in multi-modal landscapes (0/3 detection vs.\ 3/3 with WTA). However, Experiment~XV partially mitigates this concern by demonstrating that a neural network can learn a barrier well enough for the AASV Hunter to detect all violations using only $\nabla h_{\text{NN}}$ via \texttt{torch.autograd}---without any WTA decomposition or structural oracle. This works because the learned barrier's gradient implicitly captures the local geometry near failure modes, even though it does not decompose into per-spike components. Three practical mitigation strategies exist: (1)~barrier designs where the gradient naturally decomposes (e.g., per-constraint CBFs, where each constraint defines a separate $h_i$); (2)~learned surrogate models trained to approximate the barrier from data (demonstrated in Experiment~XV); and (3)~for single-mode barriers---typical of operational safety constraints such as collision boundaries---finite-difference gradients suffice without WTA (Experiment~VI, single-spike control).


% ============================================================================
% CUT LIMITATION: Code vs. Theory Scaling
% Originally in Limitations section of original_paper.md
% ============================================================================

\subsubsection{Code vs.\ Theory Scaling}
The primary figure-generation code computes Hutchinson estimates and barrier gradients using dense matrix operations ($O(n^2)$). However, the accompanying \texttt{generate\_figure\_5.py} and \texttt{generate\_figure\_7.py} now include a full JAX-based AD JVP implementation that empirically validates the $O(n)$ claim: at $n = 1024$, the AD Hutchinson pathway (0.023s) overtakes the dense implementation (0.034s), and at $n = 2048$ the AD speedup reaches $\sim$2.5$\times$. The AD estimates match the dense-matrix results to within $0.000003\%$, confirming numerical equivalence. Practitioners implementing AASV at production scale ($n > 10{,}000$) should use a true AD-based pipeline (e.g., JAX or PyTorch) to realize the claimed linear scaling.


% ============================================================================
% CUT LIMITATION: Autoregressive Deployment
% Originally in Limitations section of original_paper.md
% ============================================================================

\subsubsection{Autoregressive Deployment}
Experiment~XIV demonstrates single-pass intervention (one prompt $\to$ one forward pass $\to$ one CBF correction at layer~8$\to$9). In autoregressive generation, each new token triggers a full forward pass, and the CBF would fire at the layer-$8 \to 9$ transition of \textit{every} decoding step. Two issues arise: (1)~the cumulative effect of repeated small perturbations $u^*_t$ across $T$ tokens has not been analyzed---the union-bound trajectory extension (Proposition~\ref{prop:trajectory_bridge}) applies, but the resulting $T \cdot P_{\text{step}}$ may become non-negligible for long generations; and (2)~each intervention $u^*_t$ alters the KV-cache for subsequent tokens, creating a feedback loop between the safety filter and the model's autoregressive dynamics that is not captured by the single-step analysis. Extending the CHDBO framework to multi-step autoregressive steering with formal guarantees on output coherence is an important direction for future work.


% ============================================================================
% CUT LIMITATION: Barrier Design for Semantic Agents
% Originally in Limitations section of original_paper.md
% ============================================================================

\subsubsection{Barrier Design for Semantic Agents}
Defining the barrier function $h(x)$ for semantic agents---determining which scalar function on an embedding space demarcates ``safe'' from ``unsafe'' semantics---remains an open and fundamental challenge in AI safety that this paper begins to address but does not fully solve. Experiment~XIV demonstrates one concrete instantiation: a linear SVM barrier trained on GPT-2 hidden states achieves 84.5\% cross-validated accuracy (76.0\% on a held-out test set) on a production toxicity benchmark, and the CBF-QP filter enforces forward invariance with a 4.0\% false activation rate on safe content (10/250 prompts). However, this linear barrier captures only the linearly separable component of the safe/unsafe distinction; more nuanced safety boundaries (e.g., context-dependent toxicity, subtle manipulation, or dual-use content) will require nonlinear barriers such as neural CBFs \cite{dawson2023, robey2020}. In principle, a neural CBF $h_\theta(x) = \text{NeuralNet}_\theta(x)$ is a drop-in replacement for the linear SVM: CHDBO's MCBC verification, CBF-QP enforcement, and Hutchinson spectral margin are all agnostic to the functional form of $h$, requiring only pointwise evaluation and a gradient $\nabla_x h$. The Lipschitz constant $L_h$ would then be estimated via the network's spectral properties \cite{virmaux2018} rather than read directly from a weight vector. The ``safe backup'' strategy for semantic agents (reversion to a verified anchor embedding) may discard recent conversation context, representing a form of utility loss that practitioners must weigh against the safety guarantee.


% ============================================================================
% CUT DEPLOYMENT GUIDE: AASV Parameters
% Originally in Appendix of original_paper.md
% ============================================================================

\subsection{AASV Parameters}

\begin{itemize}
    \item \textbf{Restart count $k$:} Use $k \geq \lceil \ln(1/\delta_{\text{adv}}) / \ln(1/(1-p_{\text{hit}})) \rceil$ where $\delta_{\text{adv}}$ is the desired adversarial miss probability (Theorem~\ref{thm:aasv_detection}). For exploratory surveys, $k = 20$--$60$ suffices (Experiment~IV); for high-assurance applications, $k \geq 100$.
    \item \textbf{PGD step size $\eta$:} Set $\eta = 0.05 / L_h$ for initial exploration; decay by $0.99\times$ per step for convergence to sharp features. Momentum coefficient $\beta = 0.9$ is robust across all tested configurations.
    \item \textbf{Detection threshold $\tau_h$:} Calibrate to the barrier resolution; $\tau_h = -0.3$ captures significant violations without triggering on numerical noise. For high-precision applications, tighten to $\tau_h = -0.1$.
    \item \textbf{Orthogonal prototype threshold $\theta_{\text{sim}}$:} Use $\theta_{\text{sim}} = 0.3$ (signed cosine similarity) for adequate angular resolution; tighten to $0.15$ if angular separation between failure modes is small ($< 15^\circ$).
\end{itemize}


% ============================================================================
% CUT CONCLUSION CLAIMS (to be integrated into Paper B conclusion):
% ============================================================================

% Claim 4 (Black Swans are Huntable):
% \item \textbf{Black Swans are Huntable (with Structural Gradient Access):} The AASV module...
% [Full text preserved in the conclusion section of original_paper.md]

% Claim 6 (Framework Extends to Real Language Models):
% \item \textbf{The Framework Extends to Real Language Models:} Experiment~XIV demonstrates...
% [Full text preserved in the conclusion section of original_paper.md]


% ============================================================================
% CUT RESULTS TABLE ROWS (Experiments IV, VI, VII, XIII, XIV, XV):
% ============================================================================

% IV (AASV, 8 configs) & Black Swan Det. & 20/20~($k{=}60$) & $\dot{x}=u$ & Mom.\ PGD + WTA \\
%  & Angular Resol. & $\geq 11.5^\circ$ & & Post-hoc clustering \\
% VI (WTA vs.\ FD) & WTA Detection & 3/3~($k{=}5$) & $\dot{x}=u$ & WTA gradient + blocking \\
%  & FD Detection (1 spike) & 1/1~($k{=}5$) & & FD $\nabla h$ \\
%  & Sum $\nabla h$ (3 spikes) & 0/3~($k{=}40$) & & Centroid saddle \\
% VII (Seed sweep) & Reproducibility & 20/20 $\pm$ 0 & $\dot{x}=u$ & 10 seeds, $k{=}60$ \\
% XIII (Union Bound) & P\textsubscript{safe} at $T{=}5$k & 0.607 (empirical) & i.i.d. & Monte Carlo \\
%  & Re-cert ($T_w{=}100$) & $\approx 0.37$ at $T{=}10$k & & Union bound reset \\
% XIV (GPT-2, $n{=}768$) & Safety Rate & 0/250 violations & GPT-2 layers & CBF-QP (targeted) \\
%  & SVM 5-fold CV & 84.5\% $\pm$ 3.8\% & & Linear SVM barrier \\
%  & Held-out test acc. & 76.0\% & & $n{=}100$ test set \\
%  & CBF on safe & 10/250 (4.0\%) & & Low false activation \\
%  & CBF on toxic & 236/250 (94.4\%) & & Steered to safe \\
%  & Toxic PPL ratio & 1.007 (median) & & Output coherence \\
%  & MCBC $P_{\text{safe}}$ & 1.0 ($N{=}10$k, KNN) & & Bounded-actuation \\
% XV (Learned Barrier) & NN MAE & 0.0024 & $\dot{x}=u$ & Learned $\nabla h_{\text{NN}}$ \\
%  & Hunter Detection & 3/3 spikes & & No WTA/oracle \\
%  & MC Baseline & 0 regions & & 50k random samples \\

% CHDBO + AASV row from comparison table:
% \textbf{CHDBO + AASV} & $O(kTn)$ per step & Adversarially bounded & Yes & + WTA gradient access \\
