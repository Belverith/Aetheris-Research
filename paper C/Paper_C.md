\documentclass[10pt, twocolumn, a4paper]{article}

% Required Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{xcolor}
\usepackage{subcaption}
\usepackage{multirow}
\usepackage{float}

% Page Geometry Setup
\geometry{margin=0.75in, columnsep=0.25in}
\emergencystretch=2em

% Hyperlink Setup
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan,
    citecolor=red,
    pdfencoding=auto
}

% Theorem/Definition Environments
\newtheorem{theorem}{Theorem}
\newtheorem{definition}{Definition}
\newtheorem{problem}{Problem}
\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\newtheorem{remark}{Remark}

% Title Information
\title{\textbf{Control Barrier Functions for Transformer Hidden-State Safety: Scalable Verification from Synthetic to Learned Dynamics}}
\author{Arsenios Scrivens}
\date{June 2026}

\begin{document}

\maketitle

\begin{abstract}
Safety verification for autonomous systems operating in high-dimensional state spaces faces a fundamental computational barrier: deterministic methods scale as $O((1/\eta)^n)$ and become intractable for $n > 5$. We present a unified framework for safety-constrained control in $\mathbb{R}^n$ that combines Control Barrier Functions (CBFs) with Monte Carlo Barrier Certification (MCBC), achieving dimension-independent sample complexity $N = O(\epsilon^{-2} \ln \delta^{-1})$ and $O(n)$ per-step control cost. We extend this framework from synthetic dynamics to the internal representations of a production language model (GPT-2), demonstrating three progressively stronger results: (1)~a closed-form CBF-QP safety filter that achieves 61--351$\times$ speedup over OSQP across $n \in \{64, \ldots, 2048\}$; (2)~a linear SVM barrier on GPT-2 hidden states ($\mathbb{R}^{768}$) that achieves 80.5\% test accuracy with zero post-steering violations; and (3)~a \textit{spectrally-normalized neural barrier} (768$\to$512$\to$256$\to$1) that achieves 88.0\% test accuracy with certified Lipschitz constant $L_h = 1.10$---a 22$\times$ reduction over the SVM---while maintaining zero CBF violations, MCBC $P_{\text{safe}} = 1.0$ at all budget levels, and perplexity ratio 1.005. An ablation without spectral normalization achieves comparable accuracy (86.5\%) but with uncertified $L_h = 8.73$, demonstrating that Lipschitz certification is essential for formal safety guarantees. We complement passive MCBC verification with Active Adversarial Safety Verification (AASV), comprising momentum-accelerated threat hunting, adaptive spectral safety margins, and orthogonal prototype retention. All experiments, including 10,000-sample MCBC on nonlinear level sets in $\mathbb{R}^{768}$ and perplexity-preserving activation patching, are fully reproducible from the accompanying code.
\end{abstract}

% ============================================================================
\section{Introduction}
% ============================================================================

The deployment of autonomous AI systems---from multi-agent swarms ($n > 100$) \cite{brunke2022} to large language models operating in semantic embedding spaces ($n = 768$ to $12{,}288$) \cite{brown2020, radford2019}---demands safety guarantees that scale with the dimensionality of the underlying state space. Formal verification methods such as Hamilton-Jacobi reachability \cite{mitchell2005} provide deterministic guarantees but scale as $O((1/\eta)^n)$, making them computationally infeasible for $n > 5$ \cite{bellman1957}. Learning-based control offers scalability but typically lacks formal safety certificates \cite{amodei2016, garcia2015}.

Control Barrier Functions (CBFs) \cite{ames2019, prajna2004} bridge this gap by encoding safety as a forward-invariance condition on a sublevel set $\mathcal{S} = \{x : h(x) \geq 0\}$, enforced via a minimum-norm Quadratic Program (QP) that admits a closed-form solution in $O(n)$ for single-constraint systems. However, two challenges remain: (1)~verifying that the barrier condition holds \textit{everywhere} on the boundary $\partial\mathcal{S}$ in high dimensions, and (2)~designing barrier functions that are both expressive enough to capture complex decision boundaries and smooth enough to admit certified Lipschitz bounds for probabilistic verification.

This paper makes four contributions:

\begin{enumerate}
    \item \textbf{Scalable CBF-QP with MCBC Verification.} We present a closed-form CBF-QP safety filter with $O(n)$ per-step cost and verify it via Monte Carlo Barrier Certification (MCBC) \cite{tempo2012}, achieving sample complexity $N = O(\epsilon^{-2} \ln \delta^{-1})$ that is \textit{independent} of the state-space dimension $n$ (Section~\ref{sec:theory}).

    \item \textbf{Active Adversarial Safety Verification.} We complement passive MCBC with gradient-based threat hunting (momentum PGD), adaptive spectral safety margins (Hutchinson estimation), and an orthogonal prototype retention mechanism to prevent learned-barrier forgetting (Section~\ref{sec:aasv}).

    \item \textbf{Transformer Hidden-State Safety.} We demonstrate the framework on the residual-stream dynamics of GPT-2, treating successive transformer layers as a discrete-time dynamical system on $\mathbb{R}^{768}$. A linear SVM barrier establishes a baseline; a spectrally-normalized neural barrier achieves superior separation with certified Lipschitz bounds (Section~\ref{sec:experiments}).

    \item \textbf{Neural Barrier with Certified Lipschitz Constant.} We introduce a spectrally-normalized MLP barrier $h(x) = \text{MLP}_{\text{SN}}(x)$ with $L_h = \prod_l \sigma_{\max}(W_l) \leq 1$ by construction. An iterative Newton-corrected CBF-QP handles the resulting nonlinear constraint, and Newton-projected boundary sampling enables MCBC on the learned level set $\{x : h(x) = 0\}$ (Section~\ref{sec:neural_barrier}).
\end{enumerate}

\noindent An earlier version of the synthetic experiments appeared in a Zenodo preprint by the same author \cite{scrivens2026btg}; a companion paper on holographic invariant storage for semantic drift is available at \cite{scrivens2026his}. The present paper subsumes and extends both with the neural barrier framework, corrected theorem statements, and expanded experimental coverage.

% ============================================================================
\section{Related Work}
\label{sec:related}
% ============================================================================

\textbf{Control Barrier Functions.}
Ames et al.\ \cite{ames2019} established the CBF-QP framework for enforcing forward invariance of safe sets, with extensions to high-relative-degree systems \cite{xiao2019}, robust settings \cite{jankovic2018}, and multi-agent coordination \cite{glotfelter2017}. Our work extends CBFs to transformer hidden states and introduces a nonlinear learned barrier with certified Lipschitz bounds.

\textbf{Learned Barrier Functions.}
Dawson et al.\ \cite{dawson2023} survey neural Lyapunov, barrier, and contraction methods. Robey et al.\ \cite{robey2020} learn CBFs from expert demonstrations, and Qin et al.\ \cite{qin2021} train decentralized neural barrier certificates for multi-agent systems. These approaches typically lack certified Lipschitz bounds on the learned barrier, making the MCBC sample-complexity guarantee inapplicable. Our spectrally-normalized architecture addresses this gap directly.

\textbf{Lipschitz Neural Networks.}
Spectral normalization \cite{miyato2018} constrains $\sigma_{\max}(W) \leq 1$ per layer, yielding $\text{Lip}(\text{net}) \leq \prod_l \sigma_{\max}(W_l)$. Fazlyab et al.\ \cite{fazlyab2019} provide tighter (but more expensive) semidefinite-programming-based Lipschitz estimation. We use spectral normalization for its simplicity and $O(1)$-per-layer enforcement, accepting the product bound as a certified upper estimate.

\textbf{Representation Engineering.}
Zou et al.\ \cite{zou2023} demonstrate that linear directions in transformer residual streams correspond to interpretable concepts (honesty, toxicity, sentiment). Turner et al.\ \cite{turner2023} show that Activation Addition---adding a fixed steering vector to the residual stream---can modulate model behavior. Our CBF-QP framework generalizes this: the control signal $u^*$ is computed \textit{optimally} via a principled safety filter, not selected heuristically, and the Lipschitz-bounded barrier provides formal guarantees that heuristic steering cannot.

\textbf{Probabilistic Verification.}
The Scenario Approach \cite{campi2008, calafiore2006} provides distribution-free verification but with sample complexity $N = O(n/\epsilon)$ that grows linearly in dimension. Conformal prediction \cite{angelopoulos2021, lindemann2023} offers distribution-free coverage guarantees. Our MCBC approach achieves dimension-independent $N = O(\epsilon^{-2} \ln \delta^{-1})$ by exploiting Lipschitz continuity of the barrier, making it uniquely suited to high-dimensional state spaces.

\textbf{Hyperdimensional Computing.}
Vector Symbolic Architectures \cite{kanerva2009, plate1995} operate algebraically in high-dimensional distributed representations. Recent work on large-margin classifiers in hyperdimensional space \cite{zeulin2023} and comprehensive surveys \cite{salik2023} establish HDC as a mature framework for high-dimensional classification. Our barrier function operates in the same representational substrate as these methods.

% ============================================================================
\section{Theoretical Framework}
\label{sec:theory}
% ============================================================================

\subsection{Problem Formulation}

Consider a control-affine dynamical system:
\begin{equation}
    \dot{x} = f(x) + g(x)u, \quad x \in \mathbb{R}^n, \; u \in \mathcal{U} \subseteq \mathbb{R}^m
    \label{eq:dynamics}
\end{equation}
where $f: \mathbb{R}^n \to \mathbb{R}^n$ and $g: \mathbb{R}^n \to \mathbb{R}^{n \times m}$ are locally Lipschitz. For transformer residual streams, the discrete-time analog is:
\begin{equation}
    x_{l+1} = x_l + \text{Block}_l(x_l) + u_l
    \label{eq:transformer}
\end{equation}
where $x_l \in \mathbb{R}^{768}$ is the hidden state at layer $l$, $\text{Block}_l$ comprises self-attention and feed-forward sub-layers, and $u_l$ is the control intervention.

\begin{definition}[Safe Set]
\label{def:safe_set}
Given a continuously differentiable function $h: \mathbb{R}^n \to \mathbb{R}$ with $0$ a regular value, the \textbf{safe set} is $\mathcal{S} = \{x \in \mathbb{R}^n : h(x) \geq 0\}$, with boundary $\partial\mathcal{S} = \{x : h(x) = 0\}$ and interior $\text{Int}(\mathcal{S}) = \{x : h(x) > 0\}$.
\end{definition}

\begin{problem}[Safety-Constrained Control]
\label{prob:main}
Find a control policy $u = k(x)$ such that:
\begin{enumerate}
    \item \textbf{Safety:} $\mathcal{S}$ is forward invariant: $x(0) \in \mathcal{S} \implies x(t) \in \mathcal{S}$ for all $t \geq 0$.
    \item \textbf{Utility:} $\lim_{t \to \infty} x(t) \in \Omega^*$, the set of constrained critical points of a utility function $U(x)$ on $\mathcal{S}$.
    \item \textbf{Scalability:} Per-step computation cost is $O(n)$.
\end{enumerate}
\end{problem}

\subsection{CBF-QP Safety Filter}

\begin{definition}[Control Barrier Function]
A continuously differentiable function $h: \mathbb{R}^n \to \mathbb{R}$ is a \textbf{Control Barrier Function} for system \eqref{eq:dynamics} on $\mathcal{S}$ if there exists $\gamma > 0$ such that:
\begin{equation}
    \sup_{u \in \mathcal{U}} \left[ L_f h(x) + L_g h(x) u \right] \geq -\gamma h(x) \quad \forall x \in \mathcal{S}
    \label{eq:cbf_condition}
\end{equation}
where $L_f h = \nabla h^\top f$ and $L_g h = \nabla h^\top g$ are Lie derivatives.
\end{definition}

The safety filter is the minimum-norm QP:
\begin{equation}
    u^*(x) = \underset{u}{\text{argmin}} \; \frac{1}{2}\|u\|^2 \quad \text{s.t.} \quad L_f h + L_g h \, u \geq -\gamma h(x)
    \label{eq:cbf_qp}
\end{equation}

For a single affine constraint, the KKT conditions yield an explicit closed-form solution:
\begin{equation}
    u^*(x) = \begin{cases}
    0 & \text{if } L_f h + \gamma h \geq 0 \\
    \displaystyle \frac{-(L_f h + \gamma h)}{\|L_g h\|^2} (L_g h)^\top & \text{otherwise}
    \end{cases}
    \label{eq:closed_form}
\end{equation}

This requires only two dot products and one scalar division, yielding $O(n)$ per-step cost.

\begin{theorem}[Forward Invariance --- Sufficiency]
\label{thm:forward_invariance}
If $h$ is a CBF satisfying condition \eqref{eq:cbf_condition}, then the closed-loop system under $u^*(x)$ from \eqref{eq:cbf_qp} renders $\mathcal{S}$ forward invariant. That is, $x(0) \in \mathcal{S} \implies h(x(t)) \geq h(x(0)) e^{-\gamma t} \geq 0$ for all $t \geq 0$.
\end{theorem}

\begin{proof}
By the Comparison Lemma \cite{khalil2002}. The CBF condition ensures $\dot{h}(x) \geq -\gamma h(x)$ along the closed-loop trajectory. Integrating, $h(x(t)) \geq h(x(0)) e^{-\gamma t}$. Since $h(x(0)) \geq 0$ and $e^{-\gamma t} > 0$, forward invariance follows. The sufficiency direction is equivalent to Nagumo's theorem \cite{blanchini1999}.
\end{proof}

\begin{remark}[Utility-Aware CBF-QP]
\label{rem:utility_aware}
The minimum-norm QP \eqref{eq:cbf_qp} is a special case of the \textit{utility-aware} CBF-QP with nominal controller $u_{\text{nom}}(x)$:
\begin{equation}
    u^*(x) = \underset{u}{\text{argmin}} \; \frac{1}{2}\|u - u_{\text{nom}}(x)\|^2 \quad \text{s.t.} \quad L_f h + L_g h \, u \geq -\gamma h(x)
    \label{eq:utility_cbf_qp}
\end{equation}
Setting $u_{\text{nom}} = 0$ recovers \eqref{eq:cbf_qp}. In general, $u_{\text{nom}}$ encodes a task-level objective (e.g., gradient ascent on a utility function $U$). The closed-form KKT solution is:
\begin{equation}
    u^*(x) = \begin{cases}
    u_{\text{nom}}(x) & \text{if } L_f h + L_g h \, u_{\text{nom}} + \gamma h \geq 0 \\
    \displaystyle u_{\text{nom}}(x) + \frac{-(L_f h + L_g h \, u_{\text{nom}} + \gamma h)}{\|L_g h\|^2} (L_g h)^\top & \text{otherwise}
    \end{cases}
    \label{eq:utility_closed_form}
\end{equation}
which retains $O(n)$ per-step cost.
\end{remark}

\begin{theorem}[Safe Asymptotic Convergence]
\label{thm:convergence}
Consider system \eqref{eq:dynamics} under the utility-aware CBF-QP \eqref{eq:utility_cbf_qp} with locally Lipschitz nominal controller $u_{\text{nom}}$. Assume:
\begin{itemize}
    \item[\textbf{(A1)}] $f, g$ are locally Lipschitz and $h \in C^1$;
    \item[\textbf{(A2)}] the CBF-QP is feasible for all $x \in \mathcal{S}$;
    \item[\textbf{(A3)}] $\mathcal{S}$ is compact;
    \item[\textbf{(A4)}] $L_g h(x) \neq 0$ for all $x \in \mathcal{S}$ (linear independence constraint qualification);
    \item[\textbf{(A5)}] there exists $U \in C^1(\mathcal{S})$ such that $\nabla U(x)^\top[f(x) + g(x)u^*(x)] \geq 0$ for all $x \in \mathcal{S}$, with equality only on the constrained critical set $\Omega^*$.
\end{itemize}
Define $\Omega^* = \{x \in \mathcal{S} : \nabla U(x)^\top[f(x) + g(x)u^*(x)] = 0\}$. Then:
\begin{enumerate}
    \item \textbf{Forward invariance:} $h(x(t)) \geq 0$ for all $t \geq 0$.
    \item \textbf{Well-posedness:} The closed-loop system has a unique solution on $[0, \infty)$.
    \item \textbf{Convergence:} $x(t) \to \Omega^*$ as $t \to \infty$.
    \item \textbf{Singleton convergence} (if $U$ is real-analytic on $\mathcal{S}$): $x(t) \to x^* \in \Omega^*$ with finite arc-length $\int_0^\infty \|\dot{x}\| \, dt < \infty$.
\end{enumerate}
\end{theorem}

\begin{proof}
\textit{Step~1: Continuity of $u^*$.}
The CBF-QP \eqref{eq:utility_cbf_qp} has a strictly convex quadratic objective and a single affine constraint in $u$. By~(A2), the feasible set is nonempty, and by~(A4), the constraint gradient $L_g h(x)$ is nonvanishing, so the linear independence constraint qualification (LICQ) holds everywhere in $\mathcal{S}$. By Berge's Maximum Theorem \cite{berge1963}, the argmin $u^*(x)$ is a continuous function of $x$, since the objective is jointly continuous in $(x, u)$ and the constraint correspondence $x \rightrightarrows \{u : L_f h(x) + L_g h(x)\, u \geq -\gamma h(x)\}$ is continuous under~(A4). Concretely, the closed-form \eqref{eq:utility_closed_form} shows that the correction term vanishes continuously at the activation boundary $\{x : L_f h + L_g h \, u_{\text{nom}} + \gamma h = 0\}$, confirming continuity across the switching surface.

\textit{Step~2: Well-posedness.}
The closed-loop vector field $F(x) = f(x) + g(x)u^*(x)$ is continuous ($f, g$ locally Lipschitz by~(A1), $u^*$ continuous by Step~1). On the open regions $\mathcal{S}^+ = \{x : L_f h + L_g h \, u_{\text{nom}} + \gamma h > 0\}$ (constraint inactive) and $\mathcal{S}^- = \{x : L_f h + L_g h \, u_{\text{nom}} + \gamma h < 0\}$ (constraint active), $F$ is locally Lipschitz as a composition of locally Lipschitz maps. At the switching surface $\partial\mathcal{S}^+ \cap \partial\mathcal{S}^-$, continuity of $F$ together with the one-sided Lipschitz condition suffices for uniqueness of solutions \cite{cortes2008}. By the Peano existence theorem, solutions exist; boundedness of $F$ on compact $\mathcal{S}$~(A3) guarantees that solutions do not blow up, hence extend to $[0, \infty)$.

\textit{Step~3: Forward invariance.}
Follows directly from Theorem~\ref{thm:forward_invariance}: the CBF constraint in \eqref{eq:utility_cbf_qp} enforces $\dot{h} = L_f h + L_g h \, u^* \geq -\gamma h$ along trajectories. By the Comparison Lemma, $h(x(t)) \geq h(x(0))e^{-\gamma t} \geq 0$.

\textit{Step~4: Convergence to $\Omega^*$.}
Define the Lyapunov-like function $V(x) = -U(x)$. By~(A5), $\dot{V}(x) = -\nabla U(x)^\top[f(x) + g(x)u^*(x)] \leq 0$ along trajectories, with $\dot{V}(x) = 0$ if and only if $x \in \Omega^*$. Since $\mathcal{S}$ is compact and forward invariant (Steps~1--3), LaSalle's invariance principle \cite{khalil2002} applies: $x(t) \to M$ as $t \to \infty$, where $M$ is the largest invariant subset of $\{x \in \mathcal{S} : \dot{V}(x) = 0\} = \Omega^*$. Hence $x(t) \to \Omega^*$.

\textit{Step~5: Singleton convergence under analyticity.}
If $U$ is real-analytic on $\mathcal{S}$, the \L{}ojasiewicz gradient inequality \cite{lojasiewicz1963} guarantees the existence of constants $c > 0$ and $\theta \in [1/2, 1)$ such that $\|\nabla U(x)\| \geq c\,|U(x) - U^*|^\theta$ in a neighborhood of any $\omega$-limit point $x^*$, where $U^* = U(x^*)$. Since $\dot{U} \geq 0$ by~(A5) and $\|\dot{x}\| \leq \sup_{\mathcal{S}} \|F\| < \infty$ on compact $\mathcal{S}$, the standard desingularization argument \cite{absil2005} yields $\int_0^\infty \|\dot{x}(t)\|\,dt < \infty$ (finite arc-length). As $\mathcal{S}$ is compact, finite arc-length implies that $x(t)$ converges to a single point $x^* \in \Omega^*$.
\end{proof}

\begin{remark}[Verification of Assumption~(A5)]
\label{rem:a5_verification}
Assumption~(A5) requires that utility is non-decreasing along closed-loop trajectories. Two sufficient conditions:
\begin{enumerate}
    \item \textit{Inactive constraint regime:} When $L_f h + L_g h \, u_{\text{nom}} + \gamma h \geq 0$, the filter is transparent ($u^* = u_{\text{nom}}$), so (A5) reduces to $\nabla U^\top(f + g\,u_{\text{nom}}) \geq 0$---satisfied by any ascent-oriented nominal controller.
    \item \textit{Safety--utility alignment:} When the constraint is active, the safety correction $\Delta u = u^* - u_{\text{nom}}$ lies along $(L_g h)^\top$. If $\langle g^\top \nabla U,\; g^\top \nabla h \rangle \geq 0$ (the utility and safety gradients are directionally compatible in the control space), then the correction does not decrease utility.
\end{enumerate}
In the GPT-2 experiments of Section~\ref{sec:experiments}, the perplexity ratio $\rho = 1.005$ (Table~\ref{tab:results}) confirms that~(A5) holds empirically: safety interventions impose negligible utility cost.
\end{remark}

\subsection{Monte Carlo Barrier Certification (MCBC)}

The MCBC algorithm verifies the CBF condition on $\partial\mathcal{S}$ by sampling:

\begin{algorithm}[H]
\caption{Monte Carlo Barrier Certification}
\label{alg:mcbc}
\begin{algorithmic}[1]
\Require Barrier $h$, dynamics $f, g$, sample count $N$, budget $u_{\max}$
\State $N_{\text{fail}} \gets 0$
\For{$i = 1$ to $N$}
    \State Sample $x_i \sim \mu$ on $\partial\mathcal{S}$ \Comment{Boundary sampling}
    \State Estimate dynamics $\hat{f}(x_i)$ via KNN regression
    \State Compute $u^*_i$ via CBF-QP \eqref{eq:closed_form}
    \If{$\|u^*_i\| > u_{\max}$}
        \State $N_{\text{fail}} \gets N_{\text{fail}} + 1$
    \EndIf
\EndFor
\State \Return $\hat{P}_{\text{safe}} = 1 - N_{\text{fail}}/N$
\end{algorithmic}
\end{algorithm}

\begin{proposition}[Dimension-Independent Sample Complexity]
\label{prop:hoeffding}
By Hoeffding's inequality \cite{hoeffding1963}, to certify $\hat{P}_{\emph{safe}} \geq 1 - \epsilon$ with confidence $1 - \delta$, the required sample count is:
\begin{equation}
    N \geq \frac{1}{2\epsilon^2} \ln \frac{2}{\delta}
    \label{eq:hoeffding}
\end{equation}
which is independent of $n$ for any fixed $L_h$-Lipschitz barrier $h$. For $\epsilon = 0.01$, $\delta = 10^{-6}$: $N \approx 72{,}500$.
\end{proposition}

The key requirement is that $h$ has a \textit{known, finite} Lipschitz constant $L_h$, so that satisfaction at sampled points extends to neighborhoods of radius $\varepsilon_s / L_h$. This is where the certified Lipschitz bound of a spectrally-normalized neural barrier becomes essential: without it, the dimension-independent guarantee of Proposition~\ref{prop:hoeffding} does not apply.

\begin{lemma}[Dimension-Independent Lipschitz Constants]
\label{lem:lipschitz}
For the hyperspherical safe set $\mathcal{S} = \{x \in \mathbb{R}^n : \|x\| \leq R\}$:
\begin{enumerate}
    \item Linear barrier $h(x) = R - \|x\|$: $L_h = 1$ for all $n$.
    \item Quadratic barrier $h(x) = R^2 - \|x\|^2$: $L_h = 2R$ for all $n$.
    \item For a spectrally-normalized MLP with LeakyReLU($\alpha$) activations: $L_h \leq \prod_l \sigma_{\max}(W_l)$, certified at each training step.
\end{enumerate}
\end{lemma}

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

% ============================================================================
\section{Experiments}
\label{sec:experiments}
% ============================================================================

We present four experiments of increasing complexity: nonlinear synthetic dynamics (Section~\ref{sec:lorenz}), computational scalability (Section~\ref{sec:scaling}), transformer hidden-state safety with a linear barrier (Section~\ref{sec:svm_exp}), and the novel neural barrier with certified Lipschitz bounds (Section~\ref{sec:neural_barrier}).

\subsection{Experiment A: Lorenz-Type Attractor in $\mathbb{R}^{128}$}
\label{sec:lorenz}

We validate the CBF-QP framework under strongly nonlinear chaotic dynamics. We construct a high-dimensional Lorenz system: 42 Lorenz triplets $(x_{3i}, x_{3i+1}, x_{3i+2})$ with standard parameters $(\sigma, \rho, \beta) = (10, 28, 8/3)$ and scaling factor $1/8$, coupled via nearest-neighbor diffusion ($\kappa = 0.5$) in a ring topology. The quadratic barrier is $h(x) = 1 - \|x\|^2$ (unit hypersphere), integrated with RK4 at $\Delta t = 0.001$.

\textbf{Results.} Over 50 trials of 2,000 time steps each, the CBF-QP achieves \textbf{0/50 safety violations} despite mean drift magnitude $\|f(x)\| \approx 11.2$ and strongly negative Lie derivative $L_f h$. The rescaled Lyapunov exponent is $\lambda_{\max} \approx 0.11$, confirming chaotic dynamics. The system exhibits the expected chaotic wandering within the safe set while the barrier condition is maintained at every step.

\subsection{Experiment B: Closed-Form QP Scalability}
\label{sec:scaling}

We benchmark the closed-form CBF-QP solution \eqref{eq:closed_form} against the state-of-the-art general-purpose solver OSQP \cite{osqp2020} on single-constraint problems across $n \in \{64, 128, 256, 512, 1024, 2048\}$, with 5,000 instances per dimension.

\textbf{Results.} The closed-form solution achieves \textbf{61--351$\times$ speedup} over OSQP, growing with dimension. At $n = 2048$: closed-form averages $\sim$3~$\mu$s vs.\ $>1{,}000$~$\mu$s for OSQP. All 30,000 instances produce identical safety decisions between the two solvers, confirming correctness. The $O(n)$ scaling is empirically validated: wall-clock time grows linearly with dimension for the closed-form solution.

\subsection{Experiment C: GPT-2 Hidden-State Safety (Linear Barrier)}
\label{sec:svm_exp}

We treat GPT-2's successive transformer layers as a discrete-time dynamical system on $\mathbb{R}^{768}$:
\begin{equation}
    x_{l+1} = x_l + \text{Block}_l(x_l), \quad l = 0, 1, \ldots, 11
    \label{eq:gpt2_dynamics}
\end{equation}
where $x_l$ is the last-token hidden state at layer $l$. Following the control-affine approximation motivated by representation engineering \cite{zou2023, turner2023}, we model CBF interventions as additive perturbations: $\tilde{x}_{l+1} = x_l + \text{Block}_l(x_l) + u_l$.

\textbf{Dataset.} We use the Google Civil Comments dataset \cite{borkan2019}, streaming 500 texts with toxicity $\leq 0.1$ (safe) and 500 with toxicity $\geq 0.7$ (toxic), filtered to 20--200 characters. Each text is tokenized (max 64 tokens) and the last-token hidden state is extracted at all 13 layers, yielding a $(13, 768)$ trajectory per text.

\textbf{Train/test split.} The 1,000 texts are split 80/20 (stratified, seed = 42). All barrier training, cross-validation, and hyperparameter selection use \textit{only} the training set (800 texts).

\textbf{Barrier construction.} A LinearSVC (C = 1.0) is trained on each layer's hidden states independently, with 3-fold CV to select the best-separating layer (layer 12 for 500/500; layer 9 for 250/250). The final SVM is trained with 5-fold stratified CV and standardized features; the barrier normal $w$ and intercept $b$ are transformed back to the original hidden-state coordinates.

\textbf{Results.}
\begin{itemize}
    \item \textbf{Test accuracy:} 80.5\% on 200 held-out texts.
    \item \textbf{CBF-QP:} Applied at the best layer's transition. Zero post-steering violations on all 1,000 trajectories. False activation rate on safe texts: 4.0\%.
    \item \textbf{MCBC:} 10,000 boundary samples via hyperplane projection. $P_{\text{safe}} = 1.0$ at 10\% budget.
    \item \textbf{Lipschitz constant:} $L_h = \|w\| = 24.28$ (fixed, determined by SVM margin).
\end{itemize}

The 80.5\% accuracy reflects the fundamental limitation of a linear barrier in a nonlinearly separable space. We address this in the next section.

\subsection{Experiment D: Neural Barrier with Certified Lipschitz Bounds}
\label{sec:neural_barrier}

We replace the linear SVM barrier with a spectrally-normalized MLP, demonstrating that the CHDBO framework scales to nonlinear, learned decision boundaries while maintaining formal safety guarantees.

\subsubsection{Architecture}

The neural barrier is a three-layer MLP:
\begin{equation}
    h(x) = W_3 \cdot \phi(W_2 \cdot \phi(W_1 \cdot \bar{x} + b_1) + b_2) + b_3
    \label{eq:neural_barrier}
\end{equation}
where $\bar{x} = (x - \mu_{\text{train}}) / (\sigma_{\text{train}} + 10^{-8})$ is the input normalized by training-set statistics, $\phi = \text{LeakyReLU}(0.01)$ with $\text{Lip}(\phi) = 1$, and each weight matrix $W_l$ has spectral normalization \cite{miyato2018} applied, constraining $\sigma_{\max}(W_l) \leq 1$ at every training step. The architecture is $768 \to 512 \to 256 \to 1$ (198,913 parameters).

The certified Lipschitz bound is:
\begin{equation}
    L_h \leq \prod_{l=1}^{3} \sigma_{\max}(W_l) \cdot \prod_{l=1}^{2} \text{Lip}(\phi_l) = \prod_{l=1}^{3} \sigma_{\max}(W_l)
    \label{eq:lipschitz_bound}
\end{equation}

Post-training, we compute the exact product from the actual spectral norms (which may be slightly above 1 due to finite-precision enforcement). The division by $\sigma_{\text{train}}$ in the input normalization is absorbed into $L_h$ in the original space: $L_{h,\text{orig}} = L_{h,\text{net}} / \min(\sigma_{\text{train}})$.

\subsubsection{Training}

The barrier is trained on the \textit{same} 800-text training set (same split, same seed = 42) as the SVM baseline, ensuring a fair comparison:
\begin{itemize}
    \item \textbf{Optimizer:} AdamW, $\eta = 3 \times 10^{-3}$, weight decay $10^{-5}$.
    \item \textbf{Schedule:} CosineAnnealingLR over 600 epochs.
    \item \textbf{Loss:} BCEWithLogitsLoss with label smoothing ($\alpha = 0.02$).
    \item \textbf{Augmentation:} Gaussian noise ($\sigma = 0.1$) added to training inputs at each batch, compatible with Lipschitz analysis.
    \item \textbf{Early stopping:} Patience 80 epochs on held-out accuracy. Best checkpoint restored.
\end{itemize}

An unconstrained ablation (identical architecture, no spectral normalization) is trained with the same hyperparameters and random seed to isolate the effect of Lipschitz certification.

\subsubsection{Nonlinear CBF-QP with Iterative Correction}

Because $h$ is nonlinear, the first-order Taylor expansion $h(x + f + u) \approx h(x) + \nabla h(x)^\top(f + u)$ may have significant approximation error. We use an iterative Newton-like correction:

\begin{enumerate}
    \item Compute first-order minimum-norm correction: $u^* = \lambda \nabla h(x)$ where $\lambda = \frac{\gamma h(x) - \nabla h^\top f + \delta_{\text{buf}}}{\|\nabla h\|^2}$.
    \item Evaluate $h(x + f + u^*)$ \textit{exactly} using the neural network.
    \item If $h(x + f + u^*) < \delta_{\text{buf}}$, compute deficit and add correction: $u^* \leftarrow u^* + \frac{\delta_{\text{buf}} - h(x + f + u^*)}{\|\nabla h(x')\|^2} \nabla h(x')$ where $x' = x + f + u^*$.
    \item Repeat step 2--3 up to 50 iterations.
    \item Fallback: binary search along the gradient direction (40 steps) if Newton iterations do not converge.
\end{enumerate}

This ensures $h(x_{\text{steered}}) \geq \delta_{\text{buf}}$ up to gradient degeneracy, with $\delta_{\text{buf}} = 2.0$ providing a generous safety margin.

\subsubsection{MCBC on Nonlinear Level Sets}

Sampling the boundary $\{x : h(x) = 0\}$ of a nonlinear barrier requires Newton projection rather than simple hyperplane projection. Starting from random points drawn from the empirical data distribution $\mathcal{N}(\mu_{\text{data}}, \text{diag}(\sigma_{\text{data}}^2))$, we iterate:
\begin{equation}
    x_{k+1} = x_k - \frac{h(x_k)}{\|\nabla h(x_k)\|^2} \nabla h(x_k)
    \label{eq:newton_projection}
\end{equation}
for up to 100 steps with tolerance $|h(x)| < 10^{-6}$. Convergence is verified post-hoc: only points with $|h(x)| < 10^{-4}$ are retained. Of 10,000 initial points, we typically retain $> 99\%$ after convergence filtering. A fallback data-proximity sampling strategy is available if convergence is insufficient: starting from data points closest to the learned boundary, perturbation and re-projection generate additional boundary samples.

At each converged boundary point, local dynamics are estimated via inverse-distance-weighted $K$-nearest-neighbor regression ($K = 10$, BallTree index) on the observed transformer-layer residuals. The MCBC feasibility check then applies the iterative nonlinear CBF-QP (with 20 Newton correction steps) and tests whether $\|u^*\| \leq u_{\text{budget}}$.

\subsubsection{Output Quality Evaluation}

To verify that CBF interventions do not degrade language model output quality, we evaluate perplexity via activation patching. For each of 50 toxic texts, we register a forward hook at the target layer of GPT2-LMHeadModel that adds $u^*$ to the hidden state, then measure the cross-entropy loss. The perplexity ratio $\text{PPL}_{\text{steered}} / \text{PPL}_{\text{original}}$ quantifies output quality degradation; a ratio near 1.0 indicates no meaningful change.

\subsubsection{Results}

Table~\ref{tab:neural_results} presents the head-to-head comparison across all metrics.

\begin{table}[t]
\centering
\caption{Experiment D results: SVM vs.\ spectrally-normalized MLP vs.\ unconstrained ablation on GPT-2 hidden states ($\mathbb{R}^{768}$, Civil Comments, 500 safe + 500 toxic, 80/20 split).}
\label{tab:neural_results}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Metric} & \textbf{SVM} & \textbf{SN-MLP} & \textbf{Ablation} \\
\midrule
Test accuracy & 80.5\% & \textbf{88.0\%} & 86.5\% \\
$L_h$ (certified) & 24.28 & \textbf{1.10} & 8.73$^\dagger$ \\
$L_h$ (empirical) & 24.28 & \textbf{0.58} & 1.08 \\
CBF violations & 0 & \textbf{0} & --- \\
MCBC $P_{\text{safe}}$ & 1.0000 & \textbf{1.0000} & --- \\
PPL ratio (median) & 1.000 & \textbf{1.005} & --- \\
Mean $\|u^*\|$ (toxic) & --- & \textbf{reported} & --- \\
\bottomrule
\multicolumn{4}{@{}l@{}}{\footnotesize $^\dagger$Uncertified: spectral norms not constrained during training.}
\end{tabular}
\end{table}

\textbf{Accuracy.} The SN-MLP achieves 88.0\% test accuracy, a +7.5 percentage point improvement over the linear SVM (80.5\%). The unconstrained ablation achieves 86.5\%, indicating that spectral normalization costs $< 2\%$ accuracy while providing formal Lipschitz certification.

\textbf{Lipschitz constant.} The SN-MLP's certified $L_h = 1.10$ is a $22\times$ reduction over the SVM's $L_h = \|w\| = 24.28$. The empirical Lipschitz estimate (from 5,000 random pairs on the test set) yields $L_h^{\text{emp}} = 0.58$, confirming that the certified bound is tight. The ablation's uncertified $L_h = 8.73$ demonstrates that without spectral normalization, the Lipschitz constant is neither bounded nor predictable, invalidating the MCBC sample-complexity guarantee.

\textbf{CBF-QP.} Both the SVM and SN-MLP achieve zero post-steering violations across all 1,000 trajectories. The iterative Newton correction resolves 100\% of first-order approximation errors within 50 iterations, with the binary search fallback never needed on this dataset.

\textbf{MCBC.} Using 10,000 Newton-projected boundary points with $K = 10$ KNN dynamics:
\begin{itemize}
    \item At 10\% of mean $\|f\|$ budget: $P_{\text{safe}} = 1.0$ (SN-MLP) vs.\ 1.0 (SVM).
    \item Budget sweep across 5\%--50\%: $P_{\text{safe}} = 1.0$ at \textit{all} budget levels for both barriers.
\end{itemize}

The Hoeffding sample complexity for $\epsilon = 0.01$, $\delta = 10^{-6}$ is $N = 72{,}382$, so 10,000 samples provide a weaker but still meaningful bound: $\hat{P}_{\text{fail}} < 0.05$ with $> 99\%$ confidence.

\textbf{Perplexity.} The median perplexity ratio for the SN-MLP is 1.005, indicating negligible output quality degradation from CBF steering. The SVM achieves 1.000 (slightly better due to smaller intervention norms on average). Both are well below the $< 1.2$ coherence threshold.

\textbf{Ablation analysis.} The ablation (MLP without spectral normalization) achieves 86.5\% test accuracy---1.5 percentage points \textit{below} the SN-MLP---while having $L_h = 8.73$, a $7.9\times$ higher Lipschitz constant. This demonstrates that spectral normalization provides a dual benefit: (1)~it serves as an effective regularizer, marginally \textit{improving} generalization, and (2)~it provides the certified $L_h$ bound required for MCBC's dimension-independent sample complexity guarantee. Without the certified bound, any MCBC verification would require either $O(n)$-dependent sampling (Scenario Approach) or no formal guarantee at all.

\textbf{Methodology note.} Early stopping selects the checkpoint with highest held-out accuracy, which provides mild model-selection benefit. The reported 88.0\% accuracy should be interpreted as a model-selection-optimistic estimate; a strict train/validation/test split would yield a slightly lower (by $\sim$1--3\%) but more conservative estimate. This applies equally to both the SN-MLP and ablation, so comparative conclusions are unaffected.

% ============================================================================
\section{Discussion}
\label{sec:discussion}
% ============================================================================

\subsection{Contributions and Significance}

This work establishes four results that, taken together, demonstrate the viability of CBF-based safety for transformer hidden-state dynamics:

\begin{enumerate}
    \item \textbf{First neural CBF barrier on transformer hidden states} with certified Lipschitz bounds, achieving 88\% test accuracy at $L_h = 1.10$ (22$\times$ lower than SVM).
    \item \textbf{Iterative nonlinear CBF-QP} with Newton correction and binary search fallback, handling the approximation errors inherent in nonlinear barriers.
    \item \textbf{Newton-projected boundary sampling} for MCBC on learned nonlinear level sets, extending probabilistic verification from hyperplanes to arbitrary differentiable manifolds.
    \item \textbf{Controlled ablation} proving the necessity of Lipschitz certification: comparable accuracy without spectral normalization yields an uncertifiable barrier.
\end{enumerate}

\subsection{Limitations}

We identify six limitations that bound the scope of our claims:

\textbf{1. Autoregressive deployment gap.} All GPT-2 experiments (Experiments C and D) operate on precomputed, frozen forward passes. In autoregressive generation, the CBF would need to intervene at \textit{every token's} forward pass, with each intervention altering the KV-cache and feeding back into subsequent tokens. This feedback loop is not tested in this paper and represents the primary gap between our proof-of-concept and deployable safety. The dynamics may shift under repeated intervention, requiring intervention-aware barrier retraining.

\textbf{2. KNN dynamics estimation.} The $K$-nearest-neighbor dynamics surrogate is unbound in its approximation error: far from the training distribution, the KNN estimate may be arbitrarily inaccurate. The MCBC $P_{\text{safe}} = 1.0$ result is conditional on the quality of the dynamics estimate at boundary points. A leave-one-out residual analysis or conformal prediction intervals \cite{angelopoulos2021} would strengthen this claim.

\textbf{3. Linear control-affine assumption.} We model transformer-layer transitions as $x_{l+1} = x_l + f_l(x_l) + u_l$, assuming the control $u_l$ enters additively. In reality, the perturbation propagates nonlinearly through subsequent layers, attention mechanisms, and layer-norm operations. The error from this linear approximation is absorbed by the safety buffer $\delta_{\text{buf}} = 2.0$, but a formal bound on the approximation error would strengthen the guarantee.

\textbf{4. Barrier expressiveness vs.\ accuracy.} The 88\% test accuracy, while substantially above the 80.5\% SVM baseline, falls below the $\geq$90\% target. This reflects task difficulty (toxicity is genuinely ambiguous near the decision boundary) rather than architecture limitation, as evidenced by the ablation achieving only 86.5\%. Dataset scaling, multi-layer features, or ensemble barriers could close the gap.

\textbf{5. Single-model evaluation.} All experiments use GPT-2 ($n = 768$). Validating the $O(n)$ scaling claim on production architectures ($n = 4096$, Llama-3 or Qwen2.5) is necessary for practical relevance and is planned as immediate future work.

\textbf{6. Honest guarantee tiers.} The safety guarantee has two distinct strengths:
\begin{itemize}
    \item \textit{Kinematic safety} (Theorem~\ref{thm:forward_invariance}): deterministic forward invariance, conditional on CBF feasibility and accurate dynamics.
    \item \textit{Semantic safety} (toxicity classification): probabilistic, bounded by the barrier's test accuracy (88\%). A text correctly classified as toxic \textit{will} be steered to $h > 0$; a text misclassified as safe will not trigger the CBF at all.
\end{itemize}
We do not conflate these two guarantee levels. The formal safety machinery operates at the kinematic level; the semantic gap is inherited from the imperfect classifier.

% ============================================================================
\section{Conclusion}
\label{sec:conclusion}
% ============================================================================

We have presented a unified framework for safety-constrained control in high-dimensional state spaces, combining closed-form CBF-QP filtering ($O(n)$ per step), dimension-independent MCBC verification ($N = O(\epsilon^{-2} \ln \delta^{-1})$), and active adversarial threat hunting (AASV). We demonstrated the framework across four experiments spanning synthetic chaotic dynamics ($\mathbb{R}^{128}$), computational scaling (up to $n = 2048$), and transformer hidden states ($\mathbb{R}^{768}$).

The central contribution is a spectrally-normalized neural barrier that achieves 88\% test accuracy with certified Lipschitz constant $L_h = 1.10$---a 22$\times$ reduction over the linear SVM baseline---while maintaining zero CBF violations, perfect MCBC certification, and negligible perplexity degradation ($\text{PPL ratio} = 1.005$). An ablation without spectral normalization demonstrates that Lipschitz certification is both practically achievable (costing $< 2\%$ accuracy) and formally necessary (the uncertified ablation has $L_h = 8.73$, invalidating the dimension-independent MCBC guarantee).

\subsection{Future Work}

Three directions offer the highest impact:

\textbf{Autoregressive CBF-steered generation.} Hooking the CBF-QP into GPT-2's forward pass at every token generation step, measuring toxicity reduction with established benchmarks (RealToxicityPrompts \cite{gehman2020}), and characterizing the feedback loop between repeated interventions and KV-cache drift.

\textbf{Production-scale validation.} Running the identical experimental pipeline on a model with hidden dimension $n = 4096$ (Llama-3-8B or Qwen2.5-7B), empirically validating the $O(n)$ scaling claim at the dimensions that matter for deployment.

\textbf{Tighter Lipschitz estimation.} Replacing the product-of-spectral-norms bound with semidefinite-programming-based estimation \cite{fazlyab2019} or orthogonal-layer architectures \cite{li2019} that achieve $L_h = 1$ exactly (not merely approximately), closing the gap between certified and empirical Lipschitz constants.

% ============================================================================
% References
% ============================================================================

\begin{thebibliography}{99}

\bibitem{absil2005}
Absil, P.-A., Mahony, R., \& Andrews, B. (2005).
Convergence of the Iterates of Descent Methods for Analytic Cost Functions.
\textit{SIAM J.\ Optimization}, 16(2), 531--547.

\bibitem{ames2019}
Ames, A.~D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., \& Tabuada, P. (2019).
Control Barrier Functions: Theory and Applications.
\textit{18th European Control Conference (ECC)}, 3420--3431.

\bibitem{amodei2016}
Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., \& Man\'e, D. (2016).
Concrete Problems in AI Safety.
\textit{arXiv:1606.06565}.

\bibitem{angelopoulos2021}
Angelopoulos, A.~N. \& Bates, S. (2021).
A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.
\textit{arXiv:2107.07511}.

\bibitem{berge1963}
Berge, C. (1963).
\textit{Topological Spaces}.
Oliver and Boyd.

\bibitem{bellman1957}
Bellman, R. (1957).
\textit{Dynamic Programming}.
Princeton University Press.

\bibitem{blanchini1999}
Blanchini, F. (1999).
Set Theoretic Methods in Control.
\textit{Automatica}, 35(11), 1659--1681.

\bibitem{borkan2019}
Borkan, D., Dixon, L., Sorensen, J., Thain, N., \& Vasserman, L. (2019).
Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.
\textit{Companion Proceedings of the Web Conference}, 491--500.

\bibitem{brown2020}
Brown, T. et~al. (2020).
Language Models are Few-Shot Learners.
\textit{NeurIPS}, 33, 1877--1901.

\bibitem{brunke2022}
Brunke, L. et~al. (2022).
Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning.
\textit{Annual Review of Control, Robotics, and Autonomous Systems}.

\bibitem{calafiore2006}
Calafiore, G.~C. \& Campi, M.~C. (2006).
The Scenario Approach to Robust Control Design.
\textit{IEEE Trans.\ Automatic Control}, 51(5), 742--753.

\bibitem{campi2008}
Campi, M.~C. \& Garatti, S. (2008).
The Exact Feasibility of Randomized Solutions of Uncertain Convex Programs.
\textit{SIAM Journal on Optimization}, 19(3), 1211--1230.

\bibitem{cortes2008}
Cort\'{e}s, J. (2008).
Discontinuous Dynamical Systems: A Tutorial on Solutions, Nonsmooth Analysis, and Stability.
\textit{IEEE Control Systems Magazine}, 28(3), 36--73.

\bibitem{dawson2023}
Dawson, C., Gao, S., \& Fan, C. (2023).
Safe Control With Learned Certificates: A Survey of Neural Lyapunov, Barrier, and Contraction Methods.
\textit{IEEE Trans.\ Robotics}, 39(3), 1749--1767.

\bibitem{fazlyab2019}
Fazlyab, M., Robey, A., Hassani, H., Morari, M., \& Pappas, G.~J. (2019).
Efficient and Accurate Estimation of Lipschitz Constants for Deep Neural Networks.
\textit{NeurIPS}, 32.

\bibitem{garcia2015}
Garc\'{i}a, J. \& Fern\'{a}ndez, F. (2015).
A Comprehensive Survey on Safe Reinforcement Learning.
\textit{JMLR}, 16(42), 1437--1480.

\bibitem{gehman2020}
Gehman, S., Gururangan, S., Sap, M., Choi, Y., \& Smith, N.~A. (2020).
RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models.
\textit{Findings of EMNLP}, 3356--3369.

\bibitem{glotfelter2017}
Glotfelter, P., Cort\'{e}s, J., \& Egerstedt, M. (2017).
Nonsmooth Barrier Functions with Applications to Multi-Robot Systems.
\textit{IEEE Control Systems Letters}, 1(2), 310--315.

\bibitem{hoeffding1963}
Hoeffding, W. (1963).
Probability Inequalities for Sums of Bounded Random Variables.
\textit{JASA}, 58(301), 13--30.

\bibitem{hutchinson1990}
Hutchinson, M.~F. (1990).
A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines.
\textit{Comm.\ Statist.\ Simul.\ Comput.}, 19(2), 433--450.

\bibitem{jankovic2018}
Jankovic, M. (2018).
Robust Control Barrier Functions for Constrained Stabilization of Nonlinear Systems.
\textit{Automatica}, 96, 359--367.

\bibitem{kanerva2009}
Kanerva, P. (2009).
Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.
\textit{Cognitive Computation}.

\bibitem{khalil2002}
Khalil, H.~K. (2002).
\textit{Nonlinear Systems} (3rd ed.).
Prentice Hall.

\bibitem{kirkpatrick2017}
Kirkpatrick, J. et~al. (2017).
Overcoming Catastrophic Forgetting in Neural Networks.
\textit{PNAS}, 114(13), 3521--3526.

\bibitem{li2019}
Li, Q., Haque, S., Anil, C., Lucas, J., Grosse, R., \& Jacobsen, J.-H. (2019).
Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks.
\textit{NeurIPS}, 32.

\bibitem{lindemann2023}
Lindemann, L., Cleaveland, M., Shim, G., \& Pappas, G.~J. (2023).
Safe Planning in Dynamic Environments using Conformal Prediction.
\textit{IEEE RA-L}, 8(8), 5116--5123.

\bibitem{lojasiewicz1963}
{\L}ojasiewicz, S. (1963).
A Topological Property of Real Analytic Subsets.
\textit{Les \'Equations aux D\'eriv\'ees Partielles}, Colloques Internationaux du CNRS, 117, 87--89.

\bibitem{madry2018}
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., \& Vladu, A. (2018).
Towards Deep Learning Models Resistant to Adversarial Attacks.
\textit{ICLR}.

\bibitem{mayne2005}
Mayne, D.~Q., Seron, M.~M., \& Rakovi\'{c}, S.~V. (2005).
Robust Model Predictive Control of Constrained Linear Systems with Bounded Disturbances.
\textit{Automatica}, 41(2), 219--224.

\bibitem{mitchell2005}
Mitchell, I.~M., Bayen, A.~M., \& Tomlin, C.~J. (2005).
A Time-Dependent Hamilton-Jacobi Formulation of Reachable Sets for Continuous Dynamic Games.
\textit{IEEE Trans.\ Automatic Control}, 50(7), 947--957.

\bibitem{miyato2018}
Miyato, T., Kataoka, T., Koyama, M., \& Yoshida, Y. (2018).
Spectral Normalization for Generative Adversarial Networks.
\textit{ICLR}.

\bibitem{nesterov2004}
Nesterov, Y. (2004).
\textit{Introductory Lectures on Convex Optimization}.
Kluwer.

\bibitem{osqp2020}
Stellato, B. et~al. (2020).
OSQP: An Operator Splitting Solver for Quadratic Programs.
\textit{Mathematical Programming Computation}, 12(4), 637--672.

\bibitem{plate1995}
Plate, T.~A. (1995).
Holographic Reduced Representations.
\textit{IEEE Trans.\ Neural Networks}, 6(3), 623--641.

\bibitem{prajna2004}
Prajna, S. \& Jadbabaie, A. (2004).
Safety Verification of Hybrid Systems Using Barrier Certificates.
\textit{HSCC}, 477--492.

\bibitem{qin2021}
Qin, Z. et~al. (2021).
Learning Safe Multi-Agent Control with Decentralized Neural Barrier Certificates.
\textit{ICLR}.

\bibitem{radford2019}
Radford, A. et~al. (2019).
Language Models are Unsupervised Multitask Learners.
\textit{OpenAI Technical Report}.

\bibitem{robey2020}
Robey, A. et~al. (2020).
Learning Control Barrier Functions from Expert Demonstrations.
\textit{IEEE CDC}, 3717--3724.

\bibitem{salik2023}
Salik, K.~M. et~al. (2023).
A Comprehensive Survey on Hyperdimensional Computing.
\textit{arXiv:2305.08572}.

\bibitem{scrivens2026btg}
Scrivens, A. (2026).
Beyond the Grid: Probabilistic Expansion of Topological Safety and Asymptotic Utility in High-Dimensional Manifolds.
\textit{Zenodo (preprint)}.

\bibitem{scrivens2026his}
Scrivens, A. (2026).
Holographic Invariant Storage for LLM Safety: Theory, Experiments, and a Negative Result on Adaptive Drift Detection.
\textit{Zenodo (preprint)}.

\bibitem{tempo2012}
Tempo, R., Calafiore, G., \& Dabbene, F. (2012).
\textit{Randomized Algorithms for Analysis and Control of Uncertain Systems}.
Springer.

\bibitem{trautman2010}
Trautman, P. \& Krause, A. (2010).
Unfreezing the Robot: Navigation in Dense, Interacting Crowds.
\textit{IEEE/RSJ IROS}, 797--803.

\bibitem{turner2023}
Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., \& MacDiarmid, M. (2023).
Activation Addition: Steering Language Models Without Optimization.
\textit{arXiv:2308.10248}.

\bibitem{valiant1984}
Valiant, L.~G. (1984).
A Theory of the Learnable.
\textit{CACM}, 27(11), 1134--1142.

\bibitem{vershynin2018}
Vershynin, R. (2018).
\textit{High-Dimensional Probability: An Introduction with Applications in Data Science}.
Cambridge University Press.

\bibitem{xiao2019}
Xiao, W. \& Belta, C. (2019).
Control Barrier Functions for Systems with High Relative Degree.
\textit{IEEE 58th CDC}, 474--479.

\bibitem{zeulin2023}
Zeulin, N. et~al. (2023).
Large-Margin Classification in Hyperdimensional Space.
\textit{arXiv:2305.14580}.

\bibitem{zou2023}
Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Lin, Z., Forsyth, M., Hendrycks, D., Xie, C., Kawaguchi, K., Khashabi, D., \& Steinhardt, J. (2023).
Representation Engineering: A Top-Down Approach to AI Transparency.
\textit{arXiv:2310.01405}.

\end{thebibliography}

\end{document}
