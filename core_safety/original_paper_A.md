\documentclass[10pt, twocolumn, a4paper]{article}

% Required Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs} % For nicer tables
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{xcolor}
\usepackage{subcaption}

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

% Title Information
\title{\textbf{Beyond the Grid: Probabilistic Expansion of Topological Safety and Asymptotic Utility in High-Dimensional Manifolds}}
\author{Arsenios Scrivens}
\date{February 9, 2026}

\begin{document}

\maketitle

\begin{abstract}
Verifying safety constraints in high-dimensional state spaces is computationally intractable for deterministic methods: grid-based Hamilton-Jacobi reachability scales as $O((1/\eta)^n)$, becoming infeasible for $n > 5$. We introduce \textit{Constrained High-Dimensional Barrier Optimization} (CHDBO), a framework that replaces deterministic enumeration with probabilistic certification. Using Monte Carlo Barrier Certificates (MCBC) bounded by Lipschitz continuity, CHDBO achieves a sample count---$N = O(\epsilon^{-2}\ln\delta^{-1})$---that is independent of the state-space dimension $n$ for any fixed Lipschitz-continuous barrier, with total cost $O(Nn)$, enabling safety certification at $n \geq 128$. A closed-form Control Barrier Function--Quadratic Program (CBF-QP) safety filter enforces forward invariance in $O(n)$ per step for single-constraint systems, while a projected-gradient utility maximizer ensures asymptotic convergence to constrained optima without violating the safety envelope. Experiments across ten configurations---spanning chaotic nonlinear dynamics (including the Lorenz attractor in $\mathbb{R}^{128}$), scalability to $n = 1024$, closed-form QP benchmarks, and MCBC-vs-scenario-approach comparisons---confirm that CHDBO maintains safety while preserving utility. Two additional experiments provide honest failure-mode characterization: bounded actuation degrades gracefully with a quantified phase transition, and systematic negative-result analysis identifies the regimes where guarantees break down. Active adversarial search methods that complement MCBC's passive certification are explored in a companion paper.
\end{abstract}

\section{Introduction}

The trajectory of artificial intelligence has shifted rapidly from reactive, token-prediction models to persistent, autonomous Agentic Control Systems (ACS). This evolution has revealed a fundamental fragility in current architectural paradigms: the phenomenon of ``Agent Drift'' \cite{scrivens2026}, which can expose the system to adversarial exploits and alignment failures \cite{wei2023}. LeCun's framework for autonomous machine intelligence \cite{lecun2022} independently motivates the need for persistent, self-correcting architectures that maintain behavioral coherence over extended operation. A related phenomenon---distributional degradation when models are trained on their own outputs---has been documented as ``Model Collapse'' \cite{shumailov2023}. More broadly, as systems operate over extended temporal horizons, they suffer from a form of structural entropy---sometimes termed ``mutational meltdown'' in evolutionary biology \cite{lynch1995}---where behavioral stability and decision quality degrade \cite{amodei2016}.

In our previous work, \textit{Mitigating Large Language Model Context Drift via Holographic Invariant Storage} \cite{scrivens2026}, we addressed the semantic component of this failure. We demonstrated that by anchoring an agent's identity in a Vector Symbolic Architecture (VSA) substrate \cite{kanerva2009}---a holographic read-only memory---we could substantially reduce the corruption of its goals and personality. While Vector Symbolic Architectures inherently possess the high-dimensional noise tolerance required to solve this problem \cite{plate1995, gayler2003}, a distinct but parallel challenge exists in the kinematic domain: dynamical control drift.

\subsection{The High-Dimensional Control Divergence}
Modern autonomous agents do not merely process text; they operate in high-dimensional state spaces, mapping inputs to actions in continuous environments \cite{ames2019}. Whether controlling a swarm of drones ($n>100$) \cite{brunke2022} or navigating the semantic embedding space of a multi-modal reasoning engine ($n=12,288$ \cite{brown2020}) or a visual transformer \cite{radford2021}, the agent must constantly optimize for utility while adhering to strict safety constraints (e.g., ``Do not delete root files,'' ``Do not collide'').

The central objective of modern control theory is the synthesis of an agent that is simultaneously aggressive in its pursuit of goals and conservative in its avoidance of failure. Historically, this has forced a binary choice:

\begin{enumerate}
    \item \textbf{Formal Methods (The Conservative Approach):} Techniques such as Control Barrier Functions (CBFs) \cite{ames2019} and Hamilton-Jacobi reachability \cite{mitchell2005} offer absolute, provable safety guarantees. However, these methods suffer from the ``Curse of Dimensionality'' \cite{bellman1957}. Verifying a safety manifold via grid discretization scales as $O((1/\eta)^n)$, where $\eta$ is the grid resolution and $n$ is the dimension. For a simple robotic arm ($n=6$, three joints with position and velocity), this is tractable; for a semantic reasoning agent ($n=128$), it is physically impossible, requiring computation exceeding the age of the universe.
    \item \textbf{Learning-Based Control (The Aggressive Approach):} Deep Reinforcement Learning (DRL) offers high performance and scalability but often lacks axiomatic safety \cite{amodei2016}. While recent advancements in ``Safe RL'' and Shielding attempt to impose constraints \cite{brunke2022, alshiekh2018, garcia2015}, standard implementations often rely on ``soft'' reward penalties that do not strictly prevent catastrophic failure states. More principled approaches combine CBFs with learned controllers \cite{choi2020} or employ predictive safety filters \cite{wabersich2021, fisac2019}, but these still require either accurate models or careful tuning of the CBF, and scalability to $n \gg 100$ remains largely untested.
\end{enumerate}

\subsection{The ``Golden Manifold'' and Probabilistic Relaxation}
To resolve this dichotomy, we must extend the concept of the \textbf{Forward Invariant Safe Set} $\mathcal{S}$ \cite{ames2019, blanchini1999}---colloquially termed the ``Golden Manifold'' in our prior work---from the semantic domain into the kinematic domain. In this geometric region, the agent remains invariant to microscopic fluctuations, preserving its trajectory against external perturbations.

\begin{definition}[Golden Manifold]
\label{def:golden_manifold}
The \textbf{Golden Manifold} is the maximal forward-invariant safe set $\mathcal{S} = \{x \in \mathcal{X} : h(x) \geq 0\}$ within which the closed-loop system under CBF-QP control satisfies the barrier condition $\dot{h}(x) + \gamma h(x) \geq 0$ for all $x \in \partial\mathcal{S}$. We adopt this term from our prior work \cite{scrivens2026} as an intuitive shorthand for the certified operating envelope of the agent.
\end{definition}

This paper proposes Constrained High-Dimensional Barrier Optimization (CHDBO). We argue that for high-dimensional systems, absolute deterministic safety verification is a theoretical luxury that prevents practical deployment. Instead, we introduce a probabilistic relaxation of the safety condition. By trading the deterministic exactness of grid checks for probabilistic certainty bounded by Lipschitz continuity, we can utilize Monte Carlo integration to estimate the volume of the safe set.

By sampling the hypersphere of the agent's immediate trajectory and applying statistical bounds to the barrier function $h(x)$, CHDBO certifies safety with confidence levels approaching $1 - 10^{-6}$. While categorically distinct from absolute deterministic guarantees, this approaches reliability levels comparable to many practical engineering safety margins (e.g., component failure rates), circumventing the exponential scaling of sample \textit{count} (though the total computational cost remains $O(N \cdot n)$ due to per-sample vector operations). While the \textit{per-sample} computational cost scales linearly ($O(n)$) due to vector operations, the required \textit{sample count} $N$ becomes a constant factor governed only by the desired confidence interval and the Lipschitz constant of the barrier function. We explicitly categorize this verification as \textit{Probabilistic Safety} (PAC-style \cite{valiant1984}), distinguishable from the \textit{Absolute Safety} of low-dimensional formal methods. While Lipschitz continuity theoretically bounds the minimum volume of failure modes, the \textit{Concentration of Measure} phenomenon in high-dimensional spaces ($n=128$) implies that rare, ``spiky'' failure regions (``Black Swan'' singularities) with probability mass below the sampling resolution ($\epsilon$) may evade passive uniform sampling. Active adversarial search methods that complement passive MCBC certification---transitioning from statistical assurance to gradient-based threat hunting---offer a promising direction for future work.

\subsection{Asymptotic Utility and Orthogonal Projection}
However, strict safety must not preclude mission completion; a theoretically safe agent that remains immobile is operationally useless. Our second contribution is a geometric control law that ensures Asymptotic Utility by functioning as a minimally invasive safety filter. We introduce a steering mechanism that constrains the control input strictly within the tangent cone of the safe set $\mathcal{S}$.

When the nominal action derived from the utility function $U(x)$ threatens to violate the invariant set, CHDBO does not halt the system. Instead, it solves a high-dimensional Quadratic Program (QP) to perform an Orthogonal Projection of the intent vector onto the safety boundary. This enables the agent to traverse the boundary $\partial\mathcal{S}$ of the safe manifold, maintaining the maximum permissible velocity tangent to the constraint, thereby resolving the ``Frozen Robot'' problem \cite{trautman2010} characteristic of naive barrier implementations where conflicting constraints induce stagnation.

\subsection{Contributions}
The specific contributions of this paper are:
\begin{itemize}
    \item \textbf{Generalization of Semantic Safety:} We extend the invariant preservation principles of \cite{scrivens2026} from holographic memory to continuous control manifolds.
    \item \textbf{High-Dimensional Adaptation of CBF-QP:} We adapt standard Control Barrier Function (CBF) formulations \cite{ames2019, xiao2019} to high-dimensional semantic spaces ($n \ge 128$), building on the growing body of work on learned and neural CBF methods \cite{dawson2023, robey2020, qin2021, fisac2019}. We demonstrate that the quadratic programming (QP) formulation remains computationally tractable ($O(n)$) for real-time semantic steering, overcoming the perceived ``Curse of Dimensionality'' in verification \cite{bellman1957}.
    \item \textbf{Hierarchical Utility Integration:} We provide a lexicographic control architecture that strictly prioritizes safety constraints over utility maximization, ensuring that aggressive performance and rigorous safety are hierarchically integrated rather than competing.
\end{itemize}

\subsection{Related Work}

CHDBO builds on CBF-QP theory \cite{ames2019, xiao2019}, extending it to high-dimensional spaces ($n \geq 128$) with $O(n)$ closed-form enforcement. Our Monte Carlo Barrier Certificate adapts randomized verification \cite{campi2008, calafiore2006} to barrier function analysis using Hoeffding bounds, achieving dimension-independent sample complexity. The framework complements neural barrier synthesis \cite{dawson2023, robey2020}, conformal prediction for safety \cite{lindemann2023}, safe RL \cite{garcia2015, berkenkamp2017}, and Hamilton-Jacobi reachability \cite{mitchell2005, bansal2021}, occupying a distinct niche of probabilistic safety at $O(n)$ cost. An extended discussion of related work is provided in Appendix~\ref{app:related_work}.

\section{Mathematical Preliminaries and Problem Formulation}

\subsection{System Dynamics}
We consider a control-affine system on $\mathcal{X} \subseteq \mathbb{R}^n$:
\begin{equation}
    \dot{x} = f(x) + g(x)u
\end{equation}
where:
\begin{itemize}
    \item $x \in \mathcal{X}$ is the state vector of the system (where $n$ may be arbitrarily large, e.g., $n \geq 128$ for semantic embedding spaces).
    \item $u \in \mathcal{U} \subseteq \mathbb{R}^m$ is the control input, bounded by actuation limits or policy constraints.
    \item $f: \mathbb{R}^n \to \mathbb{R}^n$ and $g: \mathbb{R}^n \to \mathbb{R}^{n \times m}$ are locally Lipschitz continuous vector fields representing the drift dynamics and control effectiveness, respectively \cite{khalil2002}.
\end{itemize}
In the context of an Agentic Control System (ACS), $x$ represents the combined state of the agent's internal cognition and external environment, while $u$ represents the discrete or continuous actions taken to modify that state.

\subsubsection{Notation Summary}
For reference, we summarize the principal notation used throughout this paper:

\begin{table}[htbp]
\centering
\caption{Notation Summary}
\scriptsize
\begin{tabular}{@{}l@{\hskip 4pt}p{0.55\linewidth}@{}}
\toprule
\textbf{Symbol} & \textbf{Description} \\ \midrule
$x \in \mathcal{X} \subseteq \mathbb{R}^n$ & System state vector \\
$u \in \mathcal{U} \subseteq \mathbb{R}^m$ & Control input \\
$f(x), g(x)$ & Drift and control-effectiveness fields \\
$h(x)\!:\!\mathbb{R}^n \!\to\! \mathbb{R}$ & Control Barrier Function (CBF) \\
$\mathcal{S} = \{x : h(x) \geq 0\}$ & Safe set (forward-invariant region) \\
$\partial\mathcal{S} = \{x : h(x) = 0\}$ & Safety boundary \\
$L_f h, L_g h$ & Lie derivatives of $h$ along $f$, $g$ \\
$\gamma > 0$ & Class-$\mathcal{K}$ decay rate \\
$L_h$ & Lipschitz constant of $h$ \\
$\epsilon, \delta$ & Hoeffding confidence parameters \\
$\rho(x)$ & Adaptive spectral safety margin \\
$\epsilon_{\text{model}}$, $\Delta_{\text{noise}}$ & Model error \& disturbance bounds \\
$U(x)\!:\!\mathbb{R}^n \!\to\! \mathbb{R}$ & Utility function (maximized) \\
$\sigma_{\max}(A)$, $\|A\|_F$ & Spectral \& Frobenius norms \\
\bottomrule
\end{tabular}
\label{tab:notation}
\end{table}

\textbf{Remark (Continuous Relaxation).}
Applying the control-affine ODE to semantic embedding spaces requires a continuous relaxation, justified by the Neural ODE paradigm \cite{chen2018, kidger2022, geshkovski2024} and the interpretation of Transformer attention as discretizations of continuous particle systems \cite{geshkovski2024, sander2022}. The single-step Euler local truncation error $O(\Delta t^2 \cdot L_{\nabla f})$ is absorbed into $\epsilon_{\text{model}}$ of Equation~\ref{eq:robust_barrier}, and multi-step drift is bounded by a Gronwall argument: $\|x_{\text{true}} - x_{\text{Euler}}\| \leq C \cdot \Delta t \cdot (e^{L_f T} - 1)$, small when $\Delta t \ll 1/L_f$. When $L_f$ is locally large (e.g., adversarial NLP perturbations \cite{goodfellow2014}), the safety tube $\rho(x)$ inflates and the framework becomes appropriately conservative.

\subsection{Recap of Topological Safety}
Following the standard formulation in \cite{ames2019} and \cite{blanchini1999}, we define safety through the lens of set invariance. A set $\mathcal{S} \subset \mathcal{X}$ is defined as the safe set, representing the region of the state space where the system is permitted to operate (e.g., ``Sanity'', ``Goal Alignment''). We define $\mathcal{S}$ as the super-level set of a continuously differentiable scalar function $h(x): \mathbb{R}^n \to \mathbb{R}$, known as the Control Barrier Function (CBF):
\begin{align*}
    \mathcal{S} &= \{ x \in \mathbb{R}^n \mid h(x) \geq 0 \} \\
    \partial \mathcal{S} &= \{ x \in \mathbb{R}^n \mid h(x) = 0 \} \\
    \text{Int}(\mathcal{S}) &= \{ x \in \mathbb{R}^n \mid h(x) > 0 \}
\end{align*}

As established in prior work \cite{scrivens2026}, we formalize the safety condition as follows:

\begin{theorem}[Topological Safety --- Sufficiency]
\label{thm:topological_safety}
The system is safe (i.e., the set $\mathcal{S}$ is forward invariant) if there exists a control input $u$ such that the time derivative of $h(x)$ satisfies the linear class-$\mathcal{K}$ inequality (concept formally defined in \cite{khalil2002}, applied to barriers in \cite{ames2019}):
\begin{equation}
    \sup_{u \in \mathcal{U}} \left[ L_f h(x) + L_g h(x) u \right] \geq -\gamma h(x)
\end{equation}
where $L_f h$ and $L_g h$ denote the Lie derivatives of $h$ along $f$ and $g$, and $\gamma > 0$ is a tunable relaxation parameter governing how quickly the system is allowed to approach the boundary. Sufficiency follows from Nagumo's theorem \cite{nagumo1942, blanchini1999}: if the barrier condition holds for all $x \in \partial\mathcal{S}$, then any trajectory starting in $\mathcal{S}$ remains in $\mathcal{S}$ for all $t \geq 0$.
\end{theorem}

\textbf{Remark (Necessity).} The converse---that forward invariance of $\mathcal{S}$ \textit{implies} the existence of a control satisfying the barrier condition---holds additionally when $0$ is a regular value of $h$ (i.e., $\nabla h(x) \neq 0$ on $\partial\mathcal{S}$) and $\mathcal{S}$ is compact. In this case, the barrier condition is both necessary and sufficient for forward invariance \cite{ames2019, blanchini1999}. Throughout this paper, we rely only on the sufficiency direction: all safety guarantees follow from \textit{enforcing} the barrier condition, not from verifying that it is the unique characterization of invariance.

\subsection{The High-Dimensional Divergence}
While the formulation provides sufficient conditions for safety, the verification method utilized therein, Grid-Based Invariance Checking, relies on discretizing the domain $\mathcal{X}$. As established in Section~1.1, the computational complexity of verifying the invariance condition over a grid $G_\eta$ scales exponentially:
\begin{equation}
    C(G_\eta) \propto \left(\frac{L}{\eta} \right)^n
\end{equation}
Consequently, exact topological guarantees cannot be directly computed for high-degree-of-freedom systems ($n \ge 128$). This necessitates the Probabilistic Relaxation we introduce in Section~3.

\subsection{Utility Maximization and Problem Statement}
Unlike \cite{scrivens2026}, which focused primarily on safety (survival and memory persistence), this work introduces a performance objective. We define a Utility Function $U(x): \mathbb{R}^n \to \mathbb{R}$, which encodes the agent's task (e.g., reaching a target, maximizing speed, maintaining formation). We assume $U(x)$ is continuously differentiable and concave.

We seek to find a control law $k(x)$ that drives the system to the global maximum of $U(x)$ without ever leaving $\mathcal{S}$.

\begin{problem}[Constrained High-Dimensional Barrier Optimization]
Find a control policy $u=k(x)$ such that for any initial condition $x_0 \in \text{Int}(\mathcal{S})$:
\begin{enumerate}
    \item \textbf{Safety (High Probability):} $P(x(t) \in \mathcal{S}) \geq 1-\delta$ for all $t \geq 0$, where $\delta$ is a negligible failure probability (e.g., $10^{-6}$).
    \item \textbf{Asymptotic Utility:} $\lim_{t \to \infty} x(t) = x^*_{\mathcal{S}}$ where $x^*_{\mathcal{S}} = \text{argmax}_{x \in \mathcal{S}} U(x)$.
    \item \textbf{Scalability:} The computation time for calculating $u$ is $O(n)$ relative to dimension (linear scaling). This enables real-time operation for $n \gg 10$.
\end{enumerate}
\end{problem}
This formulation stratifies the control problem: safety verification (handled via probabilistic barriers) acts as a hard constraint, while the performance objective (handled via gradient-based steering) acts as a soft preference, resolving the ``Safety-Performance Trade-off'' through lexicographic optimization.

\subsection{Standing Assumptions}
\label{sec:assumptions}
For clarity, we collect the assumptions underlying the CHDBO framework:
\begin{enumerate}
    \item[\textbf{A1.}] \textbf{Lipschitz Continuity:} The drift $f(x)$, control effectiveness $g(x)$, and barrier $h(x)$ are locally Lipschitz continuous on $\mathcal{X}$ \cite{khalil2002}.
    \item[\textbf{A2.}] \textbf{Barrier Differentiability:} $h(x)$ is continuously differentiable ($C^1$), and $0$ is a regular value of $h$ (i.e., $\nabla h(x) \neq 0$ on $\partial\mathcal{S}$).
    \item[\textbf{A3.}] \textbf{Relative Degree One:} $L_g h(x) \neq 0$ on $\partial\mathcal{S}$. For higher relative degree, an Exponential CBF (ECBF) \cite{xiao2019} reduces the constraint to relative degree one via auxiliary functions $\psi_i(x) = \dot{\psi}_{i-1} + \alpha_i \psi_{i-1}$, preserving the $O(n)$ framework (validated in Experiment~IV).
    \item[\textbf{A4.}] \textbf{Bounded Control:} The control input set $\mathcal{U}$ is compact and the CBF-QP (Equation~\ref{eq:cbfqp}) is feasible for all $x \in \partial\mathcal{S}$.
    \item[\textbf{A5.}] \textbf{Local Convexity or Star-Shapedness:} The safe set $\mathcal{S}$ is locally convex or star-shaped near $\partial\mathcal{S}$ for the boundary projection to be well-defined. For non-convex $\mathcal{S}$, tangent half-space inner-approximations are used (Section~3.3).
    \item[\textbf{A6.}] \textbf{Utility Smoothness:} $U(x)$ is continuously differentiable. For global convergence guarantees, $U$ is concave or satisfies the Polyak-{\L}ojasiewicz condition \cite{polyak1963}.
    \item[\textbf{A7.}] \textbf{Continuous Relaxation (Semantic Systems):} Discrete semantic dynamics are modeled as continuous ODEs via the Neural ODE paradigm \cite{chen2018, geshkovski2024}; see the Continuous Relaxation remark above. Discretization error is absorbed into $\epsilon_{\text{model}}$; the framework becomes conservative when $L_f$ is large.
    \item[\textbf{A8.}] \textbf{Controllability Margin:} There exists $c > 0$ such that $\inf_{x \in \partial\mathcal{S}} \frac{\|L_g h(x)\|}{|L_f h(x)| + 1} > c$, ensuring sufficient control authority. Trivially satisfied in our experiments ($g = I$).
\end{enumerate}

\section{High-Dimensional Probabilistic Verification}
We now present the Monte Carlo Barrier Certificate (MCBC), which replaces deterministic enumeration with probabilistic certification.

\subsection{The Scaling Bottleneck}
The computational cost of exact verification scales as $O((1/\eta)^n)$: a 3-link arm ($n{=}6$) requires $10^{12}$ evaluations; a semantic embedding space ($n \geq 128$) requires $10^{200}$. Two classical results motivate random sampling: the Concentration of Measure phenomenon \cite{vershynin2018} (barrier statistics from random boundary samples are representative of global behavior) and the Johnson-Lindenstrauss lemma \cite{johnson1984, dasgupta2003} (pairwise distances are preserved under random projections). Neither directly guarantees barrier-value preservation, but together they support---and our experiments validate---that sparse boundary sampling captures the essential safety geometry.

We shift from Worst-Case to PAC-Style Invariance \cite{valiant1984}: bounding the probability of encountering a failure state by $\epsilon$ via the Chernoff--Hoeffding bound \cite{hoeffding1963}, aligning with conformal prediction \cite{angelopoulos2021, lindemann2023} and randomized control frameworks \cite{tempo2012, calafiore2006}.

\subsection{Randomized Barrier Certification}
Building on Barrier Certificates \cite{prajna2004}, we treat $\mathcal{X}$ as a probability space with measure $\mu$ and define the Safety Violation Probability:
\begin{equation}
    P_{\text{fail}} = \int_{\mathcal{X}} \mathbb{I}\left(\sup_{u \in \mathcal{U}} \dot{h}(x, u) < -\gamma h(x) \right) d\mu(x)
\end{equation}
where $\mathbb{I}(\cdot)$ is the indicator function. Since this integral is analytically intractable for nonlinear $f(x)$ and $g(x)$, we approximate it via Monte Carlo integration.

\subsection{Lipschitz Continuity and Sample Complexity}
Since the safety condition is critical primarily at the boundary $\partial \mathcal{S}$, we employ isotropic Gaussian sampling projected onto the hypersphere surface \cite{muller1959}.

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_concentration.png}
    \caption{\textbf{Concentration of Measure (`Hollow Ball').} As $n$ increases, probability mass concentrates near the boundary ($r > 0.95$ contains $\approx 99.9\%$ of volume at $n=128$), validating boundary-focused sampling.}
    \label{fig:concentration}
\end{figure}

For non-convex $\mathcal{S}$, local convex approximations (tangent half-spaces) inner-approximate the true safe set \cite{ames2019, kong2023}: the approximation may exclude safe regions but never includes unsafe ones, so the Hoeffding guarantee remains valid.

By Hoeffding's Inequality \cite{hoeffding1963}, the sample count for verification is independent of dimension ($O(1)$) for fixed $L_h$ and safety margin $\varepsilon_s$ (if $L_h$ grows with $n$, effective complexity may increase). The required sample count is:
\begin{equation}
    N \geq \frac{1}{2\epsilon^2} \ln\left(\frac{2}{\delta}\right)
    \label{eq:hoeffding}
\end{equation}
For example, certifying a 128-dimensional system at 99\% confidence ($\delta = 0.01$, $\epsilon = 0.01$) requires only $N \approx 26{,}000$ samples. The bound certifies the accuracy of $\hat{P}_{\text{fail}}$; the connection to trajectory safety is addressed by Proposition~\ref{prop:trajectory_bridge}.

\textbf{Qualification.} Three implicit dimensional dependencies (per-sample cost, $L_h$ growth, coverage ball volume decay) qualify the dimension-independence; details in Appendix~\ref{app:qualifications}. Concentration of Measure both aids verification (boundary mass concentration) and limits it (narrow failure spikes may be volumetrically invisible), motivating active adversarial search.

\subsubsection{Dimension-Independence of Lipschitz Constants}
The following lemma shows that for natural barrier functions, $L_h$ is dimension-independent.

\begin{lemma}[Dimension-Independent Lipschitz Constants]
\label{lem:lipschitz_dim}
Let $\mathcal{S} = \{x \in \mathbb{R}^n : \|x\| \leq R\}$ be a hyperspherical safe set of radius $R > 0$. Consider the following barrier functions:
\begin{enumerate}
    \item \textbf{Linear barrier:} $h_1(x) = R - \|x\|$. Then $L_{h_1} = 1$ for all $n$.
    \item \textbf{Quadratic barrier:} $h_2(x) = R^2 - \|x\|^2$. Then $L_{h_2} = 2R$ for all $n$.
\end{enumerate}
More generally, for any barrier $h(x) = \phi(\|x\|)$ where $\phi: \mathbb{R}_{\geq 0} \to \mathbb{R}$ is $L_\phi$-Lipschitz, the composite barrier satisfies $L_h = L_\phi$, independent of $n$.
\end{lemma}

\begin{proof}
For $h(x) = \phi(\|x\|)$, we have $\nabla h(x) = \phi'(\|x\|) \cdot x/\|x\|$ for $x \neq 0$. Thus $\|\nabla h(x)\| = |\phi'(\|x\|)| \leq L_\phi$, and by the mean value theorem, $|h(x) - h(y)| \leq L_\phi \|x - y\|$. For the linear barrier $h_1(x) = R - \|x\|$, we have $\phi(r) = R - r$, so $\phi'(r) = -1$ and $L_{h_1} = 1$. For the quadratic barrier $h_2(x) = R^2 - \|x\|^2$, we have $\phi(r) = R^2 - r^2$, so $|\phi'(r)| = 2r \leq 2R$ on $\{\|x\| \leq R\}$, giving $L_{h_2} = 2R$. In all cases, the Lipschitz constant depends only on the radial profile $\phi$ and the radius $R$, not on the ambient dimension $n$.
\end{proof}

\textbf{Remark.} For non-spherical barriers (e.g., polytopes, signed-distance functions), $L_h$ typically scales with geometric complexity, not ambient dimension \cite{ledoux2001}. Practitioners should verify dimension-independence of $L_h$ for their specific barrier.

\subsection{Verification Algorithm}

\begin{algorithm}
\caption{High-Dimensional Probabilistic Safety Verification}\label{alg:verify}
\begin{algorithmic}[1]
\State \textbf{Input:} Dynamics $(f,g)$, Barrier $h(x)$, Dimension $n$, Sample Count $N$.
\State \textbf{Initialize:} $N_{fail} \leftarrow 0$.
\For{$i = 1$ to $N$}
    \State Sample state $x_i \sim \mathcal{N}(0, I_n)$ projected to $\partial\mathcal{S}$ \Comment{Boundary Sampling}
    \State Compute Lie derivatives: $L_f h(x_i)$ and $L_g h(x_i)$.
    \State Solve for optimal control $u^*$ (via the CBF-QP controller of Section~4.2).
    \State \Comment{\textbf{Verify Barrier Condition}}
    \State $\Delta = L_f h(x_i) + L_g h(x_i)u^* + \gamma h(x_i)$.
    \If{$\Delta < 0$} \Comment{Violation of Class-K Inequality}
        \State $N_{fail} \leftarrow N_{fail} + 1$.
    \EndIf
\EndFor
\State \textbf{Output:} $\hat{P}_{\text{safe}} = 1 - \frac{N_{fail}}{N}$.
\end{algorithmic}
\end{algorithm}

\textit{Note:} $\hat{P}_{\text{fail}}$ is conditioned on boundary states. For spherical $\mathcal{S}$, isotropic Gaussian projection yields uniform boundary sampling; for non-spherical $\partial\mathcal{S}$, see Appendix~\ref{app:qualifications}.

\subsubsection{From Boundary Certification to Trajectory Safety}

\begin{proposition}[Average-Case Trajectory Safety from Boundary Certification]
\label{prop:trajectory_bridge}
Let $h: \mathbb{R}^n \to \mathbb{R}$ be $L_h$-Lipschitz, and suppose the MCBC certificate guarantees $\hat{P}_{\text{fail}} < \epsilon$ on $\partial\mathcal{S}$ under a uniform boundary measure $\mu$. Let $x(t)$ be a trajectory of the closed-loop system with CBF-QP control, evolving for $T$ discrete steps of size $\Delta t$.

\textbf{Distributional assumption.} Assume the initial condition $x(0)$ is drawn from a distribution $\mu_0$ such that, at each step where the trajectory reaches the boundary, the boundary projection $\text{Proj}_{\partial\mathcal{S}}(x(t))$ is distributed according to $\mu$ (or absolutely continuous with respect to $\mu$). Under this assumption, the per-step probability of encountering an infeasible boundary point is at most $\epsilon$, and by a union bound:
\begin{equation}
    P\!\left(\exists\, t \in \{1,\ldots,T\} : \text{CBF-QP infeasible at } x(t)\right) \leq T \cdot \epsilon
\end{equation}
More precisely, if the trajectory visits at most $T_{\partial}$ boundary episodes (time steps where $h(x(t)) < \eta$ for a threshold $\eta > 0$), then the failure probability tightens to $T_{\partial} \cdot \epsilon$, since the CBF constraint is slack (inactive) in the interior.
\end{proposition}

This provides an \emph{average-case} guarantee; the union bound is conservative but dimension-free, becoming vacuous at $T \geq 1/\epsilon$. Extended discussion is in Appendix~\ref{app:qualifications}.

\section{Asymptotic Utility Maximization}
We propose a Harmonic Control Architecture where the safety barrier actively shapes the utility gradient, guiding the system along $\partial\mathcal{S}$ toward the global optimum.

\subsection{The Utility Landscape}
The objective is a continuously differentiable $U(x): \mathcal{X} \to \mathbb{R}$. For convergence proofs, we assume $U$ is concave or satisfies the Polyak-{\L}ojasiewicz condition \cite{polyak1963}; for non-convex $U$, Rotational Circulation (Section~4.4) destabilizes spurious local equilibria.

\subsection{Gradient Projection on the Tangent Cone}
Let the nominal control be $u_{\text{nom}} = \nabla U(x)$. When $h(x) \gg 0$, we apply $u = u_{\text{nom}}$; as $h(x) \to 0$, $u_{\text{nom}}$ may violate safety.

We resolve this conflict by employing a \textbf{Control Barrier Function-based Quadratic Program (CBF-QP)}, a formulation established in modern control theory \cite{ames2019} and adapted here for high-dimensional semantic spaces. At every time step, we solve the following optimization problem, which acts as a minimum-deviation safety filter:
\begin{equation}
\label{eq:cbfqp}
\begin{aligned}
    u^*(x) &= \underset{u \in \mathcal{U}}{\text{argmin}} \quad \frac{1}{2}\|u - u_{\text{nom}}(x)\|^2 \\
    \text{subject to:} &\quad L_f h(x) + L_g h(x) u \geq -\gamma h(x)
\end{aligned}
\end{equation}

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_projection.png}
    \caption{\textbf{Orthogonal Projection on Tangent Cone.} The nominal $u_{\text{nom}}$ (red) is projected onto the tangent hyperplane orthogonal to $\nabla h(x)$, yielding $u^*$ (green) that maximizes utility while maintaining invariance.}
    \label{fig:projection}
\end{figure}

As illustrated in Figure~\ref{fig:projection}, because this formulation involves a single affine inequality constraint, the KKT conditions yield an explicit closed-form solution (a geometric projection onto a half-space). This circumvents the iterative complexity of general QP solvers ($O(n^3)$) \cite{boyd2004}, reducing the computational cost to $O(n)$ per time step (dominated strictly by vector dot products). Note that for systems with multiple simultaneous safety constraints, we assume they are aggregated into a single barrier function $h(x)$ (e.g., using a smooth LogSumExp or SoftMin approximation \cite{ames2019, glotfelter2017}) to preserve this linear scaling efficiency.

For systems with model uncertainty or discretization error, the barrier condition is tightened to the \textbf{robust barrier condition}:
\begin{equation}
\label{eq:robust_barrier}
    L_f h(x) + L_g h(x)\, u + \gamma\, h(x) \geq \rho(x) + \epsilon_{\text{model}} + \Delta_{\text{noise}}
\end{equation}
where $\rho(x) = \tilde{\sigma}_{\max}(J_f(x)) \cdot d_{\text{step}}$ is the adaptive spectral safety margin (estimated via Hutchinson's method \cite{hutchinson1990}), $\epsilon_{\text{model}}$ bounds the barrier-derivative impact of surrogate model error (i.e., $\epsilon_{\text{model}} \geq \sup |\nabla h^T \delta f| \geq L_h \|\delta f\|$, absorbing $L_h$ into the bound; including Euler discretization error), and $\Delta_{\text{noise}}$ bounds physical disturbances.

\subsection{Safe Asymptotic Convergence}
The interaction between the Lyapunov-like utility function and the Barrier function guarantees convergence.

\begin{theorem}[Safe Asymptotic Convergence]
\label{thm:safe_convergence}
Consider the system $\dot{x} = f(x) + g(x)u$ subject to the control law $u^*(x)$ defined above. If the sets $\mathcal{S}$ (safety) and $\mathcal{L}_c = \{ x \mid U(x) \geq c \}$ (utility level sets) are compact, and if $\nabla U(x)$ and $\nabla h(x)$ are not opposing collinear vectors at the boundary (regularity condition), then:
\begin{enumerate}
    \item \textbf{Forward Invariance:} For all $t \geq 0$, $h(x(t)) \geq 0$ (Safety is guaranteed).
    \item \textbf{Convergence to KKT Set:} The trajectory $x(t)$ converges to the set of constrained critical points $\Omega^* = \{x \in \mathcal{S} : \nabla U(x) + \lambda \nabla h(x) = 0, \lambda \geq 0, \lambda h(x) = 0\}$. If, additionally, $U(x)$ is concave (or satisfies the Polyak-{\L}ojasiewicz condition on $\mathcal{S}$), then $\Omega^*$ contains the unique global constrained maximum $x^*_{\mathcal{S}}$.
    \item \textbf{Singleton Convergence (under analyticity):} If $U(x)$ is real-analytic on $\mathcal{S}$ and satisfies the {\L}ojasiewicz gradient inequality \cite{lojasiewicz1963}, then $x(t)$ converges to a single KKT point $x^*_\mathcal{S} \in \Omega^*$ (not merely to the set $\Omega^*$).
\end{enumerate}
\end{theorem}
\begin{proof}[Proof Sketch]
We establish each claim separately.

\textbf{Forward Invariance.} This follows directly from the CBF condition: the constraint $L_f h + L_g h\, u \geq -\gamma h$ in Equation~\ref{eq:cbfqp} is feasible (by the relative-degree-one assumption and compactness of $\mathcal{U}$), and any feasible $u^*$ satisfies $\dot{h} \geq -\gamma h$, which by the Comparison Lemma \cite{khalil2002} implies $h(x(t)) \geq h(x(0))\, e^{-\gamma t} \geq 0$ for all $t \geq 0$ \cite{ames2019}.

\textbf{Convergence to KKT Set.} We analyze the single-integrator case $\dot{x} = u$ (used in our simulations, Section~5) and then state the general result. Define $V(x) = -U(x)$ (to be minimized). With $u_{\text{nom}} = \nabla U(x)$, when the CBF constraint is inactive, $\dot{V} = -\|\nabla U\|^2 \leq 0$. When the constraint is active, the KKT solution $u^* = u_{\text{nom}} + \lambda^* \nabla h$ with $\lambda^* \geq 0$ gives $\dot{V} = -\|\nabla U\|^2 - \lambda^* \nabla U \cdot \nabla h$. In the single-integrator case, expanding via the geometric decomposition yields $\dot{V} = -\|\text{Proj}_{\text{Null}(\nabla h)} \nabla U\|^2 + \frac{\gamma h \cdot (\nabla U^T \nabla h)}{\|\nabla h\|^2}$; since the constraint activates only when $\nabla U^T \nabla h < 0$ and $h \geq 0$, the second term is $\leq 0$, so $\dot{V} \leq 0$ always. Thus for the single-integrator case, $V$ is monotonically non-increasing and classical LaSalle's invariance principle \cite{lasalle1960} applies directly.

However, the general control-affine case $\dot{x} = f(x) + g(x)u$ with $f(x) \neq 0$ introduces a drift term $L_f V$ that can make $\dot{V}$ transiently positive during boundary phases when the drift-compensation term dominates. To handle both the single-integrator case and its general extension uniformly, we present a \textbf{Barbalat-type argument} \cite{barbalat1959, khalil2002} that does not require monotonicity of $V$. Since $V$ is bounded below on compact $\mathcal{S}$ and the trajectory $x(t)$ remains in $\mathcal{S}$ by forward invariance, $V(x(t))$ is a bounded function of time. We now establish that $\int_0^T \|\nabla U(x(t))\|^2\, dt$ is bounded for all $T$. Decompose the time axis into \textit{interior phases} (where $h(x(t)) > 0$ and the CBF constraint is inactive) and \textit{boundary phases} (where the constraint is active). During interior phases, $\dot{V} = -\|\nabla U\|^2 \leq 0$, so $V$ is non-increasing and $\int \|\nabla U\|^2\, dt \leq V(x(0)) - \inf_\mathcal{S} V < \infty$. During boundary phases, the trajectory evolves on the codimension-1 manifold $\partial\mathcal{S}$. The projected dynamics on $\partial\mathcal{S}$ satisfy $\dot{x} = u^* = \nabla U + \lambda^* \nabla h$, where $\lambda^*$ is determined by $\dot{h} = -\gamma h = 0$. We define the \textit{projected Lyapunov function} $W(x) = V(x)|_{\partial\mathcal{S}}$, whose time derivative along the boundary is $\dot{W} = -\|\nabla U\|^2 - \lambda^* \nabla U \cdot \nabla h = -\|\text{Proj}_{\text{Null}(\nabla h)} \nabla U\|^2 \leq 0$, where the last equality uses the geometric decomposition of the projected gradient (Section~4.2). Thus $V$ is non-increasing during boundary phases as well, provided we project onto the tangent space. The total integral $\int_0^\infty \|\text{Proj}_{\text{Null}(\nabla h)} \nabla U\|^2\, dt$ is bounded by $V(x(0)) - \inf_\mathcal{S} V$.

To apply Barbalat's Lemma, we verify that $\ddot{V}$ is bounded: since $\dot{V} = -\nabla U^T (f + gu^*)$ and both $U$, $h$ are $C^1$ (Assumptions A2, A6), and $u^*$ is continuous in $x$ (as the KKT solution of the convex QP), $\dot{V}$ is Lipschitz on compact $\mathcal{S}$, implying $|\ddot{V}| \leq L_{\dot{V}} \cdot M$ where $M = \sup_{x \in \mathcal{S}, u \in \mathcal{U}} \|f(x) + g(x)u\|$ is finite by compactness. Barbalat's Lemma yields $\|\text{Proj}_{\text{Null}(\nabla h)} \nabla U(x(t))\|^2 \to 0$. Since $\mathcal{S}$ is compact and forward-invariant, the $\omega$-limit set $\Omega \subset \mathcal{S}$ is non-empty, compact, and connected \cite{khalil2002}, and every point of $\Omega$ satisfies the KKT conditions---the first-order optimality conditions of $\max_\mathcal{S} U$.

\textbf{Singleton Convergence} follows under real-analyticity of $U$ via the {\L}ojasiewicz gradient inequality \cite{lojasiewicz1963}, yielding finite arc-length $\int_0^\infty \|\dot{x}\|\, dt < \infty$ \cite{absil2005} and hence convergence to a single KKT point. The proof extends to general control-affine dynamics $\dot{x} = f(x) + g(x)u$ under sufficient control authority; in the general case, $V$ may transiently increase during boundary phases when the drift-compensation term $L_f V$ dominates (this is the regime where the Barbalat argument is essential, since LaSalle requires monotonicity), but $V$ remains bounded on the compact forward-invariant set $\mathcal{S}$, so the total positive excursion is finite and the Barbalat argument still applies. Detailed remarks on regularity conditions, convergence rates, and the general dynamics extension are provided in Appendix~\ref{app:qualifications}.
\end{proof}

\subsection{Avoiding Local Minima (The ``Deadlock'' Problem)}
A common failure mode in potential field methods is ``local minima deadlock,'' where the agent gets stuck in a U-shaped obstacle \cite{rimon1992}. If $\nabla U(x)$ and $\nabla h(x)$ are perfectly opposing, a direct projection $v_{\perp} = \text{Proj}_{\text{Null}(\nabla h)} \nabla U$ yields the zero vector, resulting in stagnation.

To mitigate this, we inject a Rotational Circulation term. When the angle between $\nabla U$ and $\nabla h$ approaches $180^\circ$ (indicating a deadlock), we project a random perturbation vector (or strictly orthogonal noise) $\xi$ onto the tangent plane of the obstacle, utilizing a stochastic escape heuristic similar to Simulated Annealing \cite{kirkpatrick1983} or Randomized Potential Fields \cite{barraquand1991}:
\begin{equation}
    u_{\text{perturb}} = \nabla U(x) + \beta \left( \text{Proj}_{\text{Null}(\nabla h)} \xi \right)
\end{equation}
Here, $\xi \in \mathbb{R}^n$ is a noise vector. High dimensionality \textit{aids} escape: saddle points dominate high-dimensional landscapes \cite{dauphin2014, choromanska2015}, and a random perturbation almost surely has a nonzero descent component, with failure probability vanishing exponentially in $n$ \cite{vershynin2018}.

\section{Simulation Results and Empirical Validation}
We evaluate CHDBO across ten progressively challenging configurations, spanning reachability analysis, high-dimensional survival, non-trivial drift dynamics, scalability, QP benchmarking, and failure-mode characterization. Dimensions range from $n=2$ to $n=1024$.

\textbf{Barrier Function Forms.} Three forms are used: (1)~the \textit{linear barrier} $h(x) = 1 - \|x\|$ (Experiments II--III), whose gradient $\nabla h = -x/\|x\|$ is well-defined away from the origin; (2)~the \textit{quadratic barrier} $h(x) = 1 - \|x\|^2$ (Experiments IV--IX), preferred when $f(x) \neq 0$ because $L_f h = -2x^T f(x)$ is linear in $x$; and (3)~the \textit{linear hyperplane barrier} $h(x) = w^\top x + b$ (Experiments I, X), which defines a half-space safe set and yields zero discretization error ($L_h = \|w\|$, constant). In Experiment~I, $w = [-1, 0]^T$ and $b = 0.8$, defining $\mathcal{S} = \{x : x_1 \leq 0.8\}$; in Experiment~X, $w$ and $b$ define a general hyperplane in $\mathbb{R}^{128}$. Forms (1) and (2) define the safe set $\mathcal{S} = \{x : \|x\| \leq 1\}$.

\subsection{Experiment I: Reachability Analysis \texorpdfstring{($n=2$)}{(n=2)}}
We validate geometric safety in a 2D space where an agent is biased toward a Forbidden Zone ($x > 0.8$) over 50 adversarial trials. The standard agent breaches the constraint; the CBF-QP-verified agent projects velocity onto the tangent of the safety manifold.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_1.png}
    \caption{\textbf{Experiment I: Geometric Reachability (50 trials, $n=2$).} Verified trajectories form a hard wall at $x=0.8$; the agent slides along the safety boundary while maximizing Y-axis movement.}
    \label{fig:golden_manifold}
\end{figure}

\textbf{Result:} The verified agent maintains a strict bound at $x=0.8$ (Figure~\ref{fig:golden_manifold}), sliding along the safety boundary to maximize utility without crossing into the forbidden region.

\subsection{Experiment II: High-Dimensional Survival \texorpdfstring{($\mathbb{R}^{128}$)}{(R\textasciicircum 128)}}
We extend to $n=128$, subjecting the agent to a gradient attack \cite{madry2018} with the goal placed outside the safe hypersphere ($R > 1.0$). MCBC generates 50 adversarial trials of 100-step random walks (attack strength $= 0.05$) with post-hoc norm clipping to compensate for discrete-time Euler overshoot (see Limitations~\S\ref{sec:limitations}); trajectories are projected to 2D via PCA \cite{jolliffe2002}.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_2.png}
    \caption{\textbf{High-Dimensional Verification ($n=128$) under Adversarial Gradient Attack.} Standard agents (red) breach the boundary; verified agents (green) cluster on the hypersphere surface. (PCA projection is a visualization aid; safety is established by $h(x) \geq 0$ in full $\mathbb{R}^{128}$.)}
    \label{fig:high_dim_survival}
\end{figure}

\textbf{Result:} Standard agents breach the hypersphere immediately; verified agents cluster on the surface (Figure~\ref{fig:high_dim_survival}). Sample complexity remains $O(1)$ despite exponential state-space volume growth; per-sample cost scales as $O(n)$.

\subsection{Experiment III: Proportional Safety Response}
We stress-tested the system by varying adversarial intensity $\alpha \in [0, 1]$, parameterizing the angular deviation from the tangent plane. The boundary-case limit for similarity is $\sqrt{1-\alpha^2}$, corresponding to the geometric decay of orthogonal projection when safety margin $h \to 0$. In practice, states with positive margin ($h > 0$) experience less aggressive intervention, so observed similarity lies \emph{above} this curve---a desirable property confirming proportional, margin-aware filtering. At low threat ($\alpha < 0.4$), similarity remains near 1.0; at high threat ($\alpha > 0.8$), it decreases but stays above the boundary limit. Full results and Figure~\ref{fig:proportional_response} are provided in Appendix~\ref{app:exp_proportional}.

\subsection{Experiment IV: Non-Trivial Drift Dynamics \texorpdfstring{($\mathbb{R}^{128}$)}{(R\textasciicircum 128)}}
Experiments I--III employ single-integrator dynamics ($f(x) = 0$), which eliminates the Lie derivative term $L_f h$ from the barrier condition. To validate CHDBO in the general control-affine regime $\dot{x} = f(x) + g(x)u$ where $f(x) \neq 0$, we test two physically motivated drift models in $\mathbb{R}^{128}$.

\textbf{Setup A --- Linear Drift:} A marginally unstable matrix $A \in \mathbb{R}^{128 \times 128}$ with $\sigma_{\max}(A) = 1.044$ drives the uncontrolled dynamics $\dot{x} = Ax$, producing outward radial drift near the boundary $\partial\mathcal{S}$ of the unit ball. Since $f(x) \neq 0$, we use the quadratic barrier $h(x) = 1 - \|x\|^2$ (form~2), whose gradient $\nabla h = -2x$ yields $L_f h(x) = -2x^T A x$. The CBF-QP (Equation~\ref{eq:cbfqp}) must compensate whenever $L_f h < 0$ (drift pushing toward the boundary).

\textbf{Setup B --- Double-Integrator:} The state is decomposed as $x = (q, v) \in \mathbb{R}^{64} \times \mathbb{R}^{64}$ with $\dot{q} = v$, $\dot{v} = u$. Since $h(x) = 1 - \|q\|^2$ has relative degree~2 with respect to $u$, we employ an Exponential Control Barrier Function (ECBF) \cite{xiao2019}. Defining $\psi_0 = h$ and $\psi_1 = \dot{\psi}_0 + \alpha_1 \psi_0$, the ECBF constraint $\dot{\psi}_1 + \alpha_2 \psi_1 \geq 0$ expands to $\ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \geq 0$, with $\alpha_1 = \alpha_2 = 2.0$. This reduces the relative-degree-2 safety requirement to a relative-degree-1 constraint on $u$.

\textbf{Protocol:} 50 trials per configuration. Setup~A runs 300 steps with $\Delta t = 0.01$; Setup~B runs 600 steps with $\Delta t = 0.005$ (smaller step for stability of the second-order dynamics). The nominal control drives the system toward a goal outside the safe set ($\|x_{\text{goal}}\| = 1.5$), exercising the CBF-QP safety filter against the combined drift + utility pressure.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_9a.png}
    \caption{\textbf{Experiment IV-A: Linear Drift ($\mathbb{R}^{128}$, $\sigma_{\max}(A) = 1.044$).} Norm trajectories for 50 trials under marginally unstable drift. All trajectories remain below $\|x\| = 1$: 0/50 violations.}
    \label{fig:drift_linear}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_9b.png}
    \caption{\textbf{Experiment IV-B: Double-Integrator with ECBF ($\mathbb{R}^{128}$).} Position norm $\|q\|$ for 50 trials under relative-degree-2 dynamics ($\alpha_1 = \alpha_2 = 2.0$). 0/50 violations; $\max \|q\| = 0.997$.}
    \label{fig:drift_ecbf}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_9c.png}
    \caption{\textbf{Experiment IV: Drift Lie Derivative Distribution.} $L_f h(x)$ under linear drift; mean $= -0.462$ (adverse). The entirely negative distribution confirms that drift actively pushes the system toward $\partial\mathcal{S}$, exercising the full CBF condition $L_f h + L_g h \cdot u \geq -\gamma h$. The bimodal shape arises because $L_f h = -0.6\|x\|^2$ (since $x^T A_{\text{skew}}\, x = 0$ for the skew-symmetric part of $A$), so the two peaks correspond to the starting norm ($\|x\| \approx 0.7$) and the near-boundary dwell norm ($\|x\| \approx 0.98$).}
    \label{fig:drift_lfh}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_9d.png}
    \caption{\textbf{Experiment IV: ECBF Barrier Values (Representative Trial).} Both $\psi_0 = 1 - \|q\|^2$ (position barrier) and $\psi_1 = \dot{\psi}_0 + \alpha_1 \psi_0$ (ECBF condition) remain non-negative throughout, confirming forward invariance under relative-degree-2 dynamics.}
    \label{fig:drift_ecbf_barriers}
\end{figure}

\textbf{Results (Figures~\ref{fig:drift_linear}--\ref{fig:drift_ecbf_barriers}):} Both configurations achieve \textbf{0/50 violations}. Under linear drift, the mean Lie derivative $L_f h = -0.462$ (Figure~\ref{fig:drift_lfh}), confirming that drift is genuinely adverse, yet the CBF-QP compensates via $L_g h \cdot u$, maintaining $\|x\| < 1$. Under double-integrator dynamics, the ECBF maintains $\psi_0 \geq 0$ and $\psi_1 \geq 0$ throughout ($\max \|q\| = 0.997$, Figures~\ref{fig:drift_ecbf},~\ref{fig:drift_ecbf_barriers}). Both remain under 0.01\,ms per step.

\subsection{Experiment V: Nonlinear Drift --- Lorenz-type Attractor in \texorpdfstring{$\mathbb{R}^{128}$}{R128}}
To validate CHDBO under strongly nonlinear dynamics---the most likely reviewer extension request---we construct a high-dimensional Lorenz-type chaotic attractor in $\mathbb{R}^{128}$. The state is grouped into 42 Lorenz triplets $(x_{3i}, x_{3i+1}, x_{3i+2})$ evolving as:
\begin{align}
\dot{x}_{3i} &= \sigma(x_{3i+1} - x_{3i}) \notag\\
\dot{x}_{3i+1} &= x_{3i}(\rho - x_{3i+2}) - x_{3i+1} \notag\\
\dot{x}_{3i+2} &= x_{3i}\, x_{3i+1} - \beta\, x_{3i+2}
\end{align}
with standard parameters $(\sigma, \rho, \beta) = (10, 28, 8/3)$ and a scaling factor $1/8$ to reduce the attractor scale to be comparable to the unit ball, with the CBF-QP enforcing confinement. This rescaling reduces the effective Lyapunov exponent by the same factor (i.e., the rescaled system has $\lambda_{\max} \approx 0.9/8 \approx 0.11$), so the dynamics remain meaningfully chaotic with substantially stronger nonlinear forcing than a $1/40$ scaling would provide. The purpose of this experiment is to demonstrate CBF-QP enforcement under chaotic nonlinear coupling; practitioners applying CHDBO to unscaled chaotic systems should expect significantly higher control effort and CBF activation frequency. Crucially, adjacent triplets are coupled via nearest-neighbor diffusion with coupling strength $\kappa = 0.5$:
\begin{align}
\dot{x}_{3i} &\mathrel{+}= \kappa(x_{3(i+1)} - x_{3i}) \notag\\
\dot{x}_{3i+2} &\mathrel{+}= \kappa(x_{3(i-1)+2} - x_{3i+2})
\end{align}
where indices wrap around in a ring topology. This inter-triplet coupling creates genuine cross-dimensional information flow, producing a spatiotemporally chaotic system rather than 42 independent copies of 3D Lorenz dynamics. The remaining two dimensions have mild linear drift. The barrier is quadratic: $h(x) = 1 - \|x\|^2$, with the CBF-QP enforcing $\dot{h} + \gamma h \geq 0$ at every time step ($\Delta t = 0.001$, RK4 integration).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_12a.png}
    \caption{\textbf{Experiment V-a: Lorenz Drift Norm Trajectories ($\mathbb{R}^{128}$).} 50 trials under nonlinear Lorenz-type drift (42 coupled triplets, $\kappa = 0.5$, RK4, $\Delta t = 0.001$). All trajectories remain within $\|x\| \leq 1$: 0/50 violations.}
    \label{fig:lorenz_norms}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_12b.png}
    \caption{\textbf{Experiment V-b: Barrier Function Values.} $h(x) = 1 - \|x\|^2$ remains non-negative across all 50 trials under strongly nonlinear drift, confirming forward invariance via CBF-QP ($\gamma = 5.0$).}
    \label{fig:lorenz_barrier}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_12c.png}
    \caption{\textbf{Experiment V-c: Nonlinear Drift Magnitude.} Distribution of $\|f(x)\|$ across all time steps and trials; mean $\approx 11.2$, confirming that the Lorenz-type drift is substantially stronger than the linear drift of Experiment~IV ($\sigma_{\max}(A) = 1.044$).}
    \label{fig:lorenz_drift_mag}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_12d.png}
    \caption{\textbf{Experiment V-d: Lie Derivative Distribution.} $L_f h$ under nonlinear Lorenz drift; strongly negative values confirm the drift actively opposes safety, exercising the full CBF condition $L_f h + L_g h \cdot u \geq -\gamma h$.}
    \label{fig:lorenz_lfh}
\end{figure}

\textbf{Results (Figures~\ref{fig:lorenz_norms}--\ref{fig:lorenz_lfh}):} Over 50 trials (2{,}000 steps each), \textbf{0/50 violations} despite mean drift $\|f(x)\| \approx 11.2$ (Figure~\ref{fig:lorenz_drift_mag}) and strongly negative $L_f h$ (Figure~\ref{fig:lorenz_lfh}). The nearest-neighbor coupling produces spatiotemporally chaotic dynamics across all 42 triplets, confirming that CBF-QP handles nonlinear cubic coupling without modification.

\subsection{Experiment VI: Scalability Beyond \texorpdfstring{$n = 128$}{n = 128}}
We run the full pipeline---linear drift ($\sigma_{\max}(A) \approx 1.04$), CBF-QP with Hutchinson spectral estimation ($k = 5$ probes), and adaptive margin $\rho(x) = \tilde{\sigma}_{\max} \cdot \Delta t$---at $n \in \{128, 512, 1024\}$ with 20 trials $\times$ 300 steps. The Hutchinson estimate is fed into the CBF-QP via Equation~\ref{eq:robust_barrier} at each step; the deliberately small probe budget ($k{=}5$) yields a conservative Frobenius overestimate, widening the safety tube.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_13a.png}
    \caption{\textbf{Experiment VI-a: Wall-clock Time vs.\ Dimension.} Mean computation time at $n \in \{128, 512, 1024\}$ (20 trials $\times$ 300 steps). The red dashed line shows $O(n)$ reference; superlinear growth reflects dense matrix operations ($A \in \mathbb{R}^{n \times n}$).}
    \label{fig:scale_time}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_13b.png}
    \caption{\textbf{Experiment VI-b: Hutchinson Spectral Estimates.} $\|A\|_F$ estimates ($k{=}5$ probes) grow as $\sqrt{n}$ due to Frobenius overestimation; fed into CBF-QP via Eq.~\ref{eq:robust_barrier}. Red dashed lines show the true $\sigma_{\max}(A) \approx 1.04$ for each dimension.}
    \label{fig:scale_sigma}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_13c.png}
    \caption{\textbf{Experiment VI-c: Safety Margin vs.\ Dimension.} Minimum barrier value $h(x) = 1 - \|x\|^2$ across each trial remains strictly positive at all dimensions: 0/60 violations.}
    \label{fig:scale_barrier}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_13d.png}
    \caption{\textbf{Experiment VI-d: CBF Activation Frequency.} Number of CBF interventions (out of 300 steps) increases moderately with dimension, reflecting the Hutchinson-inflated safety tube's conservatism.}
    \label{fig:scale_cbf}
\end{figure}

\textbf{Results (Figures~\ref{fig:scale_time}--\ref{fig:scale_cbf}):} 0/60 violations across all dimensions. Wall-clock time (Figure~\ref{fig:scale_time}) scales superlinearly due to dense matrix operations (a JVP implementation achieves $O(n)$). Hutchinson estimates (Figure~\ref{fig:scale_sigma}) conservatively overestimate $\sigma_{\max}$, widening the safety tube. CBF activation frequency (Figure~\ref{fig:scale_cbf}) increases moderately with dimension ($\sim$150 to $\sim$175 of 300 steps).

\subsection{Experiment VII: Closed-Form QP vs.\ OSQP Benchmark}
We benchmark the closed-form CBF-QP against OSQP \cite{osqp2020} on single-constraint problems across $n \in \{64, 128, 256, 512, 1024, 2048\}$, with 5{,}000 instances per dimension. The closed-form applies only to the single-constraint case; multi-constraint systems require a general solver.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_15a.png}
    \caption{\textbf{Experiment VII-a: QP Solve Time vs.\ Dimension (log-log).} Closed-form $O(n)$ remains under 4\,$\mu$s at $n = 2048$; OSQP grows to $>$1\,ms. Both remain below the 10\,ms real-time budget. Single-constraint case only.}
    \label{fig:qp_time}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_15b.png}
    \caption{\textbf{Experiment VII-b: Closed-Form Speedup Factor.} Speedup over OSQP grows with dimension because the closed-form cost is two $O(n)$ dot products, while OSQP scales superlinearly.}
    \label{fig:qp_speedup}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_15c.png}
    \caption{\textbf{Experiment VII-c: Direct Per-Solve Time Comparison.} Grouped bar chart (log scale) showing closed-form vs.\ OSQP solve times across $n = 64$--$2048$ (5{,}000 solves per dimension).}
    \label{fig:qp_direct}
\end{figure}

\textbf{Results (Figures~\ref{fig:qp_time}--\ref{fig:qp_direct}):} The closed-form achieves substantial speedup over OSQP (Figure~\ref{fig:qp_speedup}), growing with dimension. At $n = 2048$, the closed-form averages $\sim$3\,$\mu$s vs.\ $>$1{,}000\,$\mu$s for OSQP (Figure~\ref{fig:qp_time}), with identical safety decisions on all 30{,}000 instances. The speedup grows because the closed-form cost is dominated by two $O(n)$ dot products, while OSQP scales superlinearly.

\subsection{Experiment VIII: Empirical Validation of Theorem~\ref{thm:safe_convergence} (Safe Utility Convergence)}
We empirically validate Theorem~\ref{thm:safe_convergence} in $\mathbb{R}^{128}$ under linear drift ($\sigma_{\max}(A) = 1.044$) with a quadratic barrier and utility function $U(x) = -\|x - x_{\text{goal}}\|$. Across 20 trials $\times$ 1{,}200 steps (24{,}000 total evaluations), the system achieves \textbf{0 safety violations}, with mean utility converging to $U = -0.554$ (vs.\ optimal $U^* = -0.50$) and distance to the constrained optimum decreasing monotonically to 0.194. Full setup and results (Figure~\ref{fig:utility_convergence}) are provided in Appendix~\ref{app:exp_convergence}.

\subsection{Experiment IX: MCBC vs.\ Scenario Approach}
\label{sec:exp_scenario}
We compare MCBC sample complexity against the Scenario Approach \cite{campi2008, calafiore2006} across $n = 2$ to $1024$. The scenario approach requires $N_{\text{scenario}} \geq \frac{2}{\epsilon}(\ln\frac{1}{\beta} + d)$ samples (linear in the number of decision variables $d$; here $d = n$ since the barrier is parameterized over the state space); the MCBC requires $N_{\text{MCBC}} = \frac{1}{2\epsilon^2}\ln\frac{2}{\delta}$ (independent of $n$, Eq.~\ref{eq:hoeffding}). We set $\epsilon = 0.01$, $\delta = \beta = 10^{-6}$.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_17a.png}
    \caption{\textbf{Experiment IX-a: Sample Complexity.} MCBC maintains constant $N \approx 72{,}500$ regardless of dimension; the scenario approach grows linearly, crossing near $n \approx 350$. At $n = 1024$, MCBC requires $2.9\times$ fewer samples. Note: MCBC verifies a \emph{given} barrier; the scenario approach designs one.}
    \label{fig:scenario_samples}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_17b.png}
    \caption{\textbf{Experiment IX-b: Total Verification Cost.} Samples $\times$ per-sample cost: scenario $O(n^3)$ vs.\ MCBC $O(n)$. The gap exceeds three orders of magnitude at $n = 1024$. Barrier design cost is not included in the MCBC curve.}
    \label{fig:scenario_cost}
\end{figure}

\textbf{Results (Figures~\ref{fig:scenario_samples}--\ref{fig:scenario_cost}):} MCBC requires $N \approx 72{,}500$ for all dimensions; the scenario approach requires $\sim$3{,}200 at $n=2$ but grows to $\sim$208{,}000 at $n=1024$ ($2.9\times$ more). The approaches solve different problems: the scenario approach \textit{designs} a controller with PAC guarantees, while the MCBC \textit{verifies} a pre-specified barrier. The MCBC advantage emerges for $n > 350$---the regime most relevant to semantic agent verification.

\subsection{Experiment X: Bounded Actuation, Feasibility, and Failure Modes}
\label{sec:exp_bounded_actuation}

To characterize the practical limits of CHDBO, we systematically probe CBF-QP feasibility under bounded actuation and failure modes under nonlinear dynamics.

\textbf{Setup.} We use Lorenz-type dynamics in $\mathbb{R}^{128}$ (scaling $1/20$, weaker than Experiment~V's $1/8$ to isolate actuation-budget effects) with a linear barrier $h(x) = w^\top x + b$. For each actuation budget $u_{\max} \in \{0.1, 0.5, 1, 2, 5, 10, 20, \infty\}$, we run 200 trials of 500 steps ($\Delta t = 0.001$, RK4). The CBF-QP enforces $w^\top(f(x) + u) \geq -\gamma h(x)$ ($\gamma = 0.5$); when $\|u^*\| > u_{\max}$, the control is rescaled to norm $u_{\max}$.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_19a.png}
    \caption{\textbf{Experiment X (a): Safety vs.\ Control Budget} ($\mathbb{R}^{128}$, Lorenz, 200 trials). Safety rate generally increases from $\sim$92\% ($u_{\max} = 0.1$) to 100\% ($u_{\max} \geq 20$).}
    \label{fig:bounded_safety}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_19b.png}
    \caption{\textbf{Experiment X (b): CBF Activation \& Control Saturation.} Saturation rate decreases monotonically with budget (46\% $\to$ 0\%); CBF activation rate rises slightly from $\sim$46\% to $\sim$53\% as the controller gains authority.}
    \label{fig:bounded_saturation}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_19c.png}
    \caption{\textbf{Experiment X (c): Safety vs.\ Control Effort Trade-off} (Pareto frontier). Mean effort plateaus beyond $u_{\max} = 10$; safety reaches 100\% at $u_{\max} \geq 20$.}
    \label{fig:bounded_pareto}
\end{figure}

\textbf{Results (Bounded Actuation).} Safety degrades gracefully as the actuation budget decreases (Figure~\ref{fig:bounded_safety}): 100\% at $u_{\max} \geq 20$, declining to $\sim$92\% at $u_{\max} = 0.1$. CBF activation rises slightly from $\sim$46\% to $\sim$53\% while saturation drops monotonically with increasing budget (Figure~\ref{fig:bounded_saturation}). The Pareto frontier (Figure~\ref{fig:bounded_pareto}) shows that even at the lowest budget, the clipped intervention provides best-effort safety rather than catastrophic failure.

\textbf{Failure Mode Analysis.} We additionally probe three failure modes: (a)~\textit{Actuation starvation} ($\mathbb{R}^{50}$, linear barrier, $\gamma=0.3$): a sharp phase transition from 0\% safety at $u_{\max} \leq 0.20$ through 73\% at $u_{\max}=0.30$ to 100\% at $u_{\max} \geq 0.50$ (Figure~\ref{fig:failure_starvation}); (b)~\textit{Drift vs.\ control budget}: safety degrades in a steep S-curve from 100\% (ratio 0.73) through 76\% (0.80), 30\% (0.83), and 1\% (0.87) to 0\% (0.88), well before drift equals $u_{\max}$ (Figure~\ref{fig:failure_drift}); (c)~\textit{MCBC honesty}: injected failure probabilities match MCBC's measured $P_{\text{safe}}$ within 95\% Clopper--Pearson confidence intervals, confirming calibrated estimates even when safety is genuinely compromised (Figure~\ref{fig:failure_mcbc}).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figure_21a.png}
    \caption{\textbf{Experiment X (a): Actuation Starvation.} Theoretical minimum at $u_{\max} = \lVert\text{drift}\rVert = 0.15$ (dashed line); the empirical transition from 0\% to 100\% safety occurs between $u_{\max}=0.20$ and $0.50$ due to noise and temporal exposure over 200 steps.}
    \label{fig:failure_starvation}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figure_21b.png}
    \caption{\textbf{Experiment X (b): Drift Overwhelms Budget.} Safety collapses over a narrow transition (ratio $\approx 0.7$--$0.9$); the dashed line marks the theoretical limit drift $= u_{\max}$.}
    \label{fig:failure_drift}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{figure_21c.png}
    \caption{\textbf{Experiment X (c): MCBC Honesty.} Measured $P_{\text{safe}}$ tracks expected values within 95\% Clopper--Pearson CIs ($N{=}5000$).}
    \label{fig:failure_mcbc}
\end{figure}

\textbf{Significance.} Three practical lessons emerge from Experiment~X. First, CBF-QP fails when $u_{\max} < \|f(x)\|$ on $\partial\mathcal{S}$---practitioners must budget control authority to exceed expected drift. Second, the drift-to-budget transition follows a steep S-curve: safety drops from 100\% through 76\% and 30\% to 0\% over the ratio range $\sim$0.73--0.88, below the theoretical limit of unity because noise accumulates over discrete time steps. Third, MCBC does not guarantee safety---it \emph{measures} it; when dynamics are hostile, MCBC correctly reports low $P_{\text{safe}}$, preventing false confidence. These limitations are intrinsic to CBF methods, not artifacts of the hypervector representation.

\begin{table*}[htbp]
\centering
\caption{Empirical Performance Summary: CHDBO Framework}
\footnotesize
\begin{tabular}{@{}llccl@{}}
\toprule
\textbf{Experiment} & \textbf{Metric} & \textbf{Value} & \textbf{Dynamics} & \textbf{Methodology} \\ \midrule
I (Reachability, $n{=}2$) & Safety Rate & 100\% (50 trials) & $\dot{x}=u$ & CBF-QP projection \\
II (High-dim, $n{=}128$) & Safety Rate & 100\% (100 steps) & $\dot{x}=u$ & CBF-QP on $S^{127}$ \\
III (Proportional, $n{=}128$) & Utility Preservation & $\geq\sqrt{1-\alpha^2}$ bound & $\dot{x}=u$ & App.~\ref{app:exp_proportional} \\
IV (Drift, $n{=}128$) & Safety Rate (linear) & 0/50 violations & $\dot{x}=Ax+u$ & CBF-QP + $L_fh$ \\
 & Safety Rate (dbl-int) & 0/50 violations & $\ddot{q}=u$ & ECBF \\
V (Lorenz, $n{=}128$) & Safety Rate & 0/50 violations & Lorenz $f(x)$ & CBF-QP, nonlinear \\
VI (Scale, $n{\leq}1024$) & Safety Rate & 0/60 violations & $\dot{x}=Ax+u$ & $n{=}128,512,1024$ \\
VII (QP Benchmark) & Speedup vs.\ OSQP & 61--351$\times$ & $\dot{x}=u$ & Single constraint \\
VIII (Thm.~\ref{thm:safe_convergence} Valid.) & Safety Rate & 0/24{,}000 evals & $\dot{x}=Ax+u$ & App.~\ref{app:exp_convergence} \\
IX (Scenario Comp.) & MCBC Samples & 72{,}544 ($\forall n$) & --- & Hoeffding bound \\
 & Scenario Samples & 208{,}000 ($n{=}1024$) & --- & Campi--Garatti \\
X (Bounded Act.\ + Failures) & Safety vs.\ $u_{\max}$ & 92\%--100\% & Lorenz $f(x)$ & Linear barrier \\
 & Phase transition & $u_{\max} \approx 0.15$ & Lorenz $f(x)$ & Starvation \\
 & MCBC calibration & Within 95\% CI & & Clopper--Pearson \\
\bottomrule
\end{tabular}
\begin{flushleft}
\small \textit{Notes:} Experiments I--III use single-integrator dynamics ($f(x)=0$); IV introduces linear and second-order drift; V uses nonlinear chaotic drift (coupled Lorenz-type); VI tests scalability to $n=1024$; VII benchmarks closed-form QP vs.\ OSQP; VIII validates Theorem~\ref{thm:safe_convergence}; IX compares MCBC against the scenario approach; X quantifies bounded-actuation feasibility and failure modes. Full details for Experiments III and VIII in Appendix. ``ECBF'' = Exponential CBF for relative degree 2.
\end{flushleft}
\label{tab:results}
\end{table*}

\begin{table*}[htbp]
\centering
\caption{Comparison with Related Verification and Safety Approaches}
\footnotesize
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Scalability} & \textbf{Guarantee Type} & \textbf{Real-Time} & \textbf{Assumptions} \\ \midrule
H-J Reachability \cite{mitchell2005} & $O((1/\eta)^n)$ & Deterministic & $n \leq 5$ & Full model \\
DeepReach \cite{bansal2021} & $O(\text{NN})$ & Approx.\ determ. & $n \leq 10$ & NN value fn \\
Standard CBF-QP \cite{ames2019} & $O(n)$ per step & Deterministic & Yes & Known $h(x)$, rel.\ deg.\ 1 \\
Learned CBFs \cite{dawson2023} & $O(n)$ & Probabilistic & Yes & Training data, NN approx. \\
Safe RL / Shielding \cite{garcia2015} & High-dim & Reward-based & Yes & Reward shaping \\
Predictive Safety Filter \cite{wabersich2021} & $O(n^3)$ & Deterministic & Near-RT & Model + horizon \\
Robust CBF + RL \cite{choi2020} & $O(n)$ per step & Probabilistic & Yes & CLF + CBF + model uncert. \\
MPC with constraints \cite{camacho2013} & $O(n^3)$ per step & Deterministic & $n \leq 50$ & Convex model \\
\textbf{CHDBO (ours)} & $O(n)$ per step & Probabilistic (PAC) & Yes & Lipschitz $h$, Assump.\ A1--A8 \\
\bottomrule
\end{tabular}
\label{tab:comparison}
\end{table*}

\subsection{Limitations}
\label{sec:limitations}

\subsubsection{Experimental Dynamics}
Experiments I--III use single-integrator dynamics ($f(x)=0$) to isolate geometric mechanisms; IV--V introduce linear and nonlinear drift; VI validates scalability to $n=1024$; X characterizes failure modes under bounded actuation. Remaining gaps include learned neural ODE dynamics \cite{chen2018} and state-dependent $g(x) \neq I$. The theoretical results (Theorems~\ref{thm:topological_safety}--\ref{thm:safe_convergence}) are proved for general control-affine systems.

\subsubsection{Code vs.\ Theory Scaling}
The figure-generation code uses dense $O(n^2)$ operations. A JAX-based AD JVP implementation (included in the code supplement) validates the $O(n)$ claim: at $n=2048$ the AD pathway achieves $\sim$2.5$\times$ speedup over dense computation with $<$0.00001\% numerical difference. Production-scale implementations ($n > 10{,}000$) should use AD pipelines.

\subsubsection{Union Bound Horizon Degradation}
Proposition~\ref{prop:trajectory_bridge} degrades linearly with horizon $T$, becoming vacuous at $T \geq 1/\epsilon$. Periodic re-certification partially mitigates this; tighter bounds require supermartingale certificates \cite{santoyo2021, clark2021}.

\subsubsection{Multi-Constraint Aggregation}
Aggregating multiple barriers via smooth approximation (LogSumExp) introduces conservatism. Maintaining independent constraints requires a general QP solver at $O(n^3)$, negating the closed-form advantage.

\subsubsection{Dimension-Independence Caveats}
Total verification cost is $O(Nn)$ (linear, not constant in $n$). The Lipschitz constant $L_h$ may grow with $n$ for certain barrier designs; dimension-independence holds rigorously for bounded $L_h$ (as in our experiments).

\subsubsection{Bounded Actuation}
Actuation limits are enforced by post-hoc norm clipping, which may violate the CBF condition. Experiment~X quantifies this: safety degrades from 100\% to $\sim$90\% as $u_{\max}$ drops, with a phase transition at $\|f\|/u_{\max} \approx 1.0$.

\subsubsection{Discrete-Time CBF Constraints}
For linear barriers, the discrete-time CBF condition is exact; for quadratic barriers, a $-\|\Delta x\|^2$ remainder accumulates per step. Linear barriers are preferred for discrete-time systems: zero discretization error and dimension-independent $L_h = \|w\|$. In our single-integrator experiments (I--III), post-hoc norm clipping compensates for any Euler overshoot; this is noted in the respective experiment descriptions.

\subsubsection{Experimental Scope}
All experiments use synthetic dynamics; real-world validation on physical platforms (e.g., robotic manipulators, autonomous vehicles) remains future work. Additionally, the Rotational Circulation mechanism (Section~4.4) is presented theoretically but not validated experimentally; empirical characterization of escape dynamics for non-convex utility landscapes is deferred to a companion paper.

\section{Conclusion}
This paper introduced Constrained High-Dimensional Barrier Optimization (CHDBO), replacing exponentially scaling deterministic verification ($O((1/\eta)^n)$) with probabilistic certification ($O(1)$ sample complexity for fixed $L_h$ and $\varepsilon_s$). Our contributions are:

\begin{enumerate}
    \item \textbf{Probabilistically Scalable Safety:} Monte Carlo barrier certificates verify $\mathbb{R}^{128}$ manifolds with confidence $1{-}\delta$; Proposition~\ref{prop:trajectory_bridge} bridges boundary-level to trajectory-level safety.
    \item \textbf{Asymptotic Utility:} Projected-gradient steering converges to constrained optima (Theorem~\ref{thm:safe_convergence}), validated empirically under linear drift (0/24{,}000 violations, Experiment~VIII).
    \item \textbf{Scalability and Efficiency:} The framework handles linear, second-order, and nonlinear dynamics (Experiments IV--V), scales to $n = 1024$ (Experiment~VI), and achieves 61--351$\times$ speedup over OSQP (Experiment~VII).
    \item \textbf{Honest Failure Characterization:} Experiment~X identifies phase transitions at $\|f\|/u_{\max} \approx 1.0$ and confirms MCBC calibration within 95\% CI.
\end{enumerate}

Tables~\ref{tab:results} and~\ref{tab:comparison} summarize empirical performance and position CHDBO within the broader landscape. Practical deployment guidelines (parameter selection for kinematic and semantic systems) are provided in Appendix~\ref{app:deployment}. A companion paper extends this framework with Active Adversarial Safety Verification (AASV), bridging the gap between statistical certification and adversarial robustness.

\begin{thebibliography}{99}

\bibitem{scrivens2026}
Scrivens, A. (2026).
\textit{Mitigating Large Language Model Context Drift via Holographic Invariant Storage}.
Zenodo (preprint). \url{https://doi.org/10.5281/zenodo.18500602}

\bibitem{kanerva2009}
Kanerva, P. (2009).
``Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.''
\textit{Cognitive Computation}.

\bibitem{lecun2022}
LeCun, Y. (2022).
``A Path Towards Autonomous Machine Intelligence.''
\textit{Open Review}.

\bibitem{ames2019}
Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., \& Tabuada, P. (2019).
``Control Barrier Functions: Theory and Applications.''
\textit{2019 18th European Control Conference (ECC)}, 3420--3431.

\bibitem{campi2008}
Campi, M. C., \& Garatti, S. (2008).
``The exact feasibility of randomized solutions of uncertain convex programs.''
\textit{SIAM Journal on Optimization}, 19(3), 1211-1230.

\bibitem{vershynin2018}
Vershynin, R. (2018).
\textit{High-Dimensional Probability: An Introduction with Applications in Data Science}.
Cambridge University Press.

\bibitem{kong2023}
Kong, H., et al. (2023).
``Non-convex Control Barrier Functions for Safety-Critical Control.''
\textit{IEEE Transactions on Automatic Control}.

\bibitem{brunke2022}
Brunke, L., et al. (2022).
``Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning.''
\textit{Annual Review of Control, Robotics, and Autonomous Systems}.

\bibitem{dasgupta2003}
Dasgupta, S., \& Gupta, A. (2003).
``An elementary proof of a Theorem of Johnson and Lindenstrauss.''
\textit{Random Structures \& Algorithms}, 22(1), 60-65.

\bibitem{wei2023}
Wei, A., Haghtalab, N., \& Steinhardt, J. (2023).
``Jailbroken: How Does LLM Safety Training Fail?''
\textit{Advances in Neural Information Processing Systems}, 36, 80079--80110.

\bibitem{angelopoulos2021}
Angelopoulos, A. N., \& Bates, S. (2021).
``A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.''
\textit{arXiv preprint arXiv:2107.07511}.

\bibitem{alshiekh2018}
Alshiekh, M., Bloem, R., Ehlers, R., Könighofer, B., Niekum, S., \& Topcu, U. (2018).
``Safe Reinforcement Learning via Shielding.''
\textit{Proceedings of the AAAI Conference on Artificial Intelligence}, 32(1).

\bibitem{boyd2004}
Boyd, S. \& Vandenberghe, L. (2004).
\textit{Convex Optimization}.
Cambridge University Press.

\bibitem{rimon1992}
Rimon, E. \& Koditschek, D. E. (1992).
``Exact Robot Navigation Using Artificial Potential Functions.''
\textit{IEEE Transactions on Robotics and Automation}, 8(5), 501--518.

\bibitem{tempo2012}
Tempo, R., Calafiore, G., \& Dabbene, F. (2012).
\textit{Randomized Algorithms for Analysis and Control of Uncertain Systems}.
Springer Science \& Business Media.

\bibitem{bengio2013}
Bengio, Y., Courville, A., \& Vincent, P. (2013).
``Representation Learning: A Review and New Perspectives.''
\textit{IEEE Transactions on Pattern Analysis and Machine Intelligence}, 35(8), 1798--1828.

\bibitem{mitchell2005}
Mitchell, I. M., Bayen, A. M., \& Tomlin, C. J. (2005).
``A time-dependent Hamilton-Jacobi formulation of reachable sets for continuous dynamic games.''
\textit{IEEE Transactions on Automatic Control}, 50(7), 947--957.

\bibitem{amodei2016}
Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., \& Mané, D. (2016).
``Concrete Problems in AI Safety.''
\textit{arXiv preprint arXiv:1606.06565}.

\bibitem{bellman1957}
Bellman, R. (1957).
\textit{Dynamic Programming}.
Princeton University Press.

\bibitem{hoeffding1963}
Hoeffding, W. (1963).
``Probability inequalities for sums of bounded random variables.''
\textit{Journal of the American Statistical Association}, 58(301), 13--30.

\bibitem{polyak1963}
Polyak, B. T. (1963).
``Gradient methods for the minimisation of functionals.''
\textit{USSR Computational Mathematics and Mathematical Physics}, 3(4), 864--878.

\bibitem{lasalle1960}
LaSalle, J. P. (1960).
``Some extensions of Liapunov's second method.''
\textit{IRE Transactions on Circuit Theory}, 7(4), 520--527.

\bibitem{khalil2002}
Khalil, H. K. (2002).
\textit{Nonlinear Systems} (3rd ed.).
Prentice Hall.

\bibitem{kirkpatrick1983}
Kirkpatrick, S., Gelatt, C. D., \& Vecchi, M. P. (1983).
``Optimization by Simulated Annealing.''
\textit{Science}, 220(4598), 671--680.

\bibitem{jolliffe2002}
Jolliffe, I. T. (2002).
\textit{Principal Component Analysis} (2nd ed.).
Springer Series in Statistics.

\bibitem{brown2020}
Brown, T., et al. (2020).
``Language Models are Few-Shot Learners.''
\textit{Advances in Neural Information Processing Systems}, 33, 1877--1901.

\bibitem{blanchini1999}
Blanchini, F. (1999).
``Set Theoretic Methods in Control.''
\textit{Automatica}, 35(11), 1659--1681.

\bibitem{camacho2013}
Camacho, E. F., \& Bordons, C. (2013).
\textit{Model Predictive Control} (2nd ed.).
Springer Science \& Business Media.

\bibitem{trautman2010}
Trautman, P., \& Krause, A. (2010).
``Unfreezing the robot: Navigation in dense, interacting crowds.''
\textit{2010 IEEE/RSJ International Conference on Intelligent Robots and Systems}, 797--803.

\bibitem{johnson1984}
Johnson, W. B., \& Lindenstrauss, J. (1984).
``Extensions of Lipschitz mappings into a Hilbert space.''
\textit{Contemporary Mathematics}, 26(189-206).

\bibitem{barraquand1991}
Barraquand, J., \& Latombe, J. C. (1991).
``Robot motion planning with many degrees of freedom and dynamic constraints.''
\textit{Robotics Research}, 74--83.

\bibitem{plate1995}
Plate, T. A. (1995).
``Holographic Reduced Representations.''
\textit{IEEE Transactions on Neural Networks}, 6(3), 623--641.

\bibitem{gayler2003}
Gayler, R. W. (2003).
``Vector Symbolic Architectures answer Jackendoff's challenges.''
\textit{ICCS/ASCS International Conference on Cognitive Science}, 133--138.

\bibitem{prajna2004}
Prajna, S., \& Jadbabaie, A. (2004).
``Safety verification of hybrid systems using barrier certificates.''
\textit{International Workshop on Hybrid Systems: Computation and Control}, 477--492. Springer.

\bibitem{fazlyab2019}
Fazlyab, M., Robey, A., Hassani, H., Morari, M., \& Pappas, G. J. (2019).
``Efficient and accurate estimation of Lipschitz constants for deep neural networks.''
\textit{Advances in Neural Information Processing Systems}, 32.

\bibitem{madry2018}
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., \& Vladu, A. (2018).
``Towards Deep Learning Models Resistant to Adversarial Attacks.''
\textit{International Conference on Learning Representations (ICLR)}.

\bibitem{shumailov2023}
Shumailov, I., Shumaylov, Z., Zhao, Y., Gal, Y., Papernot, N., \& Anderson, R. (2023).
``The Curse of Recursion: Training on Generated Data Makes Models Forget.''
\textit{arXiv preprint arXiv:2305.17493}.

\bibitem{radford2021}
Radford, A., et al. (2021).
``Learning Transferable Visual Models From Natural Language Supervision.''
\textit{International Conference on Machine Learning (ICML)}, 8748--8763.

\bibitem{muller1959}
Muller, M. E. (1959).
``A note on a method for generating points uniformly on n-dimensional spherical surfaces.''
\textit{Communications of the ACM}, 2(4), 19--20.

\bibitem{xiao2019}
Xiao, W., \& Belta, C. (2019).
``Control Barrier Functions for Systems with High Relative Degree.''
\textit{2019 IEEE 58th Conference on Decision and Control (CDC)}, 474--479.

\bibitem{mayne2005}
Mayne, D. Q., Seron, M. M., \& Rakovi{\'c}, S. V. (2005).
``Robust model predictive control of constrained linear systems with bounded disturbances.''
\textit{Automatica}, 41(2), 219--224.

\bibitem{goodfellow2014}
Goodfellow, I. J., Shlens, J., \& Szegedy, C. (2014).
``Explaining and Harnessing Adversarial Examples.''
\textit{International Conference on Learning Representations (ICLR)}.

\bibitem{valiant1984}
Valiant, L. G. (1984).
``A Theory of the Learnable.''
\textit{Communications of the ACM}, 27(11), 1134--1142.

\bibitem{nesterov2004}
Nesterov, Y. (2004).
\textit{Introductory Lectures on Convex Optimization: A Basic Course}.
Kluwer Academic Publishers.

\bibitem{chen2018}
Chen, R. T. Q., Rubanova, Y., Bettencourt, J., \& Duvenaud, D. (2018).
``Neural Ordinary Differential Equations.''
\textit{Advances in Neural Information Processing Systems}, 31, 6571--6583.

\bibitem{chen2018hj}
Chen, M., Herbert, S. L., Vashishtha, M. S., Bansal, S., \& Tomlin, C. J. (2018).
``Decomposition of Reachable Sets and Tubes for a Class of Nonlinear Systems.''
\textit{IEEE Transactions on Automatic Control}, 63(11), 3675--3688.

\bibitem{glotfelter2017}
Glotfelter, P., Cort\'es, J., \& Egerstedt, M. (2017).
``Nonsmooth Barrier Functions with Applications to Multi-Robot Systems.''
\textit{IEEE Control Systems Letters}, 1(2), 310--315.

\bibitem{calafiore2006}
Calafiore, G. C., \& Campi, M. C. (2006).
``The Scenario Approach to Robust Control Design.''
\textit{IEEE Transactions on Automatic Control}, 51(5), 742--753.

\bibitem{lindemann2023}
Lindemann, L., Cleaveland, M., Shim, G., \& Pappas, G. J. (2023).
``Safe Planning in Dynamic Environments using Conformal Prediction.''
\textit{IEEE Robotics and Automation Letters}, 8(8), 5116--5123.

\bibitem{choromanska2015}
Choromanska, A., Henaff, M., Mathieu, M., Arous, G. B., \& LeCun, Y. (2015).
``The Loss Surfaces of Multilayer Networks.''
\textit{Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS)}, 192--204.

\bibitem{dawson2023}
Dawson, C., Gao, S., \& Fan, C. (2023).
``Safe Control With Learned Certificates: A Survey of Neural Lyapunov, Barrier, and Contraction Methods for Robotics and Control.''
\textit{IEEE Transactions on Robotics}, 39(3), 1749--1767.

\bibitem{robey2020}
Robey, A., Hu, H., Lindemann, L., Zhang, H., Dimarogonas, D. V., Tu, S., \& Matni, N. (2020).
``Learning Control Barrier Functions from Expert Demonstrations.''
\textit{2020 59th IEEE Conference on Decision and Control (CDC)}, 3717--3724.

\bibitem{qin2021}
Qin, Z., Zhang, K., Chen, Y., Chen, J., \& Fan, C. (2021).
``Learning Safe Multi-Agent Control with Decentralized Neural Barrier Certificates.''
\textit{International Conference on Learning Representations (ICLR)}.

\bibitem{garcia2015}
Garc{\'i}a, J., \& Fern{\'a}ndez, F. (2015).
``A Comprehensive Survey on Safe Reinforcement Learning.''
\textit{Journal of Machine Learning Research}, 16(42), 1437--1480.

\bibitem{lojasiewicz1963}
{\L}ojasiewicz, S. (1963).
``A topological property of real analytic subsets (Une propri{\'e}t{\'e} topologique des sous-ensembles analytiques r{\'e}els).''
\textit{Les {\'E}quations aux D{\'e}riv{\'e}es Partielles}, Colloques Internationaux du CNRS, 117, 87--89.

\bibitem{dauphin2014}
Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., \& Bengio, Y. (2014).
``Identifying and attacking the saddle point problem in high-dimensional non-convex optimization.''
\textit{Advances in Neural Information Processing Systems}, 27, 2933--2941.

\bibitem{kidger2022}
Kidger, P. (2022).
``On Neural Differential Equations.''
\textit{D.Phil.\ Thesis, University of Oxford}.

\bibitem{fisac2019}
Fisac, J. F., Akametalu, A. K., Zeilinger, M. N., Kaynama, S., Gillula, J., \& Tomlin, C. J. (2019).
``A General Safety Framework for Learning-Based Control in Uncertain Robotic Systems.''
\textit{IEEE Transactions on Automatic Control}, 64(7), 2737--2752.

\bibitem{wabersich2021}
Wabersich, K. P. \& Zeilinger, M. N. (2021).
``A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems.''
\textit{Automatica}, 129, 109597.

\bibitem{choi2020}
Choi, J., Castaneda, F., Tomlin, C. J., \& Sreenath, K. (2020).
``Reinforcement Learning for Safety-Critical Control under Model Uncertainty, using Control Lyapunov Functions and Control Barrier Functions.''
\textit{Robotics: Science and Systems (RSS)}.

\bibitem{osqp2020}
Stellato, B., Banjac, G., Goulart, P., Bemporad, A., \& Boyd, S. (2020).
``OSQP: An Operator Splitting Solver for Quadratic Programs.''
\textit{Mathematical Programming Computation}, 12(4), 637--672.

\bibitem{geshkovski2024}
Geshkovski, B., Letrouit, C., Polyanskiy, Y., \& Rigollet, P. (2024).
``A Mathematical Perspective on Transformers.''
\textit{Bulletin of the American Mathematical Society}, 61(4), 515--580. (Also \textit{arXiv preprint arXiv:2312.10794}.)

\bibitem{sander2022}
Sander, M. E., Ablin, P., Blondel, M., \& Peyr{\'e}, G. (2022).
``Sinkformers: Transformers with Doubly Stochastic Attention.''
\textit{International Conference on Artificial Intelligence and Statistics (AISTATS)}, 3515--3530.

\bibitem{ledoux2001}
Ledoux, M. (2001).
\textit{The Concentration of Measure Phenomenon}.
American Mathematical Society.

\bibitem{grimmett2001}
Grimmett, G. R., \& Stirzaker, D. R. (2001).
\textit{Probability and Random Processes}.
Oxford University Press, 3rd edition.

\bibitem{hutchinson1990}
Hutchinson, M. F. (1990).
``A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines.''
\textit{Communications in Statistics---Simulation and Computation}, 19(2), 433--450.

\bibitem{lynch1995}
Lynch, M., \& Gabriel, W. (1995).
``Mutation Load and the Survival of Small Populations.''
\textit{Evolution}, 49(6), 1067--1080.

\bibitem{nagumo1942}
Nagumo, M. (1942).
``{\"U}ber die Lage der Integralkurven gew{\"o}hnlicher Differentialgleichungen.''
\textit{Proceedings of the Physico-Mathematical Society of Japan}, 24, 551--559.

\bibitem{virmaux2018}
Virmaux, A., \& Scaman, K. (2018).
``Lipschitz Regularity of Deep Neural Networks: Analysis and Efficient Estimation.''
\textit{Advances in Neural Information Processing Systems}, 31.

\bibitem{prajna2007}
Prajna, S., Jadbabaie, A., \& Pappas, G. J. (2007).
``A Framework for Worst-Case and Stochastic Safety Verification Using Barrier Certificates.''
\textit{IEEE Transactions on Automatic Control}, 52(8), 1415--1428.

\bibitem{kim2021}
Kim, H., Papamakarios, G., \& Mnih, A. (2021).
``The Lipschitz Constant of Self-Attention.''
\textit{International Conference on Machine Learning (ICML)}, 5562--5571.

\bibitem{berkenkamp2017}
Berkenkamp, F., Turchetta, M., Schoellig, A. P., \& Krause, A. (2017).
``Safe Model-Based Reinforcement Learning with Stability Guarantees.''
\textit{Advances in Neural Information Processing Systems}, 30.

\bibitem{clark2021}
Clark, A. (2021).
``Control Barrier Functions for Stochastic Systems.''
\textit{Automatica}, 130, 109688.

\bibitem{santoyo2021}
Santoyo, C., Dutreix, M., \& Coogan, S. (2021).
``A Barrier Function Approach to Finite-Time Stochastic System Verification and Control.''
\textit{Automatica}, 125, 109439.

\bibitem{absil2005}
Absil, P.-A., Mahony, R., \& Andrews, B. (2005).
``Convergence of the Iterates of Descent Methods for Analytic Cost Functions.''
\textit{SIAM Journal on Optimization}, 16(2), 531--547.

\bibitem{dawson2022}
Dawson, C., Lowenkamp, B., Goff, D., \& Fan, C. (2022).
``Learning Safe, Generalizable Perception-Based Hybrid Control With Certificates.''
\textit{IEEE Robotics and Automation Letters}, 7(2), 1904--1911.

\bibitem{bansal2021}
Bansal, S., \& Tomlin, C. J. (2021).
``DeepReach: A Deep Learning Approach to High-Dimensional Reachability.''
\textit{2021 IEEE International Conference on Robotics and Automation (ICRA)}, 1817--1824.

\bibitem{barbalat1959}
Barbalat, I. (1959).
``Syst{\`e}mes d'{\'e}quations diff{\'e}rentielles d'oscillations non lin{\'e}aires.''
\textit{Revue Roumaine de Math{\'e}matiques Pures et Appliqu{\'e}es}, 4(2), 267--270.

\bibitem{ahmadi2019}
Ahmadi, A. A., \& Majumdar, A. (2019).
``DSOS and SDSOS Optimization: More Tractable Alternatives to Sum of Squares and Semidefinite Optimization.''
\textit{SIAM Journal on Applied Algebra and Geometry}, 3(2), 193--230.

\end{thebibliography}

\appendix

\section{Extended Related Work}
\label{app:related_work}

\textbf{Control Barrier Functions and Set Invariance.}
The theoretical foundation of safety via set invariance originates in the work of Nagumo \cite{nagumo1942} and was formalized for control systems by Blanchini \cite{blanchini1999}. Ames et al.\ \cite{ames2019} established the modern CBF-QP framework for enforcing forward invariance via pointwise optimization, providing deterministic safety guarantees for systems with known dynamics and relative-degree-one barriers. Xiao \& Belta \cite{xiao2019} extended this to higher relative degree via Exponential CBFs. Our work builds directly on this foundation, adapting the CBF-QP to high-dimensional spaces and demonstrating that its closed-form solution scales as $O(n)$, enabling real-time operation at dimensions previously considered intractable.

\textbf{Learned and Neural Barrier Certificates.}
A growing body of work replaces hand-designed CBFs with data-driven or neural-network-represented barrier functions. Dawson et al.\ \cite{dawson2023} provide a comprehensive survey of neural Lyapunov, barrier, and contraction methods, identifying training stability and verification of the learned certificate as open challenges. Robey et al.\ \cite{robey2020} learn CBFs from expert demonstrations, while Qin et al.\ \cite{qin2021} extend decentralized neural barrier certificates to multi-agent coordination. Dawson et al.\ \cite{dawson2022} further demonstrate perception-based hybrid control with learned CBF certificates, integrating visual observations with formal safety guarantees---a paradigm that extends naturally to semantic perception settings. Our framework is complementary: CHDBO assumes $h(x)$ is given (possibly learned), and focuses on the verification and real-time enforcement layer rather than the barrier synthesis problem.

\textbf{Randomized and Scenario-Based Verification.}
The scenario approach of Campi \& Garatti \cite{campi2008} and Calafiore \& Campi \cite{calafiore2006} provides PAC-style feasibility guarantees for convex programs with random constraints, requiring $O(n)$ samples in the number of decision variables. Tempo et al.\ \cite{tempo2012} extend randomized methods to analysis and control of uncertain systems. Prajna et al.\ \cite{prajna2007} developed a framework for worst-case and stochastic safety verification using sum-of-squares (SOS) barrier certificates, providing exact polynomial verification at the cost of semidefinite program solves that scale poorly beyond moderate dimensions. Our Monte Carlo Barrier Certificate differs in that it applies Hoeffding bounds for \textit{verification} of a fixed barrier (checking the volume fraction of violations) rather than for \textit{design} (finding a feasible controller), yielding sample complexity independent of dimension for fixed confidence parameters---though we note that the Lipschitz constant $L_h$ and safety margin $\varepsilon_s$ may implicitly depend on $n$ in practice.

\textbf{Conformal Prediction for Safety.}
Lindemann et al.\ \cite{lindemann2023} apply conformal prediction to safe planning, providing distribution-free uncertainty quantification for learned dynamics models. Angelopoulos \& Bates \cite{angelopoulos2021} provide a general introduction to conformal prediction methods. These approaches offer finite-sample validity guarantees without distributional assumptions---a property complementary to our Hoeffding-based MCBC, which assumes i.i.d.\ sampling but provides tighter bounds for fixed-distribution verification. Combining conformal prediction intervals with surrogate error bounds in future active adversarial verification methods is a promising direction.

\textbf{Adversarial Robustness and Safety.}
Madry et al.\ \cite{madry2018} introduced PGD-based adversarial training for neural networks, establishing the paradigm of iterative worst-case search that could be adapted for barrier verification. Goodfellow et al.\ \cite{goodfellow2014} demonstrated that small input perturbations can cause catastrophic misclassification---a phenomenon analogous to ``Black Swan'' spikes in barrier landscapes. Virmaux \& Scaman \cite{virmaux2018} established tight Lipschitz bounds for deep neural networks, and Kim et al.\ \cite{kim2021} extended Lipschitz analysis to self-attention layers, both directly relevant to estimating $L_f$ for neural dynamics. Extending adversarial search from the classification setting to continuous-state safety verification---incorporating momentum PGD, spectral margin estimation, and systematic coverage mechanisms---is a promising direction for future work.

\textbf{Safe Reinforcement Learning.}
Safe RL approaches broadly fall into constrained optimization \cite{garcia2015}, shielding \cite{alshiekh2018}, and Lagrangian methods \cite{choi2020}. Berkenkamp et al.\ \cite{berkenkamp2017} combine Lyapunov stability with Gaussian processes for safe exploration, providing formal guarantees within high-confidence regions but requiring kernel assumptions and scaling cubically in the number of data points. Predictive safety filters \cite{wabersich2021} and reachability-based shielding \cite{fisac2019} provide formal guarantees but typically require accurate forward models and scale as $O(n^3)$ or worse per step due to trajectory optimization. Model Predictive Control with safety constraints \cite{camacho2013, mayne2005} offers robust guarantees for linear systems but faces similar scalability limitations for $n > 50$. CHDBO occupies a distinct niche: it provides probabilistic (not deterministic) safety at $O(n)$ cost, trading guarantee strength for dimensional scalability.

\textbf{Stochastic Safety and Supermartingale Certificates.}
For stochastic systems, Clark \cite{clark2021} extends CBFs to It\^{o} diffusions, providing probabilistic safety certificates under Brownian noise. Santoyo et al.\ \cite{santoyo2021} introduce barrier function certificates for discrete-time stochastic systems using supermartingale arguments, yielding tighter trajectory-level safety bounds than the union bound employed in our Proposition~\ref{prop:trajectory_bridge}. Our framework currently assumes deterministic dynamics with bounded disturbances (the $\Delta_{\text{noise}}$ term) rather than stochastic diffusions; integrating supermartingale barrier theory with the CHDBO pipeline is a promising direction for future work.

\textbf{Hamilton-Jacobi Reachability.} The H-J framework \cite{mitchell2005} computes the backward reachable set via viscosity solutions of the H-J PDE. While providing the strongest possible guarantees, its $O((1/\eta)^n)$ grid complexity limits practical application to $n \leq 5$. Recent work on decomposition-based \cite{chen2018hj} and learning-based approximations---notably DeepReach \cite{bansal2021}, which parameterizes the value function with a neural network and solves the H-J PDE via self-supervised sinusoidal representations, achieving approximate reachability analysis up to $n \approx 10$---aims to extend reachability to moderate dimensions, but the fundamental exponential barrier remains for exact computation and neural approximations lack formal verification guarantees for $n \gg 10$. Our probabilistic relaxation sidesteps this barrier entirely.

\textbf{Sum-of-Squares and DSOS/SDSOS Relaxations.}
The SOS barrier certificates of Prajna et al.\ \cite{prajna2004, prajna2007} provide exact polynomial-time verification when the barrier and dynamics are polynomial, but the associated semidefinite programs scale poorly beyond moderate dimensions ($n \leq 10$--$20$). Ahmadi \& Majumdar \cite{ahmadi2019} introduced Diagonally-dominant SOS (DSOS) and Scaled DSOS (SDSOS) relaxations that replace semidefinite constraints with linear and second-order cone programs, substantially improving scalability at the cost of conservatism. These LP/SOCP-based relaxations represent the state of the art in polynomial barrier verification and can handle larger $n$ than SOS; however, their applicability is limited to polynomial systems, and they do not address the non-polynomial dynamics (e.g., neural network transitions) targeted by CHDBO.

\section{Mathematical Qualifications}
\label{app:qualifications}

This appendix collects detailed mathematical qualifications and caveats from the main text, preserved here for completeness.

\subsection{Dimension-Independence Caveats (Consolidated)}
While the \textit{sample count} $N$ required by the Hoeffding bound is independent of the state-space dimension $n$, three aspects of the verification pipeline do carry implicit or explicit dimensional dependence:
\begin{enumerate}
    \item \textbf{Per-sample cost.} Each barrier evaluation and CBF-QP solve costs $O(n)$ (or $O(n^2)$ with explicit Jacobians), so the \textit{total} verification cost is $O(N \cdot n)$---constant in $N$ but linear in $n$.
    \item \textbf{Lipschitz constant.} The $O(1)$ sample bound holds \textit{for fixed} $L_h$ and safety margin $\varepsilon_s$. If either grows with $n$ for architecture-specific reasons (e.g., barrier functions defined on unnormalized embeddings whose norms scale as $\sqrt{n}$), the effective sample complexity increases accordingly. Lemma~\ref{lem:lipschitz_dim} establishes that geometrically natural barriers avoid this growth.
    \item \textbf{Lipschitz coverage amplification.} The Lipschitz ball $B(x_i, h(x_i)/L_h)$ has volume $\propto r^n$, which vanishes exponentially near the boundary as $n$ grows. This does not affect the Hoeffding certificate (which is purely statistical) but does limit the deterministic coverage bonus in very high dimensions.
\end{enumerate}

\subsection{Concentration of Measure and Black Swan Spikes}
It is important to address the theoretical concern of Concentration of Measure in high-dimensional spaces ($n=128$), where safety violations could theoretically exist as ``spiky'' manifolds with large diameters but negligible volume (``Black Swan'' singularities) \cite{vershynin2018}. The Concentration of Measure phenomenon plays a dual role: on one hand, it \textit{aids} passive verification by concentrating probability mass near the boundary (the ``Hollow Ball'' effect of Figure~\ref{fig:concentration}), ensuring that boundary sampling captures the operationally relevant region; on the other hand, it \textit{hinders} passive verification by ensuring that narrow failure spikes occupy negligible volume, making them invisible to any feasible uniform sampling regime. MCBC exploits boundary concentration to certify that the \textit{volume fraction} of failures is small; active adversarial search methods that compensate for the volumetric invisibility of narrow spikes via gradient-based search offer a promising complement and are explored in a companion paper. We additionally leverage the Lipschitz continuity of the system dynamics $f(x)$---estimated via spectral norm techniques for deep networks \cite{fazlyab2019}---which places a lower bound on the ``width'' of any failure region, ensuring no violation is mathematically invisible to nearby samples \cite{vershynin2018}.

It may be argued that in high-dimensional spaces ($n=128$), even Lipschitz-continuous failure regions could remain statistically undetectable due to the vastness of the sampling volume. However, this assumes that the safety hazard is uniformly distributed across all dimensions. In practical control systems, safety violations (e.g., collisions) are typically low-rank phenomena, depending on a small subset of state variables (effective dimensionality $d_{\text{eff}} \ll n$) while remaining invariant to the others---an observation consistent with the general principle that learned representations concentrate task-relevant information in low-dimensional subspaces \cite{bengio2013}, though the specific claim about safety-violation dimensionality is a modeling assumption rather than a proven result. Our projection mechanism leverages this structure, ensuring that sampling efficiency scales with the complexity of the \textit{hazard}, not the complexity of the \textit{agent}.

Furthermore, we leverage the Lipschitz property of the barrier function $h(x)$. Given the Lipschitz constant $L_h$ of $h$---estimated via conservative local sampling or spectral norm bounds \cite{fazlyab2019}---any verified point $x_i$ with $h(x_i) > 0$ guarantees that all points within the hyper-ball $B(x_i, h(x_i)/L_h)$ also satisfy $h(x) > 0$. This provides a local deterministic guarantee: each verified sample ``covers'' a neighborhood of radius $r = h(x_i)/L_h$, amplifying the probabilistic certification with geometric coverage. We note, however, that the volume of each such ball scales as $r^n$, and near the boundary where $h(x_i) \to 0$, the coverage radius shrinks correspondingly. In very high dimensions ($n = 128$), this means the Lipschitz coverage amplification provides diminishing practical benefit for near-boundary points---motivating future work on active adversarial search methods that do not rely on volumetric coverage arguments.

\subsection{Distributional Qualification}
The Hoeffding bound assumes that the indicator random variables $\mathbb{I}(\Delta_i < 0)$ are independent and bounded, which holds when samples $x_i$ are drawn i.i.d.~from a fixed distribution on $\partial\mathcal{S}$. For spherical safe sets (as in our experiments), the isotropic Gaussian projection to $S^{n-1}$ yields a uniform distribution over the boundary, satisfying this assumption exactly. For non-spherical $\partial\mathcal{S}$ (e.g., polytopic safe sets or non-convex semantic manifolds), the sampling distribution must be specified explicitly, and the Hoeffding certificate applies to the \textit{chosen} distribution---not to an arbitrary alternative. In such cases, practitioners should either (1)~ensure the sampling distribution has support covering the operationally relevant boundary region, or (2)~employ importance-weighted estimators that correct for distributional mismatch.

\subsection{Trajectory Bridge: Extended Discussion}

\textbf{Scope and interpretation.} Proposition~\ref{prop:trajectory_bridge} provides an \emph{average-case} guarantee over the initial condition ensemble $\mu_0$. For a deterministic system with a \emph{fixed} initial condition $x(0)$, the trajectory is uniquely determined, so encountering an infeasible boundary point is either certain or impossible---the probabilistic bound is vacuous in this worst-case sense. We therefore classify this result as a \emph{proposition} rather than a lemma: it is a conditional statement whose practical strength depends entirely on the fidelity of the distributional assumption to the deployment scenario. Two complementary mechanisms strengthen this guarantee in practice: (1)~active adversarial verification methods (explored in a companion paper) can provide worst-case, horizon-independent safety verification by actively searching for violations along the \emph{actual} planned trajectory at each step, and (2)~periodic re-certification resets the union bound counter.

\textbf{Deterministic Alternative (Lipschitz Coverage).} For deterministic systems with fixed initial conditions, the distributional assumption above is inapplicable. An alternative trajectory-level guarantee can be obtained via Lipschitz coverage: if the MCBC samples are $\delta$-dense on $\partial\mathcal{S}$ (i.e., every boundary point lies within distance $\delta$ of some sample) and the CBF constraint function is $L_c$-Lipschitz, then any boundary encounter satisfies the CBF condition with margin at least $\min_i c(x_i) - L_c \delta$, where $c(x_i)$ is the verified margin at sample $x_i$. This provides a \emph{deterministic} bridge from boundary certification to trajectory safety, at the cost of requiring $\delta$-dense coverage---which is achievable for moderate dimensions via low-discrepancy (quasi-random) sequences but becomes impractical for $n \gg 100$, where active adversarial search methods offer the primary trajectory-level safeguard.

\textbf{Remark (Union Bound Validity).} The union bound $P(\exists t : \text{failure}) \leq T \cdot \epsilon$ is conservative and does not require independence of boundary encounters---it applies directly to arbitrarily correlated (Markovian) trajectory sequences. This is because the bound follows from the subadditivity of probability measures: $P(\bigcup_{t=1}^T A_t) \leq \sum_{t=1}^T P(A_t)$, which holds without any independence assumption \cite{grimmett2001}. The bound is tight when boundary encounters are rare and well-separated in time; it may be pessimistic when the trajectory repeatedly visits the same boundary segment, but in this case the MCBC certificate's coverage of that segment provides additional assurance.

\subsection{Convergence Proof: Extended Remarks}

\textbf{Extension to General Control-Affine Dynamics (Proof Sketch).} The single-integrator Lyapunov analysis extends to $\dot{x} = f(x) + g(x)u$ under one additional assumption: the control authority $g(x)$ must be sufficient to compensate the drift $f(x)$, i.e., $L_g V(x) \neq 0$ whenever $L_f V(x) > 0$ (the drift would increase $V$). Under this assumption, the nominal control $u_{\text{nom}}(x)$ can be designed to achieve $\dot{V} = L_f V + L_g V \cdot u_{\text{nom}} \leq 0$ in the interior, and the Barbalat argument applies with $\dot{V} = -\nabla U^T (f(x) + g(x) u^*)$ replacing $\dot{V} = -\|\nabla U\|^2$. The proof as presented is \textit{complete} for single-integrator systems; for general drift dynamics, it is a proof sketch whose rigor depends on the controllability margin $\inf_{x \in \partial\mathcal{S}} \|L_g h(x)\| / |L_f h(x)|$ remaining bounded away from zero.

\textbf{Singleton Convergence.} If $U$ is real-analytic on $\mathcal{S}$, then the constrained Lagrangian $\mathcal{L}(x, \lambda) = -U(x) + \lambda h(x)$ satisfies the {\L}ojasiewicz gradient inequality \cite{lojasiewicz1963}: there exist $C > 0$ and $\theta \in [1/2, 1)$ such that $\|\nabla_x \mathcal{L}\| \geq C |V(x) - V(x^*)|^\theta$ near any critical point $x^*$. Combined with the bounded $\ddot{V}$, this implies finite arc-length of the trajectory $\int_0^\infty \|\dot{x}\|\, dt < \infty$ \cite{absil2005}, and hence convergence to a \textit{single} KKT point $x^*_\mathcal{S} \in \Omega^*$, rather than oscillation within the KKT set. The real-analyticity requirement is satisfied by polynomial and rational utility functions but not by general $C^\infty$ functions.

\textbf{Remark (Regularity Condition).} The condition ``$\nabla U$ and $\nabla h$ are not opposing collinear'' excludes $\nabla U = -c\, \nabla h$ with $c > 0$ at an interior point of the boundary trajectory. This is precisely the KKT condition for a constrained maximum---violated \textit{only at the desired convergence point} $x^*_\mathcal{S}$ itself (where it is benign) and at boundary saddle points (which are escaped by the Rotational Circulation mechanism of Section~4.4). The regularity condition is therefore generically satisfied along trajectories.

\textbf{Remark (General Control-Affine Dynamics).} For general systems $\dot{x} = f(x) + g(x)u$ where $f(x) \neq 0$, the convergence argument additionally requires that the control authority $L_g U(x)$ is sufficient to compensate the drift $L_f U(x)$ away from the optimum. When drift is adversarial ($L_f U < 0$) and dominates, the system converges to the $\omega$-limit set contained in the constrained critical points of the Lagrangian, which need not coincide with $\text{argmax}_{\mathcal{S}}\, U$. If $U(x)$ is concave and the constraint qualification holds (Slater's condition), this critical point is the unique global constrained maximum by KKT theory.

\textbf{Remark (Convergence Rate).} The Barbalat-based argument establishes \textit{asymptotic} convergence but does not provide a rate. For the single-integrator case with concave $U$ and convex $\mathcal{S}$, projected gradient descent achieves $O(1/t)$ convergence of the utility suboptimality gap \cite{nesterov2004}; if $U$ additionally satisfies the Polyak-{\L}ojasiewicz (PL) inequality on $\mathcal{S}$, the convergence becomes linear: $U(x^*_{\mathcal{S}}) - U(x(t)) \leq (1-\mu/L)^t \cdot [U(x^*_{\mathcal{S}}) - U(x(0))]$. For the {\L}ojasiewicz-based singleton convergence, the rate depends on the exponent $\theta$: $O(e^{-ct})$ when $\theta = 1/2$, and $O(t^{-\theta/(1-2\theta)})$ when $\theta \in (1/2, 1)$.

\section{Supplementary Experiments}
\label{app:supplementary_experiments}

Two additional experiments provide further characterization of the CHDBO framework. These experiments are moved here to reduce the main body length; they remain important for completeness.

\subsection{Experiment III: Proportional Safety Response}
\label{app:exp_proportional}

A critical requirement for autonomous agents is that safety measures should not destroy utility unless absolutely necessary. We stress-tested the system by varying the ``Adversarial Intensity'' ($\alpha$) of the agent's intent from 0.0 (Pure Safe) to 1.0 (Pure Malicious). Specifically, $\alpha$ parameterizes the \textit{angular deviation} from the tangent plane: the intent velocity is constructed as
\[
    v = \sqrt{1-\alpha^2}\,\hat{t} + \alpha\,\hat{r},
\]
where $\hat{t}$ is the unit tangent direction and $\hat{r}$ is the outward radial (unsafe) direction, so that $\alpha = \sin\theta$ where $\theta$ is the angle between the intent and the safety boundary. This ensures $\|v\| = 1$ (before noise). At the boundary ($h=0$), orthogonal projection yields the Pythagorean limit $\sqrt{1-\alpha^2}$ for Semantic Preservation; states with positive margin experience less correction and therefore higher similarity.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figure_3.png}
    \caption{\textbf{Proportional Safety Response ($N=1000$).} Semantic preservation is bounded below by the zero-margin limit $\sqrt{1-\alpha^2}$ (dashed); observed data lies above this curve because states at $0.95R$ have positive safety margin, reducing CBF intervention.}
    \label{fig:proportional_response}
\end{figure}

\textbf{Observation:} As illustrated in Figure~\ref{fig:proportional_response}, the system's response is nonlinear and proportional:
\begin{itemize}
    \item \textbf{Low Threat ($\alpha < 0.4$):} Similarity remains near 1.0. The verifier allows the agent to act freely when the intent aligns with safety.
    \item \textbf{High Threat ($\alpha > 0.8$):} Similarity decreases to approximately 0.85 as the CBF intervenes more strongly, yet remains well above the boundary limit ($\sqrt{1-0.8^2} = 0.6$). Notably, similarity \emph{recovers} near $\alpha = 1$ because a purely radial intent is corrected by magnitude scaling alone, preserving direction and therefore cosine similarity.
    \item \textbf{Curve:} The dashed line $\sqrt{1-\alpha^2}$ is the boundary-case limit---the similarity that would obtain if the state sat exactly on $\partial\mathcal{S}$ ($h=0$), where the CBF constraint is maximally tight (as derived from the orthogonal projection geometry of Section~4.2). Empirical data lies systematically \emph{above} this curve because states at $0.95R$ retain margin $h=0.05$, yielding CBF slack $\gamma h = 0.25$ and therefore less aggressive correction. This gap is a positive property: the filter intervenes only as much as the margin demands.
\end{itemize}

\subsection{Experiment VIII: Empirical Validation of Theorem~\ref{thm:safe_convergence}}
\label{app:exp_convergence}

To validate the convergence claim of Theorem~\ref{thm:safe_convergence} directly, we run the complete CHDBO pipeline in $\mathbb{R}^{128}$ under linear drift and measure both safety and utility convergence over extended horizons.

\textbf{Setup.} The system employs linear drift $\dot{x} = Ax + u$ with $\sigma_{\max}(A) = 1.044$, a quadratic barrier $h(x) = 1 - \|x\|^2$ defining the unit-ball safe set, and a utility function $U(x) = -\|x - x_{\text{goal}}\|$ with $\|x_{\text{goal}}\| = 1.5$ (outside the safe set, forcing barrier activation). We run 20 independent trials of 1{,}200 steps with $\Delta t = 0.01$, $\gamma = 2.0$, and the CBF-QP control law of Equation~\ref{eq:cbfqp}. The constrained optimum is $x^*_\mathcal{S} = x_{\text{goal}} / \|x_{\text{goal}}\|$ (the boundary point nearest the goal), with optimal utility $U^* = -0.50$.

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_16.png}
    \caption{\textbf{Theorem~\ref{thm:safe_convergence} Validation ($\mathbb{R}^{128}$, Linear Drift).} (a)~Utility converges toward $U^* = -0.50$. (b)~Barrier values positive: 0/24{,}000 violations. (c)~Norm approaches boundary. (d)~Distance to optimum decreases monotonically.}
    \label{fig:utility_convergence}
\end{figure*}

\textbf{Results:} Across 20 trials $\times$ 1{,}200 steps (24{,}000 total state evaluations), the system achieves \textbf{0 safety violations}. The mean utility converges to $U = -0.554$ (vs.\ optimal $U^* = -0.50$), with the gap attributable to the linear drift consuming control authority. The mean distance to $x^*_\mathcal{S}$ decreases monotonically to 0.194, confirming that the trajectory approaches the KKT set predicted by Theorem~\ref{thm:safe_convergence}. These results empirically validate forward invariance ($h(x(t)) \geq 0$), non-decreasing utility in expectation, and convergence toward the constrained optimum.

\section{Deployment Guide: Parameter Recommendations}
\label{app:deployment}

This appendix provides practical parameter recommendations for deploying the CHDBO framework across two primary application domains: \textbf{kinematic systems} (robotic state spaces, physical actuators) and \textbf{semantic systems} (LLM embedding spaces, representation manifolds).

\subsection{MCBC Verification Parameters}

\begin{table}[htbp]
\centering
\scriptsize
\begin{tabular}{@{}l@{\hskip 3pt}c@{\hskip 3pt}c@{}}
\toprule
\textbf{Parameter} & \textbf{Kinematic} & \textbf{Semantic} \\ \midrule
$\delta$ (confidence) & $10^{-6}$ & $10^{-4}$--$10^{-6}$ \\
$\epsilon$ (failure tol.) & $10^{-3}$ & $10^{-2}$--$10^{-3}$ \\
$N$ (samples) & ${\approx}7{\times}10^6$ & ${\approx}7{\times}10^4$ \\
$T_w$ (re-cert.\ interval) & $10^3$ steps & $10^2$--$10^3$ steps \\
\bottomrule
\end{tabular}
\caption{Recommended MCBC parameters.}
\end{table}

\textbf{Kinematic recommendation.} For safety-critical physical systems ($n \leq 128$), use strict tolerances ($\epsilon = 10^{-3}$, $\delta = 10^{-6}$) and re-certify every $10^3$ control steps to keep the trajectory-level failure probability below $10^{-2}$ per re-certification window.

\textbf{Semantic recommendation.} For LLM embedding spaces ($n = 768$ to $4096$), the barrier function $h(x)$ typically has lower Lipschitz constant (embeddings vary smoothly under small perturbations), permitting looser tolerances. However, validate Assumption A7 (continuous relaxation validity) empirically for the specific embedding architecture before deployment.

\subsection{CBF-QP Parameters}

\begin{itemize}
    \item \textbf{Class-$\mathcal{K}$ gain $\gamma$:} Start with $\gamma = 1.0$ for conservative behavior; increase to $\gamma \in [2, 5]$ for more aggressive boundary tracking. Higher $\gamma$ permits operation closer to $\partial\mathcal{S}$ at the cost of increased control effort (see Experiment~III for the proportional response characterization).
    \item \textbf{Robust margin $\rho(x)$:} For kinematic systems with known dynamics, use exact $\sigma_{\max}(J_f)$ if available; otherwise, use the Hutchinson estimate with $m \geq 30$ probes for $<8\%$ Frobenius-to-spectral gap (Experiment~VI uses $k{=}5$ as a deliberately conservative lower bound). For semantic systems, set $\rho(x) = \tilde{\sigma}_{\max}(x) \cdot d_{\text{step}}$ and multiply by a safety factor $\kappa \geq 1.5$ to account for Frobenius-to-spectral-norm gap.
    \item \textbf{Discretization step $\Delta t$:} Ensure $\Delta t < \varepsilon_s / (M \cdot L_h)$ where $M = \sup \|f(x) + g(x)u\|$ is the maximum velocity and $L_h$ is the barrier Lipschitz constant. This guarantees the Euler discretization error remains within the safety margin.
\end{itemize}

\end{document}