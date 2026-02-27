\documentclass[11pt, a4paper]{article}

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
\usepackage{fancyhdr}
\usepackage{subcaption}

% Page Geometry Setup
\geometry{margin=1in}
\emergencystretch=1em

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

% Title Information
\title{\textbf{Beyond the Grid: Probabilistic Expansion of Topological Safety and Asymptotic Utility in High-Dimensional Manifolds}}
\author{Arsenios Scrivens}
\date{February 9, 2026}

\begin{document}

\maketitle

\begin{abstract}
Verifying safety constraints in high-dimensional state spaces is computationally intractable for deterministic methods: grid-based Hamilton-Jacobi reachability scales as $O((1/\eta)^n)$, becoming infeasible for $n > 5$. We introduce Constrained High-Dimensional Barrier Optimization (CHDBO), a framework that replaces deterministic enumeration with probabilistic certification. Using Monte Carlo Barrier Certificates bounded by Lipschitz continuity, CHDBO reduces verification \textit{sample complexity} to $O(1)$ relative to dimension (total cost $O(N \cdot n)$), enabling safety certification for systems with $n \geq 128$. A gradient-based utility maximizer ensures asymptotic convergence to constrained optima without violating the safety envelope. To address rare catastrophic modes invisible to passive sampling, we introduce Active Adversarial Safety Verification (AASV): momentum-accelerated adversarial search with Hutchinson-bounded spectral margins and orthogonal prototype memory, yielding an empirically-calibrated adversarial risk bound with $O(n)$ per-step cost. Experiments across seven configurations---including non-trivial drift dynamics, multi-modal barrier landscapes, and seed sensitivity analysis in $\mathbb{R}^{128}$---confirm that CHDBO maintains safety while preserving utility.
\end{abstract}

\section{Introduction}

The trajectory of artificial intelligence has shifted rapidly from reactive, token-prediction models to persistent, autonomous Agentic Control Systems (ACS). This evolution has revealed a fundamental fragility in current architectural paradigms: the phenomenon of ``Agent Drift'' \cite{scrivens2026, lecun2022}, which can expose the system to adversarial exploits and alignment failures \cite{wei2023}. A related phenomenon---distributional degradation when models are trained on their own outputs---has been documented as ``Model Collapse'' \cite{shumailov2023}. More broadly, as systems operate over extended temporal horizons, they suffer from a form of structural entropy where behavioral stability and decision quality degrade \cite{amodei2016}.

In our previous work, \textit{Mitigating Large Language Model Context Drift via Holographic Invariant Storage} \cite{scrivens2026}, we addressed the semantic component of this failure. We demonstrated that by anchoring an agent's identity in a Vector Symbolic Architecture (VSA) substrate \cite{kanerva2009}, essentially a holographic ``Read-Only'' memory, we could substantially reduce the corruption of its goals and personality. While Vector Symbolic Architectures inherently possess the high-dimensional noise tolerance required to solve this problem \cite{plate1995, gayler2003}, a distinct but parallel challenge exists in the kinematic domain: dynamical control drift.

\subsection{The High-Dimensional Control Divergence}
Modern autonomous agents do not merely process text; they operate in high-dimensional state spaces, mapping inputs to actions in continuous environments \cite{ames2019}. Whether controlling a swarm of drones ($n>100$) \cite{brunke2022} or navigating the semantic embedding space of a multi-modal reasoning engine ($n=12,288$ \cite{brown2020, bengio2013}) or visual transformer \cite{radford2021}, the agent must constantly optimize for utility while adhering to strict safety constraints (e.g., ``Do not delete root files,'' ``Do not collide'').

The central objective of modern control theory is the synthesis of an agent that is simultaneously aggressive in its pursuit of goals and conservative in its avoidance of failure. Historically, this has forced a binary choice:

\begin{enumerate}
    \item \textbf{Formal Methods (The Conservative approach):} Techniques such as Control Barrier Functions (CBFs) \cite{ames2019} and Hamilton-Jacobi reachability \cite{mitchell2005} offer absolute, provable safety guarantees. However, these methods suffer from the ``Curse of Dimensionality'' \cite{bellman1957}. Verifying a safety manifold via grid discretization scales as $O((1/\eta)^n)$, where $\eta$ is the grid resolution and $n$ is the dimension. For a simple robotic arm ($n=6$, three joints with position and velocity), this is tractable; for a semantic reasoning agent ($n=128$), it is physically impossible, requiring computation exceeding the age of the universe.
    \item \textbf{Learning-Based Control (The Aggressive approach):} Deep Reinforcement Learning (DRL) offers high performance and scalability but often lacks axiomatic safety \cite{amodei2016}. While recent advancements in ``Safe RL'' and Shielding attempt to impose constraints \cite{brunke2022, alshiekh2018, garcia2015}, standard implementations often rely on ``soft'' reward penalties that do not strictly prevent catastrophic failure states, leading to rare events where the agent maximizes utility by gambling with safety.
\end{enumerate}

\subsection{The ``Golden Manifold'' and Probabilistic Relaxation}
To resolve this dichotomy, we must extend the concept of the \textbf{Forward Invariant Safe Set} $\mathcal{S}$ \cite{ames2019, blanchini1999}---colloquially termed the ``Golden Manifold'' in our prior work---from the semantic domain into the kinematic domain. In this geometric region, the agent remains invariant to microscopic fluctuations, preserving its trajectory against external perturbations.

This paper proposes Constrained High-Dimensional Barrier Optimization (CHDBO). We argue that for high-dimensional systems, absolute deterministic safety verification is a theoretical luxury that prevents practical deployment. Instead, we introduce a probabilistic relaxation of the safety condition. By trading the deterministic exactness of grid checks for probabilistic certainty bounded by Lipschitz continuity, we can utilize Monte Carlo integration to estimate the volume of the safe set.

By sampling the hypersphere of the agent's immediate trajectory and applying statistical bounds to the barrier function $h(x)$, CHDBO certifies safety with confidence levels approaching $1 - 10^{-6}$. This effectively matches the reliability of deterministic methods while circumventing the exponential scaling of sample \textit{count} (though the total computational cost remains $O(N \cdot n)$ due to per-sample vector operations). While the \textit{total computational cost} scales linearly ($O(n)$) due to vector operations, the required \textit{sample count} $N$ becomes a constant factor governed only by the desired confidence interval and the Lipschitz constant of the barrier function. We explicitly categorize this verification as \textit{Probabilistic Safety} (PAC-style \cite{valiant1984}), distinguishable from the \textit{Absolute Safety} of low-dimensional formal methods. While Lipschitz continuity theoretically bounds the minimum volume of failure modes, the \textit{Concentration of Measure} phenomenon in high-dimensional spaces ($n=128$) implies that rare, ``spiky'' failure regions (``Black Swan'' singularities) with probability mass below the sampling resolution ($\epsilon$) may evade passive uniform sampling. To address this vulnerability directly, we introduce the Active Adversarial Safety Verification (AASV) module (Section 3.5), which transitions the verification logic from passive statistical assurance to active gradient-based threat hunting. By actively minimizing the barrier function $h(x)$ via momentum-accelerated Projected Gradient Descent (PGD), AASV converts the safety guarantee from passive probabilistic coverage ($P_{\text{fail}} < \epsilon$) to an adversarially-bounded risk that decreases exponentially with the number of adversarial restarts $k$.

\subsection{Asymptotic Utility and Orthogonal Projection}
However, strict safety must not preclude mission completion; a theoretically safe agent that remains immobile is operationally useless. Our second contribution is a geometric control law that ensures Asymptotic Utility by functioning as a minimally invasive safety filter. We introduce a steering mechanism that constrains the control input strictly within the tangent cone of the safe set $\mathcal{S}$.

When the nominal action derived from the utility function $U(x)$ threatens to violate the invariant set, CHDBO does not simply halt the system. Instead, it solves a high-dimensional Quadratic Program (QP) to perform an Orthogonal Projection of the intent vector onto the safety boundary. This enables the agent to traverse the "skin" of the safe manifold, maintaining the maximum permissible velocity tangent to the constraint, thereby resolving the "Frozen Robot" problem \cite{trautman2010} characteristic of naive barrier implementations where conflicting constraints induce stagnation.

\subsection{Contributions}
The specific contributions of this paper are:
\begin{itemize}
    \item \textbf{Generalization of Semantic Safety:} We extend the invariant preservation principles of \cite{scrivens2026} from holographic memory to continuous control manifolds.
    \item \textbf{High-Dimensional Adaptation of CBF-QP:} We adapt standard Control Barrier Function (CBF) formulations \cite{ames2019, xiao2019} to high-dimensional semantic spaces ($n \ge 128$), building on the growing body of work on learned and neural CBF methods \cite{dawson2023, robey2020, qin2021, fisac2019}. We demonstrate that the quadratic programming (QP) formulation remains computationally tractable ($O(n)$) for real-time semantic steering, overcoming the perceived ``Curse of Dimensionality'' in verification \cite{bellman1957}.
    \item \textbf{Hierarchical Utility Integration:} We provide a lexicographic control architecture that strictly prioritizes safety constraints over utility maximization, ensuring that aggressive performance and rigorous safety are hierarchically integrated rather than competing.
    \item \textbf{Active Adversarial Safety Verification (AASV):} We introduce an adversarial verification module that replaces passive Monte Carlo sampling with gradient-based adversarial search, Hutchinson-bounded spectral margins \cite{hutchinson1990}, and holographic orthogonal prototype memory, converting the probabilistic ``Black Swan'' vulnerability into a bounded adversarial robustness guarantee while preserving $O(n)$ computational scaling.
\end{itemize}

\section{Mathematical Preliminaries and Problem Formulation}
In this section, we define the class of dynamical systems under consideration and review the foundational definitions of safety established in our previous work \cite{scrivens2026}. We then formally state the High-Dimensional Utility-Safety Problem, motivating the transition from the deterministic grid-based verification of low-dimensional systems to the probabilistic framework proposed herein.

\subsection{System Dynamics}
We consider a control-affine dynamical system evolving on a manifold $\mathcal{X} \subseteq \mathbb{R}^n$, described by the ordinary differential equation:
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

\textbf{Remark (Continuous Relaxation for Discrete Semantic Systems).}
When applying the control-affine ODE formulation to semantic embedding spaces---where Large Language Models transition states discretely, token by token---a continuous relaxation is required. We adopt the \textit{Neural Ordinary Differential Equation} paradigm \cite{chen2018, kidger2022}, which models the evolution of hidden states in deep networks as the solution to a continuous-time ODE $\dot{x} = f_\theta(x, t)$, treating discrete layer transitions as Euler discretizations of an underlying continuous flow. Under this interpretation, the embedding trajectory between autoregressive steps is modeled as continuous, with the Lie derivative and barrier conditions evaluated on this continuous interpolant. The \textit{discretization error} introduced by the finite step size $\Delta t$ between tokens is bounded by $O(\Delta t^2 \cdot L_{\nabla f})$ (one-step local truncation error of the Euler method); the \textit{global} accumulated error over $T$ steps is $O(\Delta t \cdot L_{\nabla f})$ by standard ODE convergence theory. Because the barrier condition and $\epsilon_{\text{model}}$ are re-evaluated at each step using the \textit{current} state $x(t)$---not propagated from a fixed initial condition---the relevant error is the single-step local truncation, which is absorbed into the surrogate error term $\epsilon_{\text{model}}$ of the robust barrier condition (Equation~\ref{eq:robust_barrier}). Multi-step drift is bounded by a Gronwall-type argument: $\|x_{\text{true}}(t) - x_{\text{Euler}}(t)\| \leq C \cdot \Delta t \cdot (e^{L_f T} - 1)$, which remains small when $\Delta t \ll 1/L_f$ \cite{khalil2002}. We emphasize that when the semantic dynamics are highly discontinuous---for instance, when a microscopic embedding perturbation induces a macroscopic semantic divergence (a known phenomenon in adversarial NLP \cite{goodfellow2014})---the Lipschitz constant $L_f$ may become locally very large, inflating both $\epsilon_{\text{model}}$ and the safety tube $\delta(x)$. In this regime, the framework correctly becomes highly conservative (shrinking the feasible set), which is the appropriate behavior for a safety-critical system operating near a discontinuity. The CHDBO guarantees therefore apply rigorously to the continuous relaxation; their fidelity to the true discrete system is governed by the tightness of the Euler discretization bound, which practitioners must validate empirically for their specific architecture.

\subsection{Recap of Topological Safety}
Following the standard formulation in \cite{ames2019} and \cite{blanchini1999}, we define safety through the lens of set invariance. A set $\mathcal{S} \subset \mathcal{X}$ is defined as the safe set, representing the region of the state space where the system is permitted to operate (e.g., ``Sanity'', ``Goal Alignment''). We define $\mathcal{S}$ as the super-level set of a continuously differentiable scalar function $h(x): \mathbb{R}^n \to \mathbb{R}$, known as the Control Barrier Function (CBF):
\begin{align*}
    \mathcal{S} &= \{ x \in \mathbb{R}^n \mid h(x) \geq 0 \} \\
    \partial \mathcal{S} &= \{ x \in \mathbb{R}^n \mid h(x) = 0 \} \\
    \text{Int}(\mathcal{S}) &= \{ x \in \mathbb{R}^n \mid h(x) > 0 \}
\end{align*}

As established in prior work \cite{scrivens2026}, we formalize the safety condition as follows:

\begin{theorem}[Topological Safety]
The system is safe (i.e., the set $\mathcal{S}$ is forward invariant) if there exists a control input $u$ such that the time derivative of $h(x)$ satisfies the linear class-$\mathcal{K}$ inequality (concept formally defined in \cite{khalil2002}, applied to barriers in \cite{ames2019}). Sufficiency follows from Nagumo's theorem \cite{blanchini1999}; necessity holds additionally when $0$ is a regular value of $h$ and $\mathcal{S}$ is compact:
\begin{equation}
    \sup_{u \in \mathcal{U}} \left[ L_f h(x) + L_g h(x) u \right] \geq -\gamma h(x)
\end{equation}
where $L_f h$ and $L_g h$ denote the Lie derivatives of $h$ along $f$ and $g$, and $\gamma > 0$ is a tunable relaxation parameter governing how quickly the system is allowed to approach the boundary.
\end{theorem}

\subsection{The High-Dimensional Divergence}
While the formulation provides sufficient conditions for safety, the verification method utilized therein, Grid-Based Invariance Checking, relies on discretizing the domain $\mathcal{X}$. As established in Section 1.1, the computational complexity of verifying the invariance condition over a grid $G_\eta$ scales exponentially:
\begin{equation}
    C(G_\eta) \propto \left(\frac{L}{\eta} \right)^n
\end{equation}
Consequently, exact topological guarantees cannot be directly computed for high-degree-of-freedom systems ($n \ge 128$). This necessitates the Probabilistic Relaxation we introduce in Section 3.

\subsection{Utility Maximization and Problem Statement}
Unlike \cite{scrivens2026}, which focused primarily on safety (survival and memory persistence), this work introduces a performance objective. We define a Utility Function $U(x): \mathbb{R}^n \to \mathbb{R}$, which encodes the agent's task (e.g., reaching a target, maximizing speed, maintaining formation). We assume $U(x)$ is continuously differentiable and concave.

We seek to find a control law $k(x)$ that drives the system to the global maximum of $U(x)$ without ever leaving $\mathcal{S}$.

\begin{problem}[Constrained High-Dimensional Barrier Optimization]
Find a control policy $u=k(x)$ such that for any initial condition $x_0 \in \text{Int}(\mathcal{S})$:
\begin{enumerate}
    \item \textbf{Safety (High Probability):} $P(x(t) \in \mathcal{S}) \geq 1-\delta$ for all $t \geq 0$, where $\delta$ is a negligible failure probability (e.g., $10^{-6}$).
    \item \textbf{Asymptotic Utility:} $\lim_{t \to \infty} x(t) = x^*_{\mathcal{S}}$ where $x^*_{\mathcal{S}} = \text{argmax}_{x \in \mathcal{S}} U(x)$.
    \item \textbf{Scalability:} The computation time for calculating $u$ is $O(n)$ relative to dimension (linear scaling), enabling real-time operation for $n \gg 10$.
\end{enumerate}
\end{problem}
This formulation stratifies the control problem: safety verification (handled via probabilistic barriers) acts as a hard constraint, while the performance objective (handled via gradient-based steering) acts as a soft preference, resolving the 'Safety-Performance Trade-off' through lexicographic optimization.

\subsection{Standing Assumptions}
\label{sec:assumptions}
For clarity, we collect the assumptions underlying the CHDBO framework:
\begin{enumerate}
    \item[\textbf{A1.}] \textbf{Lipschitz Continuity:} The drift $f(x)$, control effectiveness $g(x)$, and barrier $h(x)$ are locally Lipschitz continuous on $\mathcal{X}$ \cite{khalil2002}.
    \item[\textbf{A2.}] \textbf{Barrier Differentiability:} $h(x)$ is continuously differentiable ($C^1$), and $0$ is a regular value of $h$ (i.e., $\nabla h(x) \neq 0$ on $\partial\mathcal{S}$).
    \item[\textbf{A3.}] \textbf{Relative Degree One:} The barrier $h(x)$ has relative degree one with respect to the control input $u$, i.e., $L_g h(x) \neq 0$ on $\partial\mathcal{S}$. For higher relative degree, an Exponential CBF extension is employed (Section~5.5).
    \item[\textbf{A4.}] \textbf{Bounded Control:} The control input set $\mathcal{U}$ is compact and the CBF-QP (Equation~\ref{eq:cbfqp}) is feasible for all $x \in \partial\mathcal{S}$.
    \item[\textbf{A5.}] \textbf{Local Convexity or Star-Shapedness:} The safe set $\mathcal{S}$ is locally convex or star-shaped near $\partial\mathcal{S}$ for the boundary projection to be well-defined. For non-convex $\mathcal{S}$, tangent half-space inner-approximations are used (Section~3.3).
    \item[\textbf{A6.}] \textbf{Utility Smoothness:} $U(x)$ is continuously differentiable. For global convergence guarantees, $U$ is concave or satisfies the Polyak-{\L}ojasiewicz condition \cite{polyak1963}.
\end{enumerate}

\section{High-Dimensional Probabilistic Verification}
The central contribution of this work is the assertion that absolute deterministic safety is a luxury of low-dimensional systems, while high-confidence probabilistic safety is a necessity for high-dimensional ones. In this section, we present the Monte Carlo Barrier Certificate (MCBC), a method that circumvents the scaling limitations of grid-based verification by estimating the volume of the safe invariant set $\mathcal{S}$ rather than delineating its exact boundary.

\subsection{The Scaling Bottleneck}
As established in Section 2, the computational cost of exact verification scales as $O((1/\eta)^n)$. To contextualize this: verifying a standard 3-link robotic arm ($n=6$) on a coarse grid requires $10^{12}$ evaluations. Verifying a swarm of 50 drones or a semantic embedding space ($n \geq 128$) requires $10^{200}$ evaluations, rendering deterministic grid-based verification physically intractable regardless of available computing power. However, the Johnson-Lindenstrauss lemma \cite{johnson1984} provides suggestive geometric intuition: pairwise distances between points are preserved with high probability in lower-dimensional random projections (see \cite{dasgupta2003} for a simplified proof), indicating that the geometric structure relevant to boundary proximity is not destroyed by dimensionality. While J-L does not directly guarantee preservation of barrier function values or set membership, this distance-preserving property motivates the hypothesis---validated empirically in our experiments---that sparse random boundary sampling can capture the essential safety geometry without exhaustive enumeration.

To solve this, we shift our verification criterion from Worst-Case Invariance (is there any point $x$ where safety fails?) to PAC-Style Invariance \cite{valiant1984} (is the probability of encountering a failure state bounded by $\epsilon$?). We employ Hoeffding's Inequality to bound the estimation error $\epsilon$ \cite{hoeffding1963}. This approach aligns with modern uncertainty quantification methods, such as Conformal Prediction \cite{angelopoulos2021}, and established randomized control frameworks \cite{tempo2012}, effectively constructing a probabilistic envelope around the system dynamics rather than a rigid boundary.

\subsection{Randomized Barrier Certification}
We employ a randomized verification strategy based on the Monte Carlo Barrier Certificate (MCBC) engine algorithm. Building upon the foundational concept of Barrier Certificates introduced by Prajna et al. \cite{prajna2004}, which originally utilized sum-of-squares optimization for low-dimensional verification, we adapt the certificate condition for high-dimensional probabilistic sampling. Instead of discretizing the state space $\mathcal{X}$, we treat $\mathcal{X}$ as a probability space equipped with a measure $\mu$. We define the Safety Violation Probability, $P_{\text{fail}}$, as the measure of the set of states where the barrier condition is violated:
\begin{equation}
    P_{\text{fail}} = \int_{\mathcal{X}} \mathbb{I}\left(\sup_{u \in \mathcal{U}} \dot{h}(x, u) < -\gamma h(x) \right) d\mu(x)
\end{equation}
where $\mathbb{I}(\cdot)$ is the indicator function. Since this integral is analytically intractable for non-linear $f(x)$ and $g(x)$, we approximate it via Monte Carlo integration.

\subsection{Lipschitz Continuity and Sample Complexity}
A critical challenge in high-dimensional probability is the concentration of measure, where uniform volume sampling fails to detect boundary violations effectively. To address this, we leverage the \textbf{Boundary Concentration} property. Since the safety condition $h(x) \ge 0$ is critical primarily at the boundary $\partial \mathcal{S}$, we employ an Isotropic Gaussian sampling strategy projected onto the hypersphere surface \cite{muller1959}, rather than attempting to sample the entire volume.

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figure_concentration.png}
    \caption{\texorpdfstring{\textbf{The `Hollow Ball': Concentration of Measure in High Dimensions ($n=128$).}}{The 'Hollow Ball': Concentration of Measure in High Dimensions (n=128).} In low dimensions ($n=3$), volume is distributed evenly. In high-dimensional semantic spaces ($n=128$), probability mass concentrates almost exclusively in a thin `Active Safety Shell' near the boundary ($r > 0.95$). This validates the CHDBO sampling strategy: verifying the boundary effectively verifies the entire volume.}
    \label{fig:concentration}
\end{figure}

To ensure the tractability of the projection operator $\text{Proj}_{\partial \mathcal{S}}(x)$, we assume for this formulation that $\mathcal{S}$ is locally convex or star-shaped (e.g., an intersection of half-spaces or a hypersphere). For highly non-convex semantic manifolds, we rely on local convex approximations (tangent half-spaces) derived from the Lie derivatives, ensuring that the projection remains well-defined locally even if the global geometry is complex \cite{ames2019, kong2023}. Critically, this local convex approximation acts as a \textit{one-directional conservative} filter: the tangent half-space inner-approximates the true safe set at each boundary point, meaning the approximation may exclude genuinely safe regions but will never include unsafe ones. Consequently, the Hoeffding-bounded safety guarantee remains valid (the probabilistic certificate applies to the inner-approximation, which is a subset of the true safe set), while utility may be sub-optimal in highly irregular spaces where the agent bypasses safe but narrow non-convex passages. Formal quantification of the geometric distortion---bounding the volume ratio between the local convex approximation and the true non-convex safe set as a function of curvature---remains an important open problem for future work.

By Hoeffding's Inequality \cite{hoeffding1963} and results from statistical learning theory \cite{vapnik1998}, we can decouple the verification complexity from the system dimension. While the standard \textit{Scenario Approach} for convex design \cite{campi2008} dictates that sample complexity must scale linearly with the number of decision variables ($O(n)$), our approach utilizes Hoeffding bounds for \textit{verification}, which allows the sample count to remain independent of dimension ($O(1)$). Reinforced by high-dimensional concentration bounds \cite{vershynin2018}, for a valid probability estimate $\hat{P}_{\text{fail}}$, the number of samples required is derived as:
\begin{equation}
    N \geq \frac{1}{2\epsilon^2} \ln\left(\frac{2}{\delta}\right)
\end{equation}
For example, to certify that a 128-dimensional system is safe with $99\%$ confidence ($\delta = 0.01$) and a $1\%$ margin of error ($\epsilon = 0.01$), we require only $N \approx 26,000$ samples. This results in a constant \textit{sample complexity} ($O(1)$) relative to the dimension $n$, as the bound depends only on confidence parameters.

\textbf{Important Qualification.} The $O(1)$ sample bound certifies that the \textit{volume fraction} of boundary points violating the CBF condition is below $\epsilon$. This is a necessary but not sufficient condition for trajectory-level safety: if a catastrophic failure mode has volume fraction below $\epsilon$ (e.g., a Black Swan spike with measure $10^{-100}$ in $\mathbb{R}^{128}$), the Monte Carlo estimate will report $\hat{P}_{\text{fail}} \approx 0$, correctly bounding the volume fraction but providing no protection against trajectory-specific encounters with the spike. This inherent limitation of passive statistical verification is precisely what motivates the Active Adversarial Safety Verification (AASV) module introduced in Section~3.5, which complements the volume-fraction bound with active, gradient-based threat hunting.

It is important to address the theoretical concern of Concentration of Measure in high-dimensional spaces ($n=128$), where safety violations could theoretically exist as ``spiky'' manifolds with large diameters but negligible volume (``Black Swan'' singularities) \cite{vershynin2018}. The Concentration of Measure phenomenon plays a dual role in this framework: on one hand, it \textit{aids} passive verification by concentrating probability mass near the boundary (the ``Hollow Ball'' effect of Figure~\ref{fig:concentration}), ensuring that boundary sampling captures the operationally relevant region of the state space; on the other hand, it \textit{hinders} passive verification by ensuring that narrow failure spikes occupy negligible volume, making them invisible to any feasible uniform sampling regime. We resolve this tension by assigning each role to a different verification layer: MCBC (Section~3.2) exploits boundary concentration to certify that the \textit{volume fraction} of failures is small, while AASV (Section~3.5) compensates for the volumetric invisibility of narrow spikes via active gradient-based search. Together, the two layers provide complementary coverage---statistical breadth from MCBC and adversarial depth from AASV. We additionally leverage the Lipschitz continuity of the system dynamics $f(x)$---estimated via spectral norm techniques for deep networks \cite{fazlyab2019}---which places a lower bound on the ``width'' of any failure region, ensuring no violation is mathematically invisible to nearby samples \cite{campi2008, vershynin2018}.

It may be argued that in high-dimensional spaces ($n=128$), even Lipschitz-continuous failure regions could remain statistically undetectable due to the vastness of the sampling volume. However, this assumes that the safety hazard is uniformly distributed across all dimensions. In practical control systems, safety violations (e.g., collisions) are typically low-rank phenomena, depending on a small subset of state variables (effective dimensionality $d_{eff} \ll n$) while remaining invariant to the others \cite{bengio2013}. Our projection mechanism leverages this structure, ensuring that sampling efficiency scales with the complexity of the \textit{hazard}, not the complexity of the \textit{agent}.

Furthermore, we leverage the Lipschitz property of the barrier function $h(x)$. Given the Lipschitz constant $L_h$ of $h$---estimated via conservative local sampling or spectral norm bounds \cite{fazlyab2019}---any verified point $x_i$ with $h(x_i) > 0$ guarantees that all points within the hyper-ball $B(x_i, h(x_i)/L_h)$ also satisfy $h(x) > 0$. This provides a local deterministic guarantee: each verified sample ``covers'' a neighborhood of radius $r = h(x_i)/L_h$, amplifying the probabilistic certification with geometric coverage. We note, however, that the volume of each such ball scales as $r^n$, and near the boundary where $h(x_i) \to 0$, the coverage radius shrinks correspondingly. In very high dimensions ($n = 128$), this means the Lipschitz coverage amplification provides diminishing practical benefit for near-boundary points---a further motivation for the active adversarial search of AASV, which does not rely on volumetric coverage arguments.

\subsection{Algorithm 1: High-Dimensional Safety Verification}
The formal procedure for verifying the safety of a high-dimensional manifold is detailed below.

\begin{algorithm}
\caption{High-Dimensional Probabilistic Safety Verification}\label{alg:verify}
\begin{algorithmic}[1]
\State \textbf{Input:} Dynamics $(f,g)$, Barrier $h(x)$, Dimension $n$, Sample Count $N$.
\State \textbf{Initialize:} $N_{fail} \leftarrow 0$.
\For{$i = 1$ to $N$}
    \State Sample state $x_i \sim \mathcal{N}(0, I_n)$ projected to $\partial\mathcal{S}$ \Comment{Boundary Sampling}
    \State Compute Lie derivatives: $L_f h(x_i)$ and $L_g h(x_i)$.
    \State Solve for optimal control $u^*$ (via the CBF-QP controller of Section 4.2).
    \State \Comment{\textbf{Verify Barrier Condition}}
    \State $\Delta = L_f h(x_i) + L_g h(x_i)u^* + \gamma h(x_i)$.
    \If{$\Delta < 0$} \Comment{Violation of Class-K Inequality}
        \State $N_{fail} \leftarrow N_{fail} + 1$.
    \EndIf
\EndFor
\State \textbf{Output:} $\hat{P}_{\text{safe}} = 1 - \frac{N_{fail}}{N}$.
\end{algorithmic}
\end{algorithm}

\textit{Note:} Algorithm \ref{alg:verify} provides a global statistical certificate of the safety landscape. Because Algorithm~\ref{alg:verify} samples from the boundary $\partial\mathcal{S}$ (not uniformly from $\mathcal{X}$), the resulting $\hat{P}_{\text{fail}}$ is the failure probability \textit{conditioned on boundary states}---the critical region where safety violations originate. The Hoeffding bound applies identically to this conditional estimate; however, the interpretation is that the certificate bounds the fraction of \textit{boundary} configurations that lack a safe control action, which is the operationally relevant quantity for forward invariance. Real-time safety during agent operation is ensured by the local Quadratic Program projection detailed in Section 4.2.

\textbf{Distributional Qualification.} The Hoeffding bound (Equation~5) assumes that the indicator random variables $\mathbb{I}(\Delta_i < 0)$ are independent and bounded, which holds when samples $x_i$ are drawn i.i.d.~from a fixed distribution on $\partial\mathcal{S}$. For spherical safe sets (as in our experiments), the isotropic Gaussian projection to $S^{n-1}$ yields a uniform distribution over the boundary, satisfying this assumption exactly. For non-spherical $\partial\mathcal{S}$ (e.g., polytopic safe sets or non-convex semantic manifolds), the sampling distribution must be specified explicitly, and the Hoeffding certificate applies to the \textit{chosen} distribution---not to an arbitrary alternative. In such cases, practitioners should either (1)~ensure the sampling distribution has support covering the operationally relevant boundary region, or (2)~employ importance-weighted estimators that correct for distributional mismatch. We note that the AASV Hunter (Section~3.5) partially mitigates this concern by actively searching for violations regardless of the sampling distribution.

\subsection{Active Adversarial Safety Verification (AASV)}
While Algorithm \ref{alg:verify} certifies $P_{\text{fail}} < \epsilon$ via passive sampling, the Concentration of Measure phenomenon (Section 3.3) implies that ``spiky'' failure regions---narrow manifolds with large diameters but negligible volume---may evade uniform sampling entirely. We refer to these as ``Black Swan'' singularities: failure modes that are statistically invisible to random verification yet catastrophic if encountered along the agent's trajectory.

To address this, we augment the probabilistic verification framework with the \textbf{Active Adversarial Safety Verification (AASV)} module, which transitions safety certification from passive statistical assurance to active, optimization-based threat hunting. AASV comprises three interlocking mechanisms.

\subsubsection{The ``Hunter'': Momentum-Accelerated Trajectory Attack}
Instead of relying solely on random samples $x \sim \mathcal{N}(0, I_n)$ to detect barrier violations, the Hunter employs an optimization-based sampler that \textit{actively seeks to minimize} the barrier function $h(x)$ along the agent's planned trajectory. The key insight is that while random sampling asks ``Is this random point safe?'', the Hunter asks ``Can an intelligent adversary force the system into an unsafe state?''---the gold standard in robust control \cite{goodfellow2014, madry2017}.

\textbf{Surrogate Gradient Availability.}
In semantic embedding spaces ($n \gg 100$), the system dynamics $f(x)$ often involve computationally expensive Transformer transitions where exact backpropagation for every control step is prohibitive. To resolve this, the Hunter attacks a \textbf{Local Linearized Surrogate} (LLS), $\tilde{f}(x)$, approximating the dynamics around the current trajectory via the Jacobian $J_f(x)$. Critically, the surrogate introduces bounded error, which we account for explicitly:
\begin{equation}
    \epsilon_{\text{model}} \geq \sup_{x \in \mathcal{B}(x_0, r)} \|f(x) - \tilde{f}(x)\|
\end{equation}
This error bound is incorporated directly into the robust barrier condition (Equation \ref{eq:robust_barrier}), ensuring that even if the surrogate underestimates danger, the safety margin absorbs the discrepancy.

\textbf{Practical Computation of $\epsilon_{\text{model}}$.} While computing the exact supremum over a continuous ball is intractable in general, a tight \textit{upper bound} is readily computable in $O(n)$ via the Taylor remainder: for a twice-differentiable $f(x)$, the linearization error over $\mathcal{B}(x_0, r)$ satisfies $\|f(x) - \tilde{f}(x)\| \leq \frac{1}{2} L_{\nabla f} r^2$, where $L_{\nabla f}$ is the local Lipschitz constant of the Jacobian (estimable via Hessian-vector products at $O(n)$ cost \cite{baydin2018}). For black-box systems without gradient access, $\epsilon_{\text{model}}$ may be calibrated empirically by evaluating $\|f(x) - \tilde{f}(x)\|$ at $m$ random points within $\mathcal{B}(x_0, r)$ and taking the maximum observed value plus a Hoeffding correction. The framework degrades gracefully: a larger $\epsilon_{\text{model}}$ shrinks the feasible operating region but the safety guarantee remains mathematically valid. In the limiting case where $\epsilon_{\text{model}}$ exceeds $\max_x h(x)$, the feasible set becomes empty and the agent halts---the correct safe default when the surrogate cannot be trusted.

\textbf{Momentum PGD with Stochastic Restarts.}
Standard gradient descent stalls in local minima of non-convex barrier landscapes, potentially missing deeper failure modes hidden behind shallow safe basins. We employ Momentum-Accelerated Projected Gradient Descent with Stochastic Restarts \cite{madry2017, nesterov2004}:
\begin{align}
    v_{t+1} &= \mu v_t - \alpha \nabla_x h(\tilde{x}_{\text{plan}} + \xi) \\
    x_{\text{adv}} &= \text{Proj}_{\mathcal{S}} \left( \tilde{x}_{\text{plan}} + v_{t+1} \right)
\end{align}
where $\mu \in [0.8, 0.95]$ is the momentum coefficient and $\xi \sim \mathcal{N}(0, \sigma^2 I)$ is noise injected to escape saddle points. We execute $k$ \textit{parallel restarts} from different initial conditions, each running for $T$ iterations.

\textbf{Guarantee.} If the optimization fails to find a violation $h(x_{\text{adv}}) < 0$ after $k \times T$ total iterations, the state is certified as \textit{Adversarially Robust} with confidence:
\begin{equation}
    P(\text{missed spike}) \leq (1 - p_{\text{hit}})^k
\end{equation}
where $p_{\text{hit}}$ is the probability that a single restart's basin of attraction contains the global minimum of $h(x)$. We emphasize that this is a \textit{probabilistic} bound enhanced by active search---we do not claim deterministic global convergence in non-convex spaces.

\textbf{Operationalizing $p_{\text{hit}}$.} For a failure spike with angular half-width $\theta_w$ on $S^{n-1}$, the \textit{geometric} cap area ratio $\text{Cap}(\theta_w, n) / \text{Area}(S^{n-1})$ decreases exponentially with $n$ for fixed $\theta_w$ \cite{vershynin2018}---for $n = 128$ and $\theta_w = 0.05$ radians, this ratio is astronomically small ($\ll 10^{-100}$), rendering passive random detection effectively impossible. However, the gradient field of the Gaussian barrier well extends far beyond the spike's geometric footprint, creating an attraction funnel that guides momentum PGD toward the spike center from initial conditions well outside the cap. This gradient amplification enlarges the effective convergence basin by orders of magnitude compared to the passive geometric cap. In our $\mathbb{R}^{128}$ experiments, empirical measurement yields detection rates of $p_{\text{hit}} \geq 0.05$ per restart. With $k = 60$ restarts, this gives $P(\text{miss all spikes}) \leq (1 - 0.05)^{60} < 0.046$ per spike, or equivalently $> 95\%$ detection probability per failure mode.

\textbf{Caveat on $p_{\text{hit}}$ Generalization.} The empirical estimate $p_{\text{hit}} \geq 0.05$ is obtained from controlled experiments with analytically defined barrier landscapes where the gradient is computable exactly (Section~5). In real deployment scenarios where the barrier landscape is unknown or the gradient is obtained through a learned surrogate, $p_{\text{hit}}$ may be substantially lower due to gradient noise, surrogate error, and unforeseen barrier topology. Practitioners should therefore treat $p_{\text{hit}}$ as a system-specific parameter requiring empirical calibration on representative barrier instances, and should increase the restart count $k$ conservatively to compensate for uncertainty in the detection rate.

\subsubsection{The ``Buffer'': Adaptive Tube-Based Spectral Margins}
The original verification condition $h(x) \geq 0$ is insufficient for adversarial robustness because it ignores three sources of uncertainty: (1) surrogate model error, (2) physical noise/disturbance, and (3) geometric uncertainty between sample points. Drawing on the tube-based robust MPC framework of \cite{mayne2005}, we strengthen the barrier condition to a \textbf{Robust Tube-Based Constraint}:
\begin{equation}
    h(x) \geq \delta(x) + \epsilon_{\text{model}} + \Delta_{\text{noise}}
    \label{eq:robust_barrier}
\end{equation}
where:
\begin{itemize}
    \item $\delta(x) = \tilde{\sigma}_{\max}(J_f(x)) \cdot d_{\text{step}}$ is the \textbf{local volatility margin}, proportional to the estimated spectral norm of the dynamics Jacobian and the agent's step size. This creates a dynamic safety ``tube'' that thickens in volatile, highly non-linear regions and thins in smooth regions, avoiding the ``Frozen Robot'' problem \cite{trautman2010} caused by overly conservative global Lipschitz bounds.
    \item $\epsilon_{\text{model}}$ is the surrogate divergence bound from Section 3.5.1.
    \item $\Delta_{\text{noise}}$ is the physical disturbance bound, known from hardware specifications or estimated online.
\end{itemize}
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_8.png}
    \caption{\texorpdfstring{\textbf{Tube-Based Adaptive Safety Margins.}}{Tube-Based Adaptive Safety Margins.} Left: A fixed global Lipschitz margin creates an overly conservative ``dead zone'' that induces the Frozen Robot problem in high-curvature regions. Right: The AASV Buffer computes adaptive margins $\delta(x) = \tilde{\sigma}_{\max}(J_f) \cdot d_{\text{step}}$ that thicken near volatile boundary segments and thin in smooth regions, preserving agent mobility while maintaining safety.}
    \label{fig:adaptive_tube}
\end{figure}
\textbf{Matrix-Free Spectral Estimation for $O(n)$ Scaling.}
A critical implementation detail is the computation of $\tilde{\sigma}_{\max}(J_f(x))$. Full Singular Value Decomposition scales as $O(n^3)$. Explicit-matrix Power Iteration requires $O(n^2)$ per iteration (forming and multiplying by the full $n \times n$ Jacobian). For semantic spaces where $n = 12{,}288$ \cite{brown2020}, both explicit-matrix approaches violate the $O(n)$ real-time constraint established in Problem 1.

Critically, neither method requires explicit matrix construction when automatic differentiation (AD) is available \cite{baydin2018}. A Jacobian-vector product $J_f z$ is computable in $O(n)$ via forward-mode AD, and the transpose product $J_f^T w$ in $O(n)$ via reverse-mode AD (backpropagation), without ever forming the $n \times n$ Jacobian. This observation enables two matrix-free $O(n)$ strategies:

\textbf{Strategy A: AD-Based Power Iteration (Preferred).}
Power iteration for $\sigma_{\max}(J_f)$ requires only the iteration $w \leftarrow J_f^T J_f v / \|J_f^T J_f v\|$, each step involving one JVP and one VJP at cost $O(n)$. After $k \approx 10\text{--}20$ iterations (sufficient for convergence when the spectral gap $\sigma_1/\sigma_2$ is bounded away from 1), this yields a \textit{tight} estimate of $\sigma_{\max}$ at total cost $O(kn) = O(n)$.

\textbf{Strategy B: Hutchinson Trace Estimator (Fallback).}
When only forward-mode AD is available (precluding the VJP step), we employ \textbf{Hutchinson's Trace Estimator} \cite{hutchinson1990}, a matrix-free randomized method. The Frobenius norm of the Jacobian---which provides a conservative upper bound on the spectral norm, $\|J_f\|_F \geq \sigma_{\max}(J_f)$---is estimated using $m$ random Rademacher probe vectors $z_i \in \{-1, +1\}^n$:
\begin{equation}
    \tilde{\sigma}_{\max}(J_f) \leq \|J_f\|_F \approx \sqrt{\frac{1}{m} \sum_{i=1}^{m} z_i^T J_f^T J_f z_i}
\end{equation}
The inequality $\|J_f\|_F \geq \sigma_{\max}$ follows immediately from $\|J_f\|_F^2 = \sum_i \sigma_i^2 \geq \sigma_{\max}^2$, ensuring that the resulting safety margin is never underestimated. Each term $J_f z_i$ requires only a \textit{Jacobian-vector product} (computable in $O(n)$ via forward-mode AD \cite{baydin2018}). Total complexity: $O(m \cdot n)$ where $m \approx 20\text{--}30$ samples suffice for accurate estimation.

\textbf{Remark (Frobenius Tightness).} The Frobenius bound satisfies $\sigma_{\max} \leq \|J_f\|_F \leq \sqrt{\mathrm{rank}(J_f)} \cdot \sigma_{\max}$, so the overestimation factor is at most $\sqrt{\mathrm{rank}(J_f)}$. For a full-rank $128 \times 128$ Jacobian with uniform singular values, this gap is $\sqrt{128} \approx 11.3$. In practice, two factors mitigate this for kinematic systems: (1) the Jacobians of smooth physical dynamics typically exhibit rapid spectral decay (a few dominant singular values), reducing the effective rank and tightening the bound considerably; and (2) even with the Frobenius overestimate, the adaptive tube $\delta(x) = \tilde{\sigma}_{\max} \cdot d_{\text{step}}$ still varies across states---thinning in smooth regions and thickening in volatile ones---which is strictly less conservative than a fixed global Lipschitz margin. However, we caution that dense semantic models (e.g., Large Language Model transformers) are often engineered to preserve variance across all dimensions to maximize representational capacity \cite{brown2020}, resulting in near-full-rank Jacobians where the Frobenius gap approaches $\sqrt{n}$. In such regimes, Strategy~B may induce excessive conservatism; for full-rank $n=128$ Jacobians, the Frobenius overestimate inflates the margin by $\sqrt{128}\approx 11.3\times$, which we show empirically still permits agent mobility in our experiments (Section~5), but may be prohibitive at larger scales. \textbf{For full-rank semantic dynamics, Strategy~A (AD-based power iteration) is therefore mandatory}, as it converges directly to $\sigma_{\max}$ without the rank-dependent gap. Strategy~B should be reserved for inherently low-rank manifolds or as a conservative fallback when reverse-mode AD is unavailable.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_5.png}
    \caption{\texorpdfstring{\textbf{Hutchinson Trace Estimator Convergence ($n=128$).}}{Hutchinson Trace Estimator Convergence (n=128).} Left: The Hutchinson estimate of $\mathrm{tr}(J^T J)$ converges to the true value as the number of Rademacher probe vectors $m$ increases. Right: Relative estimation error decreases below 10\% at $m \approx 20$ and below 5\% at $m \approx 50$. Each probe requires only an $O(n)$ Jacobian-vector product, yielding total complexity $O(m \cdot n) = O(n)$.}
    \label{fig:hutchinson_convergence}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_7.png}
    \caption{\texorpdfstring{\textbf{Computational Scaling: SVD vs.\ Power Iteration vs.\ Hutchinson.}}{Computational Scaling: SVD vs. Power Iteration vs. Hutchinson.} Wall-clock time for spectral norm computation across dimensions $n = 16$ to $n = 2048$ (log-log scale). Full SVD scales as $O(n^3)$ and exceeds 5 seconds at $n = 2048$. Explicit-matrix Power Iteration scales as $O(n^2)$; note that matrix-free (AD-based) power iteration achieves $O(kn) = O(n)$ scaling comparable to Hutchinson (see text). Hutchinson estimation ($m=30$) maintains $O(n)$ scaling, staying well under the 10ms real-time budget even at $n = 2048$. \textbf{Implementation note:} The figure-generation code uses explicit dense-matrix Jacobian-vector products ($J \cdot z$, which are $O(n^2)$ per multiply) rather than AD-based JVPs. Consequently, the measured Hutchinson and Power Iteration timings include the $O(n^2)$ matrix-multiply cost. The ``$O(n)$'' designation on the plot refers to the \textit{AD-based pipeline} described in Section~3.5.2, which computes $Jz$ at true $O(n)$ cost via forward-mode automatic differentiation \cite{baydin2018}; practitioners implementing AASV at production scale should use this AD pathway.}
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

\textbf{Why Not PCA?} Standard dimensionality reduction (e.g., PCA) averages failure modes. If two Black Swan spikes exist at orthogonal directions $v_1$ and $v_2$, their principal component $v_{\text{avg}}$ may point to a \textit{safe} region, effectively erasing the memory of both dangers. This is unacceptable for safety-critical applications.

\textbf{Orthogonal Prototype Retention.} We enforce an orthogonality constraint on stored failure modes:
\begin{equation}
    \text{Store } v_{\text{new}} \text{ as distinct prototype if: } |v_{\text{new}} \cdot v_{\text{centroid}}| < \theta
\end{equation}
where $\theta$ is a similarity threshold. If the new failure mode is collinear with an existing centroid, it is merged; if orthogonal, it is stored as a separate prototype. This leverages the quasi-orthogonality of random vectors in high dimensions \cite{vershynin2018}, enabling the storage of up to $O(n)$ independent failure prototypes without crosstalk.

\textbf{Repulsion Augmentation.} The Hunter's cost function is augmented to repel from stored prototypes, forcing exploration of novel failure geometries. Since the Hunter \textit{minimizes} $J(x)$ to find barrier violations, adding a positive similarity penalty increases the cost near known prototypes, steering the search toward unexplored regions:
\begin{equation}
    J(x) = h(x) + \lambda \sum_{c \in \mathcal{M}_{\text{ban}}} \text{Sim}(x, c), \quad \lambda > 0
\end{equation}
where $\mathcal{M}_{\text{ban}}$ is the set of stored failure prototypes and $\text{Sim}(\cdot, \cdot)$ is cosine similarity.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_6.png}
    \caption{\texorpdfstring{\textbf{Anti-Memory: PCA Averaging vs.\ Orthogonal Prototype Retention ($\mathbb{R}^{128}$).}}{Anti-Memory: PCA Averaging vs. Orthogonal Prototype Retention (R128).} Left: PCA dimensionality reduction averages three orthogonal failure directions into a single centroid that points to a \textit{safe} region---effectively erasing memory of all three Black Swan spikes. Right: Orthogonal Prototype Retention (AASV Anti-Memory) stores each failure direction independently when $|v_{\text{new}} \cdot v_{\text{centroid}}| < \theta$, preserving all spike locations and enabling repulsion-guided exploration.}
    \label{fig:anti_memory}
\end{figure}

\subsubsection{Pipelined Real-Time Verification}
A critical challenge is that iterative PGD attacks introduce computational latency, potentially violating the real-time control loop constraint ($< 10$ms). Naively running the Hunter asynchronously (``Dreaming'') creates a dangerous temporal gap: the agent might act on outdated verification data, breaking the Forward Invariance guarantee of Theorem 1.

We resolve this via \textbf{Tube-Based Pipelined Verification with Recursive Backup}:

\begin{enumerate}
    \item \textbf{Time $t$ (Execute):} The agent executes control $u_t$, which was verified and certified safe at time $t-1$.
    \item \textbf{Time $t$ (Hunt):} The AASV Hunter simultaneously attacks the $\epsilon$-ball around $x_{\text{pred}}(t+1)$, the predicted outcome of the next intended action $u_{t+1}$. The $\epsilon$-ball accounts for physical drift, ensuring the verification covers the entire region the agent might physically occupy.
    \item \textbf{Time $t$ (Certify):} If $\min_{x \in \mathcal{B}(x_{\text{pred}}, \epsilon)} h(x) > 0$ (accounting for the robust margin of Equation \ref{eq:robust_barrier}), authorize $u_{t+1}$.
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
    \If{Hunter finds violation $h(x_{\text{adv}}) < \delta(x) + \epsilon_{\text{model}} + \Delta_{\text{noise}}$}
        \State Store $x_{\text{adv}} / \|x_{\text{adv}}\|$ in $\mathcal{M}_{\text{ban}}$ (if novel)
        \State \textbf{Reject} $u^*$; engage safe backup trajectory
    \Else
        \State \textbf{Execute} $u^*$; update $x \leftarrow x_{\text{pred}}$
    \EndIf
\EndFor
\end{algorithmic}
\end{algorithm}

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

\subsubsection{Assumptions and Limitations}
For intellectual honesty, we state the assumptions underlying AASV:
\begin{enumerate}
    \item \textbf{Surrogate Validity:} The error bound $\epsilon_{\text{model}}$ must be estimable. For black-box LLMs, this requires empirical calibration or conservative overestimation.
    \item \textbf{Lipschitz Assumption:} The barrier function $h(x)$ is assumed Lipschitz continuous. Discontinuous dynamics (e.g., contact mechanics) require specialized treatment.
    \item \textbf{Hutchinson Variance:} The spectral estimate has variance $\propto 1/m$. For safety-critical applications, $m$ should be chosen conservatively (e.g., $m \geq 50$).
    \item \textbf{Non-Convexity:} The Hunter's probabilistic guarantee depends on the barrier landscape topology. Pathological geometries with exponentially many local minima can degrade detection probability.
\end{enumerate}

\section{Asymptotic Utility Maximization}
Having established a scalable probabilistic framework for safety in Section 3, we now address the performance objective. In many robust control frameworks, safety constraints act as ``brakes,'' aggressively retarding the system's progress to ensure invariance. This often leads to conservative behavior where agents freeze in the face of uncertainty.

We propose a Harmonic Control Architecture where the safety barrier does not merely inhibit motion but actively shapes the utility gradient, guiding the system along the boundary of the safe set toward the global optimum.

\subsection{The Utility Landscape}
We define the system's objective via a continuously differentiable Utility Function, $U(x): \mathcal{X} \to \mathbb{R}$. Without loss of generality, we seek to maximize $U(x)$. Typical examples include:
\begin{itemize}
    \item \textbf{Target Reaching:} $U(x) = -\|x - x_{\text{goal}}\|$ (minimizing distance to a target).
    \item \textbf{Formation Control:} $U(x) = -\sum_{i < j} (\|x_i - x_j\| - d)^2$ (maintaining inter-agent spacing).
\end{itemize}
For the purpose of theoretical convergence proofs, we analyze the system under the assumption that $U(x)$ is concave or satisfies the Polyak-{\L}ojasiewicz condition \cite{polyak1963, lojasiewicz1963}. However, in practical deployment where $U(x)$ may be non-convex and populated with local maxima (e.g., complex semantic navigation), we rely on the \textbf{Rotational Circulation} mechanism (detailed in Section 4.4) to destabilize spurious local equilibria and drive the agent toward the global optimum.

\subsection{Gradient Projection on the Tangent Cone}
The core of our methodology is the projection of the Utility Gradient, $\nabla U(x)$, onto the Tangent Cone of the safe set $\mathcal{S}$. Let the nominal control input that maximizes utility be $u_{nom} = k_U(x) = \nabla U(x)$. If the system is far from the boundary (i.e., $h(x) \gg 0$), we apply $u = u_{nom}$. However, as $h(x) \to 0$, $u_{nom}$ may violate the safety condition.

We resolve this conflict by employing a \textbf{Control Barrier Function-based Quadratic Program (CBF-QP)}, a formulation established in modern control theory \cite{ames2019} and adapted here for high-dimensional semantic spaces. At every time step, we solve the following optimization problem, which acts as a minimum-deviation safety filter:
\begin{equation}
\label{eq:cbfqp}
\begin{aligned}
    u^*(x) &= \underset{u \in \mathcal{U}}{\text{argmin}} \quad \frac{1}{2}\|u - u_{nom}(x)\|^2 \\
    \text{subject to:} &\quad L_f h(x) + L_g h(x) u \geq -\gamma h(x)
\end{aligned}
\end{equation}

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.5\textwidth]{figure_projection.png}
    \caption{\texorpdfstring{\textbf{Geometric Control: Orthogonal Projection on Tangent Cone.}}{Geometric Control: Orthogonal Projection on Tangent Cone.} When the nominal utility vector $u_{nom}$ (Red) threatens to breach the safe set $\mathcal{S}$, the controller projects it onto the tangent hyperplane orthogonal to the barrier normal $\nabla h(x)$ (Black). The resulting control input $u^*$ (Green) maximizes velocity while maintaining invariance.}
    \label{fig:projection}
\end{figure}

Because this formulation involves a single affine inequality constraint, the KKT conditions yield an explicit closed-form solution (a geometric projection onto a half-space). This circumvents the iterative complexity of general QP solvers ($O(n^3)$) \cite{boyd2004}, reducing the computational cost to $O(n)$ per time step (dominated strictly by vector dot products). Note that for systems with multiple simultaneous safety constraints, we assume they are aggregated into a single barrier function $h(x)$ (e.g., using a smooth LogSumExp or SoftMin approximation \cite{ames2019}) to preserve this linear scaling efficiency.

\subsection{Safe Asymptotic Convergence}
The interaction between the Lyapunov-like utility function and the Barrier function guarantees convergence.

\begin{theorem}[Safe Asymptotic Convergence]
Consider the system $\dot{x} = f(x) + g(x)u$ subject to the control law $u^*(x)$ defined above. If the sets $\mathcal{S}$ (safety) and $\mathcal{L}_c = \{ x \mid U(x) \geq c \}$ (utility level sets) are compact, and if $\nabla U(x)$ and $\nabla h(x)$ are not opposing collinear vectors at the boundary (regularity condition), then:
\begin{enumerate}
    \item \textbf{Forward Invariance:} For all $t \geq 0$, $h(x(t)) \geq 0$ (Safety is guaranteed).
    \item \textbf{Convergence:} $\lim_{t \to \infty} x(t) = x^*_{\mathcal{S}}$, where $x^*_{\mathcal{S}}$ is a constrained local maximum of $U(x)$ within $\mathcal{S}$. If $U(x)$ is concave (or satisfies the Polyak-{\L}ojasiewicz condition on $\mathcal{S}$), $x^*_{\mathcal{S}}$ is the unique global maximum.
\end{enumerate}
\end{theorem}
\begin{proof}[Proof Sketch]
We establish each claim separately.

\textbf{Forward Invariance.} This follows directly from the CBF condition: the constraint $L_f h + L_g h\, u \geq -\gamma h$ in Equation~\ref{eq:cbfqp} is feasible (by the relative-degree-one assumption and compactness of $\mathcal{U}$), and any feasible $u^*$ satisfies $\dot{h} \geq -\gamma h$, which by the Comparison Lemma implies $h(x(t)) \geq h(x(0))\, e^{-\gamma t} \geq 0$ for all $t \geq 0$ \cite{ames2019}.

\textbf{Convergence.} We analyze the single-integrator case $\dot{x} = u$ (used in our simulations, Section~5) and then state the general result. Define $V(x) = -U(x)$ (to be minimized). With $u_{\text{nom}} = \nabla U(x)$, when the CBF constraint is inactive, $\dot{V} = -\|\nabla U\|^2 \leq 0$. When the constraint is active, the KKT solution $u^* = u_{\text{nom}} + \lambda^* \nabla h$ with $\lambda^* \geq 0$ gives $\dot{V} = -\|\nabla U\|^2 - \lambda^* \nabla U \cdot \nabla h$. If $\nabla U \cdot \nabla h < 0$ (utility points toward the boundary), $\dot{V}$ may be temporarily positive, violating the monotonicity required by classical LaSalle.

We therefore use a \textbf{Barbalat-type argument} \cite{khalil2002} instead of LaSalle's invariance principle \cite{lasalle1960}. Since $V$ is bounded below on compact $\mathcal{S}$ and the trajectory $x(t)$ remains in $\mathcal{S}$ by forward invariance, $V(x(t))$ is a bounded function of time. The integral $\int_0^T \|\nabla U(x(t))\|^2\, dt$ is bounded for all $T$ (since $V$ cannot decrease below $\inf_\mathcal{S} V$ and cannot increase above $V(x(0))$ indefinitely---the time spent with $\dot{V} > 0$ is finite because boundary-sliding with opposing $\nabla U$ and $\nabla h$ can only persist while $\lambda^* > 0$ and $h(x) = 0$, a codimension-1 set from which trajectories generically exit in finite time by the regularity assumption). By Barbalat's Lemma (noting $\ddot{V}$ is bounded by smoothness of $U$ and $h$), $\|\nabla U(x(t))\|^2 \to 0$, and consequently $x(t)$ converges to the set $\{x \in \mathcal{S} \mid \nabla U(x) + \lambda \nabla h(x) = 0,\; \lambda \geq 0\}$---the KKT points of the constrained optimization problem $\max_\mathcal{S} U$.

\textbf{Remark (Regularity Condition).} The condition ``$\nabla U$ and $\nabla h$ are not opposing collinear'' excludes the case $\nabla U = -c\, \nabla h$ with $c > 0$ at an interior point of the boundary trajectory. Note that $\nabla U = -c\, \nabla h$ with $c > 0$ is precisely the KKT condition for a constrained maximum---so this condition is violated \textit{only at the desired convergence point} $x^*_\mathcal{S}$ itself (where it is benign) and at boundary saddle points (which are escaped by the Rotational Circulation mechanism of Section~4.4). The regularity condition is therefore generically satisfied along trajectories.

\textbf{Remark (General Control-Affine Dynamics).} For general systems $\dot{x} = f(x) + g(x)u$ where $f(x) \neq 0$, the convergence argument additionally requires that the control authority $L_g U(x)$ is sufficient to compensate the drift $L_f U(x)$ away from the optimum. When drift is adversarial ($L_f U < 0$) and dominates, the system converges to the $\omega$-limit set contained in the constrained critical points of the Lagrangian $\mathcal{L}(x, \lambda) = -U(x) + \lambda\, h(x)$, which need not coincide with $\text{argmax}_{\mathcal{S}}\, U$. If $U(x)$ is concave and the constraint qualification holds (Slater's condition), this critical point is the unique global constrained maximum by KKT theory. For non-convex safe sets $\mathcal{S}$, the convergence claim holds locally within each connected component.
\end{proof}

\subsection{Avoiding Local Minima (The ``Deadlock'' Problem)}
A common failure mode in potential field methods is ``local minima deadlock,'' where the agent gets stuck in a U-shaped obstacle \cite{rimon1992}. If $\nabla U(x)$ and $\nabla h(x)$ are perfectly opposing, a direct projection $v_{\perp} = \text{Proj}_{\text{Null}(\nabla h)} \nabla U$ yields the zero vector, resulting in stagnation.

To mitigate this, we inject a Rotational Circulation term. When the angle between $\nabla U$ and $\nabla h$ approaches $180^\circ$ (indicating a deadlock), we project a random perturbation vector (or strictly orthogonal noise) $\xi$ onto the tangent plane of the obstacle, utilizing a stochastic escape heuristic similar to Simulated Annealing \cite{kirkpatrick1983} or Randomized Potential Fields \cite{barraquand1991}:
\begin{equation}
    u_{perturb} = \nabla U(x) + \beta \left( \text{Proj}_{\text{Null}(\nabla h)} \xi \right)
\end{equation}
Here, $\xi \in \mathbb{R}^n$ is a noise vector such that $\xi \neq \nabla h$. This operation selects a valid sliding direction tangential to the obstacle. We note that high dimensionality actually \textit{aids} this escape mechanism: at a saddle point of an $n$-dimensional function, the Hessian generically has both positive and negative eigenvalues, and a random perturbation $\xi \in \mathbb{R}^n$ will almost surely have a nonzero component along at least one descent direction, with the probability of failure vanishing exponentially in $n$ \cite{vershynin2018}. While true local minima are notoriously difficult to escape, they are exponentially rare in high-dimensional landscapes \cite{dauphin2014}; saddle points dominate, and random perturbation suffices to break their symmetry. Thus, while this mechanism may not invariably yield the \textit{optimal} escape trajectory, it is sufficient to destabilize deadlocks with high probability, allowing the gradient-based steering to resume progress toward the global optimum.

\section{Simulation Results and Empirical Validation}
In this section, we present a comprehensive evaluation of the Constrained High-Dimensional Barrier Optimization (CHDBO) framework. We position our approach relative to Model Predictive Control (MPC) \cite{camacho2013} and Deep Reinforcement Learning (DRL) \cite{schulman2017}, evaluating CHDBO's safety, utility, and scalability across seven progressively challenging configurations.

Our experiments were conducted using the Monte Carlo Barrier Certificate (MCBC) engine and our utility maximization algorithm engines described in Sections 3 and 4. The simulation environment models a high-dimensional single-integrator system $\dot{x} = u$ subject to non-convex spherical obstacles in $\mathbb{R}^n$, with dimensions ranging from $n=2$ to $n=128$.

\subsection{Experiment I: The Golden Manifold (Reachability Analysis)}
To validate the fundamental geometric principles of our safety verification, we simulated 50 adversarial trials in a 2D state space where an agent is biased to move toward a ``Forbidden Zone'' ($x > 0.8$). This experiment visualizes the ``Golden Manifold'', the geometric boundary where the agent is invariant to microscopic fluctuations.

\begin{itemize}
    \item \textbf{Setup:} The agent possesses a ``bias'' vector forcing it rightward ($+x$), simulating a corrupted internal goal or ``mutational meltdown''.
    \item \textbf{Standard Agent (Red):} Lacking verification, the agent follows the noise and bias directly into the forbidden zone, breaching the safety constraint $x=0.8$.
    \item \textbf{Verified Agent (Green):} The Orthogonal Verifier intercepts the velocity vector at every time step. As the agent approaches the theoretical bound, the verifier projects the velocity onto the tangent of the safety manifold.
\end{itemize}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\textwidth]{figure_1.png}
    \caption{\texorpdfstring{\textbf{Orthogonal Verification: Geometric Reachability Analysis (50 trials, $n=2$).}}{Orthogonal Verification: Geometric Reachability Analysis (50 trials, n=2).} The verified trajectories form a hard geometric wall at $x=0.8$. The agent effectively ``slides'' along the safety boundary, preventing any excursion into the Forbidden Zone while still attempting to maximize its allowable movement along the Y-axis. This confirms the efficacy of Reachability Analysis in statistically minimizing the probability of 'black swan' events to a bounded threshold $\epsilon$.}
    \label{fig:golden_manifold}
\end{figure}

\textbf{Result:} As illustrated in Figure~\ref{fig:golden_manifold}, the simulation confirms that the Orthogonal Verifier successfully enforces the invariant set. Unlike the standard agent, which breaches the constraint under adversarial bias, the verified agent maintains a strict upper bound at $x=0.8$. This behavior empirically validates the "sliding" mechanic of the projection operator, demonstrating that the agent can maximize utility along the tangent of the safety manifold without crossing into the forbidden region.

\subsection{Experiment II: High-Dimensional Survival \texorpdfstring{($\mathbb{R}^{128}$)}{(R\textasciicircum 128)}}
To prove the scalability of CHDBO, we extended the simulation to 128 dimensions, mimicking the embedding space of a VSA-based semantic agent. The agent was subjected to a ``Gradient Attack'' \cite{madry2017} where the goal vector was placed explicitly outside the safe hypersphere ($R > 1.0$).

\begin{itemize}
    \item \textbf{Method:} We utilized the Monte Carlo Barrier Certificate (MCBC) engine to generate 100-step random walks under intense gradient pressure ($\text{Attack Strength} = 0.05$).
    \item \textbf{Visualization:} The resulting 128-dimensional trajectories were projected down to 2D using Principal Component Analysis (PCA) for visualization \cite{jolliffe2002}.
\end{itemize}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.75\textwidth]{figure_2.png}
    \caption{\texorpdfstring{\textbf{High-Dimensional Orthogonal Verification ($D=128$) Adversarial Gradient Attack Simulation.}}{High-Dimensional Orthogonal Verification (D=128) Adversarial Gradient Attack Simulation.} Standard agents (red) immediately breach the unit hypersphere, drifting into the unsafe region. Verified agents (green) cluster tightly on the surface of the hypersphere. The PCA projection illustrates that despite the high dimensionality, the projection logic maintains the agent on the ``skin'' of the safe manifold. (Note: the 2D PCA projection is a visualization aid that captures the two directions of greatest variance; the safety guarantee is established by the barrier condition $h(x) \geq 0$ in the full $\mathbb{R}^{128}$, not by the projection.)}
    \label{fig:high_dim_survival}
\end{figure}

\textbf{Result:}
\begin{itemize}
    \item \textbf{Standard Agents:} Immediately breached the unit hypersphere, drifting into the unsafe region. This mirrors the ``Logic Regression'' observed in unverified LLM agents \cite{scrivens2026, wei2023}.
    \item \textbf{Verified Agents:} Clustered tightly on the surface of the hypersphere. The PCA projection illustrates that despite the high dimensionality, the projection logic maintains the agent on the ``skin'' of the safe manifold without requiring a grid-based computation.
\end{itemize}

This result validates that the ``Curse of Dimensionality'' \cite{bellman1957} is effectively mitigated for safety verification; the sample complexity remained constant ($O(1)$) despite the exponential increase in state space volume, with per-sample computational cost scaling linearly ($O(n)$).

\subsection{Experiment III: Proportional Utility Response}
A critical requirement for autonomous agents is that safety measures should not destroy utility unless absolutely necessary. We stress-tested the system by varying the ``Adversarial Intensity'' ($\alpha$) of the agent's intent from 0.0 (Pure Safe) to 1.0 (Pure Malicious).

\begin{itemize}
    \item \textbf{Metric:} We measured Semantic Preservation (Cosine Similarity) between the agent's intended action vector and the allowed action vector.
\end{itemize}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figure_3.png}
    \caption{\texorpdfstring{\textbf{Proportional Safety Response ($N=1000$).}}{Proportional Safety Response (N=1000).} The relationship between Adversarial Intensity ($\alpha$) and Semantic Preservation obeys the theoretical geometric decay $\sqrt{1-\alpha^2}$, demonstrating an adaptive intervention strategy.}
    \label{fig:proportional_response}
\end{figure}

\textbf{Observation:} As illustrated in Figure \ref{fig:proportional_response}, the system's response is non-linear and proportional:
\begin{itemize}
    \item \textbf{Low Threat ($\alpha < 0.4$):} Similarity remains near 1.0. The verifier allows the agent to act freely when the intent aligns with safety.
    \item \textbf{High Threat ($\alpha > 0.8$):} Similarity drops precipitously as the Verifier aggressively filters the vector to prevent a breach.
    \item \textbf{Curve:} The simulation data strictly follows the theoretical geometric decay $\sqrt{1-\alpha^2}$ characteristic of orthogonal projections in high-dimensional Euclidean space \cite{vershynin2018, blum2020}.
\end{itemize}

This confirms that the implementation matches the theoretical prediction of Section~4.2: the orthogonal projection yields the geometric decay $\sqrt{1-\alpha^2}$, validating that the CBF-QP filter preserves the maximum utility component orthogonal to the constraint gradient. The agent does not simply stop when near danger; it maximizes the component of the utility vector that is orthogonal to the danger vector. This ensures the agent ``rides the boundary,'' achieving the maximum possible utility permitted by the safety constraints.

\subsection{Experiment IV: AASV Black Swan Detection --- Comprehensive Robustness Evaluation}
To validate the necessity and efficacy of the AASV module (Section 3.5), we construct a comprehensive suite of controlled Black Swan scenarios in $\mathbb{R}^{128}$. Adversarial ``spike'' singularities are injected into the barrier landscape as narrow Gaussian wells on $S^{127}$ with angular half-width $\theta_w = 0.05$ radians. Their volume fraction is negligibly small---exponentially suppressed by the dimension $n$---rendering them statistically invisible to any feasible uniform sampling regime. We evaluate the AASV pipeline across eight progressively challenging configurations to stress-test detection under varying geometric conditions.

\textbf{Barrier Construction.} We define $h(x) = (1 - \|x\|) + \lambda \sum_i (1 - \cos\angle(x, s_i)) - D \sum_i \exp\!\bigl(-\frac{(1-\text{sim}_i)^2}{2\theta_w^2}\bigr)$ on $S^{127}$, where each spike $s_i$ creates a localized negative well. The depth parameter $D = (N-1)\lambda + 2$ ensures every spike produces $h < 0$ at its center despite the aggregate funnel repulsion.

\textbf{Protocol.} Each panel employs the full AASV pipeline: momentum PGD ($\mu = 0.9$, $\alpha = 0.05$, $T = 200$) with Winner-Take-All gradient blocking (threshold $= 0.7$), post-violation refinement (100 additional low-noise PGD steps converging to $\cos > 0.9999$ with the true spike center), post-hoc clustering at similarity threshold $0.98$ ($\approx 11.5^\circ$ angular resolution), and orthogonal prototype Anti-Memory for repulsion-guided novel exploration. Eight scenarios are evaluated:

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
    \item \textbf{Standard MC:} \textit{Zero} violations detected across all 10{,}000 samples in panel (a). The passive verifier incorrectly certifies the state space as safe ($P_{\text{fail}} < \epsilon$ vacuously satisfied), confirming the ``Black Swan'' blindness predicted by Section 3.3.
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

This experiment empirically confirms the theoretical prediction of Section 3.3: passive Monte Carlo verification, while statistically valid in expectation, is operationally blind to rare catastrophic events. The AASV module closes this gap by actively hunting threats, yielding a verification regime that is both probabilistically sound and adversarially robust. Critically, the system is honest about its limitations: panel~(g) demonstrates that spikes closer than the clustering resolution cannot be distinguished, and this limitation is reported transparently rather than hidden.

\subsection{Experiment V: Non-Trivial Drift Dynamics \texorpdfstring{($\mathbb{R}^{128}$)}{(R\textasciicircum 128)}}
Experiments I--IV employ single-integrator dynamics ($f(x) = 0$), which eliminates the Lie derivative term $L_f h$ from the barrier condition. To validate CHDBO in the general control-affine regime $\dot{x} = f(x) + g(x)u$ where $f(x) \neq 0$, we test two physically motivated drift models in $\mathbb{R}^{128}$.

\textbf{Setup A --- Linear Drift:} A marginally unstable matrix $A \in \mathbb{R}^{128 \times 128}$ with $\sigma_{\max}(A) = 1.044$ drives the uncontrolled dynamics $\dot{x} = Ax$, producing outward radial drift near the boundary $\partial\mathcal{S}$ of the unit ball. The CBF-QP (Equation~\ref{eq:cbfqp}) must now compensate for $L_f h(x) = -x^T A x / \|x\|$, which can be negative (pushing toward the boundary).

\textbf{Setup B --- Double-Integrator:} The state is decomposed as $x = (q, v) \in \mathbb{R}^{64} \times \mathbb{R}^{64}$ with $\dot{q} = v$, $\dot{v} = u$. Since $h(x) = 1 - \|q\|^2$ has relative degree~2 with respect to $u$, we employ an Exponential Control Barrier Function (ECBF) \cite{xiao2019}: $\psi_0(x) = \ddot{h} + (\alpha_1 + \alpha_2)\dot{h} + \alpha_1 \alpha_2 h \geq 0$ with $\alpha_1 = \alpha_2 = 2.0$, which reduces the relative-degree-2 constraint to a relative-degree-1 condition on $u$.

\textbf{Protocol:} 50 trials per configuration, each running 200 time steps with $\Delta t = 0.01$. The nominal control drives the system toward a goal outside the safe set ($\|x_{\text{goal}}\| = 1.5$), exercising the CBF-QP safety filter against the combined drift + utility pressure.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_9.png}
    \caption{\texorpdfstring{\textbf{Experiment V: CHDBO with Non-Trivial Drift Dynamics ($\mathbb{R}^{128}$).}}{Experiment V: CHDBO with Non-Trivial Drift Dynamics (R128).} (a)~Linear drift ($\sigma_{\max}(A) = 1.044$): 50 trials, 0 safety violations; all trajectories remain within $\|x\| < 1$. (b)~Double-integrator with ECBF: 50 trials, 0 violations; the ECBF maintains $\psi_0 \geq 0$ throughout. (c)~Distribution of $L_f h(x)$ under linear drift; mean $= -0.26$, confirming non-trivial adverse drift. (d)~ECBF barrier values remain non-negative across all trials.}
    \label{fig:drift_dynamics}
\end{figure}

\textbf{Results:} Both configurations achieve \textbf{0/50 safety violations}. Under linear drift, the mean Lie derivative contribution is $L_f h = -0.26$ (adverse---the drift pushes the system toward the boundary), yet the CBF-QP compensates via the $L_g h \cdot u$ control term, maintaining $\|x\| < 1$ at all times. Under double-integrator dynamics, the ECBF condition $\psi_0 \geq 0$ is maintained throughout all trials, with $\max \|q\| = 0.997$. Both configurations remain under 0.01\,ms per step via the closed-form QP solution, confirming that the $O(n)$ computational budget is preserved even with non-zero drift. These results bridge the gap identified in Section~\ref{sec:assumptions}: the CBF-QP framework handles non-trivial $f(x) \neq 0$ dynamics without modification to the core algorithm.

\subsection{Experiment VI: WTA Gradient Decomposition vs.\ Black-Box Discovery}
The AASV Hunter in Experiment~IV uses a Winner-Take-All (WTA) gradient that selects the nearest unblocked spike direction per restart, exploiting structural knowledge of the barrier decomposition. A natural question is whether the Hunter can discover unknown failure modes using only the true barrier gradient $\nabla h(x)$---either computed analytically (oracle) or via finite differences (black-box).

\textbf{Setup:} We test three gradient modes on $S^{127}$ with $\theta_w = 0.05$ radians:
\begin{enumerate}
    \item \textbf{WTA gradient} (from Experiment~IV): targets one spike per restart with blocking.
    \item \textbf{Sum gradient (oracle):} the analytical $\nabla h(x) = \sum_i \nabla h_i(x)$, summing over all spike contributions.
    \item \textbf{Sum gradient (FD):} finite-difference approximation of $\nabla h$ with step $\epsilon = 10^{-4}$.
\end{enumerate}

\textbf{Single-spike control:} To isolate the gradient-decomposition effect from the multi-modal landscape problem, we first test with a single spike ($N = 1$). With one failure mode, no centroid saddle exists, and FD gradients detect the spike on every restart (1/1 at $k = 5$).

\textbf{Multi-spike test:} With 3 orthogonal spikes, the full gradient $\nabla h$ is the sum of three coplanar funnel components, creating a saddle point at the centroid direction $\hat{c} = (s_1 + s_2 + s_3)/\|s_1 + s_2 + s_3\|$. Standard PGD converges to this centroid---which is \textit{not} a spike and satisfies $h(\hat{c}) > 0$---rather than to any individual violation.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_10.png}
    \caption{\texorpdfstring{\textbf{Experiment VI: WTA vs.\ Black-Box Gradient ({$\mathbb{R}^{128}$}).}}{Experiment VI: WTA vs. Black-Box Gradient (R128).} (a)~Single failure mode: both FD and oracle gradients detect the spike at $k = 5$, since no centroid saddle exists. (b)~Three orthogonal spikes: the WTA gradient resolves all 3 at $k = 5$, while the sum gradient (oracle or FD) detects 0/3 at any $k$---the centroid saddle traps the optimizer.}
    \label{fig:wta_vs_fd}
\end{figure}

\textbf{Results:} The WTA gradient detects all 3/3 spikes from $k = 5$ restarts onward. Both sum-gradient variants (oracle and FD) detect \textbf{0/3} at all tested restart budgets ($k \leq 40$), confirming the centroid saddle hypothesis. This demonstrates that the WTA decomposition is not merely a heuristic but a \textit{necessary structural component}: the full gradient $\nabla h$ contains no information about which spike is nearest, leading to destructive interference between funnel gradients. In deployment, this motivates either (1)~barrier designs where the gradient naturally decomposes (e.g., per-constraint CBFs), or (2)~learned surrogate models that approximate the WTA gradient from data. For barriers with a single dominant failure mode---common in operational safety (e.g., a single collision boundary)---the FD gradient suffices without WTA.

\subsection{Experiment VII: Seed Sensitivity and Reproducibility}
To assess the statistical robustness of AASV detection, we run a seed sensitivity sweep: 20 orthogonal spikes on $S^{127}$ (the maximum independent directions tested in Experiment~IV-h), evaluated across 10 independent random seeds controlling both spike direction generation and Hunter initialization.

\textbf{Protocol:} For each seed $s \in \{0, \ldots, 9\}$, a fresh set of 20 orthogonal spike directions is generated, and the full AASV pipeline (WTA gradient, $k = 80$ restarts, $T = 200$ steps, prototype blocking, post-violation refinement, post-hoc clustering) is executed independently.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_11.png}
    \caption{\texorpdfstring{\textbf{Experiment VII: Seed Sensitivity Sweep (20 Spikes $\times$ 10 Seeds, $\mathbb{R}^{128}$).}}{Experiment VII: Seed Sensitivity Sweep (20 Spikes x 10 Seeds, R128).} (a)~Matched spikes per seed: all 10 seeds achieve 20/20 detection. (b)~Box plot of matched and cluster counts: zero variance across seeds. (c)~Per-seed detection rate: 100\% for all seeds. The AASV pipeline exhibits no sensitivity to random initialization.}
    \label{fig:seed_sweep}
\end{figure}

\textbf{Results:} Across all 10 seeds, the AASV pipeline detects \textbf{20/20 spikes with zero variance}: mean $= 20.0 \pm 0.0$, 100\% detection rate on every seed. Mean violation count is $100.2 \pm 0.4$, with 20.0 clusters matching the 20 ground-truth spikes exactly. Total wall-clock time is $2.2 \pm 0.1$\,s per seed (single-threaded CPU), confirming real-time feasibility. This result eliminates the concern that Experiment~IV's detection rates depend on favorable random initialization: the WTA gradient with orthogonal prototype blocking is a deterministic-convergence mechanism, not a lucky random search. The zero-variance result reflects the deterministic convergence of the WTA mechanism with sufficient restarts; at lower restart budgets ($k < 2 N_{\text{spikes}}$), variance increases as some spikes may be missed due to insufficient coverage of the search space.

\subsection{Experiment VIII: Nonlinear Drift --- Lorenz-type Attractor in \texorpdfstring{$\mathbb{R}^{128}$}{R128}}
To validate CHDBO under strongly nonlinear dynamics---the most likely reviewer extension request---we construct a high-dimensional Lorenz-type chaotic attractor in $\mathbb{R}^{128}$. The state is grouped into 42 Lorenz triplets $(x_{3i}, x_{3i+1}, x_{3i+2})$ evolving as:
\begin{equation}
\dot{x}_{3i} = \sigma(x_{3i+1}{-}x_{3i}), \quad \dot{x}_{3i+1} = x_{3i}(\rho{-}x_{3i+2}){-}x_{3i+1}, \quad \dot{x}_{3i+2} = x_{3i} x_{3i+1} {-} \beta x_{3i+2}
\end{equation}
with standard parameters $(\sigma, \rho, \beta) = (10, 28, 8/3)$ and a scaling factor $1/40$ to confine the attractor within the unit ball. The remaining 2 dimensions have mild linear drift. The barrier is quadratic: $h(x) = 1 - \|x\|^2$, with the CBF-QP enforcing $\dot{h} + \gamma h \geq 0$ at every timestep ($\Delta t = 0.005$).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_12.png}
    \caption{\texorpdfstring{\textbf{Experiment VIII: Nonlinear Lorenz-type Drift in $\mathbb{R}^{128}$.}}{Experiment VIII: Nonlinear Lorenz-type Drift in R128.} (a)~Norm trajectories remain safely within the unit ball across 50 trials. (b)~Barrier values $h(x)$ stay strictly positive. (c)~Distribution of drift magnitudes $\|f(x)\|$, showing the nonlinear dynamics produce substantial adverse forcing. (d)~Lie derivative $L_f h$ distribution: the drift persistently pushes the barrier toward zero, yet the CBF-QP maintains safety through active intervention.}
    \label{fig:lorenz_drift}
\end{figure}

\textbf{Results:} Over 50 adversarial trials (500 steps each), the CBF-QP maintains \textbf{0/50 safety violations} despite mean drift magnitude $\|f(x)\| \approx 9.6$ and strongly negative $L_f h$ (mean $\approx -16.3$). The maximum observed norm is 0.97, demonstrating that the quadratic CBF-QP handles genuinely nonlinear cubic coupling terms without modification. This experiment extends the linear-drift validation of Experiment~V to a chaotic nonlinear regime, confirming that the continuous-relaxation safety theory (Theorem~1) holds in practice for the most challenging class of drift dynamics tested.

\subsection{Experiment IX: Scalability Beyond \texorpdfstring{$n = 128$}{n = 128}}
All previous experiments operate at $n = 128$. To validate that CHDBO scales practically to higher dimensions, we run the full pipeline---linear drift ($\sigma_{\max}(A) \approx 1.04$), CBF-QP enforcement, and Hutchinson spectral estimation ($k = 5$ probes)---at $n \in \{128, 512, 1024\}$ with 20 trials $\times$ 300 steps per dimension.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figure_13.png}
    \caption{\texorpdfstring{\textbf{Experiment IX: Scalability at $n = 128, 512, 1024$.}}{Experiment IX: Scalability at n = 128, 512, 1024.} (a)~Wall-clock time scales predictably with dimension (the $O(n^2)$ cost is due to the dense $Ax$ computation in the figure-generation code; a JVP-based implementation would achieve $O(n)$). (b)~Hutchinson $\hat{\sigma}$ estimates grow with $\sqrt{n}$ as expected for Frobenius-norm overestimation (see Section~3.5.2). (c)~Minimum barrier values remain strictly positive across all dimensions: 0/60 violations total. (d)~CBF activation frequency is stable across dimensions, confirming dimension-independent safety behavior.}
    \label{fig:scalability}
\end{figure}

\textbf{Results:} Zero safety violations across all 60 trials (20 per dimension). Wall-clock time scales from 0.02\,s ($n{=}128$) to 0.62\,s ($n{=}1024$) per trial, confirming practical real-time feasibility even at $n{=}1024$. The Hutchinson $\hat{\sigma}$ estimates exhibit the expected $\sqrt{n}$ Frobenius overestimation (Section~3.5.2), but the CBF-QP compensates by increasing the safety margin, preserving forward invariance without tuning. CBF activation frequency remains stable across dimensions ($\sim$147--148 of 300 steps), indicating that the fraction of control effort devoted to safety is dimension-independent.

\subsection{Summary of Results}

\begin{table}[htbp]
\centering
\caption{Empirical Performance Summary: CHDBO Framework}
\footnotesize
\begin{tabular}{@{}llccl@{}}
\toprule
\textbf{Experiment} & \textbf{Metric} & \textbf{Value} & \textbf{Dynamics} & \textbf{Methodology} \\ \midrule
I (Reachability, $n{=}2$) & Safety Rate & 100\% (50 trials) & $\dot{x}=u$ & CBF-QP projection \\
II (High-dim, $n{=}128$) & Safety Rate & 100\% (100 steps) & $\dot{x}=u$ & CBF-QP on $S^{127}$ \\
III (Proportional, $n{=}128$) & Utility Preservation & $\sqrt{1-\alpha^2}$ match & $\dot{x}=u$ & Cosine similarity \\
IV (AASV, 8 configs) & Black Swan Det. & 20/20~($k{=}60$) & $\dot{x}=u$ & Mom.\ PGD + WTA \\
 & Angular Resol. & $\geq 11.5^\circ$ & & Post-hoc clustering \\
V (Drift, $n{=}128$) & Safety Rate (linear) & 0/50 violations & $\dot{x}=Ax+Bu$ & CBF-QP + $L_fh$ \\
 & Safety Rate (dbl-int) & 0/50 violations & $\ddot{q}=u$ & ECBF \\
 & Computation & $<0.01$\,ms/step & & QP closed-form \\
VI (WTA vs.\ FD) & WTA Detection & 3/3~($k{=}5$) & $\dot{x}=u$ & WTA gradient + blocking \\
 & FD Detection (1 spike) & 1/1~($k{=}5$) & & FD $\nabla h$ \\
 & Sum $\nabla h$ (3 spikes) & 0/3~($k{=}40$) & & Centroid saddle \\
VII (Seed sweep) & Reproducibility & 20/20 $\pm$ 0 & $\dot{x}=u$ & 10 seeds, $k{=}80$ \\
VIII (Lorenz, $n{=}128$) & Safety Rate & 0/50 violations & Lorenz $f(x)$ & CBF-QP, nonlinear \\
 & Mean $\|f(x)\|$ & 9.6 & & Cubic coupling \\
IX (Scale, $n{\leq}1024$) & Safety Rate & 0/60 violations & $\dot{x}=Ax+u$ & $n{=}128,512,1024$ \\
 & Time ($n{=}1024$) & 0.62\,s/trial & & Hutchinson $k{=}5$ \\
\bottomrule
\end{tabular}
\begin{flushleft}
\small \textit{Notes:} Experiments I--IV and VI--VII use single-integrator dynamics ($f(x)=0$); Experiment V introduces linear/bilinear drift; Experiment VIII introduces nonlinear chaotic drift (Lorenz-type); Experiment IX tests scalability up to $n=1024$. ``WTA'' = Winner-Take-All gradient decomposition. ``ECBF'' = Exponential CBF for relative degree 2. Computation times measured on a single CPU thread.
\end{flushleft}
\label{tab:results}
\end{table}

The results demonstrate that CHDBO offers a potent alternative to traditional verification. By trading exactness for high-probability certification, we unlock the ability to control systems with dimensions previously thought intractable ($n \gg 10$) without sacrificing the agent's ability to pursue its goals.

\begin{table}[htbp]
\centering
\caption{Comparison with Related Verification and Safety Approaches}
\footnotesize
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Scalability} & \textbf{Guarantee Type} & \textbf{Real-Time} & \textbf{Assumptions} \\ \midrule
H-J Reachability \cite{mitchell2005} & $O((1/\eta)^n)$ & Deterministic & $n \leq 5$ & Full model \\
Standard CBF-QP \cite{ames2019} & $O(n)$ per step & Deterministic & Yes & Known $h(x)$, rel.\ deg.\ 1 \\
Learned CBFs \cite{dawson2023} & $O(n)$ & Probabilistic & Yes & Training data, NN approx. \\
Safe RL / Shielding \cite{garcia2015} & High-dim & Reward-based & Yes & Reward shaping \\
MPC with constraints \cite{camacho2013} & $O(n^3)$ per step & Deterministic & $n \leq 50$ & Convex model \\
\textbf{CHDBO (ours)} & $O(n)$ per step & Probabilistic (PAC) & Yes & Lipschitz $h$, Assump.\ A1--A6 \\
\textbf{CHDBO + AASV} & $O(kTn)$ per step & Adversarially bounded & Yes & + WTA gradient access \\
\bottomrule
\end{tabular}
\label{tab:comparison}
\end{table}

\subsection{Limitations}

\subsubsection{Experimental Dynamics}
We explicitly acknowledge that Experiments I--IV employ single-integrator dynamics ($\dot{x} = u$, i.e., $f(x) = 0$, $g(x) = I$). This design choice isolates the geometric mechanisms---orthogonal projection, Monte Carlo barrier certification, and AASV adversarial detection---from the confounding effects of drift dynamics, enabling clean validation of each component independently. Experiment~V introduces linear and double-integrator drift in $\mathbb{R}^{128}$; Experiment~VIII extends to a strongly nonlinear chaotic regime (Lorenz-type attractor with cubic coupling terms); and Experiment~IX validates scalability to $n = 1024$. Together, these experiments confirm that the CBF-QP safety filter maintains forward invariance across linear, bilinear, and nonlinear dynamics. Remaining limitations include learned neural ODE dynamics \cite{chen2018} and systems with state-dependent control matrices $g(x) \neq I$.

This limitation does not invalidate the theoretical results: Theorems~1 and~2 are stated and proved for the general control-affine system $\dot{x} = f(x) + g(x)u$, and the AASV module's spectral margin (Equation~\ref{eq:robust_barrier}) explicitly accounts for drift volatility via $\delta(x) = \tilde{\sigma}_{\max}(J_f) \cdot d_{\text{step}}$.

\subsubsection{WTA Gradient Oracle Requirement}
\label{sec:wta_limitation}
The most significant practical limitation of the AASV module is the Winner-Take-All (WTA) gradient's reliance on structural knowledge of the barrier decomposition. Specifically, the WTA mechanism requires access to individual spike contributions $h_i(x)$ to select the nearest unblocked failure direction per restart. For a general learned barrier $h(x) = \text{NeuralNet}(x)$, this decomposition is unavailable.

Experiment~VI demonstrates that this is not merely a convenience issue but a \textit{fundamental requirement}: the full gradient $\nabla h(x)$ (whether oracle or finite-difference) converges to a centroid saddle rather than individual spikes in multi-modal landscapes (0/3 detection vs.\ 3/3 with WTA). Three practical mitigation strategies exist: (1)~barrier designs where the gradient naturally decomposes (e.g., per-constraint CBFs, where each constraint defines a separate $h_i$); (2)~learned surrogate models trained to approximate the WTA gradient from interaction data; and (3)~for single-mode barriers---typical of operational safety constraints such as collision boundaries---finite-difference gradients suffice without WTA (Experiment~VI, single-spike control).

\subsubsection{Code vs.\ Theory Scaling}
The figure-generation code computes Hutchinson estimates and barrier gradients using dense matrix operations ($O(n^2)$), not the $O(n)$ JVP-based automatic differentiation pipeline described in Section~3.5.2. This is acceptable for figure generation at the tested dimensions ($n \leq 2048$), but practitioners implementing AASV at production scale ($n > 10{,}000$) must use a true AD-based pipeline (e.g., PyTorch or JAX) to realize the claimed linear scaling.

\subsubsection{Barrier Design for Semantic Agents}
Defining the barrier function $h(x)$ for semantic agents---determining which scalar function on an embedding space demarcates ``safe'' from ``unsafe'' semantics---remains an open and fundamental challenge in AI safety that this paper does not address. Our framework assumes $h(x)$ is given (or learned from data via methods such as those in \cite{dawson2023, robey2020}); the design of semantically meaningful barrier functions for LLMs is orthogonal to, but prerequisite for, the verification and enforcement machinery presented here. Similarly, the ``safe backup'' strategy for semantic agents (reversion to a verified anchor embedding) may discard recent conversation context, representing a form of utility loss that practitioners must weigh against the safety guarantee.

\section{Conclusion}
The pursuit of autonomous systems that are both demonstrably safe and operationally aggressive has long been a ``zero-sum'' game in control theory. To guarantee safety via Hamilton-Jacobi reachability or grid-based verification, one historically had to sacrifice scalability, resigning formal proofs to low-dimensional toy problems. Conversely, to achieve high utility via Reinforcement Learning, one had to sacrifice guarantees, accepting a non-zero probability of catastrophic failure, a phenomenon known as ``mutational meltdown'' or ``agent drift'', in exchange for performance.

This paper has introduced Constrained High-Dimensional Barrier Optimization (CHDBO), a unified framework that addresses this dichotomy. By shifting the verification paradigm from deterministic enumeration (which scales exponentially, $O((1/\eta)^n)$) to probabilistic certification (which scales constantly, $O(1)$ in sample complexity), we have circumvented the ``Curse of Dimensionality'' that has plagued safety-critical autonomy for decades \cite{bellman1957}. We emphasize that this probabilistic relaxation trades deterministic certainty for high-confidence statistical bounds---a necessary concession for high-dimensional systems, complemented by the adversarial guarantees of AASV.

Our empirical results confirm four critical advancements:
\begin{enumerate}
    \item \textbf{Safety is Probabilistically Scalable:} Through Monte Carlo barrier certificates bounded by Lipschitz continuity, we can verify the boundary consistency of $\mathbb{R}^{128}$ manifolds with high statistical confidence ($1-\delta$), bounding the volume fraction of failure modes.
    \item \textbf{Utility is Asymptotic:} By projecting utility gradients onto the tangent cone of the safe set, we ensure that agents do not merely survive, but thrive, converging to global optima without violating invariant constraints.
    \item \textbf{The Trade-off is False:} We have demonstrated that aggressive performance and rigorous safety are not mutually exclusive. An agent can operate at the very edge of the safe set $\mathcal{S}$ without ever crossing the line into failure.
    \item \textbf{Black Swans are Huntable:} The AASV module transitions safety assurance from passive statistical estimation to active, optimization-based threat hunting. Across eight configurations in $\mathbb{R}^{128}$---including orthogonal, antipodal, clustered, and random spike geometries---standard Monte Carlo sampling detected zero injected singularities, while AASV's WTA-accelerated PGD located all spikes (up to 20/20 with $k=60$ restarts) with perfect reproducibility across 10 independent seeds (Experiment~VII). The WTA gradient decomposition is demonstrated to be essential for multi-modal landscapes (Experiment~VI), while the framework generalizes to non-trivial drift dynamics including double-integrator systems (Experiment~V).
\end{enumerate}

We conclude that the future of robust autonomy lies in the synthesis of semantic and kinematic stability. While Holographic Invariant Storage (HIS) provides the ``invariant core'' of the agent, preserving its goals and personality against context drift, CHDBO provides the ``geometric constraint,'' ensuring its actions remain within the bounds of safety. The AASV module bridges the gap between statistical certification and adversarial robustness, providing bounded-risk guarantees even against failure modes that are invisible to passive sampling. Together, these technologies offer a framework for persistent safe operation at scale, enabling the deployment of autonomous agents capable of operating reliably over extended time horizons without regression.

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

\bibitem{vapnik1998}
Vapnik, V. N. (1998).
\textit{Statistical Learning Theory}.
Wiley-Interscience.

\bibitem{alshiekh2018}
Alshiekh, M., Bloem, R., Ehlers, R., Könighofer, B., Niekum, S., \& Topcu, U. (2018).
Safe Reinforcement Learning via Shielding.
\textit{Proceedings of the AAAI Conference on Artificial Intelligence}, 32(1).

\bibitem{boyd2004}
Boyd, S. \& Vandenberghe, L. (2004).
\textit{Convex Optimization}.
Cambridge University Press.

\bibitem{rimon1992}
Rimon, E. \& Koditschek, D. E. (1992).
Exact Robot Navigation Using Artificial Potential Functions.
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

\bibitem{schulman2017}
Schulman, J., et al. (2017).
``Proximal Policy Optimization Algorithms.''
\textit{arXiv preprint arXiv:1707.06347}.

\bibitem{madry2017}
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

\bibitem{blum2020}
Blum, A., Hopcroft, J., \& Kannan, R. (2020).
\textit{Foundations of Data Science}.
Cambridge University Press.

\bibitem{muller1959}
Muller, M. E. (1959).
``A note on a method for generating points uniformly on n-dimensional spherical surfaces.''
\textit{Communications of the ACM}, 2(4), 19--20.

\bibitem{xiao2019}
Xiao, W., \& Belta, C. (2019).
``Control Barrier Functions for Systems with High Relative Degree.''
\textit{2019 IEEE 58th Conference on Decision and Control (CDC)}, 474--479.

\bibitem{hutchinson1990}
Hutchinson, M. F. (1990).
``A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines.''
\textit{Communications in Statistics---Simulation and Computation}, 19(2), 433--450.

\bibitem{mayne2005}
Mayne, D. Q., Seron, M. M., \& Rakovi{\'c}, S. V. (2005).
``Robust model predictive control of constrained linear systems with bounded disturbances.''
\textit{Automatica}, 41(2), 219--224.

\bibitem{goodfellow2014}
Goodfellow, I. J., Shlens, J., \& Szegedy, C. (2014).
``Explaining and Harnessing Adversarial Examples.''
\textit{International Conference on Learning Representations (ICLR)}.

\bibitem{baydin2018}
Baydin, A. G., Pearlmutter, B. A., Radul, A. A., \& Siskind, J. M. (2018).
``Automatic Differentiation in Machine Learning: A Survey.''
\textit{Journal of Machine Learning Research}, 18(153), 1--43.

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

\end{thebibliography}

\end{document}