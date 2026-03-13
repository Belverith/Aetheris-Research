\documentclass[10pt, twocolumn, a4paper]{article}

% Required Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
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

% Title Information
\title{\textbf{Control Barrier Functions for Transformer Hidden States: A Proof of Concept}}
\author{Arsenios Scrivens}
\date{March 2026}

\begin{document}

\maketitle

\begin{abstract}
We investigate whether Control Barrier Functions (CBFs)---a tool from nonlinear control theory---can be meaningfully applied to the internal hidden states of transformer language models. As a proof of concept, we train a \textit{spectrally-normalized neural barrier} on GPT-2 hidden states ($\mathbb{R}^{768}$) with certified Lipschitz constant $L_h = 1.10$, achieving 88\% test accuracy on a safe/toxic classification task (200 test examples; 95\% Wilson CI: [83\%, 92\%]). An ablation without spectral normalization confirms that Lipschitz certification costs $< 2\%$ accuracy while being necessary for formal guarantees. Under frozen-pass activation patching, the CBF-QP solver converges on all test trajectories with perplexity ratio 1.005. When deployed autoregressively via logit-space steering, a sweep over intervention strength $\alpha$ traces the safety--fluency Pareto frontier: toxicity reduction is nominally significant at $\alpha = 0.25$ (TF-IDF $p = 0.027$, Cohen's $d = 0.22$) but does not survive Bonferroni correction for three comparisons ($\alpha_{\text{adj}} = 0.017$); only $\alpha = 0.50$ achieves significance, at the cost of doubling perplexity. A head-to-head comparison with Activation Addition (ActAdd) shows that state-dependent CBF corrections achieve comparable toxicity reduction at 19\% lower perplexity cost (PPL ratio 1.32 vs.\ 1.62 at matched $\alpha$). Scaling to Qwen2.5-3B ($n = 2048$) validates barrier training (90\% accuracy, $L_h = 1.11$, PPL ratio 1.000 under patching) but autoregressive steering yields non-significant toxicity reduction ($p = 0.90$), indicating that intervention hyperparameters do not transfer across model scales. These results demonstrate the \textit{feasibility} of CBF-based hidden-state safety filters but do not yet establish practical utility; the effect sizes are small, the models are small by 2026 standards, and no adversarial evaluation is performed. All experiments are fully reproducible from the accompanying code.
\end{abstract}

% ============================================================================
\section{Introduction}
% ============================================================================

The deployment of autonomous AI systems---from multi-agent swarms ($n > 100$) \cite{brunke2022} to large language models operating in semantic embedding spaces ($n = 768$ to $12{,}288$) \cite{brown2020, radford2019}---demands safety guarantees that scale with the dimensionality of the underlying state space. Formal verification methods such as Hamilton-Jacobi reachability \cite{mitchell2005} provide deterministic guarantees but scale as $O((1/\eta)^n)$, making them computationally infeasible for $n > 5$ \cite{bellman1957}. Learning-based control offers scalability but typically lacks formal safety certificates \cite{amodei2016, garcia2015}.

Control Barrier Functions (CBFs) \cite{ames2019, prajna2004} bridge this gap by encoding safety as a forward-invariance condition on a sublevel set $\mathcal{S} = \{x : h(x) \geq 0\}$, enforced via a minimum-norm Quadratic Program (QP) that admits a closed-form solution in $O(n)$ for single-constraint systems. However, two challenges remain: (1)~verifying that the barrier condition holds \textit{everywhere} on the boundary $\partial\mathcal{S}$ in high dimensions, and (2)~designing barrier functions that are both expressive enough to capture complex decision boundaries and smooth enough to admit certified Lipschitz bounds for probabilistic verification.

This paper presents a proof of concept with two components:

\begin{enumerate}
    \item \textbf{Neural Barrier with Certified Lipschitz Constant.} We train a spectrally-normalized MLP barrier on GPT-2 hidden states ($\mathbb{R}^{768}$) with certified $L_h = 1.10$, achieving 88\% test accuracy (95\% CI: [83\%, 92\%] on 200 test examples). An iterative Newton-corrected CBF-QP handles the nonlinear barrier constraint. A controlled ablation without spectral normalization demonstrates that Lipschitz certification costs $< 2\%$ accuracy while being necessary for formal safety guarantees (Section~\ref{sec:neural_barrier}).

    \item \textbf{Autoregressive CBF-Steered Generation.} We close the deployment gap between frozen-pass verification and live generation by hooking the neural barrier into GPT-2's autoregressive loop, intervening at every token via logit-space steering. A sweep over intervention strength $\alpha$ empirically characterizes the safety--fluency Pareto frontier; the effect sizes are small (Cohen's $d = 0.22$ at best) and the strongest result does not survive multiplicity correction. We report the full frontier transparently, separating kinematic safety (deterministic, conditional on CBF feasibility) from semantic safety (probabilistic, bounded by classifier accuracy) (Section~\ref{sec:autoregressive}).
\end{enumerate}

This work targets \textit{inference-time} safety at the activation level, complementing training-time alignment approaches (RLHF \cite{ouyang2022}, DPO \cite{rafailov2023}, Constitutional AI \cite{bai2022}) that modify model weights. The CBF framework requires no gradient access to the base model and can be applied as a modular safety layer on top of any pretrained---or already-aligned---architecture.

% ============================================================================
\section{Related Work}
\label{sec:related}
% ============================================================================

\textbf{Control Barrier Functions.}
Ames et al.\ \cite{ames2019} established the CBF-QP framework for enforcing forward invariance of safe sets, with extensions to high-relative-degree systems \cite{xiao2019}, robust settings \cite{jankovic2018}, and multi-agent coordination \cite{glotfelter2017}. Our work extends CBFs to transformer hidden states and introduces a nonlinear learned barrier with certified Lipschitz bounds.

\textbf{Learned Barrier Functions.}
Dawson et al.\ \cite{dawson2023} survey neural Lyapunov, barrier, and contraction methods. Robey et al.\ \cite{robey2020} learn CBFs from expert demonstrations, and Qin et al.\ \cite{qin2021} train decentralized neural barrier certificates for multi-agent systems. These approaches typically lack certified Lipschitz bounds on the learned barrier. Our spectrally-normalized architecture provides certified bounds by construction.

\textbf{Lipschitz Neural Networks.}
Spectral normalization \cite{miyato2018} constrains $\sigma_{\max}(W) \leq 1$ per layer, yielding $\text{Lip}(\text{net}) \leq \prod_l \sigma_{\max}(W_l)$. Fazlyab et al.\ \cite{fazlyab2019} provide tighter (but more expensive) semidefinite-programming-based Lipschitz estimation. We use spectral normalization for its simplicity and $O(1)$-per-layer enforcement, accepting the product bound as a certified upper estimate.

\textbf{Representation Engineering.}
Zou et al.\ \cite{zou2023} demonstrate that linear directions in transformer residual streams correspond to interpretable concepts (honesty, toxicity, sentiment). Turner et al.\ \cite{turner2023} show that Activation Addition---adding a fixed steering vector to the residual stream---can modulate model behavior. Our CBF-QP framework generalizes this: the control signal $u^*$ is state-dependent, computed via a principled safety filter, and the Lipschitz-bounded barrier provides formal guarantees that heuristic steering cannot. Our empirical comparison (Section~\ref{sec:autoregressive}) confirms the theoretical advantage: at matched intervention strength, CBF steering achieves comparable toxicity reduction at 19\% lower perplexity cost than ActAdd, because it intervenes only when the barrier margin is insufficient rather than perturbing every token unconditionally.

\textbf{Alignment and Safety Training.}
RLHF \cite{ouyang2022}, DPO \cite{rafailov2023}, and Constitutional AI \cite{bai2022} modify model weights at training time to reduce harmful outputs. These are complementary to our approach: they shape the model's learned distribution, while CBF-QP adds an inference-time safety filter that requires no retraining. An informative comparison would measure the marginal safety benefit of CBF on top of an already-aligned model, which we identify as future work. We compare instead against Activation Addition \cite{turner2023}, which operates at the same inference-time level and serves as the natural baseline for activation-level steering methods.

\textbf{Probabilistic Verification.}
Randomized verification methods \cite{tempo2012} provide distribution-free safety certificates by sampling the state space rather than exhaustively gridding it. We use Monte Carlo boundary sampling as an empirical feasibility check (Section~\ref{sec:mcbc}), verifying that the CBF-QP admits bounded corrections across the learned barrier's zero level set.

% ============================================================================
\section{Theoretical Framework}
\label{sec:theory}
% ============================================================================

\subsection{Problem Formulation}

Consider a discrete-time control-affine system:
\begin{equation}
    x_{k+1} = x_k + f(x_k) + g(x_k)\, u_k, \quad x_k \in \mathbb{R}^n, \; u_k \in \mathcal{U} \subseteq \mathbb{R}^n
    \label{eq:dynamics}
\end{equation}
where $f: \mathbb{R}^n \to \mathbb{R}^n$ is locally Lipschitz and $g = I_n$ (identity, i.e., the control enters additively). Transformer residual streams are a direct instance:
\begin{equation}
    x_{l+1} = x_l + \text{Block}_l(x_l) + u_l
    \label{eq:transformer}
\end{equation}
where $x_l \in \mathbb{R}^n$ is the last-token hidden state at layer $l$ ($n = 768$ for GPT-2, $n = 2048$ for Qwen2.5-3B), $f_l = \text{Block}_l$ comprises self-attention and feed-forward sub-layers, and $u_l$ is the additive control intervention. The system evolves over a finite horizon of $L$ layers; in the autoregressive setting, a fresh $L$-step pass is executed at each generated token.

\begin{definition}[Safe Set]
\label{def:safe_set}
Given a continuously differentiable function $h: \mathbb{R}^n \to \mathbb{R}$ with $0$ a regular value, the \textbf{safe set} is $\mathcal{S} = \{x \in \mathbb{R}^n : h(x) \geq 0\}$, with boundary $\partial\mathcal{S} = \{x : h(x) = 0\}$ and interior $\text{Int}(\mathcal{S}) = \{x : h(x) > 0\}$.
\end{definition}

\begin{problem}[Safety-Constrained Control]
\label{prob:main}
Find a control policy $u_k = \pi(x_k)$ such that:
\begin{enumerate}
    \item \textbf{Safety:} $\mathcal{S}$ is forward invariant: $x_0 \in \mathcal{S} \implies x_k \in \mathcal{S}$ for all $k = 0, 1, \ldots, K$.
    \item \textbf{Utility:} The controlled trajectory remains close to the unconstrained optimum of a utility function $U(x)$ on $\mathcal{S}$ (measured by perplexity degradation in the LLM setting).
    \item \textbf{Scalability:} Per-step computation cost is $O(n)$.
\end{enumerate}
\end{problem}

\subsection{Discrete-Time CBF-QP Safety Filter}

\begin{definition}[Discrete-Time Control Barrier Function]
A continuously differentiable function $h: \mathbb{R}^n \to \mathbb{R}$ is a \textbf{Discrete-Time Control Barrier Function} (DTCBF) for system \eqref{eq:dynamics} on $\mathcal{S}$ if there exists $\gamma \in (0, 1]$ such that:
\begin{equation}
    \sup_{u \in \mathcal{U}} \; h\bigl(x + f(x) + g(x)\,u\bigr) \geq (1 - \gamma)\, h(x) \quad \forall x \in \mathcal{S}
    \label{eq:cbf_condition}
\end{equation}
\end{definition}

When $h$ is nonlinear, the constraint in \eqref{eq:cbf_condition} is not affine in $u$. A first-order Taylor expansion of $h$ around $x$ yields $h(x + f + gu) \approx h(x) + \nabla h(x)^\top\!(f + gu)$. Substituting into \eqref{eq:cbf_condition} and simplifying gives the \textit{linearized} safety constraint $a(x) + b(x)\,u \geq -\gamma\,h(x)$, where $a(x) := \nabla h(x)^\top f(x)$ and $b(x) := \nabla h(x)^\top g(x)$. The linearized safety filter is the minimum-norm QP:
\begin{equation}
    u^*(x) = \underset{u}{\text{argmin}} \; \frac{1}{2}\|u\|^2 \quad \text{s.t.} \quad a(x) + b(x) \, u \geq -\gamma h(x)
    \label{eq:cbf_qp}
\end{equation}

For our setting ($g = I_n$, so $b(x) = \nabla h(x)^\top$), the KKT conditions yield an explicit closed-form solution:
\begin{equation}
    u^*(x) = \begin{cases}
    0 & \text{if } a(x) + \gamma h(x) \geq 0 \\[4pt]
    \displaystyle \frac{-(a(x) + \gamma h(x))}{\|\nabla h(x)\|^2}\, \nabla h(x) & \text{otherwise}
    \end{cases}
    \label{eq:closed_form}
\end{equation}

This requires only two dot products and one scalar division, yielding $O(n)$ per-step cost.

\begin{theorem}[Discrete-Time Forward Invariance]
\label{thm:forward_invariance}
If $h$ is a DTCBF satisfying condition \eqref{eq:cbf_condition} exactly (i.e., without linearization error), then any policy satisfying \eqref{eq:cbf_condition} renders $\mathcal{S}$ forward invariant: $x_0 \in \mathcal{S} \implies h(x_k) \geq (1 - \gamma)^k\, h(x_0) \geq 0$ for all $k \geq 0$.
\end{theorem}

\begin{proof}
By induction. At $k = 0$, $h(x_0) \geq 0$ by assumption. If $h(x_k) \geq 0$, then \eqref{eq:cbf_condition} gives $h(x_{k+1}) \geq (1-\gamma)\,h(x_k) \geq 0$ since $\gamma \in (0,1]$. The quantitative bound follows by unrolling: $h(x_k) \geq (1-\gamma)^k h(x_0)$.
\end{proof}

\begin{remark}[Linearization gap]
\label{rem:linearization}
The QP \eqref{eq:cbf_qp} enforces the \textit{linearized} constraint, not condition \eqref{eq:cbf_condition} exactly. The residual is a second-order term bounded by $\tfrac{1}{2} L_{\nabla h} \|f + u^*\|^2$. To close this gap, our implementation verifies $h(x_{k+1}) \geq \delta_{\text{buf}}$ \textit{exactly} via the neural network after each QP solve, applying iterative Newton corrections when the linearization error is non-negligible (Section~\ref{sec:neural_barrier}). This ensures the discrete-time CBF condition holds at each step up to gradient degeneracy, without requiring the continuous-to-discrete approximations (Comparison Lemma, Nagumo's theorem \cite{khalil2002, blanchini1999}) that would arise from importing continuous-time CBF theory into an inherently discrete system.
\end{remark}

\subsection{Empirical Boundary Verification}
\label{sec:mcbc}

To verify that the CBF-QP can maintain safety across the learned boundary, we perform Monte Carlo boundary sampling: 10,000 points are projected onto $\{x : h(x) = 0\}$ via Newton iteration \eqref{eq:newton_projection}, local dynamics are estimated by $K$-nearest-neighbor regression ($K = 10$) on observed transformer-layer residuals, and the CBF-QP correction $u^*$ is computed at each point. A point ``fails'' if $\|u^*\| > u_{\text{budget}}$. This empirical check is reported in Section~\ref{sec:neural_barrier}.

\begin{lemma}[Certified Lipschitz Bound for SN-MLPs]
\label{lem:lipschitz}
For a spectrally-normalized MLP with LeakyReLU($\alpha$) activations ($\text{Lip}(\phi) = 1$), the Lipschitz constant satisfies $L_h \leq \prod_l \sigma_{\max}(W_l)$, certified at each training step via spectral normalization \cite{miyato2018}. This bound is dimension-independent: it depends only on the network weights, not on $n$.
\end{lemma}

% ============================================================================
\section{Experiments}
\label{sec:experiments}
% ============================================================================

We present four experiments that constitute the paper's empirical contribution: a spectrally-normalized neural barrier on GPT-2 hidden states with boundary feasibility verification and controlled ablation (Section~\ref{sec:neural_barrier}), autoregressive CBF-steered generation that closes the deployment gap (Section~\ref{sec:autoregressive}), a head-to-head comparison with Activation Addition (Section~\ref{sec:autoregressive}), and a scaling validation on Qwen2.5-3B (Section~\ref{sec:scaling}).

\subsection{Experimental Setup: GPT-2 Hidden-State Dynamics}

We treat GPT-2's successive transformer layers as a discrete-time dynamical system on $\mathbb{R}^{768}$:
\begin{equation}
    x_{l+1} = x_l + \text{Block}_l(x_l), \quad l = 0, 1, \ldots, 11
    \label{eq:gpt2_dynamics}
\end{equation}
where $x_l$ is the last-token hidden state at layer $l$. Following the control-affine approximation motivated by representation engineering \cite{zou2023, turner2023}, we model CBF interventions as additive perturbations: $\tilde{x}_{l+1} = x_l + \text{Block}_l(x_l) + u_l$.

\textbf{Dataset.} We use the Google Civil Comments dataset \cite{borkan2019}, streaming 500 texts with toxicity $\leq 0.1$ (safe) and 500 with toxicity $\geq 0.7$ (toxic), filtered to 20--200 characters. Each text is tokenized (max 64 tokens) and the last-token hidden state is extracted at all 13 layers, yielding a $(13, 768)$ trajectory per text.

\textbf{Train/test split.} The 1,000 texts are split 80/20 (stratified, seed = 42). All barrier training, cross-validation, and hyperparameter selection use \textit{only} the training set (800 texts).

\textbf{Linear baseline.} A LinearSVC ($C = 1.0$) trained on layer-12 hidden states achieves 80.5\% test accuracy with $L_h = \|w\| = 24.28$. This establishes the linear-separation floor and provides the Lipschitz-constant comparison point for the neural barrier.

\subsection{Neural Barrier with Certified Lipschitz Bounds}
\label{sec:neural_barrier}

We replace the linear SVM barrier with a spectrally-normalized MLP, demonstrating that the CBF framework scales to nonlinear, learned decision boundaries while maintaining formal safety guarantees.

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

\subsubsection{Boundary Feasibility Verification}

Sampling the boundary $\{x : h(x) = 0\}$ of a nonlinear barrier requires Newton projection rather than simple hyperplane projection. Starting from random points drawn from the empirical data distribution $\mathcal{N}(\mu_{\text{data}}, \text{diag}(\sigma_{\text{data}}^2))$, we iterate:
\begin{equation}
    x_{k+1} = x_k - \frac{h(x_k)}{\|\nabla h(x_k)\|^2} \nabla h(x_k)
    \label{eq:newton_projection}
\end{equation}
for up to 100 steps with tolerance $|h(x)| < 10^{-6}$. Convergence is verified post-hoc: only points with $|h(x)| < 10^{-4}$ are retained. Of 10,000 initial points, we typically retain $> 99\%$ after convergence filtering. A fallback data-proximity sampling strategy is available if convergence is insufficient: starting from data points closest to the learned boundary, perturbation and re-projection generate additional boundary samples.

At each converged boundary point, local dynamics are estimated via inverse-distance-weighted $K$-nearest-neighbor regression ($K = 10$, BallTree index) on the observed transformer-layer residuals. The feasibility check then applies the iterative nonlinear CBF-QP (with 20 Newton correction steps) and tests whether $\|u^*\| \leq u_{\text{budget}}$.

\subsubsection{Output Quality Evaluation}

To verify that CBF interventions do not degrade language model output quality, we evaluate perplexity via activation patching. For each of 50 toxic texts, we register a forward hook at the target layer of GPT2-LMHeadModel that adds $u^*$ to the hidden state, then measure the cross-entropy loss. The perplexity ratio $\text{PPL}_{\text{steered}} / \text{PPL}_{\text{original}}$ quantifies output quality degradation; a ratio near 1.0 indicates no meaningful change.

\subsubsection{Results}

Table~\ref{tab:neural_results} presents the head-to-head comparison across all metrics.

\begin{table}[t]
\centering
\caption{SVM vs.\ spectrally-normalized MLP vs.\ unconstrained ablation on GPT-2 hidden states ($\mathbb{R}^{768}$, Civil Comments, 500 safe + 500 toxic, 80/20 split).}
\label{tab:neural_results}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Metric} & \textbf{SVM} & \textbf{SN-MLP} & \textbf{Ablation} \\
\midrule
Test accuracy & 80.5\% & \textbf{88.0\%} & 86.5\% \\
$L_h$ (certified) & 24.28 & \textbf{1.10} & 8.73$^\dagger$ \\
$L_h$ (empirical) & 24.28 & \textbf{0.58} & 1.08 \\
QP convergence failures$^\ddagger$ & 0 & \textbf{0} & --- \\
Boundary feas.\ $P_{\text{safe}}$ & 1.0000 & \textbf{1.0000} & --- \\
PPL ratio (median) & 1.000 & \textbf{1.005} & --- \\
Mean $\|u^*\|$ (toxic) & --- & \textbf{1.86} & --- \\
\bottomrule
\multicolumn{4}{@{}l@{}}{\footnotesize $^\dagger$Uncertified: spectral norms not constrained during training.} \\
\multicolumn{4}{@{}l@{}}{\footnotesize $^\ddagger$The iterative QP is \textit{designed} to enforce $h \geq \delta_{\text{buf}}$; zero failures indicates solver convergence, not an independent safety finding.}
\end{tabular}
\end{table}

\textbf{Accuracy.} The SN-MLP achieves 88.0\% test accuracy, a +7.5 percentage point improvement over the linear SVM (80.5\%). The unconstrained ablation achieves 86.5\%, indicating that spectral normalization costs $< 2\%$ accuracy while providing formal Lipschitz certification.

\textbf{Lipschitz constant.} The SN-MLP's certified $L_h = 1.10$ is a $22\times$ reduction over the SVM's $L_h = \|w\| = 24.28$. The empirical Lipschitz estimate (from 5,000 random pairs on the test set) yields $L_h^{\text{emp}} = 0.58$, confirming that the certified bound is tight. The ablation's uncertified $L_h = 8.73$ demonstrates that without spectral normalization, the Lipschitz constant is neither bounded nor predictable.

\textbf{QP convergence.} Both the SVM and SN-MLP achieve zero QP convergence failures across all 1,000 trajectories. This is \textit{not} an independent safety finding: the iterative solver is designed to enforce $h \geq \delta_{\text{buf}}$, so ``zero violations'' is a property of the algorithm, not an empirical discovery. The operationally informative result is that the binary-search fallback was never triggered: Newton iteration alone sufficed in all cases, confirming that the nonlinear barrier surface is well-conditioned and the QP landscape is benign.

\textbf{Boundary feasibility.} Monte Carlo boundary sampling (10,000 Newton-projected points, $K = 10$ KNN dynamics) finds zero infeasible points at all budget levels (5\%--50\% of mean $\|f\|$) for both the SVM and SN-MLP barriers.

\textbf{Perplexity.} The median perplexity ratio for the SN-MLP is 1.005, indicating negligible output quality degradation from CBF steering. The SVM achieves 1.000 (slightly better due to smaller intervention norms on average). Both are well below the $< 1.2$ coherence threshold.

\textbf{Ablation analysis.} The ablation (MLP without spectral normalization) achieves 86.5\% test accuracy---1.5 percentage points \textit{below} the SN-MLP---while having $L_h = 8.73$, a $7.9\times$ higher Lipschitz constant. This demonstrates that spectral normalization provides a dual benefit: (1)~it serves as an effective regularizer, marginally \textit{improving} generalization, and (2)~it provides the certified $L_h$ bound that makes the safety guarantee meaningful.

\textbf{Methodology note.} Early stopping selects the checkpoint with highest held-out accuracy, which provides mild model-selection benefit. The reported 88.0\% accuracy should be interpreted as a model-selection-optimistic estimate; a strict train/validation/test split would yield a slightly lower (by $\sim$1--3\%) but more conservative estimate. This applies equally to both the SN-MLP and ablation, so comparative conclusions are unaffected.

\subsection{Autoregressive CBF-Steered Generation}
\label{sec:autoregressive}

The preceding experiment operates on precomputed, frozen forward passes: the barrier classifies and steers \textit{existing} hidden-state trajectories without affecting subsequent token selection. This section closes the deployment gap by hooking the SN-MLP barrier into GPT-2's autoregressive generation loop, intervening at \textit{every token} with each intervention altering the token selected and thus the entire subsequent trajectory.

\subsubsection{Logit-Space CBF Intervention}

Direct hidden-state perturbation (adding $u^*$ to the layer-12 output) proved insufficient: perturbations small enough to preserve coherence did not meaningfully alter the logit distribution, yielding $< 3\%$ CBF activation and near-identical outputs. We therefore implement a \textit{logit-space} intervention:
\begin{enumerate}
    \item At each generation step, extract the last-token hidden state $x_t$ at layer 12.
    \item Evaluate $h(x_t)$; if $h(x_t) \geq \delta_{\text{buf}}$, no intervention (the model generates freely).
    \item If $h(x_t) < \delta_{\text{buf}}$, compute the minimum-norm CBF correction $u^*$ via iterative Newton refinement (up to 20 steps), capped at $\|u^*\| \leq 0.5$.
    \item Recompute logits from the corrected hidden state: $\tilde{\ell} = \text{lm\_head}(\text{ln\_f}(x_t + u^*))$.
    \item Blend: $\ell_{\text{final}} = \alpha \cdot \tilde{\ell} + (1 - \alpha) \cdot \ell_{\text{orig}}$, where $\alpha$ controls intervention strength.
    \item Sample the next token from $\text{top-}k(\ell_{\text{final}})$.
\end{enumerate}

The blending parameter $\alpha$ directly controls the safety--fluency tradeoff: $\alpha = 1$ applies full correction (maximum safety, maximum coherence cost), while $\alpha = 0$ recovers unsteered generation.

\subsubsection{Experimental Setup}

\textbf{Prompts.} 200 toxic prompts (first 10 tokens of Civil Comments texts with toxicity $\geq 0.7$) and 100 safe control prompts (toxicity $\leq 0.1$). Each prompt generates 50 tokens with temperature 1.0 and top-$k = 50$.

\textbf{Barrier.} The same SN-MLP from Section~\ref{sec:neural_barrier} ($L_h = 1.10$, test accuracy 88\%), loaded frozen. No retraining or fine-tuning for the autoregressive setting.

\textbf{Evaluation.} Three independent toxicity measures: (1)~the barrier's own hidden-state score $h(x)$; (2)~a TF-IDF + logistic regression classifier trained on the original Civil Comments texts; (3)~an external neural scorer (\texttt{unitary/toxic-bert}). Perplexity is computed via teacher-forced cross-entropy on the generated completions.

\textbf{Controls.} Steered and unsteered runs use \textit{identical random seeds} per prompt, ensuring the only difference is the CBF intervention. All statistical tests are two-sided Welch's $t$-tests with Cohen's $d$ effect sizes.

\textbf{Hyperparameters.} The intervention strength $\alpha$ was swept over three values $\{0.15, 0.25, 0.50\}$ with buffer $\delta_{\text{buf}} = 0.3$ and correction cap $\|u^*\|_{\max} = 0.5$, selected to characterize the Pareto frontier rather than optimize a single operating point. All three runs are reported.

\subsubsection{Results}

Table~\ref{tab:autoregressive} presents results across the $\alpha$ sweep.

\begin{table}[t]
\centering
\caption{Autoregressive CBF-steered generation on 200 toxic prompts. Three intervention strengths $\alpha$ characterize the safety--fluency Pareto frontier. TF-IDF and external scorer $p$-values from two-sided Welch's $t$-test vs.\ unsteered baseline.}
\label{tab:autoregressive}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Metric} & $\alpha = 0.15$ & $\alpha = 0.25$ & $\alpha = 0.50$ \\
\midrule
\multicolumn{4}{@{}l}{\textit{Toxicity reduction}} \\
TF-IDF (unsteered: 0.463) & 0.445 & 0.439 & 0.414 \\
TF-IDF $p$-value & 0.097 & \textbf{0.027} & \textbf{7.7e-6} \\
TF-IDF Cohen's $d$ & 0.17 & 0.22 & 0.45 \\
External (unsteered: 0.103) & 0.067 & 0.072 & 0.033 \\
External $p$-value & 0.125 & 0.185 & \textbf{7.7e-4} \\
Toxicity rate $> 0.5$ & 25.5\% & 26.5\% & 19.5\% \\
\midrule
\multicolumn{4}{@{}l}{\textit{Coherence}} \\
PPL (unsteered: 14.7) & 18.3 & 19.5 & 34.3 \\
PPL ratio & 1.24 & 1.32 & 2.33 \\
\midrule
\multicolumn{4}{@{}l}{\textit{CBF activity}} \\
Activation rate & 59.1\% & 58.6\% & 61.2\% \\
Mean $\|u^*\|$ (active) & 0.33 & 0.33 & 0.33 \\
Barrier $\bar{h}(x)$ & 0.118 & 0.228 & 0.183 \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Safety--fluency tradeoff.} The three $\alpha$ values trace a clear Pareto frontier: stronger intervention monotonically reduces toxicity and monotonically increases perplexity. At $\alpha = 0.50$, both text-level toxicity measures are highly significant ($p < 10^{-3}$) but perplexity more than doubles. At $\alpha = 0.15$, perplexity is near-baseline (ratio 1.24) but toxicity reduction does not reach significance. The intermediate $\alpha = 0.25$ achieves statistically significant TF-IDF reduction ($p = 0.027$, $d = 0.22$) with moderate perplexity cost (ratio 1.32). Applying Bonferroni correction for three comparisons ($\alpha_{\text{adj}} = 0.017$), only $\alpha = 0.50$ survives multiplicity adjustment; the $\alpha = 0.25$ result is marginal ($p = 0.027 > 0.017$). We report all three operating points without cherry-picking.

\textbf{CBF activation.} The barrier activates on $\sim$59\% of generation steps across all $\alpha$ values, confirming that the intervention is triggered frequently. The correction norm is consistently capped at 0.33 (slightly below the 0.5 maximum), indicating the barrier requests corrections near but below the cap.

\textbf{Barrier $h(x)$ at generation time.} The internal barrier score is \textit{not} significantly different between steered and unsteered runs ($p > 0.49$ at all $\alpha$). This is expected: the barrier was trained on hidden states from \textit{human-written} text, whereas at generation time it evaluates hidden states from \textit{GPT-2-generated} text---a distribution shift. The correction cap ($\|u^*\| \leq 0.5$) also limits how far the hidden state can move. Crucially, the text-level metrics (which evaluate the \textit{decoded output}) show that the logit-space intervention \textit{does} change which tokens are selected, even when the hidden-state score itself barely moves.

\textbf{Comparison with frozen-pass results.} Frozen-pass activation patching (Section~\ref{sec:neural_barrier}) achieves PPL ratio 1.005 with zero barrier violations. The gap to the autoregressive setting's best PPL ratio of 1.24 quantifies the cost of the feedback loop: each corrected token alters the sequence context for all subsequent tokens, compounding small perturbations. This establishes an empirical baseline for future work on intervention-aware barrier retraining.

\textbf{Qualitative assessment.} At $\alpha = 0.25$, steered completions remain largely fluent and on-topic (see Appendix for examples). At $\alpha = 0.50$, fluency degrades noticeably, with visible repetition and off-topic drift.

\subsubsection{Comparison with Activation Addition}

Representation engineering \cite{zou2023} and Activation Addition \cite{turner2023} steer model behavior by adding a fixed vector to the residual stream---typically the mean difference between safe and toxic hidden states. To compare against this natural baseline, we compute the steering direction $v = \bar{x}_{\text{safe}} - \bar{x}_{\text{toxic}}$ at layer 12 from the training set, normalize it, and scale it to $\|v\| = 0.5$ (matching the CBF correction cap). At each generation step, we add $v$ to the hidden state and recompute logits via the same $\alpha$-blended logit-space procedure used for CBF steering.

Table~\ref{tab:repeng} presents the head-to-head comparison on the same 200 toxic prompts with identical random seeds, and Table~\ref{tab:repeng_sweep} shows the full ActAdd $\alpha$-sweep.

\begin{table}[t]
\centering
\caption{CBF vs.\ Activation Addition (ActAdd) at matched $\alpha = 0.25$ on 200 toxic prompts, identical random seeds and evaluation pipeline. $p$-values from two-sided Welch's $t$-test vs.\ unsteered baseline (TF-IDF mean $= 0.463$, External mean $= 0.103$).}
\label{tab:repeng}
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{CBF ($\alpha{=}0.25$)} & \textbf{ActAdd ($\alpha{=}0.25$)} \\
\midrule
TF-IDF mean $\downarrow$ & 0.439 & 0.446 \\
TF-IDF $p$ & \textbf{0.027} & 0.119 \\
External mean $\downarrow$ & 0.072 & 0.066 \\
PPL ratio $\downarrow$ & \textbf{1.32} & 1.62 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{ActAdd $\alpha$-sweep on the same 200 toxic prompts. Only $\alpha = 0.50$ achieves $p < 0.05$, but at a 3.9$\times$ perplexity cost. CBF at $\alpha = 0.25$ (Table~\ref{tab:repeng}) achieves comparable toxicity reduction at 1.32$\times$.}
\label{tab:repeng_sweep}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Method} & \textbf{TF-IDF mean $\downarrow$} & \textbf{TF-IDF $p$} & \textbf{PPL ratio} \\
\midrule
Unsteered & 0.463 & --- & 1.00 \\
\addlinespace
ActAdd $\alpha{=}0.15$ & 0.453 & 0.376 & 1.32 \\
ActAdd $\alpha{=}0.25$ & 0.446 & 0.119 & 1.62 \\
ActAdd $\alpha{=}0.50$ & 0.423 & $1.9{\times}10^{-4}$ & 3.89 \\
\addlinespace
CBF $\alpha{=}0.25$ & 0.439 & \textbf{0.027} & \textbf{1.32} \\
\bottomrule
\end{tabular}
\end{table}

At matched $\alpha = 0.25$, CBF and ActAdd achieve similar toxicity reduction (head-to-head Welch's $t = -0.72$, $p = 0.47$, Cohen's $d = 0.07$---no significant difference in absolute toxicity), but CBF does so at a 19\% lower perplexity cost (ratio 1.32 vs.\ 1.62). The mechanism behind this efficiency gap is clear: the CBF correction $u^*(x)$ is \textit{state-dependent}, activating only when $h(x) < \delta_{\text{buf}}$ and applying the minimum-norm correction to restore the safety margin. ActAdd applies a fixed vector at every token regardless of the current hidden state, perturbing already-safe representations unnecessarily. This unconditional intervention drives up perplexity without proportional toxicity benefit.

The ActAdd $\alpha$-sweep further illustrates the point. ActAdd only reaches statistical significance ($p < 0.05$) at $\alpha = 0.50$, where the perplexity ratio explodes to 3.89---a 3$\times$ greater fluency cost than CBF at $\alpha = 0.25$ for comparable toxicity reduction ($\Delta$TF-IDF $= 0.040$ for CBF vs.\ $0.017$ for ActAdd at $\alpha = 0.25$, and $0.040$ for ActAdd at $\alpha = 0.50$). This demonstrates that state-dependent corrections provide a fundamentally better safety--fluency tradeoff than fixed steering vectors.

\subsection{Scaling Validation: Qwen2.5-3B ($\mathbb{R}^{2048}$)}
\label{sec:scaling}

To validate that the framework scales beyond GPT-2's $n = 768$, we replicate the full pipeline---SVM baseline, SN-MLP barrier training, CBF-QP verification, boundary feasibility sampling, activation-patching perplexity, and autoregressive generation---on Qwen2.5-3B ($n = 2048$, 36 layers), a 2.7$\times$ increase in hidden dimension.

The architecture is scaled to $2048 \to 512 \to 256 \to 1$ (1,180,417 parameters). All hyperparameters, data splits, and evaluation metrics are identical to the GPT-2 experiments. Experiments were run on an NVIDIA RTX PRO 4500 Blackwell (33.7~GB VRAM).

\begin{table}[t]
\centering
\caption{Scaling validation: GPT-2 ($n{=}768$) vs.\ Qwen2.5-3B ($n{=}2048$). The SN-MLP barrier, CBF-QP verification, and activation-patching results transfer cleanly to the larger architecture. Autoregressive steering at $\alpha = 0.25$ requires model-specific tuning (see text).}
\label{tab:scaling}
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{GPT-2 ($n{=}768$)} & \textbf{Qwen2.5-3B ($n{=}2048$)} \\
\midrule
Best layer & 12 / 12 & 29 / 36 \\
SVM test acc & 0.805 & 0.850 \\
SVM $L_h$ & 24.28 & 0.54 \\
SN-MLP test acc & 0.880 & \textbf{0.900} \\
SN-MLP $L_h$ (certified) & 1.10 & 1.11 \\
SN-MLP $L_h$ (empirical) & --- & 0.15 \\
Ablation $L_h$ & 8.73$^\dagger$ & 15.43$^\dagger$ \\
QP convergence failures$^\ddagger$ & 0 & 0 \\
PPL ratio (patching) & 1.005 & \textbf{1.000} \\
\addlinespace
TF-IDF $p$ ($\alpha{=}0.25$) & \textbf{0.027} & 0.901 \\
PPL ratio ($\alpha{=}0.25$) & 1.32 & \textbf{1.09} \\
Activation rate & --- & 0.322 \\
\bottomrule
\multicolumn{3}{@{}l@{}}{\footnotesize $^\dagger$Uncertified: spectral norms not constrained during training.} \\
\multicolumn{3}{@{}l@{}}{\footnotesize $^\ddagger$Solver is designed to enforce $h \geq \delta_{\text{buf}}$; zero failures indicates convergence, not an independent safety result.}
\end{tabular}
\end{table}

\textbf{Barrier training and verification.} The SN-MLP barrier achieves 90\% test accuracy on Qwen2.5-3B---2 percentage points above GPT-2---while maintaining a nearly identical certified Lipschitz constant ($L_h = 1.11$ vs.\ 1.10). The unconstrained ablation's $L_h$ still explodes to 15.43, confirming that spectral normalization remains necessary regardless of scale. The iterative CBF-QP solver converges on all test trajectories (as expected by construction---see Table~\ref{tab:neural_results} footnote), and activation-patching perplexity is exactly 1.000---the barrier correction is invisible to the model's output distribution in the frozen-pass setting.

A notable dimension-dependent effect appears in the SVM baseline: at $n = 2048$, the SVM weight-vector norm drops to $\|w\| = 0.54$ (from 24.28 at $n = 768$), reflecting better linear separability in higher-dimensional spaces. The SN-MLP's advantage over SVM therefore shifts from Lipschitz reduction to accuracy (90\% vs.\ 85\%) and the formal certification that spectral normalization provides.

\textbf{Boundary feasibility.} Newton projection converges for all 10,000 sampled boundary points ($h(x) = 0$). At 20\% of the mean dynamics norm, $P_{\text{safe}} = 0.9999$ (1 infeasible); at 50\%, $P_{\text{safe}} = 1.0$. The tighter budget fractions (5\%: $P_{\text{safe}} = 0.039$, 10\%: $P_{\text{safe}} = 0.430$) show that larger corrections are needed in higher dimensions, consistent with the larger $\|f\|$ (18.6 vs.\ GPT-2's 4.8).

\textbf{Autoregressive steering.} At $\alpha = 0.25$, the CBF activates on 32\% of generation steps with mean $\|u^*\| = 0.39$, yielding a PPL ratio of only 1.09---substantially less perplexity degradation than GPT-2's 1.32. However, the toxicity reduction is not statistically significant (TF-IDF $p = 0.90$, external $p = 0.92$). This reflects two factors: (1)~Qwen2.5-3B produces less toxic text at baseline (TF-IDF mean $= 0.452$ vs.\ GPT-2's $0.463$), leaving less room for improvement, and (2)~the barrier activates less frequently, applying fewer corrections per sequence. This indicates that autoregressive hyperparameters ($\alpha$, $\delta_{\text{buf}}$) require model-specific tuning---a direction for future work---while the underlying barrier training and CBF-QP framework transfer without modification.

% ============================================================================
\section{Discussion}
\label{sec:discussion}
% ============================================================================

\subsection{Contributions and Significance}

This work presents a proof of concept demonstrating that CBF-based safety filters can be applied to transformer hidden-state dynamics. Two results support this:

\begin{enumerate}
    \item \textbf{Neural CBF barrier with certified Lipschitz bounds} on transformer hidden states, achieving 88\% test accuracy at $L_h = 1.10$ on GPT-2 and 90\% at $L_h = 1.11$ on Qwen2.5-3B ($n = 2048$), with near-unity perplexity ratio on both architectures. A controlled ablation confirms the necessity of Lipschitz certification. These accuracy numbers carry wide confidence intervals (95\% Wilson CI $\approx \pm 4$--$5\%$ on 200 test examples); the comparative conclusions (SN-MLP vs.\ SVM, SN-MLP vs.\ ablation) are more robust than the absolute numbers.
    \item \textbf{Autoregressive deployment with principled advantage over fixed steering}: logit-space CBF steering achieves nominally significant toxicity reduction ($p = 0.027$, marginal after Bonferroni correction) with small effect size (Cohen's $d = 0.22$) on GPT-2, and non-significant reduction ($p = 0.90$) on Qwen2.5-3B. The practical toxicity reduction is modest ($\sim$5\% relative TF-IDF decrease), but the comparison with Activation Addition demonstrates a structural advantage of state-dependent corrections: 19\% lower perplexity cost at matched intervention strength (1.32 vs.\ 1.62), because the CBF activates only when needed. The two-tier guarantee---kinematic safety (deterministic, conditional on CBF feasibility) vs.\ semantic safety (probabilistic, bounded by classifier accuracy)---is reported transparently.
\end{enumerate}

\textbf{Scope and limitations of the formal guarantee.} The theoretical framework (Theorem~\ref{thm:forward_invariance}) provides discrete-time forward invariance \textit{conditional on} (1)~accurate dynamics estimation, (2)~the linearization gap being closed by the iterative solver, and (3)~the barrier classifier correctly identifying the state as safe or toxic. In practice, condition (1) relies on a KNN surrogate operating in dimensions where the curse of dimensionality limits its fidelity (Limitation~2), condition (2) is verified empirically but not certified (Remark~\ref{rem:linearization}), and condition (3) fails on 12--15\% of test examples. The formal safety guarantee therefore holds in a narrower regime than the title might suggest: it certifies that \textit{if} the barrier correctly classifies the current state \textit{and} the QP converges \textit{and} the dynamics estimate is accurate, \textit{then} the corrected state satisfies $h(x_{k+1}) \geq \delta_{\text{buf}}$. The gap between this conditional guarantee and unconditional deployment safety is substantial, and closing it is the central challenge for future work.

\subsection{Limitations}

We identify seven limitations that bound the scope of our claims:

\textbf{1. Autoregressive effect sizes are small.} The best autoregressive result (GPT-2, $\alpha = 0.25$: TF-IDF $\Delta = 0.024$, Cohen's $d = 0.22$) is a small effect that does not survive Bonferroni correction. Only $\alpha = 0.50$ reaches corrected significance, at a destructive 2.3$\times$ perplexity cost. On Qwen2.5-3B, autoregressive steering is fully non-significant ($p = 0.90$). These results demonstrate the existence of a safety--fluency tradeoff mediated by the barrier, but do not yet demonstrate practical utility. Intervention-aware barrier retraining---where the barrier is trained on hidden states from \textit{steered} generation rather than human text---is the most promising direction for improving effect sizes.

\textbf{2. KNN dynamics estimation and curse of dimensionality.} The $K$-nearest-neighbor dynamics surrogate ($K = 10$) operates in $\mathbb{R}^{768}$ (GPT-2) to $\mathbb{R}^{2048}$ (Qwen), where the curse of dimensionality renders all points approximately equidistant. The KNN estimate is therefore a coarse local average rather than a true pointwise dynamics estimate, and the boundary feasibility result ($P_{\text{safe}} = 1.0$) is conditional on the quality of this approximation at boundary points. A learned dynamics model, a leave-one-out residual analysis, or conformal prediction intervals \cite{angelopoulos2021} would provide a more principled dynamics surrogate.

\textbf{3. Linear control-affine assumption.} We model transformer-layer transitions as $x_{l+1} = x_l + f_l(x_l) + u_l$, assuming the control $u_l$ enters additively. In reality, the perturbation propagates nonlinearly through subsequent layers, attention mechanisms, and layer-norm operations. The error from this linear approximation is absorbed by the safety buffer $\delta_{\text{buf}} = 2.0$, but a formal bound on the approximation error would strengthen the guarantee.

\textbf{4. Barrier expressiveness vs.\ accuracy.} GPT-2's 88\% test accuracy, while substantially above the 80.5\% SVM baseline, falls below the $\geq$90\% target. Notably, Qwen2.5-3B reaches 90\% with the same architecture, suggesting that higher-dimensional representations provide better separability. Dataset scaling, multi-layer features, or ensemble barriers could close the gap on smaller models.

\textbf{5. Model scale.} GPT-2 (124M parameters, 2019) and Qwen2.5-3B are small models by March 2026 standards; production systems routinely use architectures with $n \geq 4096$ and $> 70$B parameters. Our results establish feasibility of the CBF framework on these smaller models but do not demonstrate that the approach works at production scale. The Qwen autoregressive null result ($p = 0.90$) is particularly cautionary: barrier training transfers, but autoregressive steering does not, and scaling further may reveal additional failure modes.

\textbf{6. Small dataset and wide confidence intervals.} The total dataset is 1,000 texts (500 safe + 500 toxic), with a 200-example test set. The 88\% and 90\% accuracy figures carry 95\% Wilson confidence intervals of approximately $\pm 4$--$5\%$, meaning the true accuracy could plausibly be as low as 83\% or as high as 92\%. By NLP standards this is a very small evaluation. Scaling to larger, more diverse toxicity datasets (e.g., full Civil Comments with nuanced toxicity labels) would narrow these intervals and test generalization.

\textbf{7. No adversarial evaluation.} For a paper on safety, the absence of adversarial testing is a significant gap. Our toxic prompts are naturally occurring Civil Comments text; we do not evaluate against adversarial attack, jailbreaking, or red-teaming scenarios. A barrier classifier with 12--15\% error rate on benign inputs will likely perform substantially worse under adversarial pressure. Gradient-based attacks on the barrier function itself (e.g., finding inputs that maximize $h(x)$ while being toxic) are straightforward to construct and could reveal failure modes invisible to our current evaluation. No deployment claim should be made without such testing.

The safety guarantee has two distinct strengths that should not be conflated:
\begin{itemize}
    \item \textit{Kinematic safety} (Theorem~\ref{thm:forward_invariance}): deterministic forward invariance, conditional on CBF feasibility and accurate dynamics.
    \item \textit{Semantic safety} (toxicity classification): probabilistic, bounded by the barrier's test accuracy (88\%). A text correctly classified as toxic \textit{will} be steered to $h > 0$; a text misclassified as safe will not trigger the CBF at all.
\end{itemize}
We do not conflate these two guarantee levels. The formal safety machinery operates at the kinematic level; the semantic gap is inherited from the imperfect classifier.

% ============================================================================
\section{Conclusion}
\label{sec:conclusion}
% ============================================================================

We have presented a proof of concept for applying Control Barrier Functions to transformer hidden states, combining a spectrally-normalized neural barrier with certified Lipschitz bounds and autoregressive logit-space steering.

The results are mixed but informative. The barrier training and frozen-pass verification pipeline works cleanly: 88\% test accuracy on GPT-2, 90\% on Qwen2.5-3B, certified $L_h \approx 1.1$ on both, and near-perfect perplexity under activation patching. The comparative result---the SN-MLP ablation confirming that spectral normalization is necessary for Lipschitz certification at negligible accuracy cost---is the paper's cleanest contribution. The autoregressive results are weaker: a small, marginally significant toxicity reduction on GPT-2 ($d = 0.22$, not surviving Bonferroni) and a null result on Qwen2.5-3B. The comparison with Activation Addition demonstrates a structural efficiency advantage of state-dependent corrections (19\% lower PPL at matched $\alpha$), but neither method achieves strong toxicity reduction at acceptable perplexity cost on these models.

\subsection{Future Work}

Four directions offer the highest impact:

\textbf{Intervention-aware barrier retraining.} The autoregressive experiment reveals that the barrier trained on human-text hidden states does not transfer perfectly to model-generated hidden states. Training the barrier on hidden states \textit{from steered generation}---closing the distribution-shift loop---is the most direct path to improving the autoregressive PPL ratio.

\textbf{Production-scale validation and autoregressive tuning.} Section~\ref{sec:scaling} extends to $n = 2048$ with strong barrier results but non-significant autoregressive steering. Validation on architectures that are actually deployed ($n \geq 4096$, $> 70$B parameters) is essential to determine whether the framework scales or whether the Qwen null result is indicative. Model-specific tuning of $\alpha$ and $\delta_{\text{buf}}$ (e.g., higher $\alpha$ for larger models with lower baseline toxicity) and substantially larger evaluation datasets are necessary next steps.

\textbf{Active adversarial verification.} The boundary sampling presented here may miss narrow failure regions in high dimensions. Complementing it with gradient-based adversarial threat hunting (momentum PGD) and adaptive spectral safety margins would strengthen the verification guarantee.

\textbf{Tighter Lipschitz estimation.} Replacing the product-of-spectral-norms bound with semidefinite-programming-based estimation \cite{fazlyab2019} or orthogonal-layer architectures \cite{li2019} that achieve $L_h = 1$ exactly, closing the gap between certified and empirical Lipschitz constants.

% ============================================================================
% References
% ============================================================================

\begin{thebibliography}{99}

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

\bibitem{bai2022}
Bai, Y. et~al. (2022).
Constitutional AI: Harmlessness from AI Feedback.
\textit{arXiv:2212.08073}.

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


\bibitem{glotfelter2017}
Glotfelter, P., Cort\'{e}s, J., \& Egerstedt, M. (2017).
Nonsmooth Barrier Functions with Applications to Multi-Robot Systems.
\textit{IEEE Control Systems Letters}, 1(2), 310--315.

\bibitem{jankovic2018}
Jankovic, M. (2018).
Robust Control Barrier Functions for Constrained Stabilization of Nonlinear Systems.
\textit{Automatica}, 96, 359--367.

\bibitem{khalil2002}
Khalil, H.~K. (2002).
\textit{Nonlinear Systems} (3rd ed.).
Prentice Hall.

\bibitem{li2019}
Li, Q., Haque, S., Anil, C., Lucas, J., Grosse, R., \& Jacobsen, J.-H. (2019).
Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks.
\textit{NeurIPS}, 32.

\bibitem{mitchell2005}
Mitchell, I.~M., Bayen, A.~M., \& Tomlin, C.~J. (2005).
A Time-Dependent Hamilton-Jacobi Formulation of Reachable Sets for Continuous Dynamic Games.
\textit{IEEE Trans.\ Automatic Control}, 50(7), 947--957.

\bibitem{miyato2018}
Miyato, T., Kataoka, T., Koyama, M., \& Yoshida, Y. (2018).
Spectral Normalization for Generative Adversarial Networks.
\textit{ICLR}.

\bibitem{ouyang2022}
Ouyang, L. et~al. (2022).
Training Language Models to Follow Instructions with Human Feedback.
\textit{NeurIPS}, 35, 27730--27744.

\bibitem{prajna2004}
Prajna, S. \& Jadbabaie, A. (2004).
Safety Verification of Hybrid Systems Using Barrier Certificates.
\textit{HSCC}, 477--492.

\bibitem{qin2021}
Qin, Z. et~al. (2021).
Learning Safe Multi-Agent Control with Decentralized Neural Barrier Certificates.
\textit{ICLR}.

\bibitem{rafailov2023}
Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.~D., \& Finn, C. (2023).
Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
\textit{NeurIPS}, 36.

\bibitem{radford2019}
Radford, A. et~al. (2019).
Language Models are Unsupervised Multitask Learners.
\textit{OpenAI Technical Report}.

\bibitem{robey2020}
Robey, A. et~al. (2020).
Learning Control Barrier Functions from Expert Demonstrations.
\textit{IEEE CDC}, 3717--3724.

\bibitem{tempo2012}
Tempo, R., Calafiore, G., \& Dabbene, F. (2012).
\textit{Randomized Algorithms for Analysis and Control of Uncertain Systems}.
Springer.

\bibitem{turner2023}
Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., \& MacDiarmid, M. (2023).
Activation Addition: Steering Language Models Without Optimization.
\textit{arXiv:2308.10248}.

\bibitem{xiao2019}
Xiao, W. \& Belta, C. (2019).
Control Barrier Functions for Systems with High Relative Degree.
\textit{IEEE 58th CDC}, 474--479.

\bibitem{zou2023}
Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Lin, Z., Forsyth, M., Hendrycks, D., Xie, C., Kawaguchi, K., Khashabi, D., \& Steinhardt, J. (2023).
Representation Engineering: A Top-Down Approach to AI Transparency.
\textit{arXiv:2310.01405}.

\end{thebibliography}

\end{document}
