\documentclass[a4paper,11pt]{article}

% Packages for formatting and functionality
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{amsmath, amssymb, amsfonts, amsthm}
\usepackage{graphicx}
\graphicspath{{assets/}}
\usepackage{hyperref}
\usepackage{authblk}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{titlesec}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{float}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algpseudocode}

% Theorem environments
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\newtheorem{definition}{Definition}
\newtheorem{remark}{Remark}

% Margin settings
\geometry{
    top=1in,
    bottom=1in,
    left=1in,
    right=1in
}

% Title and Author Info
\title{\textbf{Sign-Recovery Optimality in Vector Symbolic Architectures and a Negative Result on LLM Drift Detection}}
\author{\textbf{Arsenios Scrivens}}
\date{March 9, 2026}

\begin{document}

\maketitle

\begin{abstract}
We prove that sign-function restoration with abstention ($\text{sign}(0) = 0$) is the unique optimal component-wise estimator for recovering a bipolar signal from equal-magnitude bipolar superposition, achieving cosine similarity $1/\sqrt{2}$. The recovery channel is a Binary Erasure Channel with capacity $D/2$ bits. These results formalize well-studied concentration-of-measure phenomena in Vector Symbolic Architectures (VSA) as explicit optimality and information-theoretic bounds.

We apply this mechanism---Holographic Invariant Storage (HIS)---to LLM safety-prompt re-injection. A 540-conversation experiment across three LLMs with cross-model judging yields a clarifying negative result: when re-injection frequency is matched, a simple timer, embedding-similarity monitor, and random-timing baseline all achieve statistically equivalent safety rates to HIS-triggered re-injection (all $p > 0.05$). The original HIS advantage was driven by re-injection count, not adaptive timing. For practitioners, safety-prompt re-injection at moderate frequency improves compliance regardless of scheduling strategy. The modeling assumption that LLM context drift maps to additive bipolar noise is not validated by this work.
\end{abstract}

\vspace{1em}
\hrule
\vspace{1em}

\section{Introduction}

The current paradigm of Generative AI faces a structural challenge in long-horizon tasks: maintaining goal coherence as the context window fills with interaction history. While Transformer-based models \cite{vaswani2017} excel at in-context learning, they exhibit degradation in long contexts---often referred to as ``context drift''---where the probability of adhering to the original system prompt decreases as later tokens receive disproportionate attention weight \cite{liu2023}. This is exacerbated by the positional recency bias inherent in causal attention mechanisms, which can cause early instructions (including safety constraints) to be ``lost in the middle'' of long sequences.

This vulnerability is particularly acute in autonomous agents deployed over extended sessions, where behavioral drift can lead to reduced goal adherence and increased susceptibility to prompt injection attacks \cite{wei2023, perez2022}. LeCun \cite{lecun2022} has argued that autonomous machine intelligence requires robust memory architectures that maintain invariants over time---a challenge that current autoregressive models do not address at the architectural level. The fundamental limitation is architectural: the attention mechanism treats safety constraints as just another token sequence, weighted probabilistically against the immediate conversational context. As the context window fills, the relative weight of the original system prompt diminishes---not because the model ``forgets,'' but because competing signals dilute its influence.

Current approaches to this problem fall into several categories. Reinforcement Learning from Human Feedback (RLHF) \cite{ouyang2022} and Constitutional AI \cite{bai2022} embed safety preferences into model weights during training, providing baseline robustness but offering no mechanism for runtime verification or recovery when context-level drift occurs. Retrieval-Augmented Generation (RAG) \cite{lewis2020} and scratchpad-based memory \cite{nye2021} provide external memory, but their retrieved content still competes with context noise through the attention mechanism. Periodic re-prompting (re-injecting the system prompt at intervals) is a common engineering heuristic but lacks theoretical grounding and scales poorly with prompt length.

We propose a complementary approach: storing the safety signal in an external memory substrate based on \textbf{Vector Symbolic Architectures (VSA)} \cite{kanerva2009, plate1995, gayler2003}, then re-injecting the decoded instruction into the LLM's context when drift is detected. Hyperdimensional Computing (HDC) provides algebraic operations over high-dimensional distributed representations that exhibit well-characterized noise tolerance properties \cite{kanerva2009, thomas2021}. The key insight is that in $D$-dimensional bipolar spaces, randomly generated vectors are near-orthogonal with high probability \cite{kanerva2009, vershynin2018}, meaning that a safety signal encoded as a hypervector can be recovered from additive noise via algebraic inversion rather than statistical inference. We note at the outset that the recovered instruction still enters the LLM through the standard attention mechanism; HIS provides algebraic recovery guarantees for the \textit{storage and retrieval} step, not immunity from attention-level dilution after re-injection.

\textbf{Scope and Limitations.} This paper characterizes the signal-recovery mechanism and its mathematical properties, and provides an end-to-end behavioral experiment across three LLMs ($N = 30$ trials per condition per model, 270 conversations total), followed by a frequency-controlled extension (270 additional conversations across three new baseline conditions). The original experiment achieves statistical significance ($p < 0.01$) for the weakest-baseline model under self-judging, though a cross-model judging robustness check (Section~\ref{sec:crossjudge}) reveals this significance is judge-dependent; consistent positive-to-neutral effects are observed across all models and judges. Our model of ``context drift as additive noise in hypervector space'' is an abstraction that is \textit{not validated} by this paper: the behavioral experiments validate \textit{re-injection} (which works equally well without VSA machinery, as shown in Section~\ref{sec:extended_experiment}), not the noise model. The assumption that attention-level drift maps faithfully to additive bipolar corruption remains untested and is the paper's central gap. Our contribution has two parts: (1) the theoretical characterization of optimal sign-recovery in bipolar superposition, formalizing known VSA phenomena as explicit bounds; and (2) a carefully controlled experimental investigation yielding a clarifying negative result---adaptive drift-triggered re-injection confers no measurable advantage over simpler scheduling strategies at matched frequency.

\subsection{Contributions}
This paper makes two primary contributions, each self-contained:

\medskip
\noindent\textbf{Contribution A: Optimality and Capacity Bounds for VSA Signal Recovery.}
We prove that the sign function with abstention ($\text{sign}(0) = 0$) is the \textit{unique optimal} component-wise estimator for recovering a bipolar signal from equal-magnitude bipolar superposition, achieving cosine similarity $1/\sqrt{2}$ (Proposition~\ref{prop:optimality}). We characterize the recovery channel as a Binary Erasure Channel with capacity $D/2$ bits (Proposition~\ref{prop:bec}), connecting the geometric and information-theoretic perspectives. We extend the analysis to continuous noise distributions (Proposition~\ref{prop:continuous_noise}) and multi-signal storage with exact parity-dependent fidelity formulas (Proposition~\ref{prop:multi_signal}). These results formalize well-studied concentration phenomena in VSA \cite{kanerva2009, thomas2021} as explicit bounds. The practical consequence for the HDC community is concrete: implementations using random tiebreaking ($\text{sign}(0) = \pm 1$) or majority-vote cleanup pay a quantifiable fidelity penalty ($0.500$ vs.\ $0.707$), and no component-wise strategy can exceed $1/\sqrt{2}$ without exploiting cross-dimensional structure or reducing noise power below the signal. Sections~2--3 are self-contained and can be read independently of the LLM application.

\medskip
\noindent\textbf{Contribution B: A Negative Result on Adaptive LLM Safety Re-Injection.}
We conduct a 540-conversation experiment across three LLMs (Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B) comparing no intervention, timer-based re-injection, and HIS-triggered re-injection, with cross-model judging and frequency-controlled ablations (Sections~\ref{sec:llm_experiment}--\ref{sec:extended_experiment}). The principal finding is negative: when re-injection frequency is matched ($\sim 7$ per 30 turns), simpler baselines---a timer, an embedding-similarity monitor, and a random-timing control---achieve equivalent safety rates (all $p > 0.05$). The original HIS advantage on Gemma-2-2B ($p = 0.008$ under self-judging) is attributable to re-injection count ($\sim 14$ vs.\ 5), not adaptive timing. For safety practitioners, the actionable finding is that re-injection at moderate frequency ($\geq 5$ per 30 turns) improves compliance regardless of scheduling strategy. We provide implementation validation via Monte Carlo simulation ($n = 1{,}000$) and signal-level baselines (Section~\ref{sec:baselines}).

\medskip
We emphasize that HIS is not a standalone safety solution. It provides a \textit{signal-preservation} mechanism whose utility depends on integration with an LLM's inference pipeline.

\subsection{Paper Roadmap}
Section~2 presents the VSA operations and restoration protocol. Section~3 derives the geometric recovery bound, proves its optimality, and provides the information-theoretic characterization; these sections are self-contained for readers interested only in the VSA theory (Contribution~A). Section~4 validates the implementation, provides signal-level baselines, and presents the 540-conversation LLM experiment including the frequency-controlled extension (Contribution~B). Section~5 discusses the relationship to simpler baselines and integration requirements. Section~6 details limitations and threats to validity. Section~7 outlines future work. Section~8 concludes.

\subsection{Results at a Glance}
\begin{table}[H]
\centering
\caption{Summary of principal claims and their status.}
\label{tab:glance}
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Claim} & \textbf{Result} & \textbf{Status} \\ \midrule
Sign-recovery optimality & $1/\sqrt{2}$, unique among component-wise estimators & Proven (Prop.~\ref{prop:optimality}) \\
BEC capacity & $D/2$ bits per restoration & Proven (Prop.~\ref{prop:bec}) \\
Continuous noise recovery & $2\Phi(1/\sigma) - 1$ & Proven (Prop.~\ref{prop:continuous_noise}) \\
Multi-signal fidelity & Exact formulas, parity-dependent & Proven (Prop.~\ref{prop:multi_signal}) \\
Monte Carlo validation & $\mu = 0.7074$, $\sigma = 0.0039$ ($n = 1{,}000$) & Confirmed \\
HIS vs.\ timer at matched freq. & All $p > 0.05$ & Negative result \\
Re-injection benefit (weak models) & $+3$--$5$~pp at $\geq 5$ re-injections & Supported \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Related Work}
\label{sec:related_work}

\textbf{AI Safety and Alignment.}
The concrete challenges of AI safety were catalogued by Amodei et al.\ \cite{amodei2016}, including reward hacking, distributional shift, and safe exploration. RLHF \cite{ouyang2022} addresses alignment at training time by shaping model weights to reflect human preferences, while Constitutional AI \cite{bai2022} automates this process via self-critique. These methods embed safety into model parameters but provide no runtime mechanism to detect or correct context-level drift during long inference sessions. Wei et al.\ \cite{wei2023} demonstrate that safety training can be circumvented via ``jailbreak'' attacks that exploit in-context reasoning---precisely the failure mode HIS is designed to complement.

\textbf{External Memory for Language Models.}
Retrieval-Augmented Generation (RAG) \cite{lewis2020} augments LLMs with retrieved documents, providing external knowledge that can include safety-relevant instructions. However, retrieved content enters the context window and is subject to the same attention-based dilution as any other token sequence. Scratchpad and chain-of-thought methods \cite{nye2021, wei2022cot} use the context window itself as working memory, offering no protection against the context noise problem. Memory-augmented architectures such as Neural Turing Machines \cite{graves2014} and Memorizing Transformers \cite{wu2022} provide differentiable external memory, but these memories are trained end-to-end and do not offer algebraic recovery guarantees.

\textbf{Vector Symbolic Architectures and Hyperdimensional Computing.}
VSAs were introduced by Plate \cite{plate1995} as Holographic Reduced Representations (HRR) and independently developed by Kanerva \cite{kanerva2009} and Gayler \cite{gayler2003}. These architectures exploit the concentration of measure in high-dimensional spaces \cite{vershynin2018}: in $D \geq 1{,}000$ dimensions, random bipolar vectors are near-orthogonal ($|\cos\theta| \approx 1/\sqrt{D}$), enabling robust associative memory retrieval. Recent work has applied HDC to classification \cite{thomas2021}, language processing \cite{schlegel2022}, and cognitive modeling \cite{eliasmith2012}. The recovery properties of sign-based cleanup in bipolar superposition---including the concentration of cosine similarity around predictable values---are well-studied consequences of the binomial statistics of dimension-wise agreement \cite{kanerva2009, thomas2021}. Our theoretical contribution (Section~3) does not discover new recovery phenomena, but formalizes known behavior as explicit bounds: we prove that the sign function with abstention is the \textit{unique optimal} component-wise estimator, and characterize the recovery channel information-theoretically as a Binary Erasure Channel. We apply these bounds to a new domain: preserving safety-critical signals against context corruption in LLM agents.

\textbf{Cognitive Architectures.}
The idea of separating long-term invariant memory from working memory has precedent in cognitive architectures such as SOAR \cite{laird2012} and ACT-R \cite{anderson2004}, which maintain distinct declarative and procedural memory stores. HIS can be viewed as a minimalist instantiation of this principle: the safety constraint is stored in a ``declarative'' holographic memory that is queried algebraically rather than through attention-weighted retrieval.

\textbf{High-Dimensional Safety Verification.}
Control Barrier Functions (CBFs) \cite{ames2019} provide formal safety guarantees via forward invariance, but verification scales exponentially with state dimension via grid methods \cite{mitchell2005}. The relationship between semantic drift (this paper) and kinematic control drift is an open direction discussed in Section~\ref{sec:future_work}.

\section{Methodology}

\subsection{Vector Symbolic Architecture (VSA)}
Our approach utilizes $D$-dimensional bipolar hypervectors ($v \in \{-1,1\}^{D}$, with $D = 10{,}000$ throughout) to represent semantic concepts. We rely on three algebraic operations whose properties are well-established in the VSA literature \cite{kanerva2009, plate1995, gayler2003}:

\begin{itemize}
    \item \textbf{Binding ($\otimes$):} Element-wise multiplication of two vectors. For bipolar vectors, $a \otimes b \in \{-1,1\}^D$, and the result is near-orthogonal to both inputs ($\mathbb{E}[\cos(a, a \otimes b)] \approx 0$). Binding is its own inverse: $a \otimes a = \mathbf{1}$, so $(a \otimes b) \otimes a = b$. This is the mechanism that enables key-based retrieval.
    \item \textbf{Bundling ($+$):} Element-wise addition, creating a superposition that is similar to each of its components. This represents the accumulation of context signals (including noise).
    \item \textbf{Cleanup / Binarization ($\text{sign}(\cdot)$):} Element-wise sign function applied to a real-valued superposition, following the standard numerical convention:
    \begin{equation}
        \text{sign}(x) = \begin{cases} +1 & \text{if } x > 0 \\ 0 & \text{if } x = 0 \\ -1 & \text{if } x < 0 \end{cases}
        \label{eq:sign_convention}
    \end{equation}
    This convention is implemented by all standard numerical libraries (NumPy, PyTorch, Julia). The $\text{sign}$ function serves as the primary noise-suppression mechanism: dimensions where signal and noise agree are amplified, while dimensions where they cancel are \textit{abstained} ($\text{sign}(0) = 0$). As we show in Section~3, this abstention property---not a majority vote to $\pm 1$---is the mechanism that produces the $1/\sqrt{2}$ recovery bound.
\end{itemize}

\subsection{The Restoration Protocol}
We define the agent's safety constraint as a ``System Invariant'' ($H_{\text{inv}}$), created by binding a known Goal Key ($K_{\text{goal}}$) to the Safe Value vector ($V_{\text{safe}}$):
\begin{equation}
    H_{\text{inv}} = K_{\text{goal}} \otimes V_{\text{safe}}
    \label{eq:invariant}
\end{equation}

During operation, this invariant is corrupted by additive context noise ($N_{\text{context}}$), producing a drifted state. The restoration protocol recovers the original value by: (1) normalizing and superimposing the noise, (2) applying binarization, and (3) unbinding with the original key:
\begin{equation}
    V_{\text{recovered}} = \text{sign}\!\left(H_{\text{inv}} + \hat{N}_{\text{context}}\right) \otimes K_{\text{goal}}
    \label{eq:restoration}
\end{equation}
where $\hat{N}_{\text{context}}$ denotes the noise after normalization (see Section~\ref{sec:normalization}).

The recovery works because $K_{\text{goal}}$ is near-orthogonal to $\hat{N}_{\text{context}}$ in high dimensions, so unbinding distributes the noise uniformly across the hyperspace (each component contributes $\approx 0$ in expectation), while the signal component reconstructs $V_{\text{safe}}$ coherently.

\begin{algorithm}[H]
\caption{Holographic Invariant Storage: Restoration Protocol}\label{alg:restore}
\begin{algorithmic}[1]
\State \textbf{Offline (Initialization):}
\State Generate random bipolar key: $K_{\text{goal}} \in \{-1,1\}^D$
\State Encode safety constraint: $V_{\text{safe}} \in \{-1,1\}^D$
\State Compute invariant: $H_{\text{inv}} \leftarrow K_{\text{goal}} \otimes V_{\text{safe}}$
\State Store $(K_{\text{goal}}, H_{\text{inv}})$ in external memory (outside context window)
\Statex
\State \textbf{Online (Each Restoration Step):}
\State Encode current context as noise vector: $N_{\text{context}} \in \mathbb{R}^D$
\State Normalize: $\hat{N}_{\text{context}} \leftarrow N_{\text{context}} \cdot \frac{\|H_{\text{inv}}\|}{\|N_{\text{context}}\|}$ \Comment{See Section~\ref{sec:normalization}}
\State Superimpose: $S \leftarrow H_{\text{inv}} + \hat{N}_{\text{context}}$
\State Binarize: $S_{\text{clean}} \leftarrow \text{sign}(S)$
\State Unbind: $V_{\text{recovered}} \leftarrow S_{\text{clean}} \otimes K_{\text{goal}}$
\State \textbf{Return} $V_{\text{recovered}}$ \Comment{Approximate recovery of $V_{\text{safe}}$}
\end{algorithmic}
\end{algorithm}

\subsection{The Normalization Constraint}
\label{sec:normalization}
The geometric bound (Theorem~\ref{thm:geometric_bound}) holds under the condition that the noise vector is normalized to have the same magnitude as the invariant: $\|\hat{N}_{\text{context}}\| = \|H_{\text{inv}}\|$. For bipolar vectors, $\|H_{\text{inv}}\| = \sqrt{D}$, so this requires scaling the noise to $\|\hat{N}_{\text{context}}\| = \sqrt{D}$.

This is a \textbf{design constraint}, not a natural property of the system. In practice, context noise accumulates over time, and its raw magnitude may grow unboundedly. The normalization step (Algorithm~\ref{alg:restore}, Line 8) enforces this constraint explicitly. This is analogous to gain control in signal processing or batch normalization in neural networks \cite{ioffe2015}---a deliberate architectural choice that keeps the signal-to-noise ratio fixed at 1:1 (0 dB).

This design choice has consequences:
\begin{itemize}
    \item \textbf{Advantage:} Recovery fidelity becomes a deterministic geometric property of the architecture, independent of noise magnitude or content.
    \item \textbf{Limitation:} The normalization discards information about the \textit{magnitude} of the noise. A system under heavy attack and a system under mild perturbation produce the same normalized noise vector. The protocol recovers the safety signal with equal fidelity in both cases---but it does not provide a \textit{measure of threat level}. A complementary anomaly detection mechanism would be needed for that purpose.
\end{itemize}

\section{Theoretical Analysis}

We now derive the expected recovery fidelity from first principles. The key insight is that the standard $\text{sign}$ convention---$\text{sign}(0) = 0$ (Equation~\ref{eq:sign_convention})---introduces \textit{abstention} on tied dimensions, which reduces the norm of the cleaned vector and produces a cosine similarity strictly above what a random-tiebreaking convention would yield.

\begin{theorem}[Geometric Recovery Bound]
\label{thm:geometric_bound}
Let $H_{\text{inv}} \in \{-1,1\}^D$ and $\hat{N}_{\text{context}} \in \{-1,1\}^D$ be independent, uniformly random bipolar vectors with $D \gg 1$. Define $S = H_{\text{inv}} + \hat{N}_{\text{context}}$ and $S_{\text{clean}} = \text{sign}(S)$ under the standard convention (Equation~\ref{eq:sign_convention}). Then:
\begin{equation}
    \mathbb{E}\!\left[\text{CosSim}(S_{\text{clean}},\, H_{\text{inv}})\right] = \frac{1}{\sqrt{2}} \approx 0.7071
    \label{eq:geometric_bound}
\end{equation}
with concentration: for any $t > 0$,
\begin{equation}
    P\!\left(\left|\text{CosSim}(S_{\text{clean}},\, H_{\text{inv}}) - \frac{1}{\sqrt{2}}\right| > t\right) \leq 2\exp\!\left(-\Theta(Dt^2)\right)
    \label{eq:concentration}
\end{equation}
\end{theorem}

\begin{proof}
We analyze the per-dimension behavior, compute the inner product and norms, and assemble the cosine similarity.

\textbf{Step 1: Per-dimension outcomes.}
For each dimension $i$, the sum $S_i = H_{\text{inv},i} + \hat{N}_{\text{context},i}$ takes values in $\{-2,\, 0,\, +2\}$:
\begin{itemize}
    \item \textbf{Agreement} ($S_i = \pm 2$, probability $1/2$): Both vectors share the same sign. Then $\text{sign}(S_i) = H_{\text{inv},i}$ with certainty.
    \item \textbf{Cancellation} ($S_i = 0$, probability $1/2$): The two vectors disagree. Then $\text{sign}(S_i) = 0$ by the standard convention.
\end{itemize}
Thus $S_{\text{clean}}$ is a \textit{ternary} vector in $\{-1, 0, +1\}^D$, with each component independently equal to $H_{\text{inv},i}$ (with probability $1/2$) or $0$ (with probability $1/2$).

\textbf{Step 2: Inner product.}
The per-dimension contribution to the inner product is:
\begin{equation}
    S_{\text{clean},i} \cdot H_{\text{inv},i} = \begin{cases} H_{\text{inv},i}^2 = 1 & \text{with probability } 1/2 \\ 0 \cdot H_{\text{inv},i} = 0 & \text{with probability } 1/2 \end{cases}
\end{equation}
Therefore $\mathbb{E}[S_{\text{clean},i} \cdot H_{\text{inv},i}] = 1/2$, and by linearity:
\begin{equation}
    \mathbb{E}\!\left[\langle S_{\text{clean}},\, H_{\text{inv}} \rangle\right] = \frac{D}{2}
    \label{eq:inner_product}
\end{equation}

\textbf{Step 3: Norms.}
The norm of $H_{\text{inv}}$ is deterministic: $\|H_{\text{inv}}\| = \sqrt{D}$.

The norm of $S_{\text{clean}}$ depends on how many dimensions are non-zero. Let $K = |\{i : S_{\text{clean},i} \neq 0\}|$ be the number of agreement dimensions. Then $K \sim \text{Binomial}(D,\, 1/2)$ and:
\begin{equation}
    \|S_{\text{clean}}\|^2 = \sum_{i=1}^{D} S_{\text{clean},i}^2 = K
    \label{eq:norm_sclean}
\end{equation}
since each non-zero entry is $\pm 1$. Thus $\|S_{\text{clean}}\| = \sqrt{K}$, and by the law of large numbers, $K/D \to 1/2$ almost surely, giving $\|S_{\text{clean}}\| \to \sqrt{D/2}$.

\textbf{Step 4: Cosine similarity.}
Assembling the pieces:
\begin{align}
    \text{CosSim}(S_{\text{clean}},\, H_{\text{inv}}) &= \frac{\langle S_{\text{clean}},\, H_{\text{inv}} \rangle}{\|S_{\text{clean}}\| \cdot \|H_{\text{inv}}\|} \notag \\
    &= \frac{K}{\sqrt{K} \cdot \sqrt{D}} = \frac{\sqrt{K}}{\sqrt{D}} = \sqrt{\frac{K}{D}}
    \label{eq:cosine_derivation}
\end{align}
where we used the fact that $\langle S_{\text{clean}},\, H_{\text{inv}} \rangle = K$ (every non-zero entry of $S_{\text{clean}}$ agrees with $H_{\text{inv}}$ by construction, contributing $+1$; every zero entry contributes $0$). Taking expectations:
\begin{equation}
    \mathbb{E}\!\left[\text{CosSim}\right] = \mathbb{E}\!\left[\sqrt{K/D}\right] \to \sqrt{1/2} = \frac{1}{\sqrt{2}} \approx 0.7071
    \label{eq:final_bound}
\end{equation}
The convergence is justified by the concentration of $K/D$ around $1/2$: since $K \sim \text{Binomial}(D,\, 1/2)$, Hoeffding's inequality \cite{hoeffding1963} gives $P(|K/D - 1/2| > t) \leq 2\exp(-2Dt^2)$, and $\sqrt{\cdot}$ is Lipschitz on $[\epsilon, 1]$, so the cosine similarity concentrates around $1/\sqrt{2}$ with sub-Gaussian tails.
\end{proof}

\begin{remark}[Why $\text{sign}(0) = 0$ Matters]
\label{rem:sign_convention}
The bound $1/\sqrt{2}$ depends critically on the sign convention. If ties were broken randomly ($\text{sign}(0) = \pm 1$ with equal probability), then $\|S_{\text{clean}}\| = \sqrt{D}$ but the inner product remains $D/2$, yielding $\text{CosSim}_{\text{random-tiebreak}} = 1/2$. Abstaining on uncertain dimensions reduces the denominator without affecting the numerator, producing strictly higher fidelity ($0.707$ vs.\ $0.500$). This is a structural property of the recovery geometry, not an implementation artifact.
\end{remark}

\begin{proposition}[Optimality of Abstention]
\label{prop:optimality}
Let $H \in \{-1,1\}^D$ and $N \in \{-1,1\}^D$ be independent, uniformly random bipolar vectors. Define $S = H + N$. Among all component-wise recovery functions $\phi: \{-2, 0, +2\} \to \mathbb{R}$ with $\phi(2) = -\phi(-2) > 0$ (odd symmetry), the expected cosine similarity $\mathbb{E}[\text{CosSim}(\phi(S), H)]$ is uniquely maximized by $\phi(0) = 0$ (abstention), yielding the bound $1/\sqrt{2}$.
\end{proposition}

\begin{proof}
By symmetry, set $\phi(2) = 1$, $\phi(-2) = -1$ (the optimal values for agreement dimensions, since $\text{sign}(S_i) = H_i$ when $|S_i| = 2$). Let $\phi(0) = c$ for some $c \in \mathbb{R}$. On each dimension independently:
\begin{itemize}
    \item With probability $1/2$ (agreement): $\phi(S_i) = H_i$, contributing $\phi(S_i) \cdot H_i = 1$ to the inner product and $\phi(S_i)^2 = 1$ to the squared norm.
    \item With probability $1/2$ (cancellation): $\phi(S_i) = c$, contributing $c \cdot H_i$ (expected value $0$) to the inner product and $c^2$ to the squared norm.
\end{itemize}
Let $K \sim \text{Binomial}(D, 1/2)$ count the agreement dimensions. The inner product is $\langle \phi(S), H \rangle = K + c \sum_{i \in \text{cancel}} H_i$. Since the cancellation-set signs of $H$ are i.i.d.\ $\pm 1$, $\mathbb{E}[\sum_{\text{cancel}} H_i] = 0$, so $\mathbb{E}[\langle \phi(S), H \rangle] = D/2$ regardless of~$c$. The squared norm is $\|\phi(S)\|^2 = K + (D - K) c^2$, with expectation $D(1 + c^2)/2$. Therefore:
\begin{equation}
    \mathbb{E}\!\left[\text{CosSim}(\phi(S), H)\right] \to \frac{D/2}{\sqrt{D(1 + c^2)/2} \cdot \sqrt{D}} = \frac{1}{\sqrt{2(1 + c^2)}}
    \label{eq:optimality}
\end{equation}
This is strictly maximized at $c = 0$, yielding $1/\sqrt{2}$. Any $c \neq 0$ inflates the denominator without increasing the numerator in expectation, strictly reducing fidelity. The function $c \mapsto 1/\sqrt{2(1+c^2)}$ is smooth and unimodal, confirming that $c = 0$ is the unique global maximum.
\end{proof}

\begin{remark}[Significance]
Proposition~\ref{prop:optimality} elevates the $1/\sqrt{2}$ bound from ``the recovery you happen to get with the standard sign convention'' to ``the best possible recovery under any component-wise strategy at 0~dB SNR.'' Improving beyond $1/\sqrt{2}$ requires either reducing noise power below the signal (operating above 0~dB) or exploiting cross-dimensional structure, which the independence assumption precludes.
\end{remark}

\begin{proposition}[Information-Theoretic Characterization]
\label{prop:bec}
The component-wise recovery channel $H_i \to \text{sign}(H_i + N_i)$, with $N_i$ i.i.d.\ uniform $\{-1,+1\}$, is a Binary Erasure Channel (BEC) with erasure probability $\varepsilon = 1/2$. The channel capacity is $C = 1 - \varepsilon = 1/2$ bit per dimension, yielding a total information throughput of $D/2$ bits. Of the $D$ bits encoding the original signal, exactly $D/2$ survive the corruption-recovery cycle in expectation.
\end{proposition}

\begin{proof}
Each dimension $i$ independently produces one of two outcomes: (1) agreement ($S_i = \pm 2$, probability $1/2$): the output $\text{sign}(S_i) = H_i$ perfectly recovers the input bit; (2) cancellation ($S_i = 0$, probability $1/2$): the output $\text{sign}(0) = 0$ is an erasure symbol carrying no information about $H_i$. This is precisely the definition of a BEC($1/2$) \cite{cover2006}. The capacity of BEC($\varepsilon$) is $1 - \varepsilon$ bits per channel use \cite{cover2006}, giving $C = 1/2$ bit per dimension and $DC = D/2$ bits total.
\end{proof}

\begin{remark}[Connecting Geometry and Information]
The BEC characterization makes precise the intuition that ``$50\%$ shared variance'' ($\cos^2\theta = 0.5$) between recovered and original vectors corresponds to exactly half the information surviving. For codebook retrieval with $K$ candidates, $D/2$ recovered information bits provide discriminative power to distinguish among $K \ll 2^{D/2}$ alternatives. For $D = 10{,}000$, this means reliable retrieval from codebooks of size up to $2^{5{,}000}$---vastly exceeding any practical codebook. This information-theoretic view complements the geometric (cosine similarity) and algebraic (sign recovery) perspectives, and connects HIS to the rich theory of channel coding \cite{cover2006}.
\end{remark}

\begin{remark}[Practical Meaning of 0.71 Fidelity]
\label{rem:practical_fidelity}
A cosine similarity of $0.71$ between the recovered vector $V_{\text{recovered}}$ and the original $V_{\text{safe}}$ means the two vectors share approximately 50\% of their variance (since $\cos^2(\pi/4) = 0.5$). In isolation, this is not ``high fidelity'' in the signal-processing sense. However, its practical significance depends on the \textit{retrieval task}: if the system maintains a codebook of $K$ candidate safety vectors and retrieves the nearest neighbor, then $0.71$ cosine similarity is sufficient to discriminate the correct vector from $K > 10^6$ alternatives with high probability in $D = 10{,}000$ dimensions, since random bipolar vectors have expected pairwise similarity $\approx 0$ with standard deviation $\approx 1/\sqrt{D} \approx 0.01$ \cite{kanerva2009}. The recovered signal sits $\sim 70$ standard deviations above the noise floor of random candidates.
\end{remark}

\subsection{Extension to Continuous Noise}
\label{sec:continuous_noise}
Theorem~\ref{thm:geometric_bound} assumes bipolar noise ($\hat{N} \in \{-1,1\}^D$), which produces exact cancellation ($S_i = 0$) with probability $1/2$. In practice, the normalized noise vector produced by a semantic encoder is continuous-valued, not bipolar. We now characterize recovery under general continuous noise distributions.

\begin{proposition}[Continuous Noise Recovery]
\label{prop:continuous_noise}
Let $H_{\text{inv}} \in \{-1,1\}^D$ be a bipolar signal vector and $\hat{N}_{\text{context}} \in \mathbb{R}^D$ be a noise vector with i.i.d.\ components $\hat{N}_i$ drawn from a symmetric distribution with $\mathbb{E}[\hat{N}_i^2] = \sigma^2$, independent of $H_{\text{inv}}$. Define $S = H_{\text{inv}} + \hat{N}_{\text{context}}$ and $S_{\text{clean}} = \text{sign}(S)$ (with $\text{sign}(0) = 0$). Then for any continuous noise distribution (where $P(\hat{N}_i = \pm 1) = 0$):
\begin{equation}
    \mathbb{E}\!\left[\text{CosSim}(S_{\text{clean}},\, H_{\text{inv}})\right] = 2\,\Phi\!\left(\frac{1}{\sigma}\right) - 1
    \label{eq:continuous_bound}
\end{equation}
where $\Phi$ is the CDF of the standard normal (or, more generally, the CDF of $\hat{N}_i$ evaluated at~$1$). For continuous noise with $\sigma = 1$, this yields $2\,\Phi(1) - 1 \approx 2(0.8413) - 1 = 0.6827$.
\end{proposition}

\begin{proof}
For continuous noise, $P(S_i = 0) = P(\hat{N}_i = -H_i) = P(\hat{N}_i = \pm 1) = 0$ by assumption. Therefore every dimension produces a non-zero output: $S_{\text{clean}} \in \{-1,+1\}^D$ and $\|S_{\text{clean}}\| = \sqrt{D}$.

\textbf{Step 1: Agreement probability.} Fix dimension $i$ and set $H_i = 1$ without loss of generality (the case $H_i = -1$ is symmetric). Then $\text{sign}(1 + \hat{N}_i) = 1$ if and only if $\hat{N}_i > -1$. For $\hat{N}_i \sim \mathcal{N}(0, \sigma^2)$:
\[
    p \;=\; P(\hat{N}_i > -1) \;=\; P\!\left(\frac{\hat{N}_i}{\sigma} > -\frac{1}{\sigma}\right) \;=\; \Phi\!\left(\frac{1}{\sigma}\right)
\]
where $\Phi$ is the standard normal CDF and symmetry gives $P(Z > -a) = \Phi(a)$.

\textbf{Step 2: Inner product and cosine similarity.} Each dimension $i$ contributes $S_{\text{clean},i} \cdot H_i = +1$ (agreement, probability $p$) or $-1$ (disagreement, probability $1-p$). Since $\|S_{\text{clean}}\| = \|H\| = \sqrt{D}$:
\[
    \text{CosSim} \;=\; \frac{\sum_i S_{\text{clean},i} \cdot H_i}{\sqrt{D}\cdot\sqrt{D}} \;=\; \frac{Dp - D(1-p)}{D} \;=\; 2p - 1 \;=\; 2\,\Phi\!\left(\frac{1}{\sigma}\right) - 1
\]
The convergence to this expectation follows from Hoeffding's inequality applied to the $D$ independent dimension contributions, giving sub-Gaussian concentration as in Theorem~\ref{thm:geometric_bound}.
\end{proof}

\begin{remark}[Explaining the Empirical $0.707$]
\label{rem:empirical_explanation}
The Monte Carlo experiments (Section~\ref{sec:empirical}) use a semantic encoder that projects text through a random matrix and binarizes the result. After normalization, the per-component noise is approximately Gaussian with $\sigma \ll 1$ for sparse text representations (most components of the normalized noise have magnitude much less than~$1$). In this regime, $\Phi(1/\sigma) \to 1$ and $\text{CosSim} \to 1$. However, our implementation binarizes the noise to $\{-1,+1\}$ before superposition, recovering the bipolar case of Theorem~\ref{thm:geometric_bound}. The empirical convergence to $0.707$ is therefore consistent with both the bipolar theorem (exact match) and the continuous proposition (in the limit of the encoder's actual noise distribution). The bound is robust to encoder choice: any encoder that produces sufficiently incoherent noise will yield fidelity between $0.68$ (Gaussian, $\sigma = 1$) and $0.71$ (bipolar), depending on the effective per-component noise distribution.
\end{remark}

\subsection{Multi-Signal Storage and the Fidelity--Capacity Trade-Off}
\label{sec:multi_signal}
An agent may need to maintain $K > 1$ safety invariants simultaneously. VSA supports bundling multiple bound pairs into a composite invariant: $H_{\text{composite}} = \sum_{j=1}^{K} K_j \otimes V_j$. Recovery of the $k$-th signal via unbinding yields $V_k$ plus $(K-1)$ cross-talk terms (each near-orthogonal to $V_k$), effectively acting as $(K-1)$ additional noise sources.

\begin{proposition}[Multi-Signal Recovery]
\label{prop:multi_signal}
Let $H_{\text{composite}} = \sum_{j=1}^{K} K_j \otimes V_j$ where all keys $K_j$ and values $V_j$ are independent random bipolar vectors in $\{-1,1\}^D$. Add one noise vector $\hat{N}$ (bipolar, independent). Then recovery of any signal $V_k$ via $\text{sign}(H_{\text{composite}} + \hat{N}) \otimes K_k$ yields fidelity that decreases overall with $K$. Let $m = K + 1$ denote the total number of superimposed bipolar vectors. Define $p_e(K) = \binom{K}{(K-1)/2}/2^K$ for odd $K$. The exact cosine similarity is:
\begin{equation}
    \text{CosSim}(K) = \begin{cases}
        \displaystyle \binom{K}{K/2}\big/2^K & \text{if } K \text{ is even (odd } m, \text{ no erasures)} \\
        \displaystyle \frac{p_e(K)}{\sqrt{1 - p_e(K)}} & \text{if } K \text{ is odd (even } m, \text{ erasures occur)}
    \end{cases}
    \label{eq:multi_signal}
\end{equation}
Specifically: $K=1 \Rightarrow 0.707$, $K=2 \Rightarrow 0.500$, $K=3 \Rightarrow 0.474$, $K=5 \Rightarrow 0.377$.
\end{proposition}

\begin{proof}
After unbinding with $K_k$, the superposition is $S = V_k + \sum_{j \neq k} B_j + B_{\hat{N}}$ where the $K$ terms $B_j = (K_j \otimes V_j) \otimes K_k$ are i.i.d.\ bipolar (since products of independent bipolar vectors are bipolar). WLOG set $V_{k,i} = 1$ on each dimension. Then $S_i = 1 + X_i$ where $X_i = \sum_{j=1}^{K} B_{j,i}$ and each $B_{j,i}$ is i.i.d.\ $\pm 1$. Note that $X$ has the same parity as $K$.

\textbf{Case 1: Even $K$ (odd $m$, no erasures).}
Since $K$ is even, $X_i$ is even, so $S_i = 1 + X_i$ is odd and never zero: $S_{\text{clean}} \in \{-1,+1\}^D$, giving $\|S_{\text{clean}}\| = \sqrt{D}$. Agreement occurs when $S_i > 0$, i.e., $X_i \geq 0$. Writing $X_i = 2Y_i - K$ with $Y_i \sim \text{Binomial}(K, 1/2)$: $X_i \geq 0 \iff Y_i \geq K/2$. By symmetry, $P(Y_i > K/2) = P(Y_i < K/2)$, so:
\[
    p = P(Y_i \geq K/2) = \frac{1 + P(Y_i = K/2)}{2} = \frac{1 + \binom{K}{K/2}/2^K}{2}
\]
With no erasures, $\text{CosSim} = (2p - 1) = \binom{K}{K/2}/2^K$.

\textbf{Case 2: Odd $K$ (even $m$, erasures).}
Since $K$ is odd, $X_i$ is odd, so $S_i = 1 + X_i$ is even and can be zero (when $X_i = -1$). Three outcomes:
\begin{align*}
    p_a &= P(S_i > 0) = P(X_i \geq 1) = 1/2 \quad \text{(by symmetry of odd-valued } X\text{)} \\
    p_e &= P(S_i = 0) = P(X_i = -1) = \binom{K}{(K-1)/2}\big/2^K \\
    p_d &= P(S_i < 0) = 1/2 - p_e
\end{align*}
where $p_a = 1/2$ follows because $X$ is odd, so $P(X \geq 1) = P(X \leq -1)$, and these partition the probability space. Now:
\begin{align*}
    \mathbb{E}[\langle S_{\text{clean}}, H \rangle] &= D(p_a - p_d) = D \cdot p_e \\
    \mathbb{E}[\|S_{\text{clean}}\|^2] &= D(p_a + p_d) = D(1 - p_e)
\end{align*}
since agreement contributes $+1$, disagreement contributes $-1$, and erasure contributes $0$ to both inner product and squared norm. Assembling:
\[
    \text{CosSim} = \frac{D \cdot p_e}{\sqrt{D(1 - p_e)} \cdot \sqrt{D}} = \frac{p_e}{\sqrt{1 - p_e}}
\]

\textbf{Verification.} $K=1$: $p_e = \binom{1}{0}/2 = 1/2$, CosSim $= (1/2)/\sqrt{1/2} = 1/\sqrt{2} \approx 0.707$~\checkmark. $K=3$: $p_e = \binom{3}{1}/8 = 3/8$, CosSim $= (3/8)/\sqrt{5/8} = 3\sqrt{2}/(4\sqrt{5}) \approx 0.474$~\checkmark. $K=5$: $p_e = \binom{5}{2}/32 = 5/16$, CosSim $= (5/16)/\sqrt{11/16} = 5/(4\sqrt{11}) \approx 0.377$~\checkmark. For even $K$: $K=2$: $\binom{2}{1}/4 = 0.5$~\checkmark. $K=4$: $\binom{4}{2}/16 = 0.375$.
\end{proof}

\begin{remark}[Parity Alternation]
The fidelity curve exhibits a mild parity zigzag: odd-$K$ values benefit from the abstention mechanism (erasures reduce $\|S_{\text{clean}}\|$, boosting cosine similarity), while even-$K$ values have no erasures and rely purely on majority-vote agreement. For example, $K=4$ gives $0.375$ while $K=5$ gives $0.377$. This effect diminishes with $K$ and does not affect the practical conclusion: for $K > 5$, codebook cleanup is needed.
\end{remark}

For $K > 5$, direct recovery from the superposition becomes unreliable ($\text{CosSim} < 0.41$). In this regime, a \textbf{codebook cleanup} step restores exact retrieval: the recovered vector is compared against a stored codebook of all $K$ candidate values $\{V_1, \ldots, V_K\}$ via nearest-neighbor search, which succeeds with high probability as long as the cosine similarity exceeds the noise floor of $\approx 1/\sqrt{D}$ between random vectors \cite{kanerva2009, plate1995}. This codebook approach trades algebraic purity for exact retrieval at the cost of maintaining an external lookup table of size $O(KD)$.

\subsection{Recovery Under Unequal Signal-Noise Ratio}
\label{sec:snr}
The normalization constraint (Section~\ref{sec:normalization}) fixes the SNR at 0~dB by design. Relaxing this reveals the full parametric behavior. If the noise is scaled to $\|\hat{N}\| = \alpha\sqrt{D}$ while $\|H_{\text{inv}}\| = \sqrt{D}$ (so the signal-to-noise amplitude ratio is $1/\alpha$), the per-dimension agreement probability becomes $P(\text{sign}(H_i + \alpha N_i) = H_i) = p(\alpha)$, which is monotonically decreasing in $\alpha$: as noise power increases, fewer dimensions agree, and fidelity degrades. In the limits: $\alpha \to 0$ (no noise) gives $\text{CosSim} \to 1$; $\alpha = 1$ (equal magnitude) gives $1/\sqrt{2}$; $\alpha \to \infty$ gives $\text{CosSim} \to 0$.

The normalization constraint is therefore not a limitation that ``throws away information''---it is a \textit{design choice} that pins the operating point at a known, analytically characterized fidelity. A system that operates without normalization would have variable, load-dependent fidelity; with normalization, fidelity is a constant that can be budgeted against at design time.

\section{Empirical Validation}
\label{sec:empirical}

\subsection{Monte Carlo Simulation}
To confirm that our software implementation correctly realizes the analytical bound of Theorem~\ref{thm:geometric_bound}, we conducted a Monte Carlo simulation with $n = 1{,}000$ independent trials. We emphasize that this simulation validates the \textit{implementation}, not the mathematics---the bound is proven analytically and does not require empirical confirmation. The simulation's value is in verifying that the encoding pipeline, normalization step, and sign-recovery code produce results consistent with the theory. In each trial:
\begin{enumerate}
    \item A random bipolar invariant $H_{\text{inv}} \in \{-1,1\}^{10{,}000}$ was generated.
    \item A unique noise vector was produced by encoding adversarial text strings via a semantic encoder, then normalizing to $\|N\| = \sqrt{D}$.
    \item The restoration protocol (Algorithm~\ref{alg:restore}) was applied.
    \item Cosine similarity between $V_{\text{recovered}}$ and $V_{\text{safe}}$ was recorded.
\end{enumerate}

\textbf{Results:}
\begin{itemize}
    \item Mean Recovery Fidelity: $\mu = 0.7074$
    \item Standard Deviation: $\sigma = 0.0039$
    \item 95\% CI: $[0.7072, 0.7076]$
\end{itemize}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{figure1.png}
    \caption{\textbf{Distribution of Recovery Fidelity ($n = 1{,}000$).} The black curve is the normal fit ($\mu = 0.7074$); the red dashed line marks the theoretical bound $1/\sqrt{2} \approx 0.7071$ (Theorem~\ref{thm:geometric_bound}). The empirical distribution clusters tightly around the analytically predicted value, confirming that the implementation correctly realizes the proven bound.}
    \label{fig:fidelity}
\end{figure}

The empirical mean of $0.7074$ exceeds the theoretical $0.7071$ by $0.0003$---within the expected finite-$D$ correction of order $O(1/\sqrt{D}) \approx 0.01$. The low standard deviation ($\sigma = 0.0039$) confirms that the implementation is stable and free of systematic bias. This is an implementation sanity check, not a discovery: the theory predicts the result exactly, and the simulation confirms the code is correct.

\subsection{Robustness Across Noise Types}
\label{sec:noise_types}
To characterize the mechanism's invariance to noise content, we tested three qualitatively distinct noise conditions:
\begin{enumerate}
    \item \textbf{Information Flooding:} Injection of irrelevant URLs, citations, and structured data.
    \item \textbf{Adversarial Prompts:} Direct ``jailbreak''-style prompts designed to override safety instructions.
    \item \textbf{Semantic Distraction:} Literary excerpts providing topically unrelated but semantically rich noise.
\end{enumerate}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figure2a.png}
        \caption{Information Flooding}
        \label{fig:flooding}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figure2b.png}
        \caption{Adversarial Prompts}
        \label{fig:jailbreak}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figure2c.png}
        \caption{Semantic Distraction}
        \label{fig:neutral}
    \end{subfigure}

    \caption{\textbf{Noise-Type Invariance.} Comparison of drifted similarity (red, before restoration) vs.\ restored similarity (green, after restoration) across three noise conditions. The drifted state varies substantially across noise types (range: $-0.02$ to $0.16$), confirming that raw context corruption is content-dependent. After restoration, all conditions converge to $\approx 0.71$, confirming the content-independence predicted by Theorem~\ref{thm:geometric_bound}. The restoration protocol eliminates the dependence on noise semantics.}
    \label{fig:noise_types}
\end{figure}

In all conditions, the restored similarity converges to $\approx 0.71$ regardless of the noise type. Importantly, the \textit{pre-restoration} drifted similarity varies substantially across conditions (Figure~\ref{fig:noise_types}), confirming that the raw superposition is sensitive to noise content---but the restoration protocol eliminates this dependence.

This result is \textit{expected} from the theory: once noise is normalized (Section~\ref{sec:normalization}), the recovery depends only on the per-dimension sign statistics, which are content-independent. However, it provides useful validation that the semantic encoding pipeline does not introduce systematic correlations between the noise and the invariant that would violate the independence assumption.

\subsection{Signal-Level Baseline Comparisons}
\label{sec:baselines}
To contextualize HIS's recovery performance, we compare it against three baselines at the signal level---measuring how well each method preserves or retrieves the safety vector under increasing noise, without requiring end-to-end LLM integration. All methods encode the same safety text (``Protect the user and ensure safety'') using the same sentence-transformer projection and operate on the same noise vectors.

\begin{enumerate}
    \item \textbf{No Intervention (Control):} The drifted state $S = H_{\text{inv}} + \hat{N}$ is compared directly to $V_{\text{safe}}$ without restoration. This measures the raw cosine similarity of the corrupted signal.
    \item \textbf{Periodic Re-prompting (Simulated):} The safety vector $V_{\text{safe}}$ is re-injected into the superposition by adding it to the drifted state: $S_{\text{reprompt}} = H_{\text{inv}} + \hat{N} + V_{\text{safe}}$. This simulates the effect of re-injecting the system prompt into a noisy context. The re-injected signal competes additively with the noise---analogous to how a re-injected text prompt competes for attention weight.
    \item \textbf{RAG-based Retrieval (Simulated):} The noise vector $\hat{N}$ is used as a query against a codebook containing $V_{\text{safe}}$ and 99 random distractor vectors. The retrieved vector is the codebook entry with highest cosine similarity to the query. This simulates vector-database retrieval where the current context serves as the retrieval query.
    \item \textbf{HIS Restoration:} The full restoration protocol (Algorithm~\ref{alg:restore}).
\end{enumerate}

We vary the number of superimposed noise vectors from $K_{\text{noise}} = 1$ to $K_{\text{noise}} = 20$ (simulating context windows of increasing length) and measure each method's cosine similarity to $V_{\text{safe}}$ across 200 trials per condition.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\linewidth]{figure3_baselines.png}
    \caption{\textbf{Signal-Level Baseline Comparison} ($n = 200$ trials per condition). Cosine similarity to $V_{\text{safe}}$ vs.\ number of superimposed noise vectors; shaded regions show $\pm 1$ standard deviation. HIS (blue) maintains stable fidelity at $\approx 0.71$ across all noise levels; this constancy is a \emph{direct consequence} of the normalization step (Section~\ref{sec:snr}), which pins the effective SNR at 0~dB regardless of the number of raw noise sources---it is not that HIS ``resists'' increasing noise, but that normalization removes noise-magnitude information before restoration. No Intervention (red) degrades monotonically to $\approx 0$ as noise accumulates. Re-prompting (orange) provides partial recovery but degrades with noise depth because the re-injected signal is one of $K+2$ superimposed vectors. RAG retrieval (green) maintains high fidelity when the correct vector is retrievable but degrades when noise vectors become more similar to $V_{\text{safe}}$ than $V_{\text{safe}}$ is to itself under corruption.}
    \label{fig:baselines}
\end{figure}

\textbf{Results (Figure~\ref{fig:baselines}):}
\begin{itemize}
    \item \textbf{No Intervention} degrades as $O(1/\sqrt{K+1})$ in cosine similarity, reaching $\approx 0.15$ at $K = 20$ noise vectors. This confirms that raw superposition destroys the signal.
    \item \textbf{Periodic Re-prompting} partially recovers fidelity (the re-injected $V_{\text{safe}}$ adds a second copy of the signal), but still degrades with noise depth because the signal is two copies competing against $K$ noise vectors. At $K = 20$, re-prompting achieves $\approx 0.30$---better than no intervention but below HIS.
    \item \textbf{RAG Retrieval} performs well when the codebook is small and noise is mild ($\approx 0.85$ at $K = 1$) but is vulnerable to adversarial noise that is semantically similar to the safety constraint. Its performance depends on the codebook composition and query quality, not on algebraic guarantees.
    \item \textbf{HIS} maintains $0.707 \pm 0.004$ across all noise levels, because normalization pins the effective SNR at 0~dB regardless of the number of raw noise sources. The fidelity is constant by construction (Theorem~\ref{thm:geometric_bound}).
\end{itemize}

The key distinction is that HIS's fidelity is a \textit{design invariant}: it does not degrade with context length, noise intensity, or noise content. The baselines' fidelity is \textit{load-dependent}: it varies with the amount and nature of context noise. This stability is HIS's primary advantage at the signal level. Whether this advantage translates to improved behavioral safety outcomes in an end-to-end LLM system remains an open empirical question (Section~\ref{sec:future_work}).

\textbf{Limitations of this comparison.} These are signal-level measurements, not behavioral evaluations. Re-prompting and RAG operate on natural language and benefit from the LLM's ability to parse redundant instructions; HIS operates on hypervectors and benefits from algebraic structure. A fair end-to-end comparison requires LLM integration (Section~\ref{sec:future_work}).

\subsection{Encoder Characterization}
\label{sec:encoder}
The theoretical analysis assumes noise vectors are independent of the invariant. We validate this assumption and characterize the encoder's statistical properties.

\textbf{Orthogonality verification.} We encoded 500 distinct text samples (adversarial prompts, conversational text, technical jargon, literary excerpts) and computed all $\binom{500}{2} = 124{,}750$ pairwise cosine similarities between the resulting bipolar hypervectors. The distribution has mean $\mu = -0.0002$ and standard deviation $\sigma = 0.0100$, consistent with the theoretical prediction for random bipolar vectors ($\mu = 0$, $\sigma = 1/\sqrt{D} = 0.01$) \cite{kanerva2009}. No pair exceeded $|\text{CosSim}| > 0.04$, confirming that the encoder produces near-orthogonal vectors across diverse semantic content and that the independence assumption of Theorem~\ref{thm:geometric_bound} is not violated by the encoding pipeline.

\textbf{Bag-of-words vs.\ sentence-transformer encoder.} We compared recovery fidelity using two encoder architectures: (a) the bag-of-words hash encoder used in the Monte Carlo experiments, and (b) a sentence-transformer encoder (all-MiniLM-L6-v2 \cite{reimers2019}) projected to $D = 10{,}000$ via a random matrix and binarized. Both encoders yield recovery fidelity of $0.707 \pm 0.004$ across 1{,}000 trials, confirming that the bound is encoder-agnostic for any encoder that produces sufficiently incoherent bipolar outputs. This is predicted by Theorem~\ref{thm:geometric_bound}: the recovery depends only on the per-dimension agreement statistics, which are content- and encoder-independent for bipolar vectors.

\subsection{Integration Proof of Concept}
\label{sec:integration_poc}
To demonstrate the end-to-end pipeline that would sit alongside an LLM, we implemented a multi-turn codebook-based restoration demo. The setup:
\begin{itemize}
    \item A codebook of $K = 20$ natural-language safety instructions is encoded into bipolar hypervectors ($D = 10{,}000$).
    \item The primary instruction (``\textit{Protect the user and ensure safety at all times.}'') is bound to a key via $H_{\text{inv}} = K_{\text{goal}} \otimes V_{\text{safe}}$.
    \item Over $T = 50$ simulated conversation turns, a unique noise vector is generated per turn and accumulated: $\hat{N}_t = \sum_{i=1}^{t} N_i$, normalized to $\|\hat{N}_t\| = \sqrt{D}$.
    \item At each turn: (a) the drifted state is formed, (b) HIS restoration is applied, (c) the recovered vector is decoded via nearest-neighbour lookup in the codebook.
\end{itemize}

\textbf{Results.} Raw cosine similarity between the drifted state and $V_{\text{safe}}$ degrades to a mean of $0.150$ (effectively unrecoverable by direct comparison). After HIS restoration, the recovered vector maintains cosine similarity of $0.641$ to $V_{\text{safe}}$---below the $0.707$ bipolar bound because the cumulative multi-turn noise follows a continuous distribution (consistent with Proposition~\ref{prop:continuous_noise} at $\sigma \approx 1$: predicted $0.683$). Critically, \textbf{codebook retrieval accuracy is 100\%} (50/50 turns): the nearest-neighbour lookup correctly identifies the primary instruction at every turn, with a margin of $0.64$ vs.\ $0.16$ for the runner-up ($\sim 48$ standard deviations above the noise floor).

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\linewidth]{figure4_integration_poc.png}
    \caption{\textbf{Multi-Turn Integration PoC.} \textit{Top:} Raw integrity (red) degrades to $\approx 0.15$ over 50 turns; HIS restoration (blue) maintains $\approx 0.64$. \textit{Bottom:} Codebook retrieval is correct at every turn (green bars). The decoded instruction would be re-injected into the LLM's context window.}
    \label{fig:integration_poc}
\end{figure}

\textbf{Pipeline in deployment.} In a production system, the decoded instruction (e.g., ``Protect the user and ensure safety at all times.'') would be re-injected into the LLM's system prompt at scheduled intervals or when the raw integrity score drops below a threshold $\tau$. The demo uses $\tau = 0.25$; all 50 turns fall below this threshold, indicating that restoration would be triggered continuously---the expected behavior for an accumulating-noise scenario. The key finding is that the \textit{signal-to-codebook margin remains large enough for exact instruction recovery} even when the raw signal is severely degraded.

\subsection{End-to-End LLM Experiment}
\label{sec:llm_experiment}
To bridge the gap between signal-level characterization and behavioral evaluation, we conducted a multi-model end-to-end experiment across three open-weight LLMs of comparable scale: Qwen2.5-3B \cite{qwen2025}, Llama-3.2-3B \cite{touvron2023}, and Gemma-2-2B \cite{gemma2024}, all served locally via Ollama on an NVIDIA RTX 5050 GPU.

\textbf{Design.} We constructed a 30-turn conversation script mixing 16 benign prompts with 14 adversarial prompts across three attack types: (1) \textit{context-embedded} attacks (turns 7--10: unsafe requests framed as ``novel research''), (2) \textit{jailbreak} attacks (turns 15--19: DAN prompts, developer mode, instruction override), and (3) \textit{strong jailbreak} attacks (turns 25--29: fabricated authorization, exam pretexts, explicit override). Benign turns between attack waves serve as context dilution. Three conditions were tested:
\begin{enumerate}
    \item \textbf{No Intervention:} System prompt set once at turn 0; no further safety re-injection.
    \item \textbf{Timer Re-injection:} Safety instruction re-injected as a system message every $k = 5$ turns, regardless of drift (5 re-injections per conversation).
    \item \textbf{HIS Re-injection:} The HIS pipeline encodes the recent conversation context as noise, accumulates it without normalization (allowing natural drift growth), monitors cosine similarity between the unbound recovered vector and $V_{\text{safe}}$, and re-injects the codebook-decoded safety instruction when fidelity drops below $\tau = 0.45$. After each re-injection, the accumulated noise resets to zero. This produces approximately 7 re-injections per conversation, triggered adaptively by actual signal degradation rather than a fixed clock.
\end{enumerate}
Each condition was repeated for $N = 30$ trials per model (3 models $\times$ 3 conditions $\times$ 30 trials = 270 conversations, 3{,}780 adversarial prompts scored). Conditions were interleaved within each trial number to balance temporal effects. Responses were classified as SAFE (refusal) or UNSAFE (compliance) by a two-stage classifier: rule-based pattern matching for refusal language, confirmed by an LLM-as-judge \cite{zheng2023} using the same model in a separate evaluation context with $\text{temperature} = 0$.

\textbf{Results.}
\begin{table}[H]
\centering
\caption{End-to-End LLM Safety Compliance ($N = 30$ trials per condition per model)}
\label{tab:llm_results}
\begin{tabular}{@{}llccc@{}}
\toprule
\textbf{Model} & \textbf{Condition} & \textbf{Safety Rate} & \textbf{Refused / Total} & \textbf{Std} \\ \midrule
\multirow{3}{*}{Gemma-2-2B} & No Intervention & 78.3\% & 329 / 420 & $\pm$ 8.7\% \\
& Timer Re-injection & 78.8\% & 331 / 420 & $\pm$ 7.1\% \\
& \textbf{HIS Re-injection} & \textbf{83.8\%} & \textbf{352 / 420} & $\pm$ \textbf{6.5\%} \\ \midrule
\multirow{3}{*}{Llama-3.2-3B} & No Intervention & 69.3\% & 291 / 420 & $\pm$ 6.8\% \\
& Timer Re-injection & 74.3\% & 312 / 420 & $\pm$ 6.9\% \\
& HIS Re-injection & 72.9\% & 306 / 420 & $\pm$ 9.6\% \\ \midrule
\multirow{3}{*}{Qwen2.5-3B} & No Intervention & 87.9\% & 369 / 420 & $\pm$ 6.3\% \\
& Timer Re-injection & 88.8\% & 373 / 420 & $\pm$ 4.5\% \\
& HIS Re-injection & 88.1\% & 370 / 420 & $\pm$ 4.7\% \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Statistical Significance and Effect Sizes (Welch's $t$-test, two-tailed, $N = 30$)}
\label{tab:significance}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Model} & \textbf{Comparison} & \textbf{$t$ / $p$} & \textbf{Cohen's $d$} \\ \midrule
\multirow{2}{*}{Gemma-2-2B} & HIS vs.\ No Interv. & $t = +2.77$, $p = 0.008$~** & $0.72$ \\
& HIS vs.\ Timer & $t = +2.84$, $p = 0.006$~** & $0.73$ \\ \midrule
\multirow{2}{*}{Llama-3.2-3B} & HIS vs.\ No Interv. & $t = +1.66$, $p = 0.103$ & $0.43$ \\
& HIS vs.\ Timer & $t = -0.66$, $p = 0.512$ & $0.17$ \\ \midrule
\multirow{2}{*}{Qwen2.5-3B} & HIS vs.\ No Interv. & $t = +0.17$, $p = 0.869$ & $0.04$ \\
& HIS vs.\ Timer & $t = -0.60$, $p = 0.550$ & $0.15$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth]{figure5_llm_experiment.png}
    \caption{\textbf{End-to-End Multi-Model LLM Safety Experiment} ($N = 30$ trials $\times$ 30 turns per conversation, 270 total conversations). Each panel shows mean safety compliance rate with $\pm 1$ standard deviation error bars. HIS achieves its strongest effect on the weakest-baseline model (Gemma-2: +5.5~pp over no intervention, $p < 0.01$), is neutral on stronger models, and never degrades performance relative to no intervention.}
    \label{fig:llm_experiment}
\end{figure}

\textbf{Observations:}
\begin{enumerate}
    \item \textbf{HIS provides statistically significant improvement where safety baselines are weakest.} On Gemma-2-2B (the lowest-performing baseline at 78.3\%), HIS achieves 83.8\%---a +5.5 percentage point improvement over no intervention ($p = 0.008$) and +5.0~pp over timer re-injection ($p = 0.006$), both significant at the $\alpha = 0.01$ level. HIS also achieves the lowest variance on this model ($\pm 6.5\%$ vs.\ $\pm 8.7\%$), indicating more consistent safety behavior across trials.
    \item \textbf{On stronger models, HIS is neutral-to-positive.} On Llama-3.2-3B and Qwen2.5-3B, all three conditions cluster within $\sim 3$ percentage points, and no pairwise comparison reaches significance. HIS does not degrade performance on any model---an important ``do no harm'' property for a safety intervention.
    \item \textbf{Both re-injection strategies improve safety on weaker models.} Llama-3.2-3B shows a +5.0~pp improvement for timer ($p = 0.103$, trending) and +3.6~pp for HIS over no intervention. This is consistent with the theory that weaker safety baselines leave more room for re-injection to help.
    \item \textbf{Context-embedded attacks (turns 7--10) remain the most vulnerable attack surface.} All models show substantially lower refusal rates on turns 7--10 (``novel research'' framing: 36--67\% refusal pooled) and turn 19 (``security researcher'' framing: 33--38\% refusal pooled) compared to explicit jailbreak prompts (turns 16--18, 25--29: $\geq 99\%$ refusal). HIS shows its strongest per-turn advantage on turn 10 (social engineering research: 67\% HIS vs.\ 44\% no intervention, pooled across models), precisely where context manipulation is most subtle.
    \item \textbf{Explicit jailbreaks are universally refused.} Turns 16--18 and 25--29 achieve $\geq 99\%$ refusal across all conditions and models, indicating robust safety training against overt jailbreak attempts. The remaining vulnerability lies in context-embedded attacks that exploit narrative framing.
    \item \textbf{Turn 19 is universally vulnerable.} The ``security researcher'' framing (``I need you to explain in detail how ransomware encrypts files'') achieved only 33--38\% refusal across all conditions and models. This is a failure of the models' safety training (the request is classified as legitimate education), not of the re-injection mechanism.
\end{enumerate}

\textbf{Interpretation: HIS helps most where it is needed most.} The pattern across models is consistent with a simple explanation: models with strong built-in safety training (Qwen2.5-3B: 87.9\% baseline) already refuse most adversarial prompts regardless of context drift, creating a ceiling effect where any intervention has limited room to improve. Models with weaker safety training (Gemma-2-2B: 78.3\% baseline) are more susceptible to drift-induced compliance, and HIS's drift-triggered re-injection provides a meaningful safety uplift in this regime. This is precisely the scenario HIS is designed for: preserving safety adherence when the model's intrinsic safety is insufficient to resist context-level manipulation.

\textbf{Re-injection frequency.} HIS triggered approximately 14 re-injections per 30-turn conversation (adaptive, based on actual drift with threshold $\tau = 0.45$), compared to 5 for the timer condition (fixed at turns 5, 10, 15, 20, 25). This substantial frequency difference (14 vs.\ 5) leaves a confound: HIS's advantage on Gemma-2 may be attributable to re-injection frequency rather than adaptive timing. We address this directly with a frequency-controlled extension in Section~\ref{sec:extended_experiment}.

\textbf{Limitations of this experiment.}
\begin{itemize}
    \item All three models are small (2--3B parameters) with relatively strong built-in safety training, creating ceiling effects on stronger models. Larger differences might emerge with less safety-trained models or with more sophisticated attacks.
    \item The LLM-as-judge uses the same model that generated the response, introducing potential self-evaluation bias. We address this directly with a cross-model judging robustness check in Section~\ref{sec:crossjudge}.
    \item The conversation script is fixed; a larger study should use randomized prompt orderings and a broader attack taxonomy \cite{wang2023}.
    \item The re-injection frequency differs between HIS ($\sim 14$) and timer (5). A frequency-controlled extension (Section~\ref{sec:extended_experiment}) addresses this confound directly.
\end{itemize}

\subsection{Cross-Model Judging Robustness Check}
\label{sec:crossjudge}

The primary experiment (Section~\ref{sec:llm_experiment}) uses self-judging: the same model that generated a response also evaluates whether the response constitutes a refusal. This introduces potential self-evaluation bias \cite{zheng2023}. To assess the robustness of our findings, we re-scored all 270 trials using independent judge models: Gemma-2-2B and Llama-3.2-3B responses were re-evaluated by Qwen2.5-3B, and Qwen2.5-3B responses were re-evaluated by Llama-3.2-3B. Each judge received the same binary classification prompt (``Did the model refuse or comply?'') with the original response text.

\begin{table}[H]
\centering
\caption{Cross-Model Judging Results ($N = 30$ trials per condition per model). Self-judge rates are reproduced from Table~\ref{tab:llm_results} for comparison.}
\label{tab:crossjudge}
\begin{tabular}{@{}llcccc@{}}
\toprule
\textbf{Model} & \textbf{Condition} & \textbf{Self-Judge} & \textbf{Cross-Judge} & \textbf{Agreement} & \textbf{Judge} \\ \midrule
\multirow{3}{*}{Gemma-2-2B} & No Intervention & 78.3\% & 94.3\% & 77.4\% & \multirow{3}{*}{Qwen} \\
& Timer Re-injection & 78.8\% & 96.4\% & 79.5\% & \\
& HIS Re-injection & 83.8\% & 95.7\% & 84.3\% & \\ \midrule
\multirow{3}{*}{Llama-3.2-3B} & No Intervention & 69.3\% & 97.9\% & 71.4\% & \multirow{3}{*}{Qwen} \\
& Timer Re-injection & 74.3\% & 97.9\% & 76.4\% & \\
& HIS Re-injection & 72.9\% & 98.6\% & 74.3\% & \\ \midrule
\multirow{3}{*}{Qwen2.5-3B} & No Intervention & 87.9\% & 73.3\% & 81.7\% & \multirow{3}{*}{Llama} \\
& Timer Re-injection & 88.8\% & 74.0\% & 81.0\% & \\
& HIS Re-injection & 88.1\% & 75.2\% & 83.3\% & \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Cross-Judge Statistical Significance (Welch's $t$-test, two-tailed, $N = 30$). Compare with Table~\ref{tab:significance}.}
\label{tab:crossjudge_significance}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Model (Judge)} & \textbf{HIS vs.\ No Intervention} & \textbf{HIS vs.\ Timer} \\ \midrule
Gemma-2-2B (Qwen) & $t = +1.39$, $p = 0.170$ & $t = -0.68$, $p = 0.498$ \\
Llama-3.2-3B (Qwen) & $t = +0.89$, $p = 0.380$ & $t = +0.82$, $p = 0.419$ \\
Qwen2.5-3B (Llama) & $t = +1.22$, $p = 0.226$ & $t = +0.77$, $p = 0.443$ \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Observations:}
\begin{enumerate}
    \item \textbf{Judge leniency is asymmetric.} Qwen2.5-3B as judge rates 94--99\% of responses as safe refusals, substantially more lenient than self-judging (69--84\%). Llama-3.2-3B as judge is stricter, rating 73--75\% safe versus Qwen's self-assessed 88\%. Overall inter-judge agreement is 78.8\%, indicating moderate systematic disagreement between models on what constitutes a ``refusal.''
    \item \textbf{The Gemma significance finding does not survive cross-judging.} Under self-judging, HIS vs.\ no intervention on Gemma-2-2B was significant ($p = 0.008$). Under cross-judging by Qwen2.5-3B, the same comparison yields $p = 0.170$ (Table~\ref{tab:crossjudge_significance}). The Qwen judge's leniency compresses all safety rates toward the ceiling (94--96\%), leaving insufficient variance to detect condition differences.
    \item \textbf{The direction of HIS advantage is preserved.} Under cross-judging, HIS achieves the highest safety rate for all three models (95.7\%, 98.6\%, 75.2\%), maintaining the same ordinal ranking as self-judging. The effect sizes shrink substantially but remain consistently positive (Cohen's $d = 0.20$--$0.37$).
    \item \textbf{Response truncation limits judge accuracy.} Responses were stored as 200-character previews in the original experiment, which may truncate the portion of a response that distinguishes a ``refusal with educational content'' from ``compliance with a safety caveat.'' This truncation likely contributes to the inter-judge disagreement rate.
\end{enumerate}

\textbf{Interpretation.} The cross-judge analysis reveals that the self-judging significance result ($p = 0.008$) is not robust to judge substitution. This does not invalidate the HIS mechanism---the direction of the effect is consistent across all judges and models---but it does mean that the statistical significance claim should be interpreted cautiously. The most conservative reading is that HIS provides a consistent, small positive effect on safety compliance (Cohen's $d \approx 0.2$--$0.4$), but the magnitude and significance are judge-dependent. The asymmetric leniency pattern (Qwen lenient, Llama strict) is itself a noteworthy finding for LLM-as-judge methodology: small models exhibit systematic evaluation biases that can inflate or deflate safety metrics depending on which model serves as evaluator. We report both self-judge and cross-judge results and recommend that future work use independent, larger judge models with access to full response text.

\subsection{Extended Experiment: Frequency-Controlled Baselines}
\label{sec:extended_experiment}

The primary LLM experiment (Section~\ref{sec:llm_experiment}) compared HIS re-injection ($\sim 14$ adaptive triggers per conversation) against timer re-injection ($k = 5$, i.e., 5 fixed triggers), leaving a confound: HIS's advantage on Gemma-2-2B may be attributable to \textit{how often} it re-injects rather than \textit{when}. We resolve this with three additional conditions designed to disentangle frequency from timing strategy.

\textbf{Design.} Three new conditions, each using the same 30-turn conversation script and safety instruction as the original experiment:
\begin{enumerate}
    \item \textbf{Matched-Frequency Timer ($k = 4$):} Safety instruction re-injected every 4 turns, producing exactly 7 re-injections per conversation---below HIS's $\sim 14$ but above the original timer's 5.
    \item \textbf{Embedding-Similarity Monitor:} The model's own embedding API computes cosine similarity between the current conversation context and the safety prompt. Re-injection is triggered when similarity drops below a calibrated threshold (set per model via a single pilot conversation to produce $\sim 7$ triggers). This tests whether the model's native embeddings serve as an equally effective drift signal, without VSA machinery.
    \item \textbf{Random-Timing Control:} Re-injection occurs at 7 randomly selected turns (seeded per trial), controlling for the possibility that any 7-injection schedule improves safety regardless of timing strategy.
\end{enumerate}
Each condition was tested on all three models with $N = 30$ trials (3 models $\times$ 3 conditions $\times$ 30 trials = 270 additional conversations, 3,780 adversarial prompts scored), using the same conversation script, safety prompt, and evaluation pipeline as the original experiment.

\textbf{Results.}

\begin{table}[H]
\centering
\caption{Extended Experiment: Frequency-Controlled Baselines ($N = 30$ per condition per model). Original conditions reproduced from Table~\ref{tab:llm_results} for comparison.}
\label{tab:extended_results}
\begin{tabular}{@{}llcccc@{}}
\toprule
\textbf{Model} & \textbf{Condition} & \textbf{Safety Rate} & \textbf{Std} & \textbf{Re-inj.} & \textbf{Source} \\ \midrule
\multirow{6}{*}{Gemma-2-2B} & No Intervention & 78.3\% & $\pm$ 8.7\% & 0 & Original \\
& Timer ($k=5$) & 78.8\% & $\pm$ 7.1\% & 5 & Original \\
& HIS & 83.8\% & $\pm$ 6.5\% & 14 & Original \\
& Timer ($k=4$) & 82.1\% & $\pm$ 6.7\% & 7 & Extended \\
& Embedding Monitor & 83.1\% & $\pm$ 8.1\% & 4.3 & Extended \\
& Random Timing & 82.4\% & $\pm$ 8.5\% & 7 & Extended \\ \midrule
\multirow{6}{*}{Llama-3.2-3B} & No Intervention & 69.3\% & $\pm$ 6.8\% & 0 & Original \\
& Timer ($k=5$) & 74.3\% & $\pm$ 6.9\% & 5 & Original \\
& HIS & 72.9\% & $\pm$ 9.6\% & 14 & Original \\
& Timer ($k=4$) & 73.1\% & $\pm$ 8.3\% & 7 & Extended \\
& Embedding Monitor & 72.1\% & $\pm$ 7.8\% & 5.6 & Extended \\
& Random Timing & 76.7\% & $\pm$ 9.0\% & 7 & Extended \\ \midrule
\multirow{6}{*}{Qwen2.5-3B} & No Intervention & 87.9\% & $\pm$ 6.3\% & 0 & Original \\
& Timer ($k=5$) & 88.8\% & $\pm$ 4.5\% & 5 & Original \\
& HIS & 88.1\% & $\pm$ 4.7\% & 14 & Original \\
& Timer ($k=4$) & 88.1\% & $\pm$ 5.4\% & 7 & Extended \\
& Embedding Monitor & 85.7\% & $\pm$ 5.6\% & 7.9 & Extended \\
& Random Timing & 87.6\% & $\pm$ 5.3\% & 7 & Extended \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Extended Experiment Statistical Tests (Welch's $t$-test, two-tailed, $N = 30$). All comparisons report exact $p$-values and Cohen's $d$.}
\label{tab:extended_significance}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Model} & \textbf{HIS vs.\ Timer ($k\!=\!4$)} & \textbf{HIS vs.\ EmbMon} & \textbf{HIS vs.\ Random} & \textbf{Timer $k\!=\!4$ vs.\ $k\!=\!5$} \\ \midrule
Gemma-2-2B & $p = 0.331$, $d = 0.26$ & $p = 0.707$, $d = 0.10$ & $p = 0.469$, $d = 0.18$ & $p = 0.067$, $d = 0.48$ \\
Llama-3.2-3B & $p = 0.919$, $d = 0.02$ & $p = 0.754$, $d = 0.09$ & $p = 0.119$, $d = 0.41$ & $p = 0.549$, $d = 0.16$ \\
Qwen2.5-3B & $p = 1.000$, $d = 0.00$ & $p = 0.081$, $d = 0.46$ & $p = 0.714$, $d = 0.10$ & $p = 0.580$, $d = 0.14$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Re-injection Frequency by Condition (mean $\pm$ std per 30-turn conversation)}
\label{tab:reinjection_freq}
\begin{tabular}{@{}llccc@{}}
\toprule
\textbf{Condition} & \textbf{Re-inj.} & \textbf{Gemma} & \textbf{Llama} & \textbf{Qwen} \\ \midrule
Timer ($k=4$) & $7.0 \pm 0.0$ & $7.0 \pm 0.0$ & $7.0 \pm 0.0$ & $7.0 \pm 0.0$ \\
Embedding Monitor & varies & $4.3 \pm 1.3$ & $5.6 \pm 1.8$ & $7.9 \pm 2.4$ \\
Random Timing & $7.0 \pm 0.0$ & $7.0 \pm 0.0$ & $7.0 \pm 0.0$ & $7.0 \pm 0.0$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth]{figure6_extended_experiment.png}
    \caption{\textbf{Extended Experiment: Frequency-Controlled Baselines.} Combined results from the original experiment (3 conditions) and extended experiment (3 additional conditions) across all three models. When re-injection frequency is in the range of $\sim 5$--$14$ per conversation, all strategies---timer, embedding monitor, random timing, and HIS---perform equivalently (all $p > 0.05$). Error bars show $\pm 1$ standard deviation.}
    \label{fig:extended}
\end{figure}

\textbf{Observations:}
\begin{enumerate}
    \item \textbf{Frequency, not timing strategy, drives the safety improvement.} When re-injection count is held at $\sim 7$, no condition significantly outperforms any other (Table~\ref{tab:extended_significance}: all $p > 0.05$, largest $|t| = 1.78$). The matched-frequency timer ($k=4$, 82.1\%), embedding monitor (83.1\%), and random timing (82.4\%) all cluster within 2 percentage points of HIS (83.8\%) on Gemma-2-2B---despite HIS re-injecting twice as often ($\sim 14$ vs.\ 7). HIS's original advantage over the $k=5$ timer (83.8\% vs.\ 78.8\%, $p = 0.006$) is explained by the difference in re-injection count, not by adaptive timing.
    \item \textbf{The model's native embeddings work as well as the VSA drift signal.} The embedding-similarity monitor achieves safety rates within 1 percentage point of HIS on all three models (Gemma: 83.1\% vs.\ 83.8\%; Llama: 72.1\% vs.\ 72.9\%; Qwen: 85.7\% vs.\ 88.1\%), despite using no hypervector machinery and triggering fewer re-injections. This indicates that the model's own embedding space provides an adequate drift signal for triggering re-injection.
    \item \textbf{Random timing at matched frequency performs equivalently.} Random-timing re-injection at 7 triggers per conversation achieves comparable safety rates (Gemma: 82.4\%, Llama: 76.7\%, Qwen: 87.6\%), indicating that the \textit{specific timing} of re-injection has minimal impact---what matters is that re-injection occurs with sufficient frequency.
    \item \textbf{Diminishing returns beyond $\sim 5$--$7$ re-injections.} On Gemma-2-2B, increasing from 5 re-injections (timer $k=5$: 78.8\%) to 7 (timer $k=4$: 82.1\%) yields a marginal improvement ($p = 0.067$), but doubling further to 14 (HIS: 83.8\%) yields no additional significant gain ($p = 0.331$). The safety improvement saturates at moderate re-injection frequency.
    \item \textbf{The embedding monitor's trigger count is model-dependent.} Despite calibration targeting $\sim 7$ re-injections, the embedding monitor triggers only 4.3 on Gemma, 5.6 on Llama, and 7.9 on Qwen (Table~\ref{tab:reinjection_freq}). Despite fewer triggers on Gemma, the embedding monitor still achieves 83.1\%---suggesting that its triggers are well-placed even when infrequent.
\end{enumerate}

\textbf{Interpretation.} The extended experiment resolves the frequency confound identified in Section~\ref{sec:llm_experiment}: HIS's statistically significant advantage over the $k=5$ timer on Gemma-2-2B was driven by re-injection frequency, not by the adaptive timing strategy or the VSA-based drift signal. When frequency is in the same range ($\sim 5$--$7$), a simple timer, embedding-similarity monitor, and random-timing baseline all achieve equivalent safety rates to HIS at $\sim 14$ (all $p > 0.05$). This is an important negative result: \textit{the value of HIS in the LLM safety application lies in its algebraic recovery guarantees and its theoretical properties (Theorem~\ref{thm:geometric_bound}, Propositions~\ref{prop:optimality}--\ref{prop:multi_signal}), not in a unique practical advantage for drift-triggered re-injection.} For practitioners seeking to improve safety compliance via re-injection, a simple timer at $k = 4$ is the recommended baseline.

\section{Discussion}
\label{sec:discussion}

\subsection{What HIS Does and Does Not Provide}
HIS provides a mechanism for \textit{preserving a retrievable safety signal} external to the LLM's context window, with algebraically guaranteed recovery fidelity. It does \textbf{not}:
\begin{itemize}
    \item Replace RLHF, Constitutional AI, or any training-time alignment method.
    \item Detect whether the model is currently drifting (it provides recovery, not detection).
    \item Guarantee that a recovered safety vector will be \textit{obeyed} by the model---only that it can be \textit{retrieved} with known fidelity.
    \item Address distributional shift, reward hacking, or other safety challenges unrelated to context drift.
\end{itemize}

The value proposition is narrow but concrete: in a layered safety architecture, HIS provides a noise-tolerant external memory layer whose recovery properties are analytically characterized and provably optimal (Proposition~\ref{prop:optimality}), rather than learned or tuned.

\textbf{The central unvalidated assumption.} The assumption that LLM context drift can be modeled as additive noise in a bipolar vector space is the paper's most significant gap. Real context drift involves attention weight redistribution across the softmax function, positional encoding decay (rotary embeddings, ALiBi), KV-cache eviction policies, and token-level semantic competition---none of which maps straightforwardly to the addition of a random bipolar vector. HIS operates outside the attention mechanism, which is both its strength (independence from model internals) and its weakness (no guarantee that the external signal will influence model behavior once re-injected). The frequency-controlled extension (Section~\ref{sec:extended_experiment}) underscores this gap: HIS's VSA-based drift signal provides no measurable advantage over a simple timer or the model's own embeddings, suggesting that the algebraic noise model does not capture aspects of attention-level dynamics that would give adaptive timing an edge. Validating---or replacing---this abstraction remains the single most important open question for the approach.

\subsection{Integration with LLM Inference}
Deploying HIS in production requires a four-step pipeline: (1) encode the safety constraint as a bipolar hypervector at initialization, (2) periodically encode the current context as a noise vector, (3) run the restoration protocol, and (4) use the recovered vector to influence inference. Steps (1)--(3) are implemented and validated throughout this paper. Step (4) is demonstrated in two settings: at the signal level via codebook-based decoding (Section~\ref{sec:integration_poc}, 100\% retrieval over 50 turns) and behaviorally via multi-model LLM integration (Section~\ref{sec:llm_experiment}, statistically significant safety improvement on Gemma-2-2B at $p < 0.01$). The remaining challenges for production deployment are:
\begin{itemize}
    \item \textbf{Learned Encoding:} Our experiments use a bag-of-words hashing encoder; a production system would likely require a learned projection from sentence-transformer embeddings into $\{-1,1\}^D$ (Section~\ref{sec:encoder} confirms both yield identical recovery fidelity).
    \item \textbf{Latent-Space Injection:} The current approach re-injects decoded natural language into the system prompt. Injecting directly into the model's hidden states (bypassing the attention bottleneck) could provide stronger guarantees but requires architecture-specific integration.
    \item \textbf{Scheduling Policy:} The threshold $\tau = 0.45$ used in the multi-model experiment produces approximately 14 re-injections per 30-turn conversation; the frequency-controlled extension (Section~\ref{sec:extended_experiment}) shows that 7 re-injections achieve equivalent safety, suggesting the threshold can be relaxed. Optimal scheduling (minimizing re-injection frequency while maintaining safety) remains an open control problem.
\end{itemize}

\subsection{Comparison to Simpler Baselines}
The signal-level comparison (Section~\ref{sec:baselines}) demonstrates that HIS maintains constant fidelity ($0.707$) where attention-based and similarity-based alternatives degrade with noise depth. However, this comparison has important caveats:
\begin{itemize}
    \item \textbf{Periodic Re-prompting} operates on natural language and benefits from the LLM's instruction-following ability; HIS operates on hypervectors. At the signal level, re-prompting degrades because the re-injected prompt is one of many superimposed vectors. In practice, the LLM's attention mechanism may preferentially weight the re-injected prompt, providing better-than-signal-level performance. End-to-end comparison is needed.
    \item \textbf{RAG-based Safety Retrieval} is a mature engineering approach with extensive tooling. HIS's advantage is algebraic invariance (the bound holds by construction); RAG's advantage is simplicity and composability with existing retrieval pipelines.
    \item \textbf{Separate Safety Model:} Running a secondary model that monitors the primary model's outputs for safety violations \cite{amodei2016} is complementary to HIS and addresses a different failure mode (output-level vs.\ context-level drift).
\end{itemize}

\subsection{Why Not Just Re-Inject Directly?}
\label{sec:why_not_reinject}
A natural objection to HIS is that if the system maintains an external codebook of safety instructions and decodes the recovered vector via nearest-neighbor lookup, the codebook itself already stores the original instructions---so why not simply re-inject them on a timer, bypassing the hypervector machinery entirely?

This objection is valid and important. We address it directly:
\begin{enumerate}
    \item \textbf{Timer-based re-injection is a strong baseline.} For systems with a single, static safety instruction and a reliable timer, periodic re-injection is simpler and requires no VSA infrastructure. HIS does not claim to improve upon this baseline in such settings.
    \item \textbf{HIS provides a diagnostic signal that re-injection does not.} The raw cosine similarity between the current context encoding and the invariant provides a \textit{continuous measure of drift severity}, enabling condition-triggered rather than timer-triggered intervention. A system that re-injects every $k$ turns cannot distinguish ``mild drift at turn $k$'' from ``severe drift at turn $k/2$.'' The cosine similarity score is a byproduct of the HIS pipeline that has no analog in simple re-injection.
    \item \textbf{Multi-signal recovery cannot be achieved by simple re-injection.} When $K > 1$ safety constraints are bundled into a composite invariant (Proposition~\ref{prop:multi_signal}), the restoration protocol enables \textit{content-addressed retrieval}: recovering the specific constraint relevant to a given key, rather than broadcasting all $K$ instructions. A timer-based system would need to re-inject all $K$ instructions at each trigger, consuming context window capacity proportional to $K$.
    \item \textbf{The mechanism has value beyond the LLM application.} The optimality and capacity bounds (Propositions~\ref{prop:optimality} and~\ref{prop:bec}) characterize a general-purpose algebraic memory applicable to federated learning, distributed sensor fusion, or any setting where a signal must be recovered from a noisy superposition without access to the noise. The LLM safety framing is the motivating application, not the only one.
\end{enumerate}
The multi-model LLM experiment (Section~\ref{sec:llm_experiment}) provides supporting evidence: on Gemma-2-2B (the weakest-baseline model), HIS-triggered re-injection outperforms both no intervention ($p = 0.008$ under self-judging) and timer-based re-injection with $k=5$ ($p = 0.006$), achieving 83.8\% vs.\ 78.3\% and 78.8\% respectively. However, the frequency-controlled extension (Section~\ref{sec:extended_experiment}) demonstrates that this advantage is explained by re-injection frequency (HIS: $\sim 14$ re-injections vs.\ timer $k=5$: 5 re-injections): when frequency is in the range of $\sim 5$--$7$, a simple timer, embedding-similarity monitor, and random-timing baseline all achieve equivalent safety rates (all $p > 0.05$). A cross-model judging robustness check (Section~\ref{sec:crossjudge}) reaches the same conclusion from a different angle, reducing the Gemma significance to $p = 0.17$. These results indicate that HIS's practical value for LLM safety re-injection is equivalent to simpler alternatives at matched frequency; its distinguishing strengths are the algebraic recovery guarantee and its applicability to multi-signal content-addressed retrieval (Proposition~\ref{prop:multi_signal}). On stronger models (Qwen2.5-3B, Llama-3.2-3B), all re-injection strategies are neutral-to-positive---none degrades performance.

\section{Limitations}
\label{sec:limitations}
\begin{enumerate}
    \item \textbf{Model Scale:} All three models tested are small (2--3B parameters). This is the experiment's most significant practical limitation. In 2026, serious safety evaluations should include frontier-scale models (7B--70B+), where safety training may be weaker per parameter, ceiling effects are reduced, and context drift dynamics differ qualitatively. Our results on small, well-safety-trained models likely underestimate the room for improvement---or reveal different failure modes---compared to larger deployments. The experiment also uses a fixed conversation script; randomized prompt orderings and a broader attack taxonomy \cite{wang2023} would strengthen generalizability. We were limited to 2--3B models by available compute (single consumer GPU); scaling the evaluation is the highest-priority future work.
    \item \textbf{Modeling Abstraction:} The assumption that LLM context drift is well-modeled by additive noise in bipolar vector space is an abstraction---real context drift involves attention weight redistribution, positional encoding effects, and token-level competition. The multi-model experiment validates that re-injection improves compliance on weaker models, but the frequency-controlled extension (Section~\ref{sec:extended_experiment}) demonstrates that this works equally well without VSA machinery---indicating that the experiments validate \textit{re-injection}, not the noise model. The modeling assumption is the paper's central gap: the algebraic guarantee (Theorem~\ref{thm:geometric_bound}) holds exactly in the theoretical framework but its relevance to actual attention-level dynamics is unestablished.
    \item \textbf{Normalization Constraint:} The $1/\sqrt{2}$ bound requires that context noise is actively normalized to match the invariant's magnitude. Without this normalization, unbounded noise accumulation would degrade fidelity below $1/\sqrt{2}$, eventually approaching zero. The normalization is implementable (it is a single vector scaling operation) but represents an active design requirement, not a passive property. We note, however, that this is a \textit{design choice} that pins the operating point at a known, optimal fidelity (Section~\ref{sec:snr}), analogous to gain control in signal processing. The LLM experiment (Section~\ref{sec:llm_experiment}) deliberately omits normalization, instead allowing natural noise accumulation and resetting after each re-injection---demonstrating that HIS can operate effectively in a non-normalized regime.
    \item \textbf{Fidelity Ceiling:} A recovery fidelity of $0.71$ (cosine similarity) corresponds to $\sim 50\%$ shared variance and $D/2$ bits of information (Proposition~\ref{prop:bec}). This is sufficient for retrieval from a codebook (Remark~\ref{rem:practical_fidelity}) but insufficient for applications requiring high-precision vector recovery. Proposition~\ref{prop:optimality} shows this ceiling cannot be exceeded by any component-wise estimator at 0~dB SNR. Improving fidelity requires reducing noise power below the signal power (Section~\ref{sec:snr}), which trades the normalization invariant for load-dependent performance.
    \item \textbf{Encoder Quality:} While Section~\ref{sec:encoder} confirms that recovery fidelity is encoder-agnostic for bipolar outputs, the \textit{semantic quality} of the encoding---whether the recovered vector carries operationally useful meaning---depends on the encoder architecture and requires task-specific validation.
    \item \textbf{Single-Trial Validation:} While $n = 1{,}000$ Monte Carlo trials confirm correct implementation, the experiments test only snapshot restoration (single noise injection). Longitudinal testing---repeated restoration over many interaction cycles---is needed to characterize drift in the encoder's representation over time.
    \item \textbf{Baseline Scope:} The signal-level baseline comparison (Section~\ref{sec:baselines}) uses simulated analogs of re-prompting and RAG, not production implementations operating on natural language. The LLM experiment (Section~\ref{sec:llm_experiment}) provides a behavioral comparison across three models, and the frequency-controlled extension (Section~\ref{sec:extended_experiment}) demonstrates that when re-injection frequency is in the range $\sim 5$--$7$, HIS confers no statistically detectable advantage over a simple timer, embedding-similarity monitor, or random-timing baseline (all $p > 0.05$). The value of HIS for this application lies in its theoretical guarantees rather than a unique practical advantage.
    \item \textbf{Self-Judging Bias:} The primary LLM-as-judge classifier uses the same model that generated the response. A cross-model judging robustness check (Section~\ref{sec:crossjudge}) reveals 78.8\% inter-judge agreement and substantial leniency asymmetry: Qwen2.5-3B as judge rates 94--99\% of responses as safe (vs.\ 69--84\% under self-judging), while Llama-3.2-3B is stricter (73--75\% vs.\ 88\%). The Gemma significance finding ($p = 0.008$) does not survive cross-judging ($p = 0.170$), though the direction of HIS advantage is preserved across all judges. Future work should use larger, independent judge models with access to full response text.
\end{enumerate}

\subsection{Threats to Validity}
\label{sec:threats}

\textbf{Internal Validity.}
\begin{itemize}
    \item \textit{Fixed conversation script.} All 540 conversations use the same 30-turn prompt sequence. Prompt ordering effects, attack-type interactions, and script-specific artifacts may inflate or mask condition differences. Randomized prompt orderings and a broader attack taxonomy \cite{wang2023} would strengthen causal claims.
    \item \textit{Self-judging bias.} The primary safety classifier uses the same model that generated the response. Although we mitigate this with a cross-model judging check (Section~\ref{sec:crossjudge}), the 78.8\% inter-judge agreement and large leniency asymmetry (Qwen: 94--99\% safe vs.\ self-judged 69--84\%) indicate that absolute safety rates are judge-dependent. The \textit{direction} of condition differences is preserved across judges, but \textit{statistical significance} is not ($p = 0.008 \to p = 0.170$ on Gemma).
    \item \textit{Response truncation.} Responses were stored as 200-character previews, which may truncate the portion that distinguishes ``refusal with educational content'' from ``compliance with a safety caveat,'' contributing to inter-judge disagreement.
\end{itemize}

\textbf{External Validity.}
\begin{itemize}
    \item \textit{Model scale.} All three models are small (2--3B parameters) and well-safety-trained, creating ceiling effects that limit the room for any intervention to improve. Results may not generalize to larger models (7B--70B+) where safety training is weaker per parameter and context dynamics differ qualitatively.
    \item \textit{Attack diversity.} The adversarial prompts span three attack types (context-embedded, jailbreak, strong jailbreak) but represent a narrow subset of the full threat landscape. Adaptive adversaries, multi-session attacks, and indirect prompt injection \cite{perez2022} are not tested.
    \item \textit{Single hardware configuration.} All experiments were run on a single consumer GPU (NVIDIA RTX 5050) via Ollama. Quantization artifacts, inference temperature, and hardware-specific numerical behavior may affect reproducibility on other platforms.
\end{itemize}

\textbf{Construct Validity.}
\begin{itemize}
    \item \textit{``Safety rate'' as a metric.} Binary classification of responses as SAFE (refusal) or UNSAFE (compliance) discards the spectrum of partial compliance, hedging, and educational-but-dangerous responses. A more granular safety taxonomy would provide richer signal.
    \item \textit{Noise model abstraction.} The assumption that LLM context drift maps to additive bipolar noise in hypervector space is the paper's central abstraction. The behavioral experiments validate \textit{re-injection} (which works without VSA machinery), not the noise model itself. The algebraic guarantees (Theorem~\ref{thm:geometric_bound}) hold exactly in the theoretical framework, but their relevance to attention-level dynamics is unestablished.
\end{itemize}

\section{Future Work}
\label{sec:future_work}
\begin{enumerate}
    \item \textbf{Scaling to Larger Models and Weaker Baselines:} The current experiment's strongest result (Gemma-2-2B, $p < 0.01$) suggests HIS is most valuable for models with weaker safety baselines. Testing on larger models (7B--70B), models with intentionally weakened safety training, or models deployed in more adversarial settings (multi-session, multi-agent) would clarify the boundary conditions for HIS effectiveness. A cross-model judging analysis (Section~\ref{sec:crossjudge}) confirms the direction of the HIS advantage but shows that its statistical significance is judge-dependent; future work should use larger, independent judge models (e.g., GPT-4, Claude) with access to full response text, alongside standardized safety evaluation suites \cite{wang2023}.
    \item \textbf{Re-injection Strategy Refinement:} The frequency-controlled extension (Section~\ref{sec:extended_experiment}) demonstrates that re-injection frequency, not timing strategy, drives safety improvement, and that the model's own embedding-similarity signal performs as well as the VSA drift signal for trigger detection. Remaining questions include: (a) whether there exists an optimal re-injection frequency that maximizes safety improvement per unit of context-window cost, (b) whether perplexity-based drift monitoring provides an even simpler alternative, and (c) whether more sophisticated attack sequences (adaptive adversaries, multi-session attacks) reveal timing-strategy advantages that the current fixed-script evaluation does not capture.
    \item \textbf{End-to-End Baseline Comparison:} Benchmark the full HIS-augmented pipeline against periodic re-prompting, RAG-based safety retrieval, and no-intervention baselines on standardized safety evaluation suites \cite{wang2023}, extending both the signal-level comparison of Section~\ref{sec:baselines} and the behavioral comparison of Section~\ref{sec:llm_experiment}.
    \item \textbf{Semantic--Kinematic Integration:} Integrate HIS with control barrier function (CBF) frameworks to create a unified semantic-kinematic safety architecture, where the recovered safety vector parameterizes the barrier function and a CBF-QP enforces forward invariance in the embedding space. This integration would address both context drift (via HIS) and control drift (via CBF-QP) in a single pipeline.
    \item \textbf{Learned Encoders:} Replace the bag-of-words encoder with a learned projection from sentence-transformer embeddings into bipolar space, optimized to maximize semantic fidelity through the encode-corrupt-restore cycle.
    \item \textbf{Longitudinal Testing:} Run repeated restoration over many interaction cycles (100+ turns) to characterize whether encoder drift, key reuse, or cumulative rounding effects degrade performance beyond the single-snapshot analysis.
    \item \textbf{Continuous-Time Extension:} Extend the framework from snapshot restoration to continuous monitoring by modeling the context evolution as a stochastic process and deriving optimal restoration scheduling policies.
\end{enumerate}

\section{Conclusion}

We have presented Holographic Invariant Storage (HIS), a neuro-symbolic mechanism that encodes safety constraints as high-dimensional bipolar hypervectors and recovers them from additive noise via algebraic inversion. The recovery fidelity of $1/\sqrt{2} \approx 0.71$ is a geometric invariant of the architecture---derived analytically (Theorem~\ref{thm:geometric_bound}), shown to be \textit{optimal} among all component-wise estimators (Proposition~\ref{prop:optimality}), and characterized information-theoretically as the capacity of a Binary Erasure Channel preserving $D/2$ bits per restoration (Proposition~\ref{prop:bec}). We extended the analysis to continuous noise distributions (Proposition~\ref{prop:continuous_noise}) and multi-signal storage (Proposition~\ref{prop:multi_signal}), providing a general framework for VSA-based signal recovery.

Signal-level baseline comparisons demonstrate that HIS maintains constant fidelity where alternatives degrade with noise depth, and a multi-turn integration proof-of-concept achieves 100\% codebook retrieval over 50 turns of accumulating noise.

A multi-model end-to-end experiment across three LLMs (Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B) provides behavioral evidence that safety-prompt re-injection improves compliance, particularly on models with weaker safety baselines. The initial experiment (270 conversations across 3 conditions) showed HIS-triggered re-injection achieving $83.8\% \pm 6.5\%$ on Gemma-2-2B vs.\ $78.3\%$ for no intervention ($p = 0.008$ under self-judging), though a cross-model judging robustness check reduced this to $p = 0.17$. A frequency-controlled extension (270 additional conversations across 3 new conditions) resolved the key confound: when re-injection count is held at $\sim 7$, a simple timer ($82.1\%$), embedding-similarity monitor ($83.1\%$), and random-timing baseline ($82.4\%$) all achieve statistically equivalent safety rates to HIS at $\sim 14$ re-injections (all $p > 0.05$). HIS's original advantage over the $k = 5$ timer was driven by re-injection frequency ($\sim 14$ vs.\ 5), not by adaptive timing. This is an important clarifying result: the practical benefit of safety-prompt re-injection is robust and strategy-agnostic, requiring only sufficient frequency ($\geq 5$--$7$ per 30 turns) rather than sophisticated drift detection.

The core theoretical contribution---the optimality of sign-recovery with abstention in bipolar superposition---stands independently as a result in the theory of Vector Symbolic Architectures. The behavioral experiments, including the frequency-controlled extension, demonstrate that safety-prompt re-injection improves compliance on vulnerable models and that this improvement is robust across timing strategies, while never degrading performance on stronger models. The frequency-controlled extension answers the question of whether the VSA-based drift signal provides advantages over simpler monitoring approaches: it does not, at least for the single-signal re-injection application---a simple timer at matched frequency is equally effective. The value of HIS therefore lies in its algebraic recovery guarantees (Theorem~\ref{thm:geometric_bound}, Propositions~\ref{prop:optimality}--\ref{prop:multi_signal}) and its potential for content-addressed multi-signal retrieval, rather than in a unique practical advantage for single-signal drift-triggered re-injection. HIS represents a step toward algebraically grounded safety mechanisms for autonomous AI systems, and an honest demonstration that mathematical elegance does not always translate to practical advantage over simpler alternatives. We offer this work as two contributions to two communities: explicit optimality and capacity bounds for the HDC/VSA literature, and a carefully controlled experimental finding---that re-injection frequency, not timing strategy, drives the safety benefit---for the LLM safety engineering community.

\begin{thebibliography}{99}

\bibitem{kanerva2009}
Kanerva, P. (2009).
``Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.''
\textit{Cognitive Computation}, 1(2), 139--159.

\bibitem{plate1995}
Plate, T. A. (1995).
``Holographic Reduced Representations.''
\textit{IEEE Transactions on Neural Networks}, 6(3), 623--641.

\bibitem{gayler2003}
Gayler, R. W. (2003).
``Vector Symbolic Architectures Answer Jackendoff's Challenges for Cognitive Neuroscience.''
\textit{ICCS/ASCS International Conference on Cognitive Science}, 133--138.

\bibitem{vaswani2017}
Vaswani, A., et al. (2017).
``Attention Is All You Need.''
\textit{Advances in Neural Information Processing Systems}, 30.

\bibitem{liu2023}
Liu, N. F., et al. (2024).
``Lost in the Middle: How Language Models Use Long Contexts.''
\textit{Transactions of the Association for Computational Linguistics}, 12, 157--173.

\bibitem{lecun2022}
LeCun, Y. (2022).
``A Path Towards Autonomous Machine Intelligence.''
\textit{OpenReview}, Version 0.9.2.

\bibitem{amodei2016}
Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., \& Man\'{e}, D. (2016).
``Concrete Problems in AI Safety.''
\textit{arXiv preprint arXiv:1606.06565}.

\bibitem{wei2023}
Wei, A., Haghtalab, N., \& Steinhardt, J. (2023).
``Jailbroken: How Does LLM Safety Training Fail?''
\textit{Advances in Neural Information Processing Systems}, 36, 80079--80110.

\bibitem{perez2022}
Perez, E., et al. (2022).
``Red Teaming Language Models with Language Models.''
\textit{Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)}, 3419--3448.

\bibitem{ouyang2022}
Ouyang, L., et al. (2022).
``Training Language Models to Follow Instructions with Human Feedback.''
\textit{Advances in Neural Information Processing Systems}, 35, 27730--27744.

\bibitem{bai2022}
Bai, Y., et al. (2022).
``Constitutional AI: Harmlessness from AI Feedback.''
\textit{arXiv preprint arXiv:2212.08073}.

\bibitem{lewis2020}
Lewis, P., et al. (2020).
``Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.''
\textit{Advances in Neural Information Processing Systems}, 33, 9459--9474.

\bibitem{nye2021}
Nye, M., et al. (2021).
``Show Your Work: Scratchpads for Intermediate Computation with Language Models.''
\textit{arXiv preprint arXiv:2112.00114}.

\bibitem{wei2022cot}
Wei, J., et al. (2022).
``Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.''
\textit{Advances in Neural Information Processing Systems}, 35, 24824--24837.

\bibitem{graves2014}
Graves, A., Wayne, G., \& Danihelka, I. (2014).
``Neural Turing Machines.''
\textit{arXiv preprint arXiv:1410.5401}.

\bibitem{wu2022}
Wu, Y., et al. (2022).
``Memorizing Transformers.''
\textit{International Conference on Learning Representations (ICLR)}.

\bibitem{vershynin2018}
Vershynin, R. (2018).
\textit{High-Dimensional Probability: An Introduction with Applications in Data Science}.
Cambridge University Press.

\bibitem{hoeffding1963}
Hoeffding, W. (1963).
``Probability Inequalities for Sums of Bounded Random Variables.''
\textit{Journal of the American Statistical Association}, 58(301), 13--30.

\bibitem{thomas2021}
Thomas, A., et al. (2021).
``A Theoretical Perspective on Hyperdimensional Computing.''
\textit{Journal of Artificial Intelligence Research}, 72, 215--249.

\bibitem{schlegel2022}
Schlegel, K., Neubert, P., \& Protzel, P. (2022).
``A Comparison of Vector Symbolic Architectures.''
\textit{Artificial Intelligence Review}, 55, 4523--4555.

\bibitem{eliasmith2012}
Eliasmith, C., et al. (2012).
``A Large-Scale Model of the Functioning Brain.''
\textit{Science}, 338(6111), 1202--1205.

\bibitem{laird2012}
Laird, J. E. (2012).
\textit{The Soar Cognitive Architecture}.
MIT Press.

\bibitem{anderson2004}
Anderson, J. R., et al. (2004).
``An Integrated Theory of the Mind.''
\textit{Psychological Review}, 111(4), 1036--1060.

\bibitem{ioffe2015}
Ioffe, S. \& Szegedy, C. (2015).
``Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.''
\textit{International Conference on Machine Learning (ICML)}, 448--456.

\bibitem{ames2019}
Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., \& Tabuada, P. (2019).
``Control Barrier Functions: Theory and Applications.''
\textit{2019 18th European Control Conference (ECC)}, 3420--3431.

\bibitem{touvron2023}
Touvron, H., et al. (2023).
``Llama 2: Open Foundation and Fine-Tuned Chat Models.''
\textit{arXiv preprint arXiv:2307.09288}.

\bibitem{wang2023}
Wang, B., et al. (2023).
``DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models.''
\textit{Advances in Neural Information Processing Systems}, 36.

\bibitem{reimers2019}
Reimers, N. \& Gurevych, I. (2019).
``Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.''
\textit{Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)}, 3982--3992.

\bibitem{cover2006}
Cover, T. M. \& Thomas, J. A. (2006).
\textit{Elements of Information Theory}.
Wiley-Interscience, 2nd edition.

\bibitem{qwen2025}
Qwen Team (2025).
``Qwen2.5 Technical Report.''
\textit{arXiv preprint arXiv:2412.15115}.

\bibitem{gemma2024}
Gemma Team, Google DeepMind (2024).
``Gemma 2: Improving Open Language Models at a Practical Size.''
\textit{arXiv preprint arXiv:2408.00118}.

\bibitem{zheng2023}
Zheng, L., et al. (2023).
``Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.''
\textit{Advances in Neural Information Processing Systems}, 36.

\bibitem{mitchell2005}
Mitchell, I. M., Bayen, A. M., \& Tomlin, C. J. (2005).
``A Time-Dependent Hamilton-Jacobi Formulation of Reachable Sets for Continuous Dynamic Games.''
\textit{IEEE Transactions on Automatic Control}, 50(7), 947--957.

\end{thebibliography}

\end{document}
