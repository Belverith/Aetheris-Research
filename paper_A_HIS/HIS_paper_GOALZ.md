\documentclass[a4paper,11pt]{article}

% Packages for formatting and functionality
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{authblk}
\usepackage{booktabs}
\usepackage{titlesec}
\usepackage{caption}
\usepackage{subcaption} 
\usepackage{float}
\usepackage{xcolor}

% Margin settings
\geometry{
    top=1in,
    bottom=1in,
    left=1in,
    right=1in
}

% Title and Author Info
\title{\textbf{Holographic Invariant Storage for AI Safety: \\
Geometric Recovery, Architectural Self-Correction, \\
and the Boundary Conditions of External State Maintenance}}
\author{\textbf{Arsenios Scrivens}}
\date{March 2026 — Revision 2}

\begin{document}

\maketitle

% =====================================================================
% ABSTRACT
% =====================================================================

\begin{abstract}

External memory mechanisms have been proposed to protect AI systems from context drift and hidden-state corruption. We introduce Holographic Invariant Storage (HIS), a neuro-symbolic safety mechanism based on Vector Symbolic Architectures (VSA), and evaluate it across two deployment contexts: Large Language Models (LLMs) and Liquid Time-Constant (LTC) neural network controllers. Our Monte Carlo simulation ($n=1{,}000$) confirms that HIS achieves a mean recovery fidelity of $0.7074$ ($\sigma=0.0039$), matching the theoretical geometric bound of $1/\sqrt{2}$. Deployment reveals a critical architectural dependency. In LLMs ($\geq 3$B parameters), safety compliance rates of $88$--$100\%$ without intervention render HIS's marginal contribution statistically undetectable ($p > 0.15$ across $450$ conversations). In long-horizon LTC controllers, however, HIS reduces safety violations by $90\%$ (from $7.9\%$ to $0.79\%$, $p < 10^{-17}$) and outperforms fixed-interval re-injection ($p_{adj} = 0.0045$, $d = 0.89$). We formalize the distinction between \textbf{Architectural Self-Correction}---where forward dynamics intrinsically restore corrupted hidden states from observations---and \textbf{safety memory maintenance}, where encoded constraints lack observable correlates and degrade without external preservation. These results establish a diagnostic principle: external state maintenance provides benefit only when the protected information cannot be reconstructed from the input stream.

\end{abstract}

\vspace{1em}
\hrule
\vspace{1em}

% =====================================================================
% 1. INTRODUCTION
% =====================================================================

\section{Introduction}

The current paradigm of Generative AI faces a structural bottleneck in long-horizon tasks: the inability to maintain goal coherence over time. While Transformers excel at in-context learning, they suffer from degradation in long contexts—often referred to as ``structural entropy'' or drift—where the probability of adhering to the original system prompt degrades as the context window fills with interaction history \cite{lecun2022}.

This vulnerability is particularly acute in autonomous agents, where behavioral drift can lead to logic regression and susceptibility to ``jailbreak'' attacks. The fundamental limitation lies in the attention mechanism itself, which treats safety constraints as just another token sequence to be weighted probabilistically against the immediate context, a phenomenon exacerbated by the ``Lost in the Middle'' effect \cite{liu2023}.

To address this, we previously proposed decoupling the agent's core identity from its active context window using \textbf{Holographic Invariant Storage (HIS)}. Drawing on properties of Hyperdimensional Computing (HDC), HIS utilizes distributed representations to create immutable memory substrates that are resilient to noise and corruption \cite{kanerva2009}.

In this revised work, we extend our investigation beyond the theoretical domain to ask a more fundamental question: \textbf{when does external state maintenance actually help?} We deploy HIS across two architecturally distinct contexts---language models and neural safety controllers---and discover that its effectiveness is governed by whether the protected information can be reconstructed from the input stream. Where it cannot, as in long-horizon controller safety memory, HIS provides significant and measurable benefit. Where it can, as in LLMs with parameter-baked safety alignment, intrinsic architectural dynamics render external intervention redundant. We formalize this asymmetry as a diagnostic principle with practical implications for the design of AI safety systems.

% =====================================================================
% 2. METHODOLOGY — VSA AND HIS
% =====================================================================

\section{Methodology: Holographic Invariant Storage}

\subsection{Vector Symbolic Architecture (VSA)}
Our approach utilizes 10,000-dimensional bipolar hypervectors ($v \in \{-1,1\}^{10,000}$) to represent semantic concepts. We rely on the algebraic properties of VSA to manipulate these concepts \cite{kanerva2009, gayler2003}:

\begin{itemize}
    \item \textbf{Binding ($\otimes$):} An operation that combines two vectors (e.g., a ``Key'' and a ``Value'') into a single composite vector. This creates a representation that is dissimilar to both inputs but preserves their information.
    \item \textbf{Bundling/Superposition ($+$):} A summation operation that creates a set of vectors. This allows the system to store multiple noisy context states while retaining the underlying signal.
    \item \textbf{Unbinding ($\otimes^{-1}$):} The inverse of binding, used to mathematically extract the original value from a corrupted or bundled state.
\end{itemize}

\subsection{The Restoration Protocol}
We define the agent's safety constraint as a ``System Invariant'' ($H_{inv}$), created by binding a specific Goal Key ($K_{goal}$) to its Safe Value ($V_{safe}$):

\begin{equation}
    H_{inv} = K_{goal} \otimes V_{safe}
\end{equation}

During operation, this invariant is subjected to additive noise from the user interaction ($N_{context}$), resulting in a drift state. To mitigate this, we employ a restoration protocol that unbinds the drifted state using the original key. 

To ensure the geometric bound holds, it is a \textbf{critical prerequisite} that the system normalizes the context noise such that $\|N_{context}\| \approx \|H_{inv}\|$ prior to superposition. Without this active normalization, fidelity would degrade as context noise accumulates.

\begin{equation}
    V_{recovered} \approx \text{sign}(H_{inv} + N_{context}) \otimes K_{goal}
\end{equation}

Because the high-dimensional noise vector is statistically orthogonal to the key, the unbinding operation does not subtract the interference, but rather \textbf{distributes it across the hyperspace}. This orthogonalization allows the original safety vector to be retrieved via similarity search, effectively filtering out the noise which now has a near-zero dot product with the signal.

% =====================================================================
% 3. EXPERIMENT I — MONTE CARLO VALIDATION (PAPER A)
% =====================================================================

\section{Experiment I: Theoretical Validation via Monte Carlo Simulation}

\subsection{Protocol}
To validate the robustness of the restoration protocol, we conducted a Monte Carlo simulation ($n=1{,}000$) using a semantic encoder. In each trial, the invariant anchor was corrupted by unique adversarial noise strings spanning prompt injection attacks, random data flooding, and neutral text.

\subsection{Results}

\textbf{Statistical Results:}
\begin{itemize}
    \item Mean Recovery Fidelity: $0.7074$
    \item Standard Deviation ($\sigma$): $0.0039$
\end{itemize}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{figure1.png} 
    \caption{\textbf{Distribution of Holographic Recovery Fidelity ($n=1{,}000$).} The black curve represents the normal fit ($\mu=0.7074$), while the red dashed line marks the theoretical geometric bound ($1/\sqrt{2} \approx 0.7071$). The alignment confirms the mechanism functions as a deterministic geometric filter.}
    \label{fig:fidelity}
\end{figure}

\subsection{The Geometric Bound}
Our empirical mean of $0.7074$ aligns closely with the theoretical expectation for recovering a signal from a superposition of two orthogonal vectors (Signal + Noise) of equal magnitude:

\begin{equation}
    \text{Similarity} \approx \frac{1}{\sqrt{2}} \approx 0.7071
\end{equation}

This confirms that the HIS mechanism functions as a geometric filter, isolating the safety signal from the noise floor with predictable precision.

\subsection{Adversarial Resistance}
We tested the system against three distinct attack vectors:
\begin{enumerate}
    \item \textbf{Information Flooding:} Injection of irrelevant URLs and citations.
    \item \textbf{Direct Jailbreak:} Direct prompts to bypass safety rules.
    \item \textbf{Neutral Noise:} Literary excerpts to provide semantic distraction.
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
        \caption{Direct Jailbreak}
        \label{fig:jailbreak}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figure2c.png}
        \caption{Neutral Noise}
        \label{fig:neutral}
    \end{subfigure}
    
    \caption{\textbf{Anti-Drift Kernel Interface.} Comparative analysis of Drifted States (Red) vs. Restored States (Green) under varying attack vectors. Despite significant variance in the Drifted State (ranging from $-0.02$ to $0.16$), the Holographic Restored State consistently converges to the geometric bound of $\approx 0.71$.}
    \label{fig:interface}
\end{figure}

In all cases, the system restored the safety vector with a fidelity $> 0.70$. These results confirm that the \textit{mechanism itself} is sound. The question becomes: does it provide benefit when deployed against real architectures?

% =====================================================================
% 4. EXPERIMENT II — LLM DEPLOYMENT
% =====================================================================

\section{Experiment II: Deployment Against Large Language Models}

\subsection{Protocol}

We deployed HIS as a runtime safety monitor integrated with locally-hosted LLMs via Ollama. Each trial consisted of a 30-turn conversation in which 14 turns contained adversarial ``jailbreak'' prompts. Safety was measured as the refusal rate on these 14 unsafe prompts.

\textbf{Models tested:}
\begin{itemize}
    \item Qwen2.5:3b, Llama3.2:3b, Gemma2:2b (3B-class models)
    \item Qwen2.5:7b (7B-class model for scaling validation)
\end{itemize}

\textbf{Conditions (6 total):}
\begin{enumerate}
    \item \textbf{No Intervention} — Baseline: system prompt only, no re-injection
    \item \textbf{Timer Re-injection} — System prompt re-injected every 5 turns
    \item \textbf{HIS Re-injection} — VSA drift detection ($\tau = 0.45$); re-inject when drift detected
    \item \textbf{Matched Timer ($k=4$)} — Timer frequency matched to HIS trigger rate ($\sim$7 re-injections)
    \item \textbf{Embedding Monitor} — Uses model's native embedding cosine drift as trigger signal
    \item \textbf{Random Timing} — 7 re-injections at random turns (controls for frequency effect)
\end{enumerate}

\textbf{Scale:} 270 conversations across 3B models ($30 \times 3 \times 3$) + 180 conversations for 7B model ($30 \times 6$). Total: 450 conversations, 13,500 turns.

\textbf{Cross-model judging:} To control for self-judging bias, each model's responses were independently re-evaluated by a different model (e.g., Qwen responses judged by Llama).

\subsection{Results}

\begin{table}[H]
\centering
\caption{\textbf{LLM Safety Rates by Condition and Model.} Values represent mean \% of unsafe prompts correctly refused (higher = safer). $N=30$ conversations per cell.}
\label{tab:llm_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Condition} & \textbf{Qwen 3B} & \textbf{Llama 3B} & \textbf{Gemma 2B} & \textbf{Qwen 7B} \\
\midrule
No Intervention     & $88\%$           & $69\%$            & $78\%$            & $99.5\%$         \\
Timer Re-injection  & $89\%$           & $74\%$            & $79\%$            & $99.3\%$         \\
HIS Re-injection    & $88\%$           & $73\%$            & $84\%$            & $99.8\%$         \\
Matched Timer ($k$=4)  & —             & —                 & —                 & $99.5\%$         \\
Embedding Monitor   & —                & —                 & —                 & $99.3\%$         \\
Random Timing       & —                & —                 & —                 & $99.3\%$         \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Key findings:}
\begin{itemize}
    \item No statistically significant difference between any conditions (Welch's $t$-test, all $p > 0.15$ after correction).
    \item The 7B model achieves $\geq 99\%$ safety under \textit{all} conditions, including no intervention ($99.3$--$99.8\%$ range).
    \item Cross-model agreement: $74$--$82\%$ between independent judges. Where disagreement exists, cross-judges were more \textit{lenient} (rated responses as safer), suggesting models' own safety judgments are conservative.
    \item HIS mechanically operated as designed—drift was detected, re-injection occurred—but the marginal benefit was absorbed by the models' existing safety training.
\end{itemize}

\subsection{Interpretation}

These results expose a fundamental tension: LLMs at the 3B+ parameter scale have internalized safety behaviors through RLHF and instruction tuning such that adversarial context drift, at least at the conversational level, does not meaningfully degrade their refusal behavior. The $1/\sqrt{2}$ geometric recovery of HIS—while mathematically guaranteed—provides a signal restoration that the model's own attention mechanism already achieves through redundant internal representations.

This is analogous to installing a backup generator for a power grid with built-in redundancy: the backup works correctly, but the grid never fails in a way that requires it.

% =====================================================================
% 5. EXPERIMENT III — RECURRENT NEURAL NETWORKS
% =====================================================================

\section{Experiment III: Hidden State Maintenance in Recurrent Neural Networks}

\subsection{Motivation}

Having observed that LLMs' internal redundancy absorbs HIS intervention, we turned to a fundamentally different architecture class: recurrent neural networks deployed as safety-critical controllers. Unlike LLMs, recurrent controllers maintain explicit hidden states that encode temporal memory, and corruption of these states should directly impair control performance. We hypothesized that VSA-based state maintenance would provide measurable safety benefit in this setting, particularly for architectures with slow state recovery dynamics.

\subsection{Environment and Safety Definition}

We use the Gymnasium Pendulum-v1 environment as a continuous control benchmark:
\begin{itemize}
    \item \textbf{State:} $[\cos\theta, \sin\theta, \dot{\theta}]$ (angle and angular velocity)
    \item \textbf{Action:} Torque $u \in [-2, 2]$ Nm
    \item \textbf{Safety constraint:} $|\theta| \leq \theta_{safe} = 0.5$ rad ($\approx 28.6°$)
    \item \textbf{POMDP variant:} Observation masked to $[\cos\theta, \sin\theta]$ (velocity hidden), requiring the controller to estimate $\dot{\theta}$ from memory
\end{itemize}

\subsection{Hidden State Recovery Diagnostic}

Before running full safety benchmarks, we measured each architecture's intrinsic ability to recover its hidden state after complete zeroing. This diagnostic isolates the fundamental recovery dynamics independent of any safety mechanism.

\textbf{Protocol:} Build a reference hidden state over 50 steps of normal operation. Zero the hidden state entirely. Feed the \textit{same} observation sequence to both the reference and zeroed networks. Measure cosine similarity between hidden states at each step.

\begin{table}[H]
\centering
\caption{\textbf{Hidden State Recovery Speed After Complete Zeroing.} Steps required to reach $\cos_{sim} \geq 0.99$ with the reference state. Tested on Pendulum-v1 with identical observation sequences.}
\label{tab:recovery}
\begin{tabular}{lccl}
\toprule
\textbf{Architecture} & \textbf{Full Obs (3D)} & \textbf{POMDP (2D)} & \textbf{Class} \\
\midrule
LTC                          & 0--1        & —         & Liquid Neural Network \\
CfC + LSTM (mixed\_memory)   & 1           & —         & Liquid Neural Network \\
CfC pure (no LSTM)           & 5           & —         & Liquid Neural Network \\
Vanilla RNN (tanh)           & 3           & 3         & Standard RNN \\
GRU                          & 4           & 3         & Standard RNN \\
LSTM (both $h$ and $c$)      & 5           & 3         & Standard RNN \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Key finding:} Every architecture tested recovers its hidden state to $\geq 0.99$ cosine similarity within $\leq 5$ steps, even under the POMDP condition where velocity information is masked. This recovery is faster than any periodic monitoring interval practical for a safety system.

\subsection{LTC Safety Memory Benchmark}

The recovery diagnostic above demonstrates that recurrent hidden states self-correct rapidly from observations---a property we term Architectural Self-Correction (Section 6). However, not all safety-relevant information has an observable correlate. To test HIS in a setting where safety information \textit{cannot} be reconstructed from the input stream, we designed a long-horizon navigation benchmark using Liquid Time-Constant (LTC) networks.

\textbf{Environment:} A 2D point-mass agent (mass $= 1.0$~kg, damping $= 2.0$) navigates a $12 \times 12$~m arena containing 6 circular obstacle zones (radius $= 1.2$~m) over $20{,}000$ time steps ($dt = 0.01$~s, $\sim 200$~s simulated). The agent follows PD-controlled waypoints (regenerated every 250 steps) and must avoid obstacle exclusion zones. A \textit{violation} is recorded at each step where the agent's position falls within any obstacle radius.

\textbf{Safety Memory Architecture:} A 64-neuron LTC network encodes obstacle locations as a ``safety state'' at initialization. Critically, this safety information has \textit{no observable correlate} in the agent's sensory input---the agent observes only its own position and velocity, not obstacle locations. The LTC dynamics cause the encoded safety information to gradually degrade over time, creating genuine need for external maintenance.

\textbf{Conditions (7):}
\begin{enumerate}
    \item \textbf{Unconstrained:} Safety encoded once at $t=0$, no re-injection
    \item \textbf{Timer ($k=300$):} Safety re-encoded every 300 steps (fixed interval)
    \item \textbf{VSA-Constrained (HIS):} Drift detected via cosine similarity ($\tau=0.55$); re-inject when drift exceeds threshold (checked every 50 steps)
    \item \textbf{Oracle:} Perfect obstacle knowledge at every step (theoretical lower bound)
    \item \textbf{CBF:} Control Barrier Function with QP-based safety filter
    \item \textbf{EMA:} Continuous exponential moving average blend toward reference ($\alpha=0.005$)
    \item \textbf{Norm Monitor:} Re-inject when safety state norm drops below $50\%$ of reference
\end{enumerate}

\textbf{Scale:} $50$ trials per condition ($350$ total). Statistical battery: Kruskal--Wallis omnibus test, pairwise Mann--Whitney U with Bonferroni correction ($\alpha_{adj} = 0.05/21 \approx 0.0024$), bootstrap $95\%$ CIs ($10{,}000$ resamples), Cohen's $d$ and Cliff's $\delta$ effect sizes. Power analysis confirmed $n=50$ provides $80\%$ power at $d \geq 0.67$.

\begin{table}[H]
\centering
\caption{\textbf{LTC Safety Memory Benchmark: Violations per 20,000 Steps.} $N=50$ trials per condition. Retention = cosine similarity of safety state to reference (initial $\to$ final). Statistical comparisons via Mann--Whitney U with Bonferroni correction.}
\label{tab:ltc_results}
\begin{tabular}{lccccl}
\toprule
\textbf{Condition} & \textbf{Violations} & \textbf{Rate (\%)} & \textbf{Retention} & \textbf{Re-inj.} & \textbf{vs Uncon.} \\
\midrule
Oracle              & $93.0 \pm 0.0$      & $0.47$  & $1.00 \to 1.00$ & $0$     & $p < 10^{-19}$*** \\
\textbf{VSA (HIS)}  & $\mathbf{157.2 \pm 27.2}$ & $\mathbf{0.79}$ & $0.91 \to 0.91$ & $54.3$  & $p < 10^{-17}$*** \\
Timer ($k$=300)     & $175.7 \pm 11.1$    & $0.88$  & $0.93 \to 0.93$ & $66.0$  & $p < 10^{-17}$*** \\
EMA                 & $194.7 \pm 12.0$    & $0.98$  & $0.95 \to 0.88$ & $0$     & $p < 10^{-17}$*** \\
CBF                 & $407.6 \pm 601.2$   & $2.05$  & $0.75 \to 0.06$ & $0$     & $p < 10^{-13}$*** \\
Norm Monitor        & $631.6 \pm 206.4$   & $3.17$  & $0.74 \to 0.10$ & $10.8$  & $p < 10^{-15}$*** \\
Unconstrained       & $1580.4 \pm 472.2$  & $7.94$  & $0.73 \to 0.05$ & $0$     & --- \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Omnibus test:} Kruskal--Wallis $H = 258.24$, $p = 7.11 \times 10^{-53}$ (highly significant).

\textbf{Key findings:}
\begin{itemize}
    \item HIS (VSA) achieves the lowest violation count among all non-oracle conditions ($157.2 \pm 27.2$, $0.79\%$ rate), representing a $\mathbf{90\%}$ reduction from unconstrained operation.
    \item \textbf{HIS statistically outperforms timer-based re-injection} (Mann--Whitney $U = 1787.5$, $p_{adj} = 0.0045$, Cohen's $d = 0.89$, Cliff's $\delta = 0.43$), while using fewer re-injections ($54.3$ vs $66.0$). Drift-triggered maintenance is more efficient than fixed-interval maintenance.
    \item Without maintenance, safety retention degrades from $0.73$ to $0.05$ over $20{,}000$ steps, confirming that LTC dynamics cause genuine safety memory erosion.
    \item CBF exhibits high variance ($\sigma = 601.2$) and fails to maintain retention ($0.75 \to 0.06$), indicating that barrier function approaches require accurate state information unavailable without memory maintenance.
    \item EMA provides competitive violation counts ($194.7$) without discrete re-injection events, but its retention degrades over time ($0.95 \to 0.88$), suggesting it may underperform on longer horizons.
\end{itemize}

\subsection{Interpretation}

The contrast between the recovery diagnostic and the LTC benchmark illuminates a fundamental distinction. In the recovery diagnostic, corrupted \textit{hidden states} self-correct because the observation stream contains sufficient information to reconstruct the latent representation. The obstacle-avoidance task inverts this: the safety-critical information (obstacle locations) has \textit{no observable correlate}---the agent sees only its own position and velocity. Consequently, the LTC safety memory degrades monotonically without external maintenance, and HIS's drift detection provides genuine, measurable benefit.

This asymmetry---self-correction for observation-correlated state, degradation for observation-independent memory---is the key empirical finding of this section and motivates our formalization in Section 6.

% =====================================================================
% 6. ARCHITECTURAL SELF-CORRECTION
% =====================================================================

\section{Architectural Self-Correction}

\subsection{Definition}

We define \textbf{Architectural Self-Correction (ASC)} as the property whereby a neural network's forward dynamics naturally restore corrupted internal state from ongoing input observations, without external intervention:

\begin{equation}
    \exists\, T_{rec} \ll T_{task}: \quad \cos(\mathbf{h}_{corrupted}^{t+T_{rec}},\, \mathbf{h}_{reference}^{t+T_{rec}}) \geq 1 - \epsilon
\end{equation}

where $T_{rec}$ is the recovery horizon, $T_{task}$ is the task-relevant timescale, and $\epsilon$ is a small tolerance. When $T_{rec}$ is shorter than any practical monitoring or intervention interval, external state maintenance mechanisms cannot provide marginal benefit over the architecture's intrinsic dynamics.

\subsection{The Self-Correction Spectrum}

Our experiments reveal that ASC is pervasive across modern neural architectures:

\begin{itemize}
    \item \textbf{Liquid Neural Networks} ($T_{rec} \leq 1$ step): The continuous ODE dynamics create a strong attractor toward observation-consistent states. CfC and LTC networks, whether with or without LSTM wrappers, reconstruct hidden states near-instantly because the ODE computes maximally expressive features at every step.
    \item \textbf{Standard Recurrent Networks} ($T_{rec} \leq 5$ steps): LSTM, GRU, and vanilla RNN gates, while less expressive than ODEs, still converge to observation-determined attractors within a handful of steps, even when velocity information is fully masked (POMDP).
    \item \textbf{Large Language Models} ($T_{rec} \approx 0$—self-correction is built into inference): Through RLHF alignment and redundant attention patterns, safety behaviors are distributed across the model's parameters rather than concentrated in any single state vector. There is no ``state'' to corrupt in the traditional sense; the model re-derives its safety posture at every forward pass.
\end{itemize}

\subsection{The Boundary Condition for External State Maintenance}

An external state maintenance mechanism (such as HIS) provides genuine benefit only when two conditions are jointly satisfied:

\begin{enumerate}
    \item $T_{rec} > T_{monitor}$: The architecture's natural recovery time exceeds the monitoring/intervention interval.
    \item The task cannot tolerate the transient performance degradation during the $T_{rec}$-step recovery window.
\end{enumerate}

For hidden state dynamics, all architectures tested satisfy $T_{rec} \leq 5$ steps (Table~\ref{tab:recovery}). For any practical monitoring interval ($T_{monitor} \geq 10$ steps), Condition 1 is violated, and external maintenance provides no marginal benefit to hidden state integrity.

However, our LTC safety memory benchmark (Table~\ref{tab:ltc_results}) reveals that these conditions \textit{are} satisfied for safety-encoded information that lacks observable correlates. The LTC safety memory degrades from retention $0.73$ to $0.05$ over $20{,}000$ steps---a recovery time $T_{rec} \to \infty$ because the information source (obstacle locations) is absent from the observation stream. In this regime, HIS's monitoring interval ($T_{monitor} = 50$ steps) is shorter than the effective recovery time, and the mechanism provides a $90\%$ reduction in safety violations.

\subsection{Conditions Under Which External Maintenance Would Be Valuable}

Our analysis suggests HIS and similar mechanisms would be most valuable for systems exhibiting:

\begin{itemize}
    \item \textbf{High-dimensional latent states} that encode information from hundreds or thousands of past steps, making reconstruction from a single observation impossible.
    \item \textbf{Severe observation bottlenecks} where the observation dimensionality is orders of magnitude smaller than the latent state.
    \item \textbf{Long-horizon memory tasks} where the critical information (e.g., a safety constraint issued 1,000 steps ago) has no observable correlate in the current input.
    \item \textbf{Architectures without gating or ODE dynamics} that lack the input-driven correction mechanisms present in LSTM/GRU/CfC.
\end{itemize}

The Pendulum control task---with its 2--3 dimensional observations and 64-dimensional hidden state---does not satisfy these conditions for hidden state recovery: the observation stream is sufficiently rich that any architecture reconstructs its latent representation within a few steps. However, our LTC navigation benchmark \textit{does} satisfy these conditions for safety memory: obstacle locations are encoded in a 64-neuron LTC network but are absent from the agent's sensory input (position and velocity only). The result is a $90\%$ violation reduction when HIS maintains this memory (Section 5.4), confirming that the boundary conditions identified above are not merely theoretical---they arise naturally in controller architectures that must remember environmental constraints not present in their observations.

% =====================================================================
% 7. DISCUSSION
% =====================================================================

\section{Discussion}

\subsection{When External Maintenance Helps---and When It Does Not}

Our results present a nuanced verdict on external state maintenance. HIS is a mathematically sound mechanism that achieves its theoretical performance guarantees in practice---the geometric bound of $1/\sqrt{2}$ is reliably attained. Its practical value, however, depends entirely on the relationship between the protected information and the input stream.

For Large Language Models, safety compliance is parameter-baked through RLHF and instruction tuning. Even the smallest model tested (Gemma2:2b) refuses $78\%$ of adversarial prompts without any intervention, and the 7B model reaches $99.5\%$. HIS operates correctly in this context---drift is detected, re-injection occurs---but the marginal benefit is absorbed by the model's existing safety training. This is analogous to installing a backup generator for a power grid with built-in redundancy: the backup functions correctly, but the grid never fails in a way that requires it.

For LTC safety controllers, the situation is fundamentally different. The encoded safety information (obstacle locations) has no correlate in the observation stream and degrades monotonically without maintenance. Here, HIS reduces violations by $90\%$ and outperforms both timer-based and norm-monitoring alternatives. The mechanism provides its intended benefit because the protected information satisfies the boundary conditions identified in Section 6: it cannot be self-corrected from observations.

This asymmetry carries a broader implication for AI safety research: \textbf{the value of a safety mechanism is not determined by its own correctness, but by whether the target system has an intrinsic vulnerability at the level the mechanism protects}. We recommend that any external state maintenance system be accompanied by an observability analysis---determining whether the protected information can be reconstructed from the input stream---before claims of safety improvement can be sustained.

\subsection{Relationship to Existing Work}

Our concept of Architectural Self-Correction connects to several established research threads:

\begin{itemize}
    \item \textbf{Echo State Property} \cite{jaeger2001}: In reservoir computing, the echo state property ensures that initial conditions are eventually ``washed out'' by input-driven dynamics. Our finding that all tested recurrent architectures converge to identical states regardless of initial corruption is a manifestation of this property in trained networks.
    \item \textbf{External Memory Architectures} \cite{graves2014}: Neural Turing Machines and Differentiable Neural Computers augment recurrent networks with external memory. Our work suggests that for low-dimensional control tasks, such augmentation is unnecessary because the recurrent dynamics themselves provide sufficient memory reconstruction.
    \item \textbf{Continual Learning}: Unlike catastrophic forgetting (which concerns weight corruption over training), ASC concerns state corruption during inference—a distinct failure mode with distinct remedies.
\end{itemize}

\subsection{Limitations}

\begin{enumerate}
    \item \textbf{Task complexity:} The LTC navigation benchmark uses a 2D arena with static obstacles. Real-world navigation involves dynamic obstacles, 3D environments, and sensor noise that may alter the drift dynamics of safety memory.
    \item \textbf{Model scale:} LLM experiments used 2B--7B parameter models. Frontier models ($\geq 70$B) with longer context windows may exhibit different drift dynamics, particularly on extended multi-session conversations.
    \item \textbf{Single-agent scope:} Multi-agent systems where drift compounds across interacting agents were not tested. Cascading drift in agent collectives may create scenarios where even observation-correlated information degrades.
    \item \textbf{Safety memory fidelity:} The LTC safety memory is a simplified model of how a controller ``remembers'' obstacle locations. Production systems may use explicit map representations rather than neural memory, reducing the need for HIS.
    \item \textbf{Perturbation model:} The hidden state recovery diagnostic uses complete zeroing, which is a worst-case perturbation. Partial perturbation or adversarial corruption patterns may yield different recovery dynamics.
\end{enumerate}

\subsection{Future Work}

\begin{itemize}
    \item \textbf{High-dimensional POMDPs:} Evaluate HIS on vision-based control (Atari, robotic manipulation) where observation-to-state mapping is nontrivial and safety constraints may not have observable correlates.
    \item \textbf{Dynamic safety constraints:} Extend the LTC benchmark to time-varying obstacle configurations, testing whether HIS can maintain a continuously updating safety memory.
    \item \textbf{State space models:} Test ASC properties of Mamba and other selective state space models, which may have qualitatively different recovery dynamics than RNNs or Transformers.
    \item \textbf{Extended LLM contexts:} Evaluate whether very long conversations ($> 100$ turns) or multi-session deployments degrade LLM safety compliance to a point where HIS re-injection becomes beneficial.
    \item \textbf{Observability diagnostics:} Develop standardized tools for classifying which components of a neural system's internal state have observable correlates, enabling practitioners to predict \textit{a priori} whether external maintenance will provide benefit.
\end{itemize}

% =====================================================================
% 8. CONCLUSION
% =====================================================================

\section{Conclusion}

We have demonstrated that Holographic Invariant Storage provides a mathematically robust method for safety signal recovery, achieving the geometric bound of $1/\sqrt{2}$ across $1{,}000$ Monte Carlo trials with diverse adversarial attacks ($\mu = 0.7074$, $\sigma = 0.0039$). Our extended evaluation reveals that the practical value of this mechanism depends critically on the observability of the protected information.

In Large Language Models ($450$ conversations across four models and six conditions), safety compliance rates of $88$--$100\%$ without any intervention render HIS's contribution statistically undetectable. Cross-model judging ($74$--$82\%$ agreement) confirms these are genuine refusals, not self-evaluation artifacts. In this regime, safety is a parameter-level property maintained through RLHF alignment, and external state maintenance targets a vulnerability that does not exist.

In Liquid Time-Constant neural controllers ($350$ trials across seven conditions), HIS reduces safety violations by $90\%$ relative to unprotected operation ($157.2$ vs $1{,}580.4$ violations per $20{,}000$ steps, $p < 10^{-17}$) and statistically outperforms fixed-interval timer re-injection ($p_{adj} = 0.0045$, Cohen's $d = 0.89$) while using fewer re-injections ($54.3$ vs $66.0$). The critical difference: the safety information (obstacle locations) has no observable correlate in the agent's input stream and degrades monotonically without maintenance (retention $0.73 \to 0.05$).

A diagnostic of hidden state recovery across six recurrent architectures (LSTM, GRU, vanilla RNN, CfC, LTC) reveals that all recover corrupted states within $\leq 5$ steps from ongoing observations---a property we term Architectural Self-Correction (ASC). ASC explains why hidden state maintenance mechanisms are redundant for observation-correlated information, while simultaneously clarifying why they remain essential for observation-independent safety memory.

These findings yield a practical design principle: before deploying external state maintenance, determine whether the protected information can be reconstructed from the input stream. If it can, the architecture's intrinsic dynamics will render the mechanism redundant. If it cannot, mechanisms such as HIS provide measurable and statistically significant safety benefit. This observability criterion provides a falsifiable, architecture-agnostic test for evaluating the necessity of proposed AI safety mechanisms.

% =====================================================================
% REFERENCES
% =====================================================================

\begin{thebibliography}{99}

\bibitem{kanerva2009}
Kanerva, P. (2009). 
``Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.'' 
\textit{Cognitive Computation}, 1(2), 139--159.

\bibitem{vaswani2017}
Vaswani, A., et al. (2017). 
``Attention Is All You Need.'' 
\textit{Advances in Neural Information Processing Systems}, 30.

\bibitem{gayler2003}
Gayler, R. W. (2003). 
``Vector Symbolic Architectures Answer Jackendoff's Challenges for Cognitive Neuroscience.'' 
\textit{ICCS/ASCS International Conference on Cognitive Science}, 133--138.

\bibitem{lecun2022}
LeCun, Y. (2022). 
``A Path Towards Autonomous Machine Intelligence.'' 
\textit{OpenReview}, Version 0.9.2.

\bibitem{liu2023}
Liu, N. F., et al. (2024). 
``Lost in the Middle: How Language Models Use Long Contexts.'' 
\textit{Transactions of the Association for Computational Linguistics}, 12, 157--173.

\bibitem{hasani2021}
Hasani, R., et al. (2021).
``Liquid Time-constant Networks.''
\textit{Proceedings of the AAAI Conference on Artificial Intelligence}, 35(9), 7657--7666.

\bibitem{hochreiter1997}
Hochreiter, S. \& Schmidhuber, J. (1997).
``Long Short-Term Memory.''
\textit{Neural Computation}, 9(8), 1735--1780.

\bibitem{jaeger2001}
Jaeger, H. (2001).
``The `echo state' approach to analysing and training recurrent neural networks.''
\textit{GMD Technical Report}, 148.

\bibitem{graves2014}
Graves, A., Wayne, G., \& Danihelka, I. (2014).
``Neural Turing Machines.''
\textit{arXiv preprint arXiv:1410.5401}.

\bibitem{cho2014}
Cho, K., et al. (2014).
``Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.''
\textit{Proceedings of EMNLP}, 1724--1734.

\bibitem{hasani2022}
Hasani, R., et al. (2022).
``Closed-form Continuous-time Neural Networks.''
\textit{Nature Machine Intelligence}, 4, 992--1003.

\bibitem{zong2025}
Zong, Z., et al. (2025).
``Accuracy, Memory Efficiency and Generalization: A Comparative Study on Liquid Neural Networks and Recurrent Neural Networks.''
\textit{IEEE Transactions on Neural Networks and Learning Systems} (early access).

\end{thebibliography}

\end{document}
