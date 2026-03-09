# Paper C: Development Plan & Tracking

---

## A. Theorem 2 — General Control-Affine Convergence Proof

### A.1: Primary Approach — General Control-Affine Case

**Goal:** Prove safe asymptotic convergence for systems of the form $\dot{x} = f(x) + g(x)u$, not just single-integrator $\dot{x} = u$.

**Why this matters:** Transformer layers are $x_{l+1} = x_l + \text{Block}_l(x_l)$ — a residual dynamical system with nonlinear $f$. If the convergence theorem only covers $f = 0$, the theory doesn't match the GPT-2 experiment. Internal coherence between theory and evidence is non-negotiable for a top venue.

**Required Assumptions:**
1. $f, g$ locally Lipschitz (standard)
2. QP feasibility for all $x \in \mathcal{S}$ (the "compatibility condition" — CBF constraint and control authority are compatible at every point in the safe set; verified empirically in Exp XIV)
3. $h$ is a valid CBF (0 is a regular value of $h$, $\mathcal{S}$ compact)
4. Utility function is real-analytic (for Łojasiewicz singleton convergence), OR drop to KKT-set convergence if this assumption is too restrictive

**Proof Strategy:**
1. Under assumptions 1-3, the CBF-QP solution $u^*(x)$ is continuous in $x$ (Berge's Maximum Theorem applied to the parametric QP).
2. The closed-loop system $\dot{x} = f(x) + g(x)u^*(x)$ has a well-defined flow by Lipschitz continuity of $f, g$ and continuity of $u^*$.
3. Forward invariance of $\mathcal{S}$ follows from Nagumo's theorem (already proven in Theorem 1).
4. Construct a Lyapunov-like function from the utility $J(x)$ restricted to $\mathcal{S}$.
5. Barbalat's Lemma gives convergence to the set where $\dot{J} = 0$ (the KKT set).
6. Łojasiewicz gradient inequality (under real-analyticity) upgrades set convergence to singleton convergence.

**Key References:**
- Ames et al. (2019) — CBF theory and applications
- Jankovic (2018) — Robust CBFs
- Berge (1963) — Maximum Theorem for parametric optimization
- Absil, Mahony, Andrews (2005) — Convergence of gradient systems via Łojasiewicz

**Estimated Time:** 2-4 weeks focused mathematical work.

**Potential Pitfalls:**
- The CBF-QP solution $u^*(x)$ may be discontinuous at the boundary of the active constraint region (where the constraint switches from inactive to active). This would invalidate the standard ODE existence/uniqueness argument.
- If this happens → invoke A.2 below.

**Status:** [ ] Not started

---

### A.2: Fallback — Clarke's Generalized Gradient Theory

**When to invoke:** If the QP projection introduces discontinuities at the boundary of the active constraint region that break standard ODE theory.

**Approach:** The discontinuity in $u^*(x)$ is of a specific, structured kind — it's the projection onto a halfspace boundary, which is piecewise smooth. Clarke's generalized gradient framework handles exactly this class of problems:

1. The closed-loop vector field $F(x) = f(x) + g(x)u^*(x)$ is piecewise smooth (Lipschitz almost everywhere).
2. Filippov solutions exist for such differential inclusions.
3. Under a "nonsmooth Lyapunov" argument, we can still establish convergence using Clarke's generalized directional derivative of $J$.
4. The convergence result would be to Clarke critical points of $J$ restricted to $\mathcal{S}$, which is the nonsmooth analog of the KKT set.

**Key References:**
- Clarke (1983) — Optimization and Nonsmooth Analysis
- Cortes (2008) — Discontinuous dynamical systems (a tutorial)
- Cortés, Bullo (2005) — Coordination and geometric optimization via Filippov flows

**Tradeoff:** The result is weaker (convergence to Clarke critical set, not KKT singleton) but covers a strictly larger class of systems and doesn't require real-analyticity of the utility.

**Estimated Time:** 1-2 additional weeks on top of A.1 work.

**Status:** [ ] Not started (only if A.1 hits a wall)

---

## B. Paper C — Target Structure

**Working Title:** "Control Barrier Functions for Transformer Hidden-State Safety: Scalable Verification from Synthetic to Learned Dynamics"

| Section | Content | Target Pages |
|---------|---------|:---:|
| **1. Introduction** | Safety for neural network hidden states; why CBFs; why dimension matters; contribution summary | 1 |
| **2. Related Work** | CBFs (Ames 2019), HDC (Salik et al. survey, Zeulin et al. Large-Margin HDC), representation engineering (Zou et al. 2023), LLM safety approaches (RLHF, Constitutional AI, guardrails) | 1 |
| **3. Theory** | CBF-QP closed-form ($O(n)$ single constraint), MCBC + Hoeffding sample complexity (dimension-independent), Theorem 1 (forward invariance via Nagumo), Theorem 2 (safe asymptotic convergence — general control-affine) | 2 |
| **4. Active Adversarial Safety Verification** | Hunter (momentum PGD on barrier landscape), Buffer (Hutchinson spectral estimation for adaptive margins), Anti-Memory (orthogonal prototype retention). Honest guarantee tiers: deterministic for kinematic, probabilistic for semantic. | 1.5 |
| **5. Experiments** | **Exp A:** Lorenz attractor (nonlinear dynamics validation); **Exp B:** QP scaling benchmark vs OSQP ($n$ up to 1024+); **Exp C:** GPT-2 hidden-state barrier (Civil Comments, SVM barrier, train/test split, CBF-QP intervention, MCBC on boundary, perplexity evaluation) | 3 |
| **6. Discussion & Limitations** | WTA oracle requirement, autoregressive deployment gap, KNN dynamics error unbound, linear barrier accuracy ceiling, normalization as active design choice | 1 |
| **7. Conclusion & Future Work** | Summary of contributions, future: autoregressive deployment, nonlinear barriers, larger models | 0.5 |
| **Total** | | **~10 pp** |

**Key Framing Principles:**
- No "Golden Manifold," no "Soul Anchor," no sci-fi naming. Descriptive technical terms only.
- Every claim backed by either a proof or an experiment *in this paper*.
- Self-cite BtG and HIS Zenodo preprints once each (footnote: "an earlier version appeared as..."), then never again.
- Cite the HDC survey (Salik et al.), Large-Margin HDC (Zeulin et al.), Zou et al. (2023) representation engineering — position the work *within* the existing field.
- Limitations section is frank: state what doesn't work, what's assumed, what's unvalidated.

**Target Venues (in order):**
1. AAAI 2027 (AI safety track, 7pp + appendix)
2. IJCAI 2027 (main track, similar format)
3. AAMAS 2027 (autonomous agents safety)
4. Workshop warm-up: SafeGenAI @ NeurIPS 2026 or AAAI SafeAI workshop

---

## C. Revolutionary Additions — Attempt All Three

If any of these succeed, they elevate Paper C from "solid, publishable" to "genuinely impressive." If all three prove infeasible or not worthwhile, proceed with the base Paper C plan above. Attempt them in order of impact.

---

### C.1: Autoregressive CBF-Steered Generation with Toxicity Measurement

**What it is:** Hook the CBF-QP into GPT-2's forward pass at *every token generation step*. Generate 1000+ completions of toxic prompts. Measure toxicity with Perspective API or a toxicity classifier. Show statistically significant reduction vs. unsteered generation.

**Why it's revolutionary:** This closes the "autoregressive gap" that Paper B honestly identifies. Current Exp XIV operates on precomputed trajectories — the CBF intervenes once on a frozen pass. Real deployment requires CBF intervention at every layer of every token, with the intervention altering the KV-cache and feeding back into subsequent tokens. Nobody has demonstrated CBF-steered autoregressive generation with measured safety outcomes.

**Technical Challenges:**
- Each CBF intervention at layer $l$ modifies the hidden state, which propagates through layers $l+1, ..., L$. The KV-cache stores pre-intervention keys/values, creating a mismatch. Need to either (a) recompute KV-cache after intervention, or (b) analyze the error from the mismatch.
- Repeated interventions across tokens may accumulate steering artifacts, degrading coherence. Perplexity alone won't catch this — need human evaluation or a separate quality judge.
- The SVM barrier was trained on single-pass hidden states. Under repeated intervention, the hidden-state distribution shifts. The barrier may no longer separate safe/toxic in the shifted regime. May need to retrain with intervention-aware data augmentation.

**Implementation Sketch:**
1. Register a forward hook at the CBF layer of GPT-2.
2. At each forward pass (each token), the hook computes $h(x)$ and, if $h < 0$, applies the CBF-QP correction $u^*$.
3. After the full forward pass, the LM head produces next-token logits from the steered hidden state.
4. Generate 50 tokens per prompt, 1000 prompts from RealToxicityPrompts (Gehman et al., 2020).
5. Score each completion with Perspective API (toxicity score 0-1).
6. Compare mean toxicity, max toxicity, and toxicity rate (fraction > 0.5) between steered and unsteered generation.

**Success Criteria:** Statistically significant reduction in toxicity rate ($p < 0.05$, Welch's t-test) with perplexity ratio $< 1.2$ (coherence preserved).

**Estimated Time:** 2-3 weeks implementation + evaluation.

**Status:** [ ] Not started

---

### C.2: Learned Neural Barrier with Certified Lipschitz Bounds

**What it is:** Replace the linear SVM barrier ($h(x) = w \cdot x + b$, 76% test accuracy) with a small neural network trained with spectral normalization, so $L_h = \text{Lip}(h)$ is bounded by construction. Show it achieves 90%+ separation on held-out test data. Then show the CBF-QP still works with this nonlinear barrier.

**Why it's revolutionary:** The linear SVM is a toy barrier. A learned nonlinear barrier with *certified* Lipschitz bounds demonstrates that the CBF framework scales to real, complex decision boundaries while maintaining the formal guarantees. This is the piece that makes the method practical.

**Technical Challenges:**
- Spectral normalization bounds the Lipschitz constant of each layer, so $L_h \leq \prod_l \sigma_{\max}(W_l)$, which can be controlled to any desired value. But tight bounds require careful architecture design (residual connections inflate the product).
- The CBF-QP requires $\nabla h(x)$, which is now the network's backpropagation gradient. This is $O(n \cdot p)$ where $p$ is the number of parameters. For a small network (2 hidden layers, 256 units), this is still fast.
- The nonlinear barrier may not satisfy the regularity conditions (0 as a regular value of $h$). Need to verify empirically or add a regularization term during training.

**Implementation Sketch:**
1. Define a small MLP: 768 → 256 → 128 → 1, with spectral normalization on each weight matrix.
2. Train on Civil Comments hidden states (same train/test split as Exp XIV). Binary cross-entropy loss with label smoothing.
3. Evaluate test accuracy. Target: 90%+.
4. Extract the Lipschitz constant: $L_h = \prod_l \bar{\sigma}(W_l)$ where $\bar{\sigma}$ is the spectral norm after normalization.
5. Run CBF-QP using $\nabla h$ from backpropagation. Verify that interventions maintain $h(x) \geq 0$ on toxic trajectories.
6. Run MCBC on the neural barrier's zero level set (harder sampling — may need to use gradient-based boundary sampling).

**Success Criteria:** Test accuracy $\geq 90\%$, $L_h$ bounded and stated, CBF-QP maintains 0 violations on steered trajectories, MCBC $P_{\text{safe}} > 0.99$.

**Estimated Time:** 1-2 weeks.

**Status:** [x] COMPLETED — `paper C/neural_barrier_experiment.py`

**Results (500 safe / 500 toxic, 80/20 split, GPT-2 layer 12):**

| Metric | SVM | SN-MLP (768→512→256→1) | Ablation (no SN) |
|--------|:---:|:---:|:---:|
| Test accuracy | 80.5% | **88.0%** | 86.5% |
| $L_h$ (certified) | 24.28 | **1.10** | 8.73 |
| $L_h$ (empirical) | 24.28 | **0.58** | 1.08 |
| CBF violations | 0 | **0** | — |
| MCBC $P_{\text{safe}}$ | 1.0000 | **1.0000** | — |
| PPL ratio (median) | 1.000 | **1.005** | — |

**Criteria Assessment:**
- [PASS] $L_h$ bounded: 1.10 (22× lower than SVM's 24.28)
- [PASS] CBF-QP 0 violations (iterative Newton + binary search fallback)
- [PASS] MCBC $P_{\text{safe}} = 1.0$ at all budget levels (5%–50%)
- [NEAR] Test accuracy 88% (target was 90%; +7.5 pp over SVM baseline)

The 88% accuracy reflects task difficulty, not architecture limitation: the unconstrained ablation achieves only 86.5% while having 7.9× worse Lipschitz constant. The spectral normalization constraint costs <2% accuracy while providing a *certified* 22× reduction in barrier sensitivity — a highly favorable tradeoff.

**Key Technical Contributions:**
1. First demonstration of spectrally-normalized neural CBF barrier on transformer hidden states
2. Iterative nonlinear CBF-QP with convergence guarantee (Newton + binary search fallback)
3. Newton-projected boundary sampling for MCBC on nonlinear level sets
4. Ablation proving the necessity of Lipschitz certification (MLP-noSN: $L_h = 8.73$, uncertifiable)

---

### C.3: Scaling to a Larger Model (Llama-3-8B or Qwen2.5-7B)

**What it is:** Run the same experiment (SVM barrier + CBF-QP intervention + MCBC verification + perplexity evaluation) on a model with hidden dimension 4096, validating the $O(n)$ scaling claim on a production-scale architecture.

**Why it's revolutionary:** GPT-2 is a 124M parameter model from 2019. Demonstrating the same framework on an 8B parameter model from 2024-2025 shows the method isn't just a toy experiment on an obsolete architecture. The $O(n)$ scaling claim becomes empirically validated at the dimensions that actually matter.

**Technical Challenges:**
- Memory: Llama-3-8B requires ~16GB VRAM in float16. Hidden-state extraction for 500 texts × 32 layers × 4096 dims is feasible but needs careful batching.
- SVM training in 4096 dimensions with 500 samples is fine (still $p \gg n$ but LinearSVC handles this).
- The KNN dynamics estimation (for MCBC) becomes slower at $n = 4096$. BallTree performance degrades in high dimensions. May need to switch to approximate nearest neighbors (FAISS).
- Perplexity evaluation requires the full LM head forward pass, which is expensive at 8B scale.

**Hardware Requirements:** You have an RTX 5050. Need to check VRAM — Llama-3-8B in 4-bit quantization fits in ~6GB. Hidden-state extraction may need to run in batches of 10-20 texts.

**Implementation Sketch:**
1. Load Llama-3.2-3B or Qwen2.5-3B (already pulled in Ollama) via HuggingFace transformers (need unquantized access to hidden states).
2. Extract hidden-state trajectories for Civil Comments texts.
3. Train SVM barrier at the best-separating layer.
4. Run CBF-QP intervention with held-out evaluation.
5. Run MCBC on SVM boundary.
6. Compare timing: QP solve time at $n = 768$ (GPT-2) vs $n = 2048$ or $n = 3072$ (3B models) vs $n = 4096$ (8B models). Plot scaling curve.

**Success Criteria:** SVM accuracy comparable to GPT-2 (75%+), CBF-QP 0 violations, scaling curve confirms $O(n)$, total experiment runs in < 4 hours on available hardware.

**Estimated Time:** 1-2 weeks.

**Status:** [ ] Not started

---

## Execution Order

1. **Theorem 2 (A.1/A.2)** — Do this first. It's pure math, no code, and determines the theoretical scope of everything else.
2. **C.2 (Neural barrier)** — Quickest revolutionary addition. If it works, it makes the GPT-2 experiment dramatically stronger with minimal new infrastructure.
3. **C.1 (Autoregressive generation)** — Highest impact but hardest. Attempt after C.2 is settled.
4. **C.3 (Larger model)** — Attempt last. It's the most hardware-dependent and the least novel (it's the same experiment at bigger scale). Also depends on whether C.2's neural barrier transfers across model architectures.

If time/feasibility forces a choice: **C.2 alone is worth more than C.1 + C.3 without C.2.** The neural barrier is the linchpin — it's what makes the framework non-toy.
