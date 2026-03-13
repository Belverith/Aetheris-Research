# Hypervectors — AI Safety Research

Research repository by Arsenios Scrivens covering hypervector-based safety mechanisms, control barrier functions, and autonomous agent verification.

---

## Repository Structure

### [`paper_A_HIS/`](paper_A_HIS/) — Holographic Invariant Storage (arXiv)
**"Holographic Invariant Storage for AI Safety"** — Published on arXiv.  
Neuro-symbolic safety mechanism based on Vector Symbolic Architectures (VSA) for mitigating LLM context drift. Demonstrates recovery fidelity matching the geometric bound of 1/√2 via Monte Carlo simulation, with LTC neural network controller experiments in the appendix.

- `HIS_paper_REVISED.tex` — **Final arXiv submission**
- `experiments/` — Monte Carlo, LLM behavioral, and cross-model judge experiments
- `ltc_experiments/` — Appendix D: Liquid Time-Constant controller experiments
- `demos/` — Mind server, anti-drift demos, dashboard
- `figures/`, `results/`, `logs/` — Experimental outputs

### [`paper_B_CHDBO/`](paper_B_CHDBO/) — Constrained High-Dimensional Barrier Optimization
**"Beyond the Grid"** — Probabilistic safety verification in high-dimensional manifolds.  
Introduces CHDBO framework: Monte Carlo Barrier Certificates, orthogonal projection verification, and gradient-based utility maximization. Includes Active Adversarial Safety Verification (AASV) module.

- `original_paper_A.md` — Core CHDBO theory
- `original_paper_B.md` — AASV / WTA / GPT-2 experiment (split content)
- `experiments/` — Safety proofs and verification scripts
- `figure_generators/` — Scripts for generating all paper figures
- `figures/` — Generated figures
- `paper_storage/` — Original preserved versions (Do Not Edit or Delete)

### [`paper_C_CBF/`](paper_C_CBF/) — Control Barrier Functions for Transformers
**"Control Barrier Functions for Transformer Hidden States: A Proof of Concept"**  
Investigates applying CBFs from nonlinear control theory to transformer hidden states. Trains spectrally-normalized neural barriers on GPT-2/Qwen2.5 with autoregressive safety steering.

- `Paper_C.md` — Main paper
- `experiments/` — Autoregressive, neural barrier, LTC, LSTM-POMDP experiments
- `checkpoints/` — Trained model weights and checkpoints
- `results/`, `figures/` — Experimental outputs

### [`aetheris/`](aetheris/) — Aetheris Cognitive Synthesis
Theoretical framework for autonomous machine intelligence — the Aetheris Master Equation unifying hierarchical cognitive dynamics, temporal synchronization, safety verification, and self-evolutionary protocols.

- `reference_papers/` — Related work from the literature

### [`conversation_history/`](conversation_history/) — Research Conversation Logs

### [`_archive/`](_archive/) — Older File Versions
Root-level files superseded by organized versions within their respective paper directories.
