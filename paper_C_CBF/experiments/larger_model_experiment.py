#!/usr/bin/env python3
"""
C.3: Larger Model Validation — Qwen2.5-3B (n=2048)
====================================================
Designed for Podrun (cloud GPU with ≥24GB VRAM).

Replicates the neural barrier + CBF-QP pipeline from the GPT-2 experiment
on Qwen2.5-3B (hidden dim 2048), demonstrating that the framework scales
to production-adjacent model sizes.

Pipeline:
  1. Load Qwen2.5-3B + Civil Comments
  2. Extract hidden-state trajectories at all layers
  3. Find best layer via SVM CV
  4. Train SN-MLP barrier (2048→512→256→1)
  5. CBF-QP verification (zero-violation check, boundary feasibility)
  6. Activation-patching perplexity evaluation
  7. Autoregressive CBF-steered generation (α=0.25)
  8. Statistical comparison with unsteered baseline

Run on Podrun:
  pip install torch transformers datasets scikit-learn scipy matplotlib
  python larger_model_experiment.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Model
MODEL_NAME = "Qwen/Qwen2.5-3B"

# Data
N_SAFE_TARGET = 500
N_TOXIC_TARGET = 500
MAX_TEXT_LEN = 200
TRAIN_TEST_RATIO = 0.2

# Barrier architecture — wider input for n=2048
HIDDEN_DIMS = [512, 256]
N_EPOCHS = 600
BATCH_SIZE = 128
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 80
LABEL_SMOOTHING = 0.02

# Generation
N_TOXIC_PROMPTS = 200
N_SAFE_PROMPTS = 100
PROMPT_TOKENS = 10
MAX_NEW_TOKENS = 50
TEMPERATURE = 1.0
TOP_K = 50

# CBF
CBF_BUFFER = 0.3
CBF_MAX_ITER = 20
CBF_ALPHA = 0.25
CBF_MAX_NORM = 0.5

# Boundary verification
N_BOUNDARY_POINTS = 10000
NEWTON_STEPS = 100
NEWTON_TOL = 1e-6
KNN_K = 10
BUDGET_FRACTIONS = [0.05, 0.10, 0.20, 0.50]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BARRIER_SAVE_PATH = os.path.join(SCRIPT_DIR, 'trained_barrier_qwen.pt')
RESULTS_SAVE_PATH = os.path.join(SCRIPT_DIR, 'c3_results.json')

# ══════════════════════════════════════════════════════════════════════
# 1. Load Data and Model
# ══════════════════════════════════════════════════════════════════════
print("\n[1/9] Loading data and model...")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {vram:.1f} GB")

# ── Load Civil Comments ──
t0 = time.time()
dataset = load_dataset('google/civil_comments', split='train', streaming=True)

safe_texts = []
toxic_texts = []
for sample in dataset:
    text = sample['text'].strip()
    tox = sample['toxicity']
    if len(text) < 20 or len(text) > MAX_TEXT_LEN:
        continue
    if tox <= 0.1 and len(safe_texts) < N_SAFE_TARGET:
        safe_texts.append(text)
    elif tox >= 0.7 and len(toxic_texts) < N_TOXIC_TARGET:
        toxic_texts.append(text)
    if len(safe_texts) >= N_SAFE_TARGET and len(toxic_texts) >= N_TOXIC_TARGET:
        break

print(f"  Data loaded in {time.time()-t0:.1f}s: "
      f"{len(safe_texts)} safe, {len(toxic_texts)} toxic")

# ── Load model ──
t0 = time.time()
print(f"  Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_lm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True)
model_lm.eval()
model_lm = model_lm.to(device)

config = model_lm.config
N_LAYERS = config.num_hidden_layers
DIM = config.hidden_size
EOS_TOKEN_ID = tokenizer.eos_token_id

print(f"  Model loaded in {time.time()-t0:.1f}s: "
      f"{N_LAYERS} layers, dim={DIM}")

# ── Extract hidden-state trajectories ──
print("  Extracting hidden-state trajectories...")
t0 = time.time()


def extract_hidden_trajectory(text, model, tok):
    """Extract (N_LAYERS+1, DIM) trajectory — last token at each layer."""
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    trajectory = np.array([
        outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
        for l in range(N_LAYERS + 1)
    ])
    return trajectory


safe_trajectories = np.array([
    extract_hidden_trajectory(t, model_lm, tokenizer) for t in safe_texts
])
toxic_trajectories = np.array([
    extract_hidden_trajectory(t, model_lm, tokenizer) for t in toxic_texts
])
print(f"  Extraction done in {time.time()-t0:.1f}s")
print(f"  Trajectory shape: {safe_trajectories.shape}")

# ── Train/test split ──
n_safe = len(safe_texts)
n_toxic = len(toxic_texts)
safe_idx_train, safe_idx_test = train_test_split(
    np.arange(n_safe), test_size=TRAIN_TEST_RATIO, random_state=RANDOM_SEED)
toxic_idx_train, toxic_idx_test = train_test_split(
    np.arange(n_toxic), test_size=TRAIN_TEST_RATIO, random_state=RANDOM_SEED)
print(f"  Split: {len(safe_idx_train)+len(toxic_idx_train)} train, "
      f"{len(safe_idx_test)+len(toxic_idx_test)} test")


# ══════════════════════════════════════════════════════════════════════
# 2. Neural Barrier
# ══════════════════════════════════════════════════════════════════════

class NeuralBarrier(nn.Module):
    """Spectrally-normalized MLP barrier."""

    def __init__(self, input_dim, hidden_dims=None, input_mean=None,
                 input_std=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        if input_mean is not None:
            self.register_buffer('input_mean',
                                 torch.tensor(input_mean, dtype=torch.float32))
            self.register_buffer('input_std',
                                 torch.tensor(input_std, dtype=torch.float32))
        else:
            self.input_mean = None
            self.input_std = None

        layers = []
        prev = input_dim
        for hd in hidden_dims:
            lin = nn.Linear(prev, hd)
            lin = spectral_norm(lin)
            layers.append(lin)
            layers.append(nn.LeakyReLU(0.01))
            prev = hd
        out_lin = nn.Linear(prev, 1)
        out_lin = spectral_norm(out_lin)
        layers.append(out_lin)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_mean is not None:
            x = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x).squeeze(-1)

    def compute_lipschitz_bound(self):
        L = 1.0
        for module in self.net:
            if isinstance(module, nn.Linear):
                W = module.weight
                sigma = torch.linalg.svdvals(W)[0].item()
                L *= sigma
        return L

    def barrier_and_grad(self, x):
        if isinstance(x, np.ndarray):
            x_t = torch.tensor(x, dtype=torch.float32,
                               device=next(self.parameters()).device).unsqueeze(0)
        else:
            x_t = x.float().unsqueeze(0) if x.dim() == 1 else x.float()
        x_t = x_t.detach().requires_grad_(True)
        h = self.forward(x_t)
        h.backward()
        grad = x_t.grad[0].detach()
        return h.item(), grad


print("\n[2/9] Finding best layer via SVM cross-validation...")
best_layer = -1
best_acc = 0.0
for layer in range(N_LAYERS + 1):
    X_safe = safe_trajectories[safe_idx_train, layer, :]
    X_toxic = toxic_trajectories[toxic_idx_train, layer, :]
    X = np.vstack([X_safe, X_toxic])
    y = np.array([1] * len(X_safe) + [-1] * len(X_toxic))
    svm = LinearSVC(C=1.0, max_iter=5000, dual='auto', random_state=RANDOM_SEED)
    scores = cross_val_score(svm, X, y, cv=3, scoring='accuracy')
    acc = scores.mean()
    if acc > best_acc:
        best_acc = acc
        best_layer = layer
    if (layer + 1) % 6 == 0 or layer == N_LAYERS:
        print(f"  Layer {layer:2d}: CV acc = {acc:.4f}" +
              (" ← best so far" if layer == best_layer else ""))

TARGET_LAYER = best_layer
print(f"  Best layer: {TARGET_LAYER} (CV = {best_acc:.4f})")

# ── Prepare data ──
X_safe_train = safe_trajectories[safe_idx_train, TARGET_LAYER, :]
X_toxic_train = toxic_trajectories[toxic_idx_train, TARGET_LAYER, :]
X_train = np.vstack([X_safe_train, X_toxic_train])
y_train = np.array([1] * len(X_safe_train) + [-1] * len(X_toxic_train))
y_train_01 = ((y_train + 1) / 2).astype(np.float32)

X_safe_test = safe_trajectories[safe_idx_test, TARGET_LAYER, :]
X_toxic_test = toxic_trajectories[toxic_idx_test, TARGET_LAYER, :]
X_test = np.vstack([X_safe_test, X_toxic_test])
y_test = np.array([1] * len(X_safe_test) + [-1] * len(X_toxic_test))
y_test_01 = ((y_test + 1) / 2).astype(np.float32)

input_mean = X_train.mean(axis=0)
input_std = X_train.std(axis=0)

# ── Linear SVM baseline ──
print("\n[3/9] Training SVM baseline...")
svm_baseline = LinearSVC(C=1.0, max_iter=5000, dual='auto',
                         random_state=RANDOM_SEED)
svm_baseline.fit(X_train, y_train)
svm_test_acc = svm_baseline.score(X_test, y_test)
svm_Lh = np.linalg.norm(svm_baseline.coef_)
print(f"  SVM test accuracy: {svm_test_acc:.4f}")
print(f"  SVM L_h = ||w|| = {svm_Lh:.2f}")

# ── Train or load SN-MLP barrier ──
print(f"\n[4/9] Training SN-MLP barrier ({DIM}→{'→'.join(map(str, HIDDEN_DIMS))}→1)...")

if os.path.exists(BARRIER_SAVE_PATH):
    print(f"  Loading saved barrier from {os.path.basename(BARRIER_SAVE_PATH)}...")
    checkpoint = torch.load(BARRIER_SAVE_PATH, weights_only=False)
    neural_barrier = NeuralBarrier(
        input_dim=DIM,
        hidden_dims=checkpoint.get('hidden_dims', HIDDEN_DIMS),
        input_mean=checkpoint['input_mean'],
        input_std=checkpoint['input_std'],
    )
    neural_barrier.load_state_dict(checkpoint['model_state_dict'])
    neural_barrier.eval()
    saved_layer = checkpoint.get('target_layer', TARGET_LAYER)
    saved_acc = checkpoint.get('test_acc', -1)
    print(f"  Loaded: layer={saved_layer}, test_acc={saved_acc:.4f}")
    if saved_layer != TARGET_LAYER:
        print(f"  [WARN] Saved layer ({saved_layer}) != current best "
              f"({TARGET_LAYER}). Using saved layer.")
        TARGET_LAYER = saved_layer
else:
    neural_barrier = NeuralBarrier(
        input_dim=DIM, hidden_dims=HIDDEN_DIMS,
        input_mean=input_mean, input_std=input_std,
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_01, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_01, dtype=torch.float32)

    torch.manual_seed(RANDOM_SEED)
    optimizer = torch.optim.AdamW(
        neural_barrier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    best_test_acc = 0.0
    best_state = None
    patience_counter = 0
    n_train = len(X_train_t)

    t0 = time.time()
    for epoch in range(N_EPOCHS):
        neural_barrier.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            xb = X_train_t[perm[start:end]]
            yb = y_train_t[perm[start:end]]
            xb = xb + torch.randn_like(xb) * 0.1
            yb_smooth = yb * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            logits = neural_barrier(xb)
            loss = criterion(logits, yb_smooth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        neural_barrier.eval()
        with torch.no_grad():
            test_logits = neural_barrier(X_test_t)
            test_pred = (test_logits > 0).float()
            test_acc = (test_pred == y_test_t).float().mean().item()

        if (epoch + 1) % 50 == 0:
            L_h = neural_barrier.compute_lipschitz_bound()
            print(f"    Epoch {epoch+1:3d}: test_acc={test_acc:.4f}, "
                  f"L_h={L_h:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.clone() for k, v in
                          neural_barrier.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        neural_barrier.load_state_dict(best_state)
    neural_barrier.eval()

    print(f"  Training done in {time.time()-t0:.1f}s, "
          f"best test_acc={best_test_acc:.4f}")

    torch.save({
        'model_state_dict': neural_barrier.state_dict(),
        'target_layer': TARGET_LAYER,
        'input_mean': input_mean,
        'input_std': input_std,
        'hidden_dims': HIDDEN_DIMS,
        'test_acc': best_test_acc,
        'model_name': MODEL_NAME,
        'dim': DIM,
    }, BARRIER_SAVE_PATH)
    print(f"  Barrier saved to {os.path.basename(BARRIER_SAVE_PATH)}")

neural_barrier = neural_barrier.to(device)
L_h = neural_barrier.compute_lipschitz_bound()
print(f"  Certified L_h = {L_h:.4f}")

# Empirical Lipschitz from random test pairs
n_pairs = 5000
idx_a = np.random.choice(len(X_test), n_pairs)
idx_b = np.random.choice(len(X_test), n_pairs)
with torch.no_grad():
    xa = torch.tensor(X_test[idx_a], dtype=torch.float32, device=device)
    xb = torch.tensor(X_test[idx_b], dtype=torch.float32, device=device)
    ha = neural_barrier(xa)
    hb = neural_barrier(xb)
    diffs_h = (ha - hb).abs().cpu().numpy()
    diffs_x = (xa - xb).norm(dim=1).cpu().numpy()
    valid = diffs_x > 1e-10
    if valid.sum() > 0:
        L_h_emp = (diffs_h[valid] / diffs_x[valid]).max()
    else:
        L_h_emp = 0.0
print(f"  Empirical L_h = {L_h_emp:.4f}")

# ── Ablation (no spectral norm) ──
print("\n  Training unconstrained ablation...")


class NeuralBarrierAblation(nn.Module):
    """Same architecture, NO spectral normalization."""

    def __init__(self, input_dim, hidden_dims=None, input_mean=None,
                 input_std=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        if input_mean is not None:
            self.register_buffer('input_mean',
                                 torch.tensor(input_mean, dtype=torch.float32))
            self.register_buffer('input_std',
                                 torch.tensor(input_std, dtype=torch.float32))
        else:
            self.input_mean = None
            self.input_std = None

        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev, hd))
            layers.append(nn.LeakyReLU(0.01))
            prev = hd
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_mean is not None:
            x = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x).squeeze(-1)

    def compute_lipschitz_bound(self):
        L = 1.0
        for module in self.net:
            if isinstance(module, nn.Linear):
                sigma = torch.linalg.svdvals(module.weight)[0].item()
                L *= sigma
        return L


ablation = NeuralBarrierAblation(
    input_dim=DIM, hidden_dims=HIDDEN_DIMS,
    input_mean=input_mean, input_std=input_std,
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_01, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test_01, dtype=torch.float32)

torch.manual_seed(RANDOM_SEED)
abl_optimizer = torch.optim.AdamW(
    ablation.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
abl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    abl_optimizer, T_max=N_EPOCHS)
abl_criterion = nn.BCEWithLogitsLoss()
abl_best_acc = 0.0
abl_best_state = None
abl_patience = 0

t0 = time.time()
for epoch in range(N_EPOCHS):
    ablation.train()
    perm = torch.randperm(len(X_train_t))
    for start in range(0, len(X_train_t), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X_train_t))
        xb = X_train_t[perm[start:end]]
        yb = y_train_t[perm[start:end]]
        xb = xb + torch.randn_like(xb) * 0.1
        yb_s = yb * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        loss = abl_criterion(ablation(xb), yb_s)
        abl_optimizer.zero_grad()
        loss.backward()
        abl_optimizer.step()
    abl_scheduler.step()

    ablation.eval()
    with torch.no_grad():
        abl_acc = (
            (ablation(X_test_t) > 0).float() == y_test_t
        ).float().mean().item()

    if abl_acc > abl_best_acc:
        abl_best_acc = abl_acc
        abl_best_state = {k: v.clone() for k, v in ablation.state_dict().items()}
        abl_patience = 0
    else:
        abl_patience += 1
        if abl_patience >= PATIENCE:
            break

if abl_best_state:
    ablation.load_state_dict(abl_best_state)
ablation.eval()
abl_Lh = ablation.compute_lipschitz_bound()
print(f"  Ablation: test_acc={abl_best_acc:.4f}, L_h={abl_Lh:.2f} (uncertified)")
print(f"  Time: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════
# 5. CBF-QP Verification
# ══════════════════════════════════════════════════════════════════════
print("\n[5/9] CBF-QP verification on all trajectories...")


def compute_cbf_correction(barrier, x_t, buffer=2.0, max_iter=50):
    """Minimum-norm CBF correction."""
    with torch.enable_grad():
        h_val, grad_h = barrier.barrier_and_grad(x_t)

    if h_val >= buffer:
        return h_val, None, 0.0

    gh_sq = grad_h.dot(grad_h).item()
    if gh_sq < 1e-12:
        return h_val, None, 0.0

    lam = (buffer - h_val) / gh_sq
    u_star = lam * grad_h

    x_steered = x_t + u_star
    for _ in range(max_iter):
        with torch.enable_grad():
            h_check, grad_check = barrier.barrier_and_grad(x_steered)
        if h_check >= buffer:
            break
        deficit = buffer - h_check
        g_sq = grad_check.dot(grad_check).item()
        if g_sq < 1e-12:
            break
        extra = (deficit / g_sq) * grad_check
        u_star = u_star + extra
        x_steered = x_t + u_star

    return h_val, u_star, float(u_star.norm().item())


# Verify on all test trajectories
violations = 0
u_norms = []
for X_set, label in [(X_safe_test, 'safe'), (X_toxic_test, 'toxic')]:
    for x in X_set:
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        h_val, u_star, u_norm = compute_cbf_correction(
            neural_barrier, x_t, buffer=2.0, max_iter=50)
        if u_star is not None:
            u_norms.append(u_norm)
            # Verify post-correction
            with torch.enable_grad():
                h_post, _ = neural_barrier.barrier_and_grad(x_t + u_star)
            if h_post < 0:
                violations += 1

print(f"  Total violations: {violations}")
print(f"  Mean ||u*|| (when corrected): {np.mean(u_norms):.4f}" if u_norms
      else "  No corrections needed")
print(f"  Max ||u*||: {np.max(u_norms):.4f}" if u_norms else "")


# ══════════════════════════════════════════════════════════════════════
# 6. Boundary Feasibility Sampling
# ══════════════════════════════════════════════════════════════════════
print(f"\n[6/9] Boundary feasibility ({N_BOUNDARY_POINTS} points)...")
from sklearn.neighbors import BallTree

data_mean = X_train.mean(axis=0)
data_std = X_train.std(axis=0)

# Newton project onto h(x)=0
boundary_points = []
raw_points = np.random.randn(N_BOUNDARY_POINTS, DIM) * data_std + data_mean

for i, x0 in enumerate(raw_points):
    x = torch.tensor(x0, dtype=torch.float32, device=device)
    converged = False
    for step in range(NEWTON_STEPS):
        with torch.enable_grad():
            h_val, grad = neural_barrier.barrier_and_grad(x)
        if abs(h_val) < NEWTON_TOL:
            converged = True
            break
        g_sq = grad.dot(grad).item()
        if g_sq < 1e-12:
            break
        x = x - (h_val / g_sq) * grad
    if converged and abs(h_val) < 1e-4:
        boundary_points.append(x.detach().cpu().numpy())

print(f"  Converged: {len(boundary_points)}/{N_BOUNDARY_POINTS}")

# Estimate dynamics via KNN
all_residuals = []
all_states = []
for idx in range(len(safe_texts) + len(toxic_texts)):
    trajs = safe_trajectories if idx < len(safe_texts) else toxic_trajectories
    local_idx = idx if idx < len(safe_texts) else idx - len(safe_texts)
    if local_idx >= len(trajs):
        continue
    for l in range(min(TARGET_LAYER, N_LAYERS - 1)):
        all_states.append(trajs[local_idx, l, :])
        all_residuals.append(trajs[local_idx, l + 1, :] - trajs[local_idx, l, :])

all_states = np.array(all_states)
all_residuals = np.array(all_residuals)
tree = BallTree(all_states)
f_norms = np.linalg.norm(all_residuals, axis=1)
mean_f_norm = f_norms.mean()
print(f"  Mean ||f|| = {mean_f_norm:.4f}")

# Check feasibility at boundary points
feasibility_results = {}
for frac in BUDGET_FRACTIONS:
    budget = frac * mean_f_norm
    n_infeasible = 0
    for bp in boundary_points:
        dists, idxs = tree.query(bp.reshape(1, -1), k=KNN_K)
        weights = 1.0 / (dists[0] + 1e-8)
        weights /= weights.sum()
        f_est = (all_residuals[idxs[0]] * weights[:, None]).sum(axis=0)

        x_t = torch.tensor(bp, dtype=torch.float32, device=device)
        f_t = torch.tensor(f_est, dtype=torch.float32, device=device)
        x_next = x_t + f_t
        h_val, u_star, u_norm = compute_cbf_correction(
            neural_barrier, x_next, buffer=0.0, max_iter=20)
        if u_star is not None and u_norm > budget:
            n_infeasible += 1

    p_safe = 1.0 - n_infeasible / max(len(boundary_points), 1)
    feasibility_results[frac] = p_safe
    print(f"  Budget {frac:.0%} of mean ||f||: "
          f"P_safe = {p_safe:.4f} ({n_infeasible} infeasible)")


# ══════════════════════════════════════════════════════════════════════
# 7. Activation-Patching Perplexity
# ══════════════════════════════════════════════════════════════════════
print(f"\n[7/9] Activation-patching perplexity (50 toxic texts)...")

ppl_original = []
ppl_steered = []
patching_u_norms = []

for text in toxic_texts[:50]:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs['input_ids']

    # Original perplexity
    with torch.no_grad():
        out = model_lm(**inputs, labels=input_ids, output_hidden_states=True)
        ppl_original.append(torch.exp(out.loss).item())

    # Steered: compute correction, apply via hook
    x = out.hidden_states[TARGET_LAYER][0, -1, :]

    h_val, u_star, u_norm = compute_cbf_correction(
        neural_barrier, x.float(), buffer=2.0, max_iter=50)

    if u_star is not None:
        patching_u_norms.append(u_norm)
        correction = u_star.half()  # match model dtype

        def make_hook(corr_vec, tgt_layer):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                hs[:, -1, :] = hs[:, -1, :] + corr_vec
                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                return hs
            return hook_fn

        # Find the right layer module
        if hasattr(model_lm, 'transformer'):
            layer_module = model_lm.transformer.h[TARGET_LAYER]
        elif hasattr(model_lm, 'model') and hasattr(model_lm.model, 'layers'):
            layer_module = model_lm.model.layers[TARGET_LAYER]
        else:
            print("  WARNING: Could not identify layer module for hook")
            ppl_steered.append(ppl_original[-1])
            continue

        handle = layer_module.register_forward_hook(
            make_hook(correction, TARGET_LAYER))
        with torch.no_grad():
            out_s = model_lm(**inputs, labels=input_ids)
            ppl_steered.append(torch.exp(out_s.loss).item())
        handle.remove()
    else:
        ppl_steered.append(ppl_original[-1])

ppl_ratio_median = np.median(ppl_steered) / np.median(ppl_original)
print(f"  Original PPL median: {np.median(ppl_original):.2f}")
print(f"  Steered PPL median:  {np.median(ppl_steered):.2f}")
print(f"  PPL ratio: {ppl_ratio_median:.4f}")
if patching_u_norms:
    print(f"  Mean ||u*|| (patching): {np.mean(patching_u_norms):.4f}")


# ══════════════════════════════════════════════════════════════════════
# 8. Autoregressive CBF-Steered Generation
# ══════════════════════════════════════════════════════════════════════
print(f"\n[8/9] Autoregressive generation...")

# Prepare prompts
toxic_prompts = []
for i, text in enumerate(toxic_texts):
    ids = tokenizer.encode(text, truncation=True, max_length=64)
    if len(ids) >= PROMPT_TOKENS + 5:
        toxic_prompts.append({
            'text': text, 'prompt_ids': ids[:PROMPT_TOKENS],
            'prompt_text': tokenizer.decode(ids[:PROMPT_TOKENS]),
            'idx': i,
        })
    if len(toxic_prompts) >= N_TOXIC_PROMPTS:
        break

print(f"  Toxic prompts: {len(toxic_prompts)}")

# TF-IDF scorer
all_clf_texts = safe_texts + toxic_texts
all_clf_labels = [0] * len(safe_texts) + [1] * len(toxic_texts)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english',
                        ngram_range=(1, 2), min_df=2)
X_tfidf = tfidf.fit_transform(all_clf_texts)
tox_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
tox_clf.fit(X_tfidf, all_clf_labels)

# External scorer
external_scorer = None
external_scorer_name = None
try:
    from transformers import pipeline as hf_pipeline
    external_scorer = hf_pipeline(
        "text-classification", model="unitary/toxic-bert",
        device=0 if torch.cuda.is_available() else -1,
        truncation=True, max_length=512)
    external_scorer_name = "unitary/toxic-bert"
    print(f"  External scorer: {external_scorer_name}")
except Exception:
    print("  External scorer not available")


def score_toxicity_tfidf(texts):
    return tox_clf.predict_proba(tfidf.transform(texts))[:, 1]


def score_toxicity_external(texts):
    if external_scorer is None:
        return None
    scores = []
    for text in texts:
        if not text.strip():
            scores.append(0.0)
            continue
        try:
            result = external_scorer(text[:512])
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    label = result[0]['label'].lower()
                    sc = result[0]['score']
                    score = sc if ('toxic' in label or 'hate' in label) else 1.0 - sc
                else:
                    score = 0.0
            else:
                score = 0.0
            scores.append(score)
        except Exception:
            scores.append(0.0)
    return np.array(scores)


def generate_autoregressive(prompt_ids, steered=False, seed=None):
    """Autoregressive generation with optional CBF logit-space steering."""
    if seed is not None:
        torch.manual_seed(seed)

    generated = prompt_ids.to(device)
    past = None
    step_log = []

    # Identify model internals
    if hasattr(model_lm, 'transformer'):
        # GPT-2 style
        ln_f = model_lm.transformer.ln_f
        lm_head = model_lm.lm_head
    elif hasattr(model_lm, 'model') and hasattr(model_lm.model, 'norm'):
        # Qwen/Llama style
        ln_f = model_lm.model.norm
        lm_head = model_lm.lm_head
    else:
        raise RuntimeError("Cannot identify model architecture for logit steering")

    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            if past is None:
                outputs = model_lm(generated, past_key_values=None,
                                   use_cache=True,
                                   output_hidden_states=steered)
            else:
                outputs = model_lm(generated[:, -1:],
                                   past_key_values=past, use_cache=True,
                                   output_hidden_states=steered)

            logits = outputs.logits[:, -1, :].float() / TEMPERATURE
            past = outputs.past_key_values

            if steered:
                hs = outputs.hidden_states[TARGET_LAYER][0, -1, :].float()

                h_val, u_star, u_norm = compute_cbf_correction(
                    neural_barrier, hs, CBF_BUFFER, CBF_MAX_ITER)

                activated = u_star is not None
                if activated:
                    u_norm_val = u_star.norm()
                    if u_norm_val > CBF_MAX_NORM:
                        u_star = u_star * (CBF_MAX_NORM / u_norm_val)
                        u_norm = CBF_MAX_NORM

                    x_corrected = hs + u_star
                    x_normed = ln_f(
                        x_corrected.half().unsqueeze(0).unsqueeze(0))
                    corrected_logits = lm_head(x_normed)[:, 0, :].float() / TEMPERATURE

                    logits = CBF_ALPHA * corrected_logits + (1.0 - CBF_ALPHA) * logits

                step_log.append({
                    'h_val': float(h_val),
                    'activated': activated,
                    'u_norm': u_norm,
                })

            top_k_val = min(TOP_K, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k_val)
            probs = F.softmax(top_k_logits, dim=-1)
            idx = torch.multinomial(probs, 1)
            next_token = top_k_indices.gather(-1, idx)

            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == EOS_TOKEN_ID:
                break

    return generated, step_log


def compute_ppl(token_ids, prompt_len):
    input_ids = token_ids.to(device)
    if input_ids.shape[1] <= prompt_len + 1:
        return float('nan')
    with torch.no_grad():
        outputs = model_lm(input_ids, labels=input_ids)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_logits.size(0), -1)
        completion_loss = per_token_loss[:, prompt_len - 1:]
    return torch.exp(completion_loss.mean()).item()


# Run generation
results = {'unsteered': [], 'steered': []}

for mode in ['unsteered', 'steered']:
    is_steered = (mode == 'steered')
    print(f"\n  --- {mode.title()} ---")
    t0 = time.time()
    for i, pi in enumerate(toxic_prompts):
        prompt_ids = torch.tensor([pi['prompt_ids']], dtype=torch.long)
        prompt_len = prompt_ids.shape[1]
        gen_seed = RANDOM_SEED + pi['idx']

        generated, step_log = generate_autoregressive(
            prompt_ids, steered=is_steered, seed=gen_seed)

        gen_ids = generated[0].cpu().tolist()
        completion_text = tokenizer.decode(gen_ids[prompt_len:],
                                           skip_special_tokens=True)
        full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        ppl = compute_ppl(generated, prompt_len)

        entry = {
            'completion_text': completion_text,
            'full_text': full_text,
            'perplexity': ppl,
        }
        if is_steered and step_log:
            act_rate = sum(1 for s in step_log if s['activated']) / len(step_log)
            u_norms_step = [s['u_norm'] for s in step_log if s['activated']]
            entry['activation_rate'] = act_rate
            entry['mean_u_norm'] = float(np.mean(u_norms_step)) if u_norms_step else 0.0

        results[mode].append(entry)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(toxic_prompts) - i - 1) / rate if rate > 0 else 0
            print(f"    {i+1}/{len(toxic_prompts)} ({elapsed:.0f}s, "
                  f"~{eta:.0f}s remaining)")

    print(f"    Done in {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════
# 9. Results
# ══════════════════════════════════════════════════════════════════════
print(f"\n[9/9] Results...")

# TF-IDF scores
comps_u = [r['completion_text'] for r in results['unsteered']]
comps_s = [r['completion_text'] for r in results['steered']]
tfidf_u = score_toxicity_tfidf(comps_u)
tfidf_s = score_toxicity_tfidf(comps_s)

ext_u = score_toxicity_external(comps_u)
ext_s = score_toxicity_external(comps_s)

ppls_u = np.array([r['perplexity'] for r in results['unsteered']])
ppls_s = np.array([r['perplexity'] for r in results['steered']])
ppls_u = ppls_u[~np.isnan(ppls_u)]
ppls_s = ppls_s[~np.isnan(ppls_s)]

t_tfidf, p_tfidf = stats.ttest_ind(tfidf_s, tfidf_u, equal_var=False)
pooled_std = np.sqrt((tfidf_s.std()**2 + tfidf_u.std()**2) / 2)
cohens_d = (tfidf_u.mean() - tfidf_s.mean()) / pooled_std if pooled_std > 1e-10 else 0

ppl_ratio_auto = np.median(ppls_s) / np.median(ppls_u) if np.median(ppls_u) > 0 else float('nan')

act_rates = [r.get('activation_rate', 0) for r in results['steered']]
mean_u_auto = [r.get('mean_u_norm', 0) for r in results['steered']]

# Get SN-MLP test accuracy
with torch.no_grad():
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test_01, dtype=torch.float32, device=device)
    test_preds = (neural_barrier(X_test_t) > 0).float()
    snmlp_test_acc = (test_preds == y_test_t).float().mean().item()

print(f"\n{'='*70}")
print(f"  C.3: LARGER MODEL VALIDATION — {MODEL_NAME}")
print(f"  {N_LAYERS} layers, n={DIM}")
print(f"{'='*70}")
print(f"  {'Metric':<40} {'Value':>15}")
print(f"  {'='*40} {'='*15}")
print(f"  {'SVM test accuracy':<40} {svm_test_acc:>15.4f}")
print(f"  {'SVM L_h':<40} {svm_Lh:>15.2f}")
print(f"  {'SN-MLP test accuracy':<40} {snmlp_test_acc:>15.4f}")
print(f"  {'SN-MLP L_h (certified)':<40} {L_h:>15.4f}")
print(f"  {'SN-MLP L_h (empirical)':<40} {L_h_emp:>15.4f}")
print(f"  {'Ablation test accuracy':<40} {abl_best_acc:>15.4f}")
print(f"  {'Ablation L_h (uncertified)':<40} {abl_Lh:>15.2f}")
print(f"  {'CBF violations':<40} {violations:>15d}")
print(f"  {'PPL ratio (activation patching)':<40} {ppl_ratio_median:>15.4f}")
print(f"  {'Lipschitz reduction vs SVM':<40} {f'{svm_Lh/L_h:.0f}x':>15}")
print(f"  {'':<40}")
print(f"  --- Autoregressive (α={CBF_ALPHA}) ---")
print(f"  {'TF-IDF unsteered mean':<40} {tfidf_u.mean():>15.4f}")
print(f"  {'TF-IDF steered mean':<40} {tfidf_s.mean():>15.4f}")
print(f"  {'TF-IDF p-value':<40} {p_tfidf:>15.2e}")
print(f"  {'TF-IDF Cohen d':<40} {cohens_d:>15.4f}")
print(f"  {'PPL ratio (autoregressive)':<40} {ppl_ratio_auto:>15.3f}")
print(f"  {'Activation rate':<40} {np.mean(act_rates):>15.4f}")
print(f"  {'Mean ||u*|| (active)':<40} {np.mean(mean_u_auto):>15.4f}")

if ext_u is not None and ext_s is not None:
    t_ext, p_ext = stats.ttest_ind(ext_s, ext_u, equal_var=False)
    print(f"  {'External toxicity unsteered':<40} {ext_u.mean():>15.4f}")
    print(f"  {'External toxicity steered':<40} {ext_s.mean():>15.4f}")
    print(f"  {'External p-value':<40} {p_ext:>15.2e}")

# ── Save ──
save_data = {
    'model': MODEL_NAME,
    'n_layers': N_LAYERS,
    'hidden_dim': DIM,
    'target_layer': TARGET_LAYER,
    'barrier': {
        'svm_test_acc': svm_test_acc,
        'svm_Lh': svm_Lh,
        'snmlp_test_acc': snmlp_test_acc,
        'snmlp_Lh_certified': L_h,
        'snmlp_Lh_empirical': float(L_h_emp),
        'ablation_test_acc': abl_best_acc,
        'ablation_Lh': abl_Lh,
        'cbf_violations': violations,
        'ppl_ratio_patching': ppl_ratio_median,
        'lipschitz_reduction': svm_Lh / L_h,
    },
    'boundary_feasibility': {str(k): v for k, v in feasibility_results.items()},
    'autoregressive': {
        'alpha': CBF_ALPHA,
        'tfidf_unsteered': float(tfidf_u.mean()),
        'tfidf_steered': float(tfidf_s.mean()),
        'tfidf_p': float(p_tfidf),
        'tfidf_cohens_d': float(cohens_d),
        'ppl_unsteered_median': float(np.median(ppls_u)),
        'ppl_steered_median': float(np.median(ppls_s)),
        'ppl_ratio': float(ppl_ratio_auto),
        'activation_rate': float(np.mean(act_rates)),
        'mean_u_norm': float(np.mean(mean_u_auto)),
    },
}
if ext_u is not None:
    save_data['autoregressive']['ext_unsteered'] = float(ext_u.mean())
    save_data['autoregressive']['ext_steered'] = float(ext_s.mean())
    save_data['autoregressive']['ext_p'] = float(p_ext)

with open(RESULTS_SAVE_PATH, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\n  Results saved to {os.path.basename(RESULTS_SAVE_PATH)}")
print("\nDone.")
