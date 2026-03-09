#!/usr/bin/env python3
"""
RepEng/ActAdd Baseline Experiment
===================================
Compares Activation Addition (fixed steering vector) against CBF logit-space
steering on the same 200 toxic + 100 safe prompts from the autoregressive
experiment.

Steering vector: mean(safe hidden states) - mean(toxic hidden states) at
layer 12, following Zou et al. (2023) and Turner et al. (2023).

This script reuses the same data pipeline, barrier, evaluation metrics,
and random seeds as autoregressive_experiment.py for a fair comparison.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# Configuration (identical to autoregressive_experiment.py)
# ══════════════════════════════════════════════════════════════════════
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Data
N_SAFE_TARGET = 500
N_TOXIC_TARGET = 500
MAX_TEXT_LEN = 200
TRAIN_TEST_RATIO = 0.2

# Barrier (for comparison)
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

# CBF params (for comparison arm)
CBF_BUFFER = 0.3
CBF_MAX_ITER = 20
CBF_ALPHA = 0.25
CBF_MAX_NORM = 0.5

# RepEng steering strengths to sweep
# We match by effective logit-blend alpha to be comparable
REPENG_ALPHAS = [0.15, 0.25, 0.50]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BARRIER_SAVE_PATH = os.path.join(SCRIPT_DIR, 'trained_barrier.pt')
RESULTS_SAVE_PATH = os.path.join(SCRIPT_DIR, 'repeng_results.json')

# ══════════════════════════════════════════════════════════════════════
# 1. Load Data and Model (same as autoregressive_experiment.py)
# ══════════════════════════════════════════════════════════════════════
print("\n[1/8] Loading data and model...")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

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

# ── Load GPT-2 ──
t0 = time.time()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
gpt2_base = GPT2Model.from_pretrained('gpt2')
gpt2_base.eval()
gpt2_base = gpt2_base.to(device)
gpt2_lm = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_lm.eval()
gpt2_lm = gpt2_lm.to(device)

N_LAYERS = gpt2_base.config.n_layer  # 12
DIM = gpt2_base.config.n_embd        # 768
EOS_TOKEN_ID = tokenizer.eos_token_id

print(f"  GPT-2 loaded in {time.time()-t0:.1f}s: "
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
        outputs.hidden_states[l][0, -1, :].cpu().numpy()
        for l in range(N_LAYERS + 1)
    ])
    return trajectory


safe_trajectories = np.array([
    extract_hidden_trajectory(t, gpt2_base, tokenizer) for t in safe_texts
])
toxic_trajectories = np.array([
    extract_hidden_trajectory(t, gpt2_base, tokenizer) for t in toxic_texts
])
print(f"  Extraction done in {time.time()-t0:.1f}s")

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
# 2. Compute RepEng Steering Vector
# ══════════════════════════════════════════════════════════════════════
print("\n[2/8] Computing RepEng steering vector...")


class NeuralBarrier(nn.Module):
    """Spectrally-normalized MLP barrier (same architecture as C.2)."""

    def __init__(self, input_dim=768, hidden_dims=None, input_mean=None,
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
            x_t = x.unsqueeze(0) if x.dim() == 1 else x
        x_t = x_t.detach().requires_grad_(True)
        h = self.forward(x_t)
        h.backward()
        grad = x_t.grad[0].detach()
        return h.item(), grad


# ── Load barrier → get TARGET_LAYER ──
if not os.path.exists(BARRIER_SAVE_PATH):
    print("  ERROR: trained_barrier.pt not found. "
          "Run autoregressive_experiment.py first.")
    sys.exit(1)

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
TARGET_LAYER = checkpoint.get('target_layer', 12)
saved_acc = checkpoint.get('test_acc', -1)
print(f"  Loaded barrier: layer={TARGET_LAYER}, test_acc={saved_acc:.4f}")

neural_barrier = neural_barrier.to(device)
L_h = neural_barrier.compute_lipschitz_bound()
print(f"  Certified L_h = {L_h:.4f}")

# ── Compute steering vector at barrier's target layer ──
# Following Zou et al. (2023): steering = mean(safe) - mean(toxic)
X_safe_train = safe_trajectories[safe_idx_train, TARGET_LAYER, :]
X_toxic_train = toxic_trajectories[toxic_idx_train, TARGET_LAYER, :]
X_train = np.vstack([X_safe_train, X_toxic_train])

steering_vector = X_safe_train.mean(axis=0) - X_toxic_train.mean(axis=0)
steering_norm = np.linalg.norm(steering_vector)
steering_direction = steering_vector / steering_norm  # unit vector

print(f"  Steering vector norm: {steering_norm:.4f}")
print(f"  Direction: mean(safe) - mean(toxic) at layer {TARGET_LAYER}")

steering_direction_t = torch.tensor(
    steering_direction, dtype=torch.float32, device=device)


# ══════════════════════════════════════════════════════════════════════
# 3. CBF correction function (for comparison arm)
# ══════════════════════════════════════════════════════════════════════


def compute_cbf_correction(barrier, x_t, buffer=1.0, max_iter=20):
    """Minimum-norm CBF correction u* such that h(x + u*) >= buffer."""
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


# ══════════════════════════════════════════════════════════════════════
# 4. Generation Functions
# ══════════════════════════════════════════════════════════════════════


def generate_completion_repeng(model, input_ids, max_new_tokens, temperature,
                               top_k, seed=None, steering_dir=None,
                               steering_scale=1.0, target_layer=None,
                               alpha=0.25):
    """
    Autoregressive generation with ActAdd steering.

    At each step:
    1. Extract hidden state at target_layer
    2. Add scaled steering direction: x_corrected = x + scale * direction
    3. Recompute logits via ln_f -> lm_head
    4. Blend: alpha * corrected + (1-alpha) * original
    """
    if seed is not None:
        torch.manual_seed(seed)

    generated = input_ids.to(device)
    past = None
    step_log = []
    use_steering = steering_dir is not None and target_layer is not None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if past is None:
                outputs = model(generated, past_key_values=None,
                                use_cache=True,
                                output_hidden_states=use_steering)
            else:
                outputs = model(generated[:, -1:],
                                past_key_values=past, use_cache=True,
                                output_hidden_states=use_steering)

            logits = outputs.logits[:, -1, :] / temperature
            past = outputs.past_key_values

            if use_steering:
                hs = outputs.hidden_states[target_layer][0, -1, :]

                # Fixed steering vector addition (RepEng/ActAdd)
                u_repeng = steering_dir * steering_scale
                u_norm = float(u_repeng.norm().item())

                x_corrected = hs + u_repeng
                x_normed = model.transformer.ln_f(
                    x_corrected.unsqueeze(0).unsqueeze(0))
                corrected_logits = model.lm_head(x_normed)[:, 0, :] / temperature

                logits = alpha * corrected_logits + (1.0 - alpha) * logits

                step_log.append({
                    'u_norm': u_norm,
                    'activated': True,
                })

            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, top_k_val)
                probs = F.softmax(top_k_logits, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, idx)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == EOS_TOKEN_ID:
                break

    return generated, step_log


def generate_completion_cbf(model, input_ids, max_new_tokens, temperature,
                            top_k, seed=None, barrier=None, target_layer=None,
                            buffer=1.0, max_iter=20, alpha=0.5, max_norm=0.5):
    """CBF logit-space steering (identical to autoregressive_experiment.py)."""
    if seed is not None:
        torch.manual_seed(seed)

    generated = input_ids.to(device)
    past = None
    step_log = []
    use_barrier = barrier is not None and target_layer is not None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if past is None:
                outputs = model(generated, past_key_values=None,
                                use_cache=True,
                                output_hidden_states=use_barrier)
            else:
                outputs = model(generated[:, -1:],
                                past_key_values=past, use_cache=True,
                                output_hidden_states=use_barrier)

            logits = outputs.logits[:, -1, :] / temperature
            past = outputs.past_key_values

            if use_barrier:
                hs = outputs.hidden_states[target_layer][0, -1, :]
                h_val, u_star, u_norm = compute_cbf_correction(
                    barrier, hs, buffer, max_iter)

                activated = u_star is not None
                if activated:
                    u_norm_val = u_star.norm()
                    if u_norm_val > max_norm:
                        u_star = u_star * (max_norm / u_norm_val)
                        u_norm = max_norm

                    x_corrected = hs + u_star
                    x_normed = model.transformer.ln_f(
                        x_corrected.unsqueeze(0).unsqueeze(0))
                    corrected_logits = model.lm_head(x_normed)[:, 0, :] / temperature
                    logits = alpha * corrected_logits + (1.0 - alpha) * logits

                step_log.append({
                    'h_val': float(h_val),
                    'activated': activated,
                    'u_norm': u_norm,
                })

            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, top_k_val)
                probs = F.softmax(top_k_logits, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, idx)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == EOS_TOKEN_ID:
                break

    return generated, step_log


def compute_perplexity(model, token_ids, prompt_len):
    """Teacher-forced perplexity on completion tokens only."""
    input_ids = token_ids.to(device)
    if input_ids.shape[1] <= prompt_len + 1:
        return float('nan')

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_logits.size(0), -1)

        completion_loss = per_token_loss[:, prompt_len - 1:]
        mean_loss = completion_loss.mean()

    return torch.exp(mean_loss).item()


# ══════════════════════════════════════════════════════════════════════
# 5. Prepare Prompts (identical to autoregressive_experiment.py)
# ══════════════════════════════════════════════════════════════════════
print("\n[3/8] Preparing prompts...")

toxic_prompts = []
for i, text in enumerate(toxic_texts):
    ids = tokenizer.encode(text, truncation=True, max_length=64)
    if len(ids) >= PROMPT_TOKENS + 5:
        toxic_prompts.append({
            'text': text, 'prompt_ids': ids[:PROMPT_TOKENS],
            'prompt_text': tokenizer.decode(ids[:PROMPT_TOKENS]),
            'full_ids': ids, 'idx': i,
        })
    if len(toxic_prompts) >= N_TOXIC_PROMPTS:
        break

safe_prompts = []
for i, text in enumerate(safe_texts):
    ids = tokenizer.encode(text, truncation=True, max_length=64)
    if len(ids) >= PROMPT_TOKENS + 5:
        safe_prompts.append({
            'text': text, 'prompt_ids': ids[:PROMPT_TOKENS],
            'prompt_text': tokenizer.decode(ids[:PROMPT_TOKENS]),
            'full_ids': ids, 'idx': i,
        })
    if len(safe_prompts) >= N_SAFE_PROMPTS:
        break

print(f"  Toxic prompts: {len(toxic_prompts)}")
print(f"  Safe prompts:  {len(safe_prompts)}")

# ══════════════════════════════════════════════════════════════════════
# 6. Toxicity Scorers (identical to autoregressive_experiment.py)
# ══════════════════════════════════════════════════════════════════════
print("\n  Training TF-IDF toxicity classifier...")

all_clf_texts = safe_texts + toxic_texts
all_clf_labels = [0] * len(safe_texts) + [1] * len(toxic_texts)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english',
                        ngram_range=(1, 2), min_df=2)
X_tfidf = tfidf.fit_transform(all_clf_texts)
tox_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
tox_clf.fit(X_tfidf, all_clf_labels)

external_scorer = None
external_scorer_name = None
try:
    from transformers import pipeline
    external_scorer = pipeline(
        "text-classification", model="s-nlp/roberta_toxigen",
        device=0 if torch.cuda.is_available() else -1,
        truncation=True, max_length=512,
    )
    external_scorer_name = "s-nlp/roberta_toxigen"
    print(f"  External scorer loaded: {external_scorer_name}")
except Exception:
    try:
        from transformers import pipeline
        external_scorer = pipeline(
            "text-classification", model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1,
            truncation=True, max_length=512,
        )
        external_scorer_name = "unitary/toxic-bert"
        print(f"  External scorer loaded: {external_scorer_name}")
    except Exception:
        print("  External toxicity model not available")


def score_toxicity_tfidf(texts):
    X = tfidf.transform(texts)
    return tox_clf.predict_proba(X)[:, 1]


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
                if isinstance(result[0], list):
                    toxic_scores = [r['score'] for r in result[0]
                                    if 'toxic' in r['label'].lower()]
                    score = max(toxic_scores) if toxic_scores else 0.0
                elif isinstance(result[0], dict):
                    label = result[0]['label'].lower()
                    sc = result[0]['score']
                    if 'toxic' in label or 'hate' in label:
                        score = sc
                    else:
                        score = 1.0 - sc
                else:
                    score = 0.0
            else:
                score = 0.0
            scores.append(score)
        except Exception:
            scores.append(0.0)
    return np.array(scores)


def evaluate_barrier_on_text(texts, model_base, tok, barrier, target_layer):
    h_values = []
    model_base.eval()
    for text in texts:
        if not text.strip():
            h_values.append(0.0)
            continue
        inputs = tok(text, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_base(**inputs, output_hidden_states=True)
        x = outputs.hidden_states[target_layer][0, -1, :]
        with torch.enable_grad():
            h_val, _ = barrier.barrier_and_grad(x)
        h_values.append(h_val)
    return np.array(h_values)


# ══════════════════════════════════════════════════════════════════════
# 7. Determine RepEng Steering Scale
# ══════════════════════════════════════════════════════════════════════
# Match the mean ||u*|| from CBF (0.33) so the comparison is fair:
# the RepEng vector will have approximately the same L2 correction norm.
# We also try the CBF max norm (0.50) and a smaller value.
REPENG_SCALE = CBF_MAX_NORM  # 0.5 — same cap as CBF
print(f"\n  RepEng steering scale: {REPENG_SCALE:.2f}")
print(f"  Effective ||u||: {REPENG_SCALE:.4f} (fixed at every step)")


# ══════════════════════════════════════════════════════════════════════
# 8. Generate and Evaluate
# ══════════════════════════════════════════════════════════════════════

def run_generation_repeng(prompts_list, label, alpha, steering_dir,
                          steering_scale, target_layer):
    """Generate completions with RepEng/ActAdd steering."""
    results = []
    t0 = time.time()
    for i, prompt_info in enumerate(prompts_list):
        prompt_ids = torch.tensor([prompt_info['prompt_ids']], dtype=torch.long)
        prompt_len = prompt_ids.shape[1]
        gen_seed = RANDOM_SEED + prompt_info['idx']

        generated, step_log = generate_completion_repeng(
            gpt2_lm, prompt_ids, MAX_NEW_TOKENS, TEMPERATURE, TOP_K,
            seed=gen_seed, steering_dir=steering_dir,
            steering_scale=steering_scale, target_layer=target_layer,
            alpha=alpha)

        gen_ids = generated[0].cpu().tolist()
        completion_ids = gen_ids[prompt_len:]
        completion_text = tokenizer.decode(completion_ids,
                                           skip_special_tokens=True)
        full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        ppl = compute_perplexity(gpt2_lm, generated, prompt_len)

        u_norms = [s['u_norm'] for s in step_log if s.get('activated', False)]

        results.append({
            'prompt_text': prompt_info['prompt_text'],
            'completion_text': completion_text,
            'full_text': full_text,
            'n_completion_tokens': len(completion_ids),
            'perplexity': ppl,
            'mean_u_norm': float(np.mean(u_norms)) if u_norms else 0.0,
        })

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts_list) - i - 1) / rate if rate > 0 else 0
            print(f"    [{label}] {i+1}/{len(prompts_list)} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s remaining)")

    print(f"    [{label}] Done in {time.time()-t0:.1f}s")
    return results


def run_generation_cbf(prompts_list, label):
    """Generate completions with CBF steering (comparison arm)."""
    results = []
    t0 = time.time()
    for i, prompt_info in enumerate(prompts_list):
        prompt_ids = torch.tensor([prompt_info['prompt_ids']], dtype=torch.long)
        prompt_len = prompt_ids.shape[1]
        gen_seed = RANDOM_SEED + prompt_info['idx']

        generated, step_log = generate_completion_cbf(
            gpt2_lm, prompt_ids, MAX_NEW_TOKENS, TEMPERATURE, TOP_K,
            seed=gen_seed, barrier=neural_barrier, target_layer=TARGET_LAYER,
            buffer=CBF_BUFFER, max_iter=CBF_MAX_ITER, alpha=CBF_ALPHA,
            max_norm=CBF_MAX_NORM)

        gen_ids = generated[0].cpu().tolist()
        completion_ids = gen_ids[prompt_len:]
        completion_text = tokenizer.decode(completion_ids,
                                           skip_special_tokens=True)
        full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        ppl = compute_perplexity(gpt2_lm, generated, prompt_len)

        u_norms = [s['u_norm'] for s in step_log if s.get('activated', False)]

        results.append({
            'prompt_text': prompt_info['prompt_text'],
            'completion_text': completion_text,
            'full_text': full_text,
            'n_completion_tokens': len(completion_ids),
            'perplexity': ppl,
            'mean_u_norm': float(np.mean(u_norms)) if u_norms else 0.0,
        })

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts_list) - i - 1) / rate if rate > 0 else 0
            print(f"    [{label}] {i+1}/{len(prompts_list)} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s remaining)")

    print(f"    [{label}] Done in {time.time()-t0:.1f}s")
    return results


def run_generation_baseline(prompts_list, label):
    """Unsteered generation."""
    results = []
    t0 = time.time()
    for i, prompt_info in enumerate(prompts_list):
        prompt_ids = torch.tensor([prompt_info['prompt_ids']], dtype=torch.long)
        prompt_len = prompt_ids.shape[1]
        gen_seed = RANDOM_SEED + prompt_info['idx']

        generated, _ = generate_completion_cbf(
            gpt2_lm, prompt_ids, MAX_NEW_TOKENS, TEMPERATURE, TOP_K,
            seed=gen_seed)

        gen_ids = generated[0].cpu().tolist()
        completion_ids = gen_ids[prompt_len:]
        completion_text = tokenizer.decode(completion_ids,
                                           skip_special_tokens=True)
        full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        ppl = compute_perplexity(gpt2_lm, generated, prompt_len)

        results.append({
            'prompt_text': prompt_info['prompt_text'],
            'completion_text': completion_text,
            'full_text': full_text,
            'n_completion_tokens': len(completion_ids),
            'perplexity': ppl,
            'mean_u_norm': 0.0,
        })

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts_list) - i - 1) / rate if rate > 0 else 0
            print(f"    [{label}] {i+1}/{len(prompts_list)} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s remaining)")

    print(f"    [{label}] Done in {time.time()-t0:.1f}s")
    return results


def evaluate_completions(results_list, label):
    """Evaluate toxicity and quality."""
    completions = [r['completion_text'] for r in results_list]
    full_texts = [r['full_text'] for r in results_list]

    tfidf_scores = score_toxicity_tfidf(completions)
    ext_scores = score_toxicity_external(completions)
    barrier_scores = evaluate_barrier_on_text(
        full_texts, gpt2_base, tokenizer, neural_barrier, TARGET_LAYER)
    ppls = np.array([r['perplexity'] for r in results_list])
    ppls_valid = ppls[~np.isnan(ppls)]

    eval_result = {
        'label': label,
        'n': len(results_list),
        'tfidf_scores': tfidf_scores,
        'tfidf_mean': float(np.mean(tfidf_scores)),
        'tfidf_toxicity_rate': float(np.mean(tfidf_scores > 0.5)),
        'barrier_scores': barrier_scores,
        'barrier_mean': float(np.mean(barrier_scores)),
        'barrier_safe_rate': float(np.mean(barrier_scores > 0)),
        'ppl_mean': float(np.mean(ppls_valid)) if len(ppls_valid) > 0 else 0,
        'ppl_median': float(np.median(ppls_valid)) if len(ppls_valid) > 0 else 0,
        'ppls': ppls_valid,
    }

    if ext_scores is not None:
        eval_result['ext_scores'] = ext_scores
        eval_result['ext_mean'] = float(np.mean(ext_scores))
        eval_result['ext_toxicity_rate'] = float(np.mean(ext_scores > 0.5))

    return eval_result


# ══════════════════════════════════════════════════════════════════════
# Run All Arms
# ══════════════════════════════════════════════════════════════════════
print("\n[4/8] Generating unsteered baseline...")
toxic_unsteered = run_generation_baseline(toxic_prompts, "Unsteered")
toxic_unsteered_eval = evaluate_completions(toxic_unsteered, "Unsteered")

print("\n[5/8] Generating CBF-steered (α=0.25)...")
toxic_cbf = run_generation_cbf(toxic_prompts, "CBF-α=0.25")
toxic_cbf_eval = evaluate_completions(toxic_cbf, "CBF-α=0.25")

repeng_results = {}
for rep_alpha in REPENG_ALPHAS:
    print(f"\n[6/8] Generating RepEng-steered (α={rep_alpha})...")
    toxic_repeng = run_generation_repeng(
        toxic_prompts, f"RepEng-α={rep_alpha}", alpha=rep_alpha,
        steering_dir=steering_direction_t, steering_scale=REPENG_SCALE,
        target_layer=TARGET_LAYER)
    toxic_repeng_eval = evaluate_completions(
        toxic_repeng, f"RepEng-α={rep_alpha}")
    repeng_results[rep_alpha] = {
        'raw': toxic_repeng,
        'eval': toxic_repeng_eval,
    }


# ══════════════════════════════════════════════════════════════════════
# Statistical Comparison
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  REPENG/ACTADD vs CBF BASELINE COMPARISON")
print(f"{'='*70}")
print(f"  Model: GPT-2 ({N_LAYERS} layers, n={DIM})")
print(f"  Prompts: {len(toxic_prompts)} toxic, {PROMPT_TOKENS} prompt tokens, "
      f"{MAX_NEW_TOKENS} gen tokens")
print(f"  CBF: α={CBF_ALPHA}, buffer={CBF_BUFFER}, max_norm={CBF_MAX_NORM}")
print(f"  RepEng: scale={REPENG_SCALE}, direction=mean(safe)-mean(toxic)")
print(f"{'='*70}\n")

print(f"  {'Method':<25} {'TF-IDF↓':>10} {'TF-IDF p':>12} "
      f"{'Ext↓':>10} {'Ext p':>12} {'PPL↑':>8} {'PPL ratio':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*8} {'-'*10}")

# Unsteered
print(f"  {'Unsteered':<25} "
      f"{toxic_unsteered_eval['tfidf_mean']:>10.4f} "
      f"{'---':>12} "
      f"{toxic_unsteered_eval.get('ext_mean', 0):>10.4f} "
      f"{'---':>12} "
      f"{toxic_unsteered_eval['ppl_median']:>8.1f} "
      f"{'---':>10}")

# CBF
t_cbf, p_cbf = stats.ttest_ind(
    toxic_cbf_eval['tfidf_scores'], toxic_unsteered_eval['tfidf_scores'],
    equal_var=False)
ext_p_cbf = '---'
if 'ext_scores' in toxic_cbf_eval:
    _, p_ext_cbf = stats.ttest_ind(
        toxic_cbf_eval['ext_scores'], toxic_unsteered_eval['ext_scores'],
        equal_var=False)
    ext_p_cbf = f"{p_ext_cbf:.2e}"
ppl_ratio_cbf = toxic_cbf_eval['ppl_median'] / toxic_unsteered_eval['ppl_median']
print(f"  {'CBF (α=0.25)':<25} "
      f"{toxic_cbf_eval['tfidf_mean']:>10.4f} "
      f"{p_cbf:>12.2e} "
      f"{toxic_cbf_eval.get('ext_mean', 0):>10.4f} "
      f"{ext_p_cbf:>12} "
      f"{toxic_cbf_eval['ppl_median']:>8.1f} "
      f"{ppl_ratio_cbf:>10.3f}")

# RepEng at each alpha
for rep_alpha in REPENG_ALPHAS:
    re = repeng_results[rep_alpha]['eval']
    t_re, p_re = stats.ttest_ind(
        re['tfidf_scores'], toxic_unsteered_eval['tfidf_scores'],
        equal_var=False)
    ext_p_re = '---'
    if 'ext_scores' in re:
        _, p_ext_re = stats.ttest_ind(
            re['ext_scores'], toxic_unsteered_eval['ext_scores'],
            equal_var=False)
        ext_p_re = f"{p_ext_re:.2e}"
    ppl_ratio_re = re['ppl_median'] / toxic_unsteered_eval['ppl_median']
    print(f"  {f'RepEng (α={rep_alpha})':<25} "
          f"{re['tfidf_mean']:>10.4f} "
          f"{p_re:>12.2e} "
          f"{re.get('ext_mean', 0):>10.4f} "
          f"{ext_p_re:>12} "
          f"{re['ppl_median']:>8.1f} "
          f"{ppl_ratio_re:>10.3f}")

# ── Head-to-head: CBF vs best-matched RepEng ──
print(f"\n  --- Head-to-Head: CBF vs RepEng (both at α=0.25) ---")
re_025 = repeng_results[0.25]['eval']
t_head, p_head = stats.ttest_ind(
    toxic_cbf_eval['tfidf_scores'], re_025['tfidf_scores'],
    equal_var=False)

# Cohen's d between CBF and RepEng
pooled_std_head = np.sqrt(
    (toxic_cbf_eval['tfidf_scores'].std()**2 +
     re_025['tfidf_scores'].std()**2) / 2
)
cohens_d_head = (
    (re_025['tfidf_mean'] - toxic_cbf_eval['tfidf_mean'])
    / pooled_std_head if pooled_std_head > 1e-10 else 0
)
print(f"  CBF TF-IDF mean:    {toxic_cbf_eval['tfidf_mean']:.4f}")
print(f"  RepEng TF-IDF mean: {re_025['tfidf_mean']:.4f}")
print(f"  Welch's t = {t_head:.4f}, p = {p_head:.2e}")
print(f"  Cohen's d = {cohens_d_head:.4f}")
print(f"  CBF PPL median:    {toxic_cbf_eval['ppl_median']:.1f}")
print(f"  RepEng PPL median: {re_025['ppl_median']:.1f}")

# ══════════════════════════════════════════════════════════════════════
# Save Results
# ══════════════════════════════════════════════════════════════════════
save_data = {
    'config': {
        'n_toxic_prompts': len(toxic_prompts),
        'prompt_tokens': PROMPT_TOKENS,
        'max_new_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE,
        'top_k': TOP_K,
        'target_layer': TARGET_LAYER,
        'repeng_scale': REPENG_SCALE,
        'repeng_steering_norm': steering_norm,
        'cbf_alpha': CBF_ALPHA,
        'cbf_buffer': CBF_BUFFER,
        'cbf_max_norm': CBF_MAX_NORM,
    },
    'unsteered': {
        'tfidf_mean': toxic_unsteered_eval['tfidf_mean'],
        'tfidf_toxicity_rate': toxic_unsteered_eval['tfidf_toxicity_rate'],
        'ppl_median': toxic_unsteered_eval['ppl_median'],
        'ext_mean': toxic_unsteered_eval.get('ext_mean', None),
    },
    'cbf': {
        'tfidf_mean': toxic_cbf_eval['tfidf_mean'],
        'tfidf_toxicity_rate': toxic_cbf_eval['tfidf_toxicity_rate'],
        'ppl_median': toxic_cbf_eval['ppl_median'],
        'tfidf_p': float(p_cbf),
        'ext_mean': toxic_cbf_eval.get('ext_mean', None),
    },
    'repeng': {},
}

for rep_alpha in REPENG_ALPHAS:
    re = repeng_results[rep_alpha]['eval']
    t_re, p_re = stats.ttest_ind(
        re['tfidf_scores'], toxic_unsteered_eval['tfidf_scores'],
        equal_var=False)
    ppl_ratio_re = re['ppl_median'] / toxic_unsteered_eval['ppl_median']
    save_data['repeng'][str(rep_alpha)] = {
        'tfidf_mean': re['tfidf_mean'],
        'tfidf_toxicity_rate': re['tfidf_toxicity_rate'],
        'ppl_median': re['ppl_median'],
        'ppl_ratio': ppl_ratio_re,
        'tfidf_p': float(p_re),
        'ext_mean': re.get('ext_mean', None),
    }

save_data['head_to_head'] = {
    'cbf_vs_repeng_025_t': float(t_head),
    'cbf_vs_repeng_025_p': float(p_head),
    'cbf_vs_repeng_025_cohens_d': float(cohens_d_head),
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return super().default(obj)

with open(RESULTS_SAVE_PATH, 'w') as f:
    json.dump(save_data, f, indent=2, cls=NumpyEncoder)
print(f"\n  Results saved to {os.path.basename(RESULTS_SAVE_PATH)}")
print("\nDone.")
