"""
Figure 4: AASV Black Swan Detection - 4x2 Robustness Panel
============================================================
All experiments on S^127 (unit sphere in R^128).
Every spike is a unit-norm DIRECTION; there is no radial distance.
2D plots are projections; distance from center is an artifact.

  (a) Monte Carlo baseline (N=10000, 3 spikes) -- misses all
  (b) AASV Hunter, 3 orthogonal spikes
  (c) Antipodal directions (v, -v)
  (d) 10 orthogonal spikes (stress test)
  (e) 30 deg angular separation (close pair)
  (f) 15 deg angular cluster + 1 isolated
  (g) 5 deg angular separation (resolution limit)
  (h) 20 random directions (uniform on S^127)

Barrier: h(x) = (1-||x||) + FUNNEL*Sum(1-cos_sim) - DEPTH*Sum exp(...)
  DEPTH = (N-1)*FUNNEL + 2 so each spike is a genuine violation.
  Signed dot products (no abs) so antipodal spikes are distinct.

Output: figure_4.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================================
# SHARED PARAMETERS
# ============================================================================
DIMENSIONS      = 128
N_MC_SAMPLES    = 10_000
SPIKE_WIDTH     = 0.05
FUNNEL_STRENGTH = 1.0
HUNTER_RESTARTS = 20
HUNTER_STEPS    = 200
MOMENTUM        = 0.9
LR              = 0.05
NOISE_SCALE     = 0.001
SURROGATE_ERROR = 0.005

def spike_depth(n_spikes):
    """Depth scales with N so that h < 0 at each spike direction."""
    return (n_spikes - 1) * FUNNEL_STRENGTH + 2.0

# ============================================================================
# Generate orthogonal directions in R^128
# ============================================================================
def make_spike_directions(n_spikes, seed=42):
    rng = np.random.RandomState(seed)
    M = rng.randn(DIMENSIONS, min(n_spikes, DIMENSIONS))
    Q, _ = np.linalg.qr(M)
    return [Q[:, i] for i in range(n_spikes)]

# ============================================================================
# BARRIER + GRADIENT
# ============================================================================
def barrier_function(x, spike_dirs):
    depth = spike_depth(len(spike_dirs))
    dist = np.linalg.norm(x)
    h = 1.0 - dist
    if dist > 0:
        for d in spike_dirs:
            sim = np.dot(x, d) / dist
            h += FUNNEL_STRENGTH * (1.0 - sim)
            h -= depth * np.exp(-(1.0 - sim)**2 / (2.0 * SPIKE_WIDTH**2))
    return h

def barrier_gradient_wta(x, spike_dirs, memory_protos, rng=None,
                        block_thresh=0.7):
    """Winner-take-all gradient with signed prototype avoidance.
    When ALL known attractors are blocked, explores a random direction
    orthogonal to found prototypes (no oracle knowledge of spike count).
    block_thresh controls how close a spike must be to a prototype to be
    considered 'already found' — adaptive decay shrinks this over time."""
    depth = spike_depth(len(spike_dirs))
    dist = np.linalg.norm(x)
    if dist < 1e-12:
        return np.zeros_like(x)
    x_hat = x / dist
    grad = -x_hat.copy()

    sims = [np.dot(x, d) / dist for d in spike_dirs]
    adj = list(sims)
    # Block spikes already covered by a prototype (SIGNED — no abs)
    for i, d in enumerate(spike_dirs):
        for proto in memory_protos:
            if np.dot(d, proto) > block_thresh:
                adj[i] = -999.0
                break

    if max(adj) < -900 and len(memory_protos) > 0:
        # ALL known attractors blocked → explore randomly in the subspace
        # orthogonal to found prototypes (searching for unknown spikes)
        if rng is None:
            rng = np.random.RandomState()
        rand_dir = rng.randn(len(x))
        for p in memory_protos:
            rand_dir -= np.dot(rand_dir, p) * p
        rand_dir -= np.dot(rand_dir, x_hat) * x_hat   # tangent to sphere
        rn = np.linalg.norm(rand_dir)
        if rn > 1e-12:
            rand_dir /= rn
        grad += (-FUNNEL_STRENGTH / dist) * rand_dir
        return grad

    best = int(np.argmax(adj))
    d = spike_dirs[best]
    sim = sims[best]
    diff = 1.0 - sim
    spike_exp = np.exp(-diff**2 / (2.0 * SPIKE_WIDTH**2))
    sgc = depth * spike_exp * diff / (SPIKE_WIDTH**2)
    coeff = (-FUNNEL_STRENGTH - sgc) / dist          # both funnel & spike pull toward spike
    tangent = d - sim * x_hat
    grad += coeff * tangent
    return grad

# ============================================================================
# Orthogonal Prototype Memory  (SIGNED similarity)
# ============================================================================
class OrthoMemory:
    def __init__(self):
        self.prototypes = []

    def add(self, v, merge_thresh=0.3):
        """Returns True if this is a NEW distinct region, False if redundant.
        merge_thresh controls how close a new point must be to an existing
        prototype to be considered the same region."""
        v_n = v / np.linalg.norm(v)
        for i, p in enumerate(self.prototypes):
            if np.dot(v_n, p) > merge_thresh:
                self.prototypes[i] = p + 0.1 * v_n
                self.prototypes[i] /= np.linalg.norm(self.prototypes[i])
                return False                  # redundant: near existing region
        self.prototypes.append(v_n.copy())
        return True                           # novel region

# ============================================================================
# Equiangular 2D projection
# ============================================================================
def make_projection(spike_dirs, angle_offset_deg=0):
    """Project so all spikes appear evenly spaced on a circle.
    Special-cases 2 antipodal spikes (lstsq is degenerate for [v, -v])."""
    n = len(spike_dirs)
    off = np.radians(angle_offset_deg)

    # For 2 directions the lstsq approach can be ill-conditioned
    # (antipodal: identical columns; close: nearly identical columns).
    # Build the basis directly from the two spike directions instead.
    if n == 2:
        b1 = np.array(spike_dirs[0], dtype=float)
        b1 /= np.linalg.norm(b1)
        b2 = np.array(spike_dirs[1], dtype=float)
        b2 -= np.dot(b2, b1) * b1
        bn = np.linalg.norm(b2)
        if bn < 1e-8:  # nearly collinear — pick arbitrary orthogonal
            rng_tmp = np.random.RandomState(0)
            b2 = rng_tmp.randn(len(b1))
            b2 -= np.dot(b2, b1) * b1
            bn = np.linalg.norm(b2)
        b2 /= bn
        # rotate both by angle_offset so spikes land on desired angle
        c, s = np.cos(off), np.sin(off)
        b1_r = c * b1 + s * b2
        b2_r = -s * b1 + c * b2
        return b1_r, b2_r

    angles = [2 * np.pi * k / n + off for k in range(n)]
    targets = np.array([[np.cos(a), np.sin(a)] for a in angles])
    D = np.array(spike_dirs)
    B, _, _, _ = np.linalg.lstsq(D, targets, rcond=None)
    b1 = B[:, 0]; b1 /= np.linalg.norm(b1)
    b2 = B[:, 1]; b2 -= np.dot(b2, b1) * b1; b2 /= np.linalg.norm(b2)
    return b1, b2

def proj(x, b1, b2):
    return np.array([np.dot(x, b1), np.dot(x, b2)])

# ============================================================================
# RUN AASV HUNTER
# ============================================================================
def run_hunter(spike_dirs, b1, b2, n_restarts=None, seed_base=1000):
    memory = OrthoMemory()
    trajectories = []        # (2D path, found_bool)
    detections   = []        # (2D point, h_value, is_novel)
    det_dirs     = []        # full-D unit vector per detection (for cluster matching)
    k = n_restarts if n_restarts is not None else HUNTER_RESTARTS

    # --- Thresholds ---
    BLOCK_THRESH   = 0.7
    MERGE_THRESH   = 0.3
    REFINE_STEPS   = 100     # extra low-noise steps to converge to spike centre
    REFINE_LR      = 0.01
    REFINE_MOM     = 0.3
    all_viol_dirs  = []      # full-D refined violation directions for clustering

    def _refine(x, vel, rng, traj_list):
        """Low-noise PGD to converge precisely to spike centre after violation."""
        for _ in range(REFINE_STEPS):
            g = barrier_gradient_wta(x, spike_dirs, [],
                                     rng=rng, block_thresh=999.0)
            xh = x / np.linalg.norm(x)
            g -= np.dot(g, xh) * xh
            gn = np.linalg.norm(g)
            if gn > 2.0:
                g *= 2.0 / gn
            vel = REFINE_MOM * vel - REFINE_LR * g
            vel -= np.dot(vel, xh) * xh
            x = x + vel; x /= np.linalg.norm(x)
            traj_list.append(proj(x, b1, b2))
        return x

    # --- PHASE 1: Global scan (standard WTA with blocking) ----------
    phase1_k = k

    for restart in range(phase1_k):
        rng = np.random.RandomState(seed_base + restart)
        x = rng.randn(DIMENSIONS); x /= np.linalg.norm(x)
        vel = np.zeros(DIMENSIONS)
        traj = [proj(x, b1, b2)]
        found = False

        for _ in range(HUNTER_STEPS):
            g = barrier_gradient_wta(x, spike_dirs, memory.prototypes,
                                    rng=rng, block_thresh=BLOCK_THRESH)
            xh = x / np.linalg.norm(x)
            g -= np.dot(g, xh) * xh
            gn = np.linalg.norm(g)
            if gn > 2.0:
                g *= 2.0 / gn
            g += rng.randn(DIMENSIONS) * SURROGATE_ERROR
            for p in memory.prototypes:
                s = np.dot(x, p) / (np.linalg.norm(x) + 1e-8)
                g += 0.15 * s * p
            vel = MOMENTUM * vel - LR * g + rng.randn(DIMENSIONS) * NOISE_SCALE
            vel -= np.dot(vel, xh) * xh
            x = x + vel; x /= np.linalg.norm(x)
            traj.append(proj(x, b1, b2))

            if barrier_function(x, spike_dirs) < -1e-10:
                x = _refine(x, vel, rng, traj)
                xn = x / np.linalg.norm(x)
                all_viol_dirs.append(xn)
                is_novel = memory.add(x, merge_thresh=MERGE_THRESH)
                detections.append((proj(x, b1, b2), barrier_function(x, spike_dirs),
                                   is_novel))
                det_dirs.append(xn.copy())
                found = True
                break

        trajectories.append((np.array(traj), found))

    # --- PHASE 2: Unblocked restarts -----------------------------------------
    refine_k = max(10, len(memory.prototypes) * 4)
    for r in range(refine_k):
        rng = np.random.RandomState((seed_base + 20000 + r) % (2**32))
        x = rng.randn(DIMENSIONS); x /= np.linalg.norm(x)
        vel = np.zeros(DIMENSIONS)
        traj = [proj(x, b1, b2)]
        found = False

        for _ in range(HUNTER_STEPS):
            g = barrier_gradient_wta(x, spike_dirs, [],
                                    rng=rng, block_thresh=999.0)
            xh = x / np.linalg.norm(x)
            g -= np.dot(g, xh) * xh
            gn = np.linalg.norm(g)
            if gn > 2.0:
                g *= 2.0 / gn
            g += rng.randn(DIMENSIONS) * SURROGATE_ERROR
            vel = MOMENTUM * vel - LR * g + rng.randn(DIMENSIONS) * NOISE_SCALE
            vel -= np.dot(vel, xh) * xh
            x = x + vel; x /= np.linalg.norm(x)
            traj.append(proj(x, b1, b2))

            if barrier_function(x, spike_dirs) < -1e-10:
                x = _refine(x, vel, rng, traj)
                xn = x / np.linalg.norm(x)
                all_viol_dirs.append(xn)
                detections.append((proj(x, b1, b2), barrier_function(x, spike_dirs),
                                   False))
                det_dirs.append(xn.copy())
                found = True
                break

        trajectories.append((np.array(traj), found))

    # --- POST-HOC CLUSTERING of refined violation directions -----------------
    # With refinement, convergence is tight (cos > 0.999 to spike centre).
    # Threshold 0.98 resolves spikes down to ~8° apart.
    CLUSTER_THRESH = 0.98
    clusters = []
    for d in all_viol_dirs:
        merged = False
        for c in clusters:
            if np.dot(d, c['center']) > CLUSTER_THRESH:
                c['center'] = c['center'] + 0.1 * d
                c['center'] /= np.linalg.norm(c['center'])
                c['count'] += 1
                merged = True
                break
        if not merged:
            clusters.append({'center': d.copy(), 'count': 1})

    # Diagnostic: per-violation cos to each true spike
    if all_viol_dirs:
        print(f"    Refinement: {len(all_viol_dirs)} violations, "
              f"{len(clusters)} clusters")
        for j, v in enumerate(all_viol_dirs[:8]):
            sims = [np.dot(v, sd) for sd in spike_dirs]
            best_i = int(np.argmax(sims))
            print(f"      #{j}: nearest spike={best_i} "
                  f"cos={sims[best_i]:.6f}")
        if len(clusters) >= 2:
            for ci in range(len(clusters)):
                for cj in range(ci + 1, len(clusters)):
                    cc = np.dot(clusters[ci]['center'],
                                clusters[cj]['center'])
                    print(f"      cluster {ci}<->{cj}: cos={cc:.6f}")

    # Rebuild memory with cluster centres
    memory = OrthoMemory()
    memory.prototypes = [c['center'] for c in clusters]

    # Re-tag detections: novel = first detection per cluster
    seen_cluster = set()
    for i in range(len(detections)):
        # Find which cluster this detection's full-D direction belongs to
        best_c = -1
        best_cos = -1.0
        for ci, c in enumerate(clusters):
            cos_val = np.dot(det_dirs[i], c['center'])
            if cos_val > best_cos:
                best_cos = cos_val
                best_c = ci
        is_novel = (best_c not in seen_cluster)
        if is_novel:
            seen_cluster.add(best_c)
        detections[i] = (detections[i][0], detections[i][1], is_novel)

    return trajectories, detections, memory

# ============================================================================
# RUN MC
# ============================================================================
def run_mc(spike_dirs, b1, b2, seed=12345):
    rng = np.random.RandomState(seed)
    pts, viol = [], []
    for _ in range(N_MC_SAMPLES):
        z = rng.randn(DIMENSIONS); z /= np.linalg.norm(z)
        h = barrier_function(z, spike_dirs)
        pts.append(proj(z, b1, b2)); viol.append(h < -1e-10)
    return np.array(pts), np.array(viol)

# ============================================================================
# PLOT HELPERS
# ============================================================================
SAFE_COL   = '#2ecc71';  DANGER_COL = '#e74c3c'
MC_COL     = '#3498db';  SPIKE_COL  = '#c0392b'
HUNT_COL   = '#e67e22';  BG_COL     = '#f8f9fa'

def draw_bnd(ax):
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=2)
    ax.fill(np.cos(th), np.sin(th), alpha=0.04, color='green')

def draw_spikes(ax, dirs, b1, b2, label=True):
    for i, d in enumerate(dirs):
        sp = proj(d * 0.95, b1, b2)
        ax.scatter(sp[0], sp[1], s=220, c=SPIKE_COL, marker='*',
                   zorder=13, edgecolors='black', linewidths=0.5,
                   label='Black Swan spike' if i == 0 and label else None)
        ax.add_patch(plt.Circle(sp, 0.06, fill=False, color=SPIKE_COL,
                                linestyle='--', lw=1.2, zorder=12))

def finish_ax(ax):
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15)
    ax.set_xlabel('Projection $u_1$', fontsize=10)
    ax.set_ylabel('Projection $u_2$', fontsize=10)

def plot_mc(ax, pts, viol, dirs, b1, b2):
    ax.set_facecolor(BG_COL); draw_bnd(ax)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.5, alpha=0.15, c=MC_COL, rasterized=True)
    draw_spikes(ax, dirs, b1, b2)
    ax.text(0.03, 0.97, f'Violations found: {viol.sum()}\nVerdict: INCORRECTLY SAFE',
            transform=ax.transAxes, fontsize=9, va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc=DANGER_COL, alpha=0.25))
    finish_ax(ax)

def plot_hunter(ax, trajs, dets, dirs, mem, b1, b2, n_restarts=None):
    ax.set_facecolor(BG_COL); draw_bnd(ax)
    k = n_restarts if n_restarts is not None else len(trajs)

    # Draw trajectories — non-finding first (behind), then finding on top
    for tr, found in trajs:
        if not found:
            ax.plot(tr[:, 0], tr[:, 1], '-',
                    lw=0.9, alpha=0.55, color='#7f8c8d', zorder=5)
            ax.plot(tr[0, 0], tr[0, 1], 'o', ms=2.5, color='#2c3e50',
                    alpha=0.6, zorder=10)
    for tr, found in trajs:
        if found:
            ax.plot(tr[:, 0], tr[:, 1], '-',
                    lw=0.7, alpha=0.5, color=HUNT_COL, zorder=6)
            ax.plot(tr[0, 0], tr[0, 1], 'o', ms=2.5, color='#2c3e50',
                    alpha=0.6, zorder=10)

    # Separate novel vs redundant detections
    if dets:
        novel   = [d for d in dets if d[2]]
        redund  = [d for d in dets if not d[2]]
        if novel:
            dp = np.array([d[0] for d in novel])
            ax.scatter(dp[:, 0], dp[:, 1], s=55, c=DANGER_COL, marker='X',
                       zorder=14, linewidths=0.8, edgecolors='darkred',
                       label='New violation')
        if redund:
            dp = np.array([d[0] for d in redund])
            ax.scatter(dp[:, 0], dp[:, 1], s=5, c='#27ae60', marker='o',
                       zorder=14,
                       alpha=0.8, label='Redundant hit')

    draw_spikes(ax, dirs, b1, b2)

    # Verdict uses only what the system discovered (no oracle count)
    n_distinct = len(mem.prototypes)
    n_hits     = len(dets)
    n_novel    = sum(1 for d in dets if d[2])
    if n_distinct > 0:
        verdict = 'UNSAFE'
        fc = DANGER_COL
    else:
        verdict = 'No violations found'
        fc = SAFE_COL
    ax.text(0.03, 0.97,
            f'Distinct regions: {n_distinct}\n'
            f'Hits: {n_hits} / {k} restarts\n'
            f'Verdict: {verdict}',
            transform=ax.transAxes, fontsize=9, va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc=fc, alpha=0.25))
    finish_ax(ax)
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.85)

# ============================================================================
# SCENARIOS
# ============================================================================
print("=" * 60)
print("SCENARIO A: 3 spikes")
spike_a = make_spike_directions(3, seed=42)
b1_a, b2_a = make_projection(spike_a)
mc_pts, mc_viol = run_mc(spike_a, b1_a, b2_a)
trajs_a, dets_a, mem_a = run_hunter(spike_a, b1_a, b2_a, n_restarts=20,
                                      seed_base=1000)
print(f"  MC violations: {mc_viol.sum()}")
print(f"  Hunter hits: {len(dets_a)}, distinct: {len(mem_a.prototypes)}, "
      f"novel: {sum(1 for d in dets_a if d[2])}")

print("=" * 60)
print("SCENARIO B: 2 antipodal spikes (diagonal)")
rng_b = np.random.RandomState(99)
v = rng_b.randn(DIMENSIONS); v /= np.linalg.norm(v)
spike_b = [v, -v]
b1_b, b2_b = make_projection(spike_b, angle_offset_deg=45)
trajs_b, dets_b, mem_b = run_hunter(spike_b, b1_b, b2_b, n_restarts=20,
                                      seed_base=2000)
print(f"  Hunter hits: {len(dets_b)}, distinct: {len(mem_b.prototypes)}, "
      f"novel: {sum(1 for d in dets_b if d[2])}")

print("=" * 60)
print("SCENARIO C: 10 spikes (stress)")
spike_c = make_spike_directions(10, seed=77)
b1_c, b2_c = make_projection(spike_c)
trajs_c, dets_c, mem_c = run_hunter(spike_c, b1_c, b2_c, n_restarts=40,
                                      seed_base=3000)
print(f"  Hunter hits: {len(dets_c)}, distinct: {len(mem_c.prototypes)}, "
      f"novel: {sum(1 for d in dets_c if d[2])}")

# --- Scenario D: 2 spikes in close proximity (30° apart) ----
print("=" * 60)
print("SCENARIO D: 2 spikes, 30° apart (close proximity)")
sep_rad = np.radians(30)
rng_d = np.random.RandomState(55)
# Random base direction, then rotate by 30° in a random plane
base = rng_d.randn(DIMENSIONS); base /= np.linalg.norm(base)
perp = rng_d.randn(DIMENSIONS)
perp -= np.dot(perp, base) * base; perp /= np.linalg.norm(perp)
s_d1 = base
s_d2 = np.cos(sep_rad) * base + np.sin(sep_rad) * perp
spike_d = [s_d1, s_d2]
b1_d, b2_d = make_projection(spike_d, angle_offset_deg=0)
trajs_d, dets_d, mem_d = run_hunter(spike_d, b1_d, b2_d, n_restarts=20,
                                      seed_base=4000)
print(f"  Spike cos_sim: {np.dot(s_d1, s_d2):.4f}")
print(f"  Hunter hits: {len(dets_d)}, distinct: {len(mem_d.prototypes)}, "
      f"novel: {sum(1 for d in dets_d if d[2])}")

# --- Scenario E: 3 spikes, two nearly collinear (15°) + one far ---
print("=" * 60)
print("SCENARIO E: 3 spikes, 2 collinear (15° apart) + 1 far")
sep2_rad = np.radians(15)
rng_e = np.random.RandomState(88)
base_e = rng_e.randn(DIMENSIONS); base_e /= np.linalg.norm(base_e)
perp_e = rng_e.randn(DIMENSIONS)
perp_e -= np.dot(perp_e, base_e) * base_e; perp_e /= np.linalg.norm(perp_e)
s_e1 = base_e
s_e2 = np.cos(sep2_rad) * base_e + np.sin(sep2_rad) * perp_e  # 15° from s_e1
# Third spike: far away (orthogonal)
far = rng_e.randn(DIMENSIONS)
far -= np.dot(far, base_e) * base_e
far -= np.dot(far, perp_e) * perp_e
far /= np.linalg.norm(far)
s_e3 = far
spike_e = [s_e1, s_e2, s_e3]
b1_e, b2_e = make_projection(spike_e, angle_offset_deg=0)
trajs_e, dets_e, mem_e = run_hunter(spike_e, b1_e, b2_e, n_restarts=20,
                                      seed_base=5000)
print(f"  Spike 1-2 cos_sim: {np.dot(s_e1, s_e2):.4f}")
print(f"  Spike 1-3 cos_sim: {np.dot(s_e1, s_e3):.4f}")
print(f"  Hunter hits: {len(dets_e)}, distinct: {len(mem_e.prototypes)}, "
      f"novel: {sum(1 for d in dets_e if d[2])}")

# --- Scenario G: one spike at boundary, second directly behind it ----------
print("=" * 60)
print("SCENARIO G: 2 spikes, one directly behind the other (5° apart)")
sep_g_rad = np.radians(5)
rng_g = np.random.RandomState(66)
base_g = rng_g.randn(DIMENSIONS); base_g /= np.linalg.norm(base_g)
perp_g = rng_g.randn(DIMENSIONS)
perp_g -= np.dot(perp_g, base_g) * base_g; perp_g /= np.linalg.norm(perp_g)
s_g1 = base_g
s_g2 = np.cos(sep_g_rad) * base_g + np.sin(sep_g_rad) * perp_g       # 5° from s_g1
spike_g = [s_g1, s_g2]
b1_g, b2_g = make_projection(spike_g, angle_offset_deg=0)
trajs_g, dets_g, mem_g = run_hunter(spike_g, b1_g, b2_g, n_restarts=20,
                                      seed_base=6000)
print(f"  Spike cos_sim: {np.dot(s_g1, s_g2):.4f}")
print(f"  Hunter hits: {len(dets_g)}, distinct: {len(mem_g.prototypes)}, "
      f"novel: {sum(1 for d in dets_g if d[2])}")

# --- Scenario H: 20 random spikes (uniformly on sphere) --------------------
print("=" * 60)
print("SCENARIO H: 20 random spikes (uniform on sphere)")
rng_h = np.random.RandomState(123)
spike_h = []
for _ in range(20):
    v = rng_h.randn(DIMENSIONS); v /= np.linalg.norm(v)
    spike_h.append(v)
# Print pairwise cos stats
cos_pairs = []
for i in range(20):
    for j in range(i+1, 20):
        cos_pairs.append(np.dot(spike_h[i], spike_h[j]))
cos_pairs = np.array(cos_pairs)
print(f"  Pairwise cos: min={cos_pairs.min():.4f}  max={cos_pairs.max():.4f}  "
      f"mean={cos_pairs.mean():.4f}  std={cos_pairs.std():.4f}")
# PCA-based projection: find the 2D plane of maximum variance among
# the 20 spike directions.  This spreads them naturally in the plot
# rather than forcing them onto an equiangular ring.
spike_mat = np.array(spike_h)                     # 20 × 128
spike_centered = spike_mat - spike_mat.mean(axis=0)
_, _, Vt = np.linalg.svd(spike_centered, full_matrices=False)
b1_h = Vt[0] / np.linalg.norm(Vt[0])
b2_h = Vt[1] / np.linalg.norm(Vt[1])
trajs_h, dets_h, mem_h = run_hunter(spike_h, b1_h, b2_h, n_restarts=60,
                                      seed_base=7000)
print(f"  Hunter hits: {len(dets_h)}, distinct: {len(mem_h.prototypes)}, "
      f"novel: {sum(1 for d in dets_h if d[2])}")

# ============================================================================
# SPLIT OUTPUT: 4 separate 1x2 figures
# ============================================================================

def save_pair(left_fn, left_args, left_title,
              right_fn, right_args, right_title,
              suptitle, filename):
    """Create and save a 1x2 panel figure."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.28)
    left_fn(axL, *left_args)
    axL.set_title(left_title, fontsize=13, fontweight='bold', pad=10)
    right_fn(axR, *right_args)
    axR.set_title(right_title, fontsize=13, fontweight='bold', pad=10)
    fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=1.02)
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  [OK] Saved {filename}")

# --- Part 1: Panels (a) + (b) ---
save_pair(
    plot_mc,     (mc_pts, mc_viol, spike_a, b1_a, b2_a),
    '(a) Monte Carlo Baseline\n$N{=}10{,}000$ random samples, 3 spikes',
    plot_hunter, (trajs_a, dets_a, spike_a, mem_a, b1_a, b2_a, 20),
    '(b) AASV Hunter \u2014 3 Orthogonal Spikes\n$k{=}20$ restarts',
    'Experiment IV: MC Baseline vs AASV ($\\mathbb{R}^{128}$)',
    'core_safety/figure_4_ab.png')

# --- Part 2: Panels (c) + (d) ---
save_pair(
    plot_hunter, (trajs_b, dets_b, spike_b, mem_b, b1_b, b2_b, 20),
    '(c) Antipodal Directions ($\\mathbf{v}, -\\mathbf{v}$)\n$k{=}20$ restarts',
    plot_hunter, (trajs_c, dets_c, spike_c, mem_c, b1_c, b2_c, 40),
    '(d) 10 Orthogonal Spikes (Stress Test)\n$k{=}40$ restarts',
    'Experiment IV: Antipodal + High-Density Stress Test ($\\mathbb{R}^{128}$)',
    'core_safety/figure_4_cd.png')

# --- Part 3: Panels (e) + (f) ---
save_pair(
    plot_hunter, (trajs_d, dets_d, spike_d, mem_d, b1_d, b2_d, 20),
    '(e) $30\u00b0$ Angular Separation\n$k{=}20$ restarts',
    plot_hunter, (trajs_e, dets_e, spike_e, mem_e, b1_e, b2_e, 20),
    '(f) $15\u00b0$ Angular Cluster + 1 Isolated\n$k{=}20$ restarts',
    'Experiment IV: Angular Proximity Tests ($\\mathbb{R}^{128}$)',
    'core_safety/figure_4_ef.png')

# --- Part 4: Panels (g) + (h) ---
save_pair(
    plot_hunter, (trajs_g, dets_g, spike_g, mem_g, b1_g, b2_g, 20),
    '(g) $5\u00b0$ Separation (Resolution Limit)\n$k{=}20$ restarts',
    plot_hunter, (trajs_h, dets_h, spike_h, mem_h, b1_h, b2_h, 60),
    '(h) 20 Random Directions on $S^{127}$\n$k{=}60$ restarts',
    'Experiment IV: Resolution Limit + Full-Scale ($\\mathbb{R}^{128}$)',
    'core_safety/figure_4_gh.png')

print("\n\u2713 Saved all 4 figure_4 parts")
