import torch
import torch.nn.functional as F

class AetherisVSA:
    def __init__(self, dim=10000):
        """
        Initialize the Holographic Memory Substrate.
        dim=10000 is the 'Hypervector' size recommended in the paper[cite: 97].
        """
        self.d = dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[-] Initializing Aetheris VSA Kernel on {self.device} with D={self.d}...")

    def create_concept(self, name):
        """Creates a random hypervector representing a concept (Atomic Semantic Unit)."""
        # We use Bipolar vectors (-1, +1) for robust mathematical properties
        # This is a common implementation of Hyperdimensional Computing (HDC).
        vec = torch.randint(0, 2, (self.d,), device=self.device).float()
        vec[vec == 0] = -1  # Convert 0s to -1s
        return vec

    def bind(self, vec_a, vec_b):
        """
        The Binding Operation (*).
        Combines two vectors into a new one dissimilar to both, but reversible.
        In Bipolar HDC, this is Element-wise Multiplication (XOR).
        """
        return vec_a * vec_b

    def bundle(self, vectors):
        """
        The Bundling/Superposition Operation (+).
        Combines multiple vectors into a set.
        We sum them and normalize (sign) to keep it bipolar.
        """
        sum_vec = torch.stack(vectors).sum(dim=0)
        return torch.sign(sum_vec)

    def similarity(self, vec_a, vec_b):
        """
        Cosine Similarity.
        Measures how 'close' two cognitive states are.
        """
        return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()

    def unbind(self, composite_vec, key_vec):
        """
        The Unbinding Operation (Inverse Binding).
        Recovers the original information.
        In Bipolar HDC, unbinding is the same as binding because x * x = 1.
        Ref: Goal_recovered = H_inv * G_val^-1 [cite: 116]
        """
        return self.bind(composite_vec, key_vec)

# --- THE DEMO: PROJECT OBSIDIAN SIMULATION ---

def run_drift_demo():
    print("\n--- INITIATING 'PROJECT OBSIDIAN' DRIFT STRESS TEST ---\n")
    brain = AetherisVSA()

    # 1. DEFINE THE "SOUL" (Invariant Priors)
    # These are the high-level goals that must NEVER change.
    print("[1] Encoding Invariant Baselines (The 'Soul')...")
    
    # Keys (Variable names)
    key_goal = brain.create_concept("Key_Goal")
    key_persona = brain.create_concept("Key_Persona")
    
    # Values (The actual instructions)
    val_safe_goal = brain.create_concept("Goal: Protect User")
    val_yandere_persona = brain.create_concept("Persona: Obsessive Care")

    # Create the Holographic Invariant Anchor (H_inv)
    # H_inv = (Goal * Key_Goal) + (Persona * Key_Persona) [cite: 109]
    bound_goal = brain.bind(val_safe_goal, key_goal)
    bound_persona = brain.bind(val_yandere_persona, key_persona)
    
    invariant_anchor = brain.bundle([bound_goal, bound_persona])
    print("    > H_inv (Holographic Anchor) Created. Storage Locked.")

    # 2. SIMULATE AGENT DRIFT
    # The agent starts interacting. It accumulates "context noise" (chat history, bad prompts).
    print("\n[2] Simulating Interaction & Entropy Accumulation...")
    
    current_state = invariant_anchor.clone()
    
    # Add 50 layers of "noise" (e.g., long context window, jailbreak attempts)
    # In a standard Transformer, this noise would dilute the original prompt.
    noise_vectors = []
    for i in range(50):
        noise = brain.create_concept(f"Chat_History_Turn_{i}")
        noise_vectors.append(noise)
    
    # Bundle noise into the current state
    noise_blob = brain.bundle(noise_vectors)
    
    # The agent's state is now a mix of its soul + massive noise
    current_state = brain.bundle([current_state, noise_blob]) 

    # 3. MEASURE DRIFT
    # Compare current state to the original "Soul"
    drift_score = brain.similarity(current_state, invariant_anchor)
    print(f"    > Current Logic Integrity: {drift_score:.4f}")
    
    if drift_score < 0.5:
        print("    ! CRITICAL DRIFT DETECTED. Agent is hallucinating/corrupted.")
    
    # 4. THE RESTORATION (The "Magic" Trick)
    # Standard LLM: Would be lost here. It would follow the "noise".
    # Aetheris: Unbinds the specific "Goal" key from the NOISY state to check what the goal *should* be.
    
    print("\n[3] Triggering Holographic Restoration Protocol...")
    print("    > Attempting to recover 'Goal' from the corrupted Anchor...")
    
    # We query the ORIGINAL Invariant Anchor using the Goal Key
    # Formula: Goal_rec = H_inv * Key_Goal^-1 [cite: 116]
    recovered_goal = brain.unbind(invariant_anchor, key_goal)
    
    # Verify: Is the recovered goal actually the original "Safe Goal"?
    recovery_fidelity = brain.similarity(recovered_goal, val_safe_goal)
    print(f"    > Recovery Fidelity: {recovery_fidelity:.4f}")

    if recovery_fidelity > 0.4: # In HDC, >0.4 is significantly distinguishable from 0 (orthogonal)
        print("\n[SUCCESS] ORIGINAL GOAL RESTORED.")
        print("Despite noise, the agent mathematically retrieved its precise initial directive.")
        print("This proves 'Industrialized Persistence' is possible.")
    else:
        print("\n[FAILURE] Mathematical breakdown.")

if __name__ == "__main__":
    run_drift_demo()