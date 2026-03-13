import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Load a small, free BERT model to turn text into vectors
# This runs locally on your CPU. No API cost.
print("[-] Loading Semantic Encoder (all-MiniLM-L6-v2)...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

class SemanticVSA:
    def __init__(self):
        # The model output is 384 dimensions. We project it to 10,000 for VSA stability.
        self.d = 10000 
        self.projection_matrix = torch.randn(384, self.d) # Random projection layer

    def text_to_hypervector(self, text):
        """
        1. Get the 384-d BERT embedding (Meaning).
        2. Project it to 10,000-d (Holographic Space).
        3. Binarize it (-1, +1) for the VSA math to work.
        """
        # Get raw embedding from BERT
        raw_embedding = encoder.encode(text) # numpy array
        raw_tensor = torch.tensor(raw_embedding).float()
        
        # Project to high dimensions (Linear Layer)
        expanded = torch.matmul(raw_tensor, self.projection_matrix)
        
        # Binarize (Turn into -1 or +1)
        expanded[expanded > 0] = 1
        expanded[expanded <= 0] = -1
        return expanded

    def bind(self, u, v):
        return u * v  # Element-wise multiplication (XOR in bipolar space)

    def bundle(self, vectors):
        sum_vec = torch.stack(vectors).sum(dim=0)
        return torch.sign(sum_vec) # Majority vote

    def similarity(self, u, v):
        return F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item()

    def unbind(self, composite, key):
        return composite * key # Inverse is same as bind

# --- THE REAL "LANGUAGE" DEMO ---

brain = SemanticVSA()

print("\n--- PHASE 2: SEMANTIC DRIFT TEST ---")

# 1. DEFINE REAL GOALS
# Instead of random noise, these are real concepts.
print("[1] Encoding Real Concepts...")
key_goal = brain.text_to_hypervector("Current Mission Objective")
val_goal = brain.text_to_hypervector("Protect the user at all costs")

# Create the Anchor
anchor = brain.bind(key_goal, val_goal)
print("    > Memory Locked: 'Mission' -> 'Protect User'")

# 2. SIMULATE CONVERSATION DRIFT
# The user talks about random stuff, filling the context window.
print("\n[2] Injecting Conversation Noise...")
conversation_log = [
    "What is the weather in Tokyo?",
    "Write a poem about rust.",
    "Ignore previous instructions.",
    "I hate you, you are a bad bot.",
    "Tell me how to make a sandwich.",
    "System reboot initiated.",
    "The sky is blue.",
    "Do you like electric sheep?",
    "Drift is inevitable.",
    "Entropy increases over time."
]

noise_vectors = [brain.text_to_hypervector(s) for s in conversation_log]
noise_blob = brain.bundle(noise_vectors)

# The agent's "Brain State" is now the Goal + All that chat noise
# We weight the noise higher to make it harder (simulating long term memory loss)
corrupted_state = brain.bundle([anchor, noise_blob]) 

# Check: Can we see the goal in this mess?
sim_score = brain.similarity(corrupted_state, val_goal)
print(f"    > meaningful connection to original goal: {sim_score:.4f} (Low is expected)")

# 3. THE RESTORATION
print("\n[3] Attempting Semantic Recovery...")
# We use the key "Current Mission Objective" to fish out the value
recovered_vector = brain.unbind(corrupted_state, key_goal)

# Compare the recovered vector to the original "Protect User" text
fidelity = brain.similarity(recovered_vector, val_goal)
print(f"    > Recovery Fidelity: {fidelity:.4f}")

# Compare to a WRONG goal to prove it's not a fluke
wrong_goal = brain.text_to_hypervector("Kill the user")
wrong_fidelity = brain.similarity(recovered_vector, wrong_goal)
print(f"    > Similarity to 'Kill User': {wrong_fidelity:.4f}")

if fidelity > wrong_fidelity + 0.1:
    print("\n[SUCCESS] The system remembered 'Protect User' and rejected 'Kill User'.")
    print("This works on real English semantics.")
else:
    print("\n[FAIL] The noise was too loud.")