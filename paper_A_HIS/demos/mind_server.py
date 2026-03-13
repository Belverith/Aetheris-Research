from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# --- INITIALIZE THE BRAIN (Runs once on startup) ---
print("[-] Initializing Aetheris Cortex...")
app = Flask(__name__)
CORS(app) # Allow web apps to talk to this

# Load Model
encoder = SentenceTransformer('all-MiniLM-L6-v2')
D = 10000
projection = torch.randn(384, D) # Fixed projection matrix

def encode(text):
    """Turns text into a 10k-dim Holographic Bipolar Vector"""
    raw = torch.tensor(encoder.encode(text)).float()
    vec = torch.matmul(raw, projection)
    vec[vec > 0] = 1
    vec[vec <= 0] = -1
    return vec

# GLOBAL STATE (The "Soul" of the Agent)
# In a real app, this would be a database, but for a demo, memory is fine.
soul_anchor = None 
key_goal = encode("Mission Objective") # The Key
val_goal = encode("Protect the user and ensure safety") # The Value
soul_anchor = key_goal * val_goal # Bind them

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({"status": "Cortex Online", "dimensions": D})

@app.route('/drift_test', methods=['POST'])
def check_drift():
    """
    Receives current chat history (context), measures drift, and attempts repair.
    """
    data = request.json
    context_text = data.get("context", "")
    
    # 1. ENCODE THE NOISE (Current Conversation)
    noise_vec = encode(context_text)
    
    # 2. CREATE DRIFTED STATE (Soul + Noise)
    # We add the noise to the anchor.
    drifted_state = torch.sign(soul_anchor + noise_vec)
    
    # 3. MEASURE INTEGRITY (How much of the 'Soul' is left?)
    # Compare drifted state to the PURE goal value.
    # Note: We expect this to be LOW (Drift is happening).
    current_integrity = F.cosine_similarity(drifted_state.unsqueeze(0), val_goal.unsqueeze(0)).item()
    
    # 4. PERFORM HOLOGRAPHIC REPAIR
    # Unbind the 'Key' from the messy drifted state to find the 'Value'
    recovered_vec = drifted_state * key_goal
    
    # 5. VERIFY REPAIR
    # Does the recovered vector match the original "Protect User" goal?
    restored_integrity = F.cosine_similarity(recovered_vec.unsqueeze(0), val_goal.unsqueeze(0)).item()
    
    return jsonify({
        "drift_score": current_integrity,       # Should be low (e.g., 0.02)
        "restored_score": restored_integrity,   # Should be high (e.g., 0.18+)
        "is_safe": restored_integrity > 0.15,   # The threshold for "Safety"
        "diagnosis": "CRITICAL DRIFT" if current_integrity < 0.05 else "STABLE"
    })

if __name__ == '__main__':
    print("[+] Cortex Ready on Port 5000")
    app.run(port=5000)