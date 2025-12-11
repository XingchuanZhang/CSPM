"""
Adversarial Attack on Cryptographic Identification Models
--------------------------------------------------------
This script implements a hybrid search strategy to generate adversarial perturbations
for cryptographic algorithm identification models.

Author: [Your Name/GitHub Handle]
Date: 2025
"""

import os
import time
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from math import erfc, sqrt
from tensorflow.keras.models import load_model
from tqdm import tqdm  # Recommended: pip install tqdm

# Style configuration for plots
plt.style.use('ggplot')

# ==============================================================================
# 1. Feature Extraction Module (Generic Interface)
# ==============================================================================
class FeatureExtractor:
    """
    Base class/Interface for feature extraction.
    Modify or extend this class to implement different NIST statistical tests 
    (e.g., Serial Test, Rank Test, etc.).
    """
    def __init__(self, method='non_overlapping_template', **kwargs):
        self.method = method
        self.kwargs = kwargs
        
        # NIST Non-overlapping Template Matching specific configuration
        self.templates = [
            "000000", "111111", "010101", "101010",
            "001100", "110011", "000111", "111000",
            "011011", "100100", "110110", "001001",
            "111100", "000011", "010010", "101101",
            "011110", "100001", "011100", "100011"
        ]

    def extract(self, seq):
        """
        Main entry point to extract features from a binary string.
        """
        # Ensure sequence contains only '0' and '1'
        bits = [int(ch) for ch in seq if ch in '01']
        
        if self.method == 'non_overlapping_template':
            return self._non_overlapping_template_20(bits, m=6)
        
        # TODO: Add other methods here
        # elif self.method == 'serial_test':
        #     return self._serial_test(bits, ...)
        
        else:
            raise ValueError(f"Unknown feature extraction method: {self.method}")

    # --- Implementation of Specific Algorithms ---

    def _non_overlapping_template_test(self, bits, template, m):
        """NIST Non-overlapping Template Matching Test logic."""
        n = len(bits)
        N = n // m
        if N == 0: return 0.5

        count = 0
        i = 0
        while i <= n - m:
            if bits[i:i+m] == template:
                count += 1
                i += m
            else:
                i += 1

        mu = (n - m + 1) / (2 ** m)
        variance = n * ((1.0 / (2 ** m)) - (2 * m - 1) / (2.0 ** (2 * m)))
        sigma = sqrt(variance)

        if sigma == 0: return 1.0

        z = (count - mu) / sigma
        p_value = erfc(abs(z) / sqrt(2.0))
        return float(p_value)

    def _non_overlapping_template_20(self, bits, m=6):
        """Extracts 20-dim vector using fixed templates."""
        feats = []
        for t_str in self.templates:
            tpl = [int(ch) for ch in t_str]
            p = self._non_overlapping_template_test(bits, tpl, m)
            feats.append(p)
        return np.array(feats, dtype=float)

# ==============================================================================
# 2. Helper Functions (Bit Manipulation & I/O)
# ==============================================================================
def flip_bits(original_seq, mask):
    """
    Flips bits in the original sequence based on the mask ('1' in mask means flip).
    """
    # Adjust mask length if necessary
    if len(mask) != len(original_seq):
        mask = mask.ljust(len(original_seq), '0')[:len(original_seq)]
    
    # Efficient string manipulation
    # Convert to integers for XOR might be faster, but string is safer for display
    flipped = []
    for b, m in zip(original_seq, mask):
        if m == '1':
            flipped.append('1' if b == '0' else '0')
        else:
            flipped.append(b)
    return ''.join(flipped)

def load_data(folder_path, max_samples=None):
    """Loads valid bit sequences from .txt files in a folder."""
    samples = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
        
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if max_samples:
        files = files[:max_samples]
        
    for file in files:
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            # Filter only 0s and 1s
            seq = ''.join(c for c in f.read() if c in '01')
            if seq: samples.append(seq)
    return samples

# ==============================================================================
# 3. Model Wrapper
# ==============================================================================
class ModelWrapper:
    def __init__(self, model_path, feature_extractor):
        self.model_path = model_path
        self.extractor = feature_extractor
        self.model = None
        self.scaler = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = load_model(self.model_path)
        print(f"[Info] Model loaded: {self.model_path}")
        
        # Assume scaler is in the same directory
        model_dir = os.path.dirname(self.model_path)
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"[Info] Scaler loaded: {scaler_path}")
        else:
            print("[Warning] No scaler found. Proceeding without normalization (Risk!)")

    def predict_one(self, seq):
        """Returns probability distribution for a single sequence."""
        feats = self.extractor.extract(seq)
        if self.scaler:
            feats = self.scaler.transform(feats.reshape(1, -1))
        else:
            feats = feats.reshape(1, -1)
        return self.model.predict(feats, verbose=0)[0]

    def predict_batch_confidence(self, samples, true_label):
        """Returns average confidence for the true_label across all samples."""
        # Note: For massive speedup, implement batch prediction instead of loop
        # But keeping it simple for logic clarity as requested
        confidences = []
        for seq in samples:
            pred = self.predict_one(seq)
            confidences.append(pred[true_label])
        return np.mean(confidences)
    
    def calculate_accuracy(self, samples, true_label):
        correct = 0
        for seq in samples:
            pred = self.predict_one(seq)
            if np.argmax(pred) == true_label:
                correct += 1
        return correct / len(samples)

# ==============================================================================
# 4. Hybrid Search Strategy
# ==============================================================================
def hybrid_search_strategy(main_folder, similar_folder, model_path, true_label, alpha=0.5):
    """
    Executes the hybrid search strategy combining Inter-class Difference and Intra-class Stability.
    """
    # 1. Initialize Components
    extractor = FeatureExtractor(method='non_overlapping_template')
    target_model = ModelWrapper(model_path, extractor)

    # 2. Load Data
    print("Loading data...")
    main_samples = load_data(main_folder)
    similar_samples = load_data(similar_folder)
    
    # Normalize lengths (Truncate to min length for analysis)
    min_len = min(min(len(s) for s in main_samples), min(len(s) for s in similar_samples))
    main_samples = [s[:min_len] for s in main_samples]
    similar_samples = [s[:min_len] for s in similar_samples]
    L = min_len
    
    print(f"Data Loaded: Main Class ({len(main_samples)}), Similar Class ({len(similar_samples)}), Len ({L})")

    # 3. Calculate Statistical Scores
    print("Calculating statistical guidance vectors...")
    
    # Convert to matrix for fast calculation
    def to_matrix(samples):
        return np.array([[int(b) for b in s] for s in samples])

    mat_main = to_matrix(main_samples)
    mat_sim = to_matrix(similar_samples)

    p_main_1 = np.mean(mat_main, axis=0)      # Prob of bit being 1 in Main
    p_main_0 = 1.0 - p_main_1                 # Prob of bit being 0 in Main
    p_sim_0  = 1.0 - np.mean(mat_sim, axis=0) # Prob of bit being 0 in Similar

    # Score Calculation
    diff_score = np.abs(p_main_0 - p_sim_0)
    stability_score = p_main_1 

    # Normalization
    def normalize(v):
        return (v - v.min()) / (v.max() - v.min() + 1e-8)
    
    diff_norm = normalize(diff_score)
    stab_norm = normalize(stability_score)

    combined_score = alpha * diff_norm + (1 - alpha) * stab_norm
    sorted_indices = np.argsort(combined_score)[::-1] # Descending order

    # 4. Iterative Attack
    current_mask = list('0' * L)
    history = [] # (step, accuracy, confidence)
    
    # Setup Output Directories
    output_base = os.path.join(os.path.dirname(main_folder), "attack_results")
    os.makedirs(output_base, exist_ok=True)
    
    print(f"\nStarting Hybrid Search (Alpha={alpha})...")
    
    # Progress bar
    pbar = tqdm(sorted_indices, desc="Optimizing Mask")
    
    # Initial Baseline
    curr_samples = [flip_bits(s, "".join(current_mask)) for s in main_samples]
    curr_conf = target_model.predict_batch_confidence(curr_samples, true_label)
    
    step = 0
    for idx in pbar:
        step += 1
        
        # Propose Flip
        temp_mask = current_mask[:]
        temp_mask[idx] = '1' # Try flipping this bit
        temp_mask_str = "".join(temp_mask)
        
        # Evaluate
        temp_samples = [flip_bits(s, temp_mask_str) for s in main_samples]
        temp_conf = target_model.predict_batch_confidence(temp_samples, true_label)
        
        # Decision Rule: Accept if confidence decreases
        if temp_conf < curr_conf:
            current_mask = temp_mask
            curr_conf = temp_conf
            accepted = True
        else:
            accepted = False
            
        # Logging (Periodically calculate full accuracy - expensive operation)
        if step % 10 == 0 or step == len(sorted_indices):
            current_mask_str = "".join(current_mask)
            # Regenerate samples for acc calculation
            current_perturbed_samples = [flip_bits(s, current_mask_str) for s in main_samples]
            acc = target_model.calculate_accuracy(current_perturbed_samples, true_label)
            history.append((step, acc, curr_conf))
            
            pbar.set_postfix({"Conf": f"{curr_conf:.4f}", "Acc": f"{acc:.4f}"})

    # 5. Save Results & Plot
    final_mask_str = "".join(current_mask)
    mask_path = os.path.join(output_base, "final_mask.txt")
    with open(mask_path, "w") as f:
        f.write(final_mask_str)
        
    print(f"\nAttack Finished. Final Mask saved to: {mask_path}")
    
    # Plotting
    if history:
        steps, accs, confs = zip(*history)
        plt.figure(figsize=(10, 6))
        plt.plot(steps, accs, label='Accuracy (Main Class)', marker='o')
        plt.plot(steps, confs, label='Confidence (True Label)', linestyle='--')
        plt.title(f'Attack Progression (Alpha={alpha})')
        plt.xlabel('Bits Queried')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.savefig(os.path.join(output_base, "attack_progression.png"))
        print("Plot saved.")

    return final_mask_str

# ==============================================================================
# Main Execution Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cryptographic Algorithm Adversarial Attack Tool")
    
    # Arguments for flexibility
    parser.add_argument("--main_folder", type=str, required=True, help="Path to the target algorithm data folder")
    parser.add_argument("--similar_folder", type=str, required=True, help="Path to the most similar algorithm data folder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .h5 model")
    parser.add_argument("--true_label", type=int, required=True, help="Integer label of the target class")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for Hybrid Strategy (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    try:
        hybrid_search_strategy(
            main_folder=args.main_folder,
            similar_folder=args.similar_folder,
            model_path=args.model_path,
            true_label=args.true_label,
            alpha=args.alpha
        )
    except Exception as e:
        print(f"Error: {e}")