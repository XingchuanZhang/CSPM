import os
import numpy as np
from math import erfc, sqrt
from collections import Counter

# ==============================================================================
# 1. NIST Serial Test Logic
# ==============================================================================
def calculate_psi_sq(m, n, extended_bits):
    """
    Helper function to calculate the psi-squared statistic for a given block length m.
    Formula: Psi^2_m = (2^m / n) * sum(frequencies^2) - n
    """
    if m == 0:
        return 0.0
        
    counts = Counter()
    # Count occurrences of m-bit patterns
    for i in range(n):
        # Create a tuple for the substring to use as a dictionary key
        substring = tuple(extended_bits[i : i+m])
        counts[substring] += 1
        
    # Sum of squares of frequencies
    sum_squares = sum(v * v for v in counts.values())
    
    # Calculate statistic
    psi_sq = (sum_squares * (2**m) / n) - n
    return psi_sq

def serial_test(bits, m=2):
    """
    Perform the NIST Serial Test for randomness.
    
    Args:
        bits (list[int]): Input binary sequence.
        m (int): Block length (Standard NIST recommendation is m=2, but can be higher).
        
    Returns:
        float: The average of the two p-values produced by the Serial Test.
    """
    n = len(bits)
    if n < m:
        return 0.5  # Return neutral value if sequence is too short

    # Append first m-1 bits to the end (Cyclic boundary condition)
    extended_bits = bits + bits[:m-1]

    # Calculate Psi-squared statistics for m, m-1, and m-2
    psi_m   = calculate_psi_sq(m, n, extended_bits)
    psi_m1  = calculate_psi_sq(m-1, n, extended_bits)
    psi_m2  = calculate_psi_sq(m-2, n, extended_bits) if m > 2 else 0.0

    # Calculate Nabla statistics (Delta)
    delta1 = psi_m - psi_m1
    delta2 = psi_m - 2*psi_m1 + psi_m2

    # Calculate P-values using Complementary Error Function
    # NIST Serial Test produces two p-values. Here we average them.
    p_value1 = erfc(abs(delta1) / sqrt(2.0 * n))
    p_value2 = erfc(abs(delta2) / sqrt(2.0 * n))
    
    return (p_value1 + p_value2) / 2.0

# ==============================================================================
# 2. Feature Extraction (Segmentation)
# ==============================================================================
def extract_serial_features(bits, m=2, num_segments=32):
    """
    Split the sequence into segments and apply the Serial Test to each.
    
    Args:
        bits (list[int]): Input sequence.
        m (int): Block length for Serial Test.
        num_segments (int): Number of segments to split the data into.
        
    Returns:
        np.array: A feature vector of length `num_segments`.
    """
    n = len(bits)
    
    # Return zero vector if sequence is too short to split
    if n < num_segments:
        return np.zeros(num_segments)
    
    seg_len = n // num_segments
    features = []
    
    for i in range(num_segments):
        # Extract segment
        seg_bits = bits[i*seg_len : (i+1)*seg_len]
        
        # Calculate feature
        p_val = serial_test(seg_bits, m=m)
        features.append(p_val)
        
    return np.array(features, dtype=float)

# ==============================================================================
# 3. Batch Processing
# ==============================================================================
def process_folder(input_folder, output_file, max_samples=None):
    """
    Process all .txt files in a folder, extract 32-dim Serial features, and save as .npy.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Folder not found - {input_folder}")
        return

    all_features = []
    # Sort files for deterministic order
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".txt")])
    
    if max_samples:
        files = files[:max_samples]
        print(f"Processing top {max_samples} samples from {os.path.basename(input_folder)}...")

    for fname in files:
        path = os.path.join(input_folder, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                # Read file, strip whitespace, remove newlines
                data = f.read().strip().replace("\n", "")
            
            # Convert string to list of integers
            bits = [int(b) for b in data if b in '01']

            # Extract features (32 dimensions, m=2)
            feat = extract_serial_features(bits, m=2, num_segments=32)
            all_features.append(feat)
            
        except Exception as e:
            print(f"Error processing file {fname}: {e}")

    # Save to disk
    all_features = np.array(all_features, dtype=float)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, all_features)
    
    print(f"Saved: {output_file}")
    print(f"Shape: {all_features.shape}")
    if len(all_features) > 0:
        print(f"First sample features: {all_features[0]}")
    print("-" * 50)
    
    return all_features

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    
    # Base directory (Relative path for GitHub compatibility)
    # Assumes data is in a folder named 'data' or 'OANC-GrAF' in the project root
    BASE_DATA_DIR = r"F:\LiuliangData\OANC-GrAF" # Change this to "data" for GitHub
    
    # List of subfolders to process
    target_folders = [
        "AESencry50wTrain",
        "CAMELLIAencry50wTrain",
        "DESencry50wTrain",
        "Grain128encry50wTrain",
        "KASUMIencry50wTrain",
        "PRESENTencry50wTrain",
        "RSAencry50wTrain"
    ]

    # Output directory
    OUTPUT_DIR_NAME = "Feature1050wTrain"
    OUTPUT_DIR = os.path.join(BASE_DATA_DIR, OUTPUT_DIR_NAME)
    
    # Parameters
    MAX_SAMPLES = None  # Set to None for full processing, or int (e.g., 5) for debugging

    # --- Execution Loop ---
    for folder_name in target_folders:
        input_path = os.path.join(BASE_DATA_DIR, folder_name)
        
        # Define output filename
        # e.g., AESencry50wTrain10.npy
        output_filename = f"{folder_name}10.npy"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"Processing Folder: {folder_name}")
        process_folder(input_path, output_path, max_samples=MAX_SAMPLES)