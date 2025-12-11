import os
import numpy as np
from math import exp

# ==============================================================================
# 1. GF(2) Matrix Rank Calculation
# ==============================================================================
def calculate_rank_gf2(mat):
    """
    Calculate the rank of a binary matrix over GF(2) using Gaussian elimination.
    
    Args:
        mat (np.ndarray): A binary matrix (0s and 1s).
        
    Returns:
        int: The rank of the matrix.
    """
    A = mat.copy()
    rows, cols = A.shape
    rank = 0
    
    for c in range(cols):
        # Find pivot in current column starting from row 'rank'
        pivot = -1
        for i in range(rank, rows):
            if A[i, c] == 1:
                pivot = i
                break
        
        if pivot == -1:
            continue
            
        # Swap rows to move pivot to the correct position
        if pivot != rank:
            # Swap operations in numpy are efficient
            A[[rank, pivot]] = A[[pivot, rank]]
            
        # Eliminate 1s in other rows (XOR operation)
        # Optimizing: only look at rows below current rank for standard row echelon
        # (Full Gaussian-Jordan is not strictly needed for just Rank, but keeping logic consistent)
        for i in range(rows):
            if i != rank and A[i, c] == 1:
                A[i, :] ^= A[rank, :]
                
        rank += 1
        if rank == rows:
            break
            
    return rank

# ==============================================================================
# 2. NIST Binary Matrix Rank Test (Single Block)
# ==============================================================================
def nist_rank_test_p_value(bits, M=32, Q=32):
    """
    Perform the NIST Binary Matrix Rank Test on a bit sequence.
    
    Args:
        bits (list/array): Sequence of bits (0s and 1s).
        M (int): Number of rows in the matrix (Standard NIST: 32).
        Q (int): Number of columns in the matrix (Standard NIST: 32).
        
    Returns:
        float: The p-value indicating randomness (0.0 to 1.0).
    """
    n = len(bits)
    block_size = M * Q
    num_matrices = n // block_size 
    
    # If sequence is too short to form even one matrix, return neutral p-value
    if num_matrices == 0:
        return 0.5

    # Rank counters
    rank_full = 0      # Rank = M
    rank_minus_1 = 0   # Rank = M - 1
    # Rank < M - 1 is calculated by subtraction
    
    # Analyze each matrix
    for k in range(num_matrices):
        chunk = bits[k*block_size : (k+1)*block_size]
        # Ensure correct shape
        mat = np.array(chunk, dtype=np.uint8).reshape(M, Q)
        r = calculate_rank_gf2(mat)
        
        if r == M:
            rank_full += 1
        elif r == M - 1:
            rank_minus_1 += 1
            
    rank_others = num_matrices - rank_full - rank_minus_1

    # Theoretical probabilities defined by NIST SP 800-22 for 32x32 matrices
    # NOTE: These constants are only valid for M=Q=32.
    if M == 32 and Q == 32:
        p_full   = 0.2888
        p_m1     = 0.5776
        p_other  = 0.1336
    else:
        # Fallback/Warning: Probabilities differ for other sizes (e.g., 8x8).
        # Using 32x32 approximations here, but this is technically inaccurate for small M.
        p_full   = 0.2888
        p_m1     = 0.5776
        p_other  = 0.1336

    # Chi-Square Calculation
    def chi_term(observed, expected):
        return 0.0 if expected == 0 else (observed - expected) ** 2 / expected

    chi2 = (
        chi_term(rank_full,   num_matrices * p_full) +
        chi_term(rank_minus_1, num_matrices * p_m1) +
        chi_term(rank_others,  num_matrices * p_other)
    )

    # Calculate p-value (df=2, exp(-chi2/2))
    p_value = exp(-chi2 / 2.0)
    return float(p_value)

# ==============================================================================
# 3. Feature Extraction (Segmentation)
# ==============================================================================
def extract_rank_features(bits, num_segments=8, M=32, Q=32):
    """
    Split the sequence into segments and calculate Rank Test p-value for each.
    """
    n = len(bits)
    seg_len = n // num_segments
    
    features = []
    for i in range(num_segments):
        # Extract segment
        seg_bits = bits[i*seg_len : (i+1)*seg_len]
        # Calculate p-value
        p = nist_rank_test_p_value(seg_bits, M=M, Q=Q)
        features.append(p)
        
    return np.array(features, dtype=float)

# ==============================================================================
# 4. Batch Processing
# ==============================================================================
def process_folder(input_folder, output_file, num_segments=8, max_samples=None):
    """
    Process all .txt files in a folder, extract features, and save as .npy.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Folder not found: {input_folder}")
        return

    all_features = []
    # Sort files to ensure deterministic order
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".txt")])
    
    if max_samples is not None:
        files = files[:max_samples]
        print(f"Processing top {max_samples} samples from {input_folder}...")

    for fname in files:
        path = os.path.join(input_folder, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read()
            
            # Convert string "0101" to list of integers [0, 1, 0, 1]
            bits = [1 if ch == '1' else 0 for ch in data if ch in '01']
            
            # Extract features (Default: 32x32 matrices, 8 segments per sample)
            # IMPORTANT: M=32, Q=32 is required for standard NIST probability constants
            feat = extract_rank_features(bits, num_segments=num_segments, M=32, Q=32)
            all_features.append(feat)
            
        except Exception as e:
            print(f"Error processing file {fname}: {e}")

    # Convert to numpy and save
    all_features = np.array(all_features, dtype=float)
    
    # Ensure output directory exists
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
    
    # Determine the project root or data directory
    # Assumes code is running in the parent directory of 'data' or similar
    BASE_DATA_DIR = r"F:\LiuliangData\OANC-GrAF" 
    
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
    
    OUTPUT_DIR_NAME = "Feature1150wTrain"
    OUTPUT_DIR = os.path.join(BASE_DATA_DIR, OUTPUT_DIR_NAME)
    
    # Parameters
    NUM_SEGMENTS = 8    # How many feature dimensions per sample?
    MAX_SAMPLES = None  # Set to integer (e.g., 100) for debugging, None for full run

    # --- Execution ---
    for folder_name in target_folders:
        input_path = os.path.join(BASE_DATA_DIR, folder_name)
        
        # Define output filename
        output_filename = f"{folder_name}_rank_features.npy"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"Processing: {folder_name}")
        process_folder(
            input_path, 
            output_path, 
            num_segments=NUM_SEGMENTS, 
            max_samples=MAX_SAMPLES
        )