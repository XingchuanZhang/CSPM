import os
import numpy as np
from math import erfc, sqrt
from collections import Counter

# ==============================================================================
# Configuration: Default NIST-like Templates (m=6)
# ==============================================================================
# You can modify this list to include different templates as needed.
DEFAULT_TEMPLATES_M6 = [
    "000000", "111111", "010101", "101010",
    "001100", "110011", "000111", "111000",
    "011011", "100100", "110110", "001001",
    "111100", "000011", "010010", "101101",
    "011110", "100001", "011100", "100011"
]

# ==============================================================================
# 1. NIST Non-overlapping Template Matching Test Logic
# ==============================================================================
def non_overlapping_template_test(bits, template, m):
    """
    Perform the NIST Non-overlapping Template Matching Test.
    
    Args:
        bits (list[int]): The binary sequence to test.
        template (list[int]): The specific pattern to search for.
        m (int): The length of the template.
        
    Returns:
        float: The p-value indicating the randomness of the sequence regarding the template.
    """
    n = len(bits)
    
    # Check if sequence is long enough for at least one block
    if n < m:
        return 0.5

    # Count non-overlapping occurrences
    count = 0
    i = 0
    while i <= n - m:
        # Check if the slice matches the template
        if bits[i : i+m] == template:
            count += 1
            i += m  # Move index forward by m (Non-overlapping)
        else:
            i += 1  # Move index forward by 1

    # Calculate Theoretical Mean (mu) and Variance (sigma^2)
    # Based on NIST SP 800-22 definitions
    mu = (n - m + 1) / (2 ** m)
    variance = n * ((1.0 / (2 ** m)) - (2 * m - 1) / (2.0 ** (2 * m)))
    sigma = sqrt(variance)

    # Avoid division by zero
    if sigma == 0:
        return 1.0

    # Calculate Chi-square / Z-score statistics
    z = (count - mu) / sigma
    
    # Calculate p-value using Complementary Error Function (erfc)
    p_value = erfc(abs(z) / sqrt(2.0))
    
    return float(p_value)

# ==============================================================================
# 2. Feature Extraction (20 Templates)
# ==============================================================================
def extract_template_features(bits, m=6, templates=None):
    """
    Run the Non-overlapping Template Test for a specific set of templates.
    Returns a feature vector where each dimension is the p-value for a specific template.
    
    Args:
        bits (list[int]): Input binary sequence.
        m (int): Length of the template.
        templates (list[str]): List of template strings (e.g., ["000", "001"]). 
                               If None, uses the default 20 templates.
                               
    Returns:
        np.array: A 20-dimensional feature vector (floats).
    """
    if templates is None:
        templates = DEFAULT_TEMPLATES_M6

    features = []
    for t_str in templates:
        # Convert string template "101" to list [1, 0, 1]
        tpl = [int(ch) for ch in t_str]
        
        p_val = non_overlapping_template_test(bits, tpl, m)
        features.append(p_val)
        
    return np.array(features, dtype=float)

# ==============================================================================
# 3. Batch Processing
# ==============================================================================
def process_folder(input_folder, output_file, max_samples=None):
    """
    Read all .txt files in a folder, extract 20-dim template features, and save as .npy.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Folder not found - {input_folder}")
        return

    all_features = []
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".txt")])
    
    if max_samples is not None:
        files = files[:max_samples]
        print(f"Processing top {max_samples} samples from {os.path.basename(input_folder)}...")

    for fname in files:
        path = os.path.join(input_folder, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read()
            
            # Convert content to binary list
            bits = [1 if ch == '1' else 0 for ch in data if ch in '01']

            # Extract features (20-dimensional vector)
            feat = extract_template_features(bits, m=6)
            all_features.append(feat)
            
        except Exception as e:
            print(f"Error processing file {fname}: {e}")

    # Convert list to numpy array
    all_features = np.array(all_features, dtype=float)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, all_features)
    
    print(f"Saved: {output_file}")
    print(f"Shape: {all_features.shape}")
    if len(all_features) > 0:
        print(f"Sample Feature Vector (First 5 dims): {all_features[0][:5]}")
    print("-" * 50)
    
    return all_features

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    
    # Base directory for data (Relative path)
    # Assumes your data is in a folder named 'OANC-GrAF' inside the project folder
    # or you can change this to match your structure.
    BASE_DATA_DIR = r"F:\LiuliangData\OANC-GrAF" # Ideally, change this to "data" for GitHub
    
    # Subfolders to process
    target_folders = [
        "AESencry50wTrain",
        "CAMELLIAencry50wTrain",
        "DESencry50wTrain",
        "Grain128encry50wTrain",
        "KASUMIencry50wTrain",
        "PRESENTencry50wTrain",
        "RSAencry50wTrain"
    ]

    # Output directory name
    OUTPUT_DIR_NAME = "Feature1250wTrain"
    OUTPUT_DIR = os.path.join(BASE_DATA_DIR, OUTPUT_DIR_NAME)

    # --- Execution Loop ---
    for folder_name in target_folders:
        input_path = os.path.join(BASE_DATA_DIR, folder_name)
        
        # Define output filename (e.g., AESencry50wTrain12.npy)
        output_filename = f"{folder_name}12.npy"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"Processing Folder: {folder_name}")
        
        # Run processing (set max_samples=5 for debugging)
        process_folder(input_path, output_path, max_samples=None)