import os
import numpy as np
from scipy.spatial.distance import euclidean, cosine

def get_min_sequence_length(folders):
    """
    Get the minimum sequence length across all files in the provided folders.
    This ensures all vectors are aligned to the same length for comparison.
    """
    min_length = float('inf')
    
    for folder in folders:
        if not os.path.exists(folder):
            continue
            
        files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        if not files:
            continue
            
        for file in files:
            file_path = os.path.join(folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read content and strip whitespace
                seq = f.read().strip()
                if len(seq) < min_length:
                    min_length = len(seq)
    
    return min_length

def calculate_zero_proportions(folder_path, min_length):
    """
    Calculate the 'zero proportion vector' for a given folder.
    It considers only the first `min_length` bits of each sequence.
    
    Returns:
        numpy.ndarray: A vector where each element represents the proportion 
                       of '0's at that bit position across all samples.
    """
    if not os.path.exists(folder_path):
        return None

    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not files:
        return None
    
    sequences = []
    for file in files:
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            # Truncate sequence to the minimum length
            seq = f.read().strip()[:min_length]
            sequences.append(seq)
    
    if not sequences:
        return None

    # Convert to numpy array (num_samples, min_length)
    # Assumes input files contain string representations of bits (e.g., "010101")
    try:
        seq_array = np.array([[int(bit) for bit in seq] for seq in sequences])
    except ValueError:
        print(f"Error: Non-integer data found in folder {folder_path}")
        return None
    
    # Calculate proportion of 1s at each position
    one_proportions = np.mean(seq_array, axis=0)
    
    # Calculate proportion of 0s (1 - proportion of 1s)
    zero_proportions = 1 - one_proportions
    
    return zero_proportions

def find_closest_folder(main_folder, compare_folders):
    """
    Find the folder in `compare_folders` that is most similar to `main_folder`.
    Uses Euclidean distance and Cosine similarity based on zero-proportion vectors.
    """
    # 1. Determine minimum length across all datasets
    all_folders = [main_folder] + compare_folders
    min_length = get_min_sequence_length(all_folders)
    
    if min_length == float('inf'):
        print("Error: No valid sample files found in any folder.")
        return None
    
    print(f"Minimum sequence length across all datasets: {min_length}")
    
    # 2. Calculate vector for the main folder
    main_vector = calculate_zero_proportions(main_folder, min_length)
    if main_vector is None:
        print(f"Error: No samples found in target folder: {main_folder}")
        return None
    
    # print(f"Target Folder Vector (First 10 dims): {main_vector[:10]}...") # Optional debug
    
    # 3. Calculate vectors for comparison folders
    folder_vectors = {}
    for folder in compare_folders:
        vector = calculate_zero_proportions(folder, min_length)
        if vector is not None:
            folder_vectors[folder] = vector
            # print(f"Processed: {folder}")
        else:
            print(f"Warning: Skipping empty or invalid folder: {folder}")
    
    if not folder_vectors:
        print("Error: No valid comparison folders found.")
        return None
    
    # 4. Compute distances and similarities
    distances = {}
    similarities = {}
    
    for folder, vector in folder_vectors.items():
        # Euclidean Distance (Lower is better)
        dist = euclidean(main_vector, vector)
        distances[folder] = dist
        
        # Cosine Similarity (Higher is better)
        # scipy cosine returns distance (1 - similarity), so we do 1 - dist
        sim = 1 - cosine(main_vector, vector)
        similarities[folder] = sim
    
    # 5. Find best matches
    closest_by_distance = min(distances, key=distances.get)
    min_distance = distances[closest_by_distance]
    
    closest_by_similarity = max(similarities, key=similarities.get)
    max_similarity = similarities[closest_by_similarity]
    
    print("-" * 50)
    print("ANALYSIS RESULTS")
    print("-" * 50)
    print(f"Target Folder: {main_folder}\n")
    print(f"Closest by Euclidean Distance (Min):")
    print(f"  Folder: {closest_by_distance}")
    print(f"  Value:  {min_distance:.4f}")
    print("-" * 30)
    print(f"Closest by Cosine Similarity (Max):")
    print(f"  Folder: {closest_by_similarity}")
    print(f"  Value:  {max_similarity:.4f}")
    print("-" * 50)
    
    return {
        "closest_by_distance": closest_by_distance,
        "min_distance": min_distance,
        "closest_by_similarity": closest_by_similarity,
        "max_similarity": max_similarity
    }

if __name__ == "__main__":
    # --- Configuration ---
    # Assuming the script is run from the project root.
    # Data is expected to be in a 'data' subfolder.
    
    base_data_path = "data" # Or "OANC-GrAF" depending on your repo structure
    
    # Target folder to analyze
    target_folder_name = "Grain128encry50wRaodong"
    main_folder_path = os.path.join(base_data_path, target_folder_name)
    
    # List of folders to compare against
    compare_folder_names = [
        "AESencry50wRaodong",
        "RSAencry50wRaodong",
        "CAMELLIAencry50wRaodong",
        "KASUMIencry50wRaodong",
        "PRESENTencry50wRaodong", 
        "DESencry50wRaodong"
    ]
    
    # Construct relative paths for comparison folders
    compare_folder_paths = [
        os.path.join(base_data_path, name) for name in compare_folder_names
    ]
    
    # Run analysis
    print(f"Starting analysis on {len(compare_folder_paths)} folders...")
    result = find_closest_folder(main_folder_path, compare_folder_paths)
    
    if result:
        print("\nProcess completed successfully.")