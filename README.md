# Adversarial Perturbations for Defeating Cryptographic Algorithm Identification
Recent advances in machine learning have enabled highly effective ciphertext-based cryptographic algorithm identification, posing a potential threat to encrypted communication. Inspired by adversarial example techniques, we present CSPM (Class-Specific Perturbation Mask Generation), a novel adversarial-defense framework that enhances ciphertext unidentifiability through misleading machine-learning-based cipher classifiers. CPSM constructs lightweight, reversible bit-level perturbations that alter statistical ciphertext features without affecting legitimate decryption.
The method leverages class prototypes to capture representative bit-distribution patterns for each cryptographic algorithm and integrates two complementary mechanisms—mimicry-based perturbing, which steers ciphertexts toward similar cipher classes, and distortion-based perturbing, which disrupts distinctive statistical traits—through a ranking-based greedy search.
Extensive experiments on seven widely used cryptographic algorithms and fifteen NIST statistical feature configurations demonstrate that CSPM consistently reduces algorithm-identification accuracy by over 25\%. These results confirm that perturbation position selection, rather than magnitude, dominates attack efficacy.
CSPM provides a practical defense mechanism, offering a new perspective for safeguarding encrypted communications against statistical and machine-learning-based traffic analysis.

## Data
The data we get from the OANC (https://anc.org/).

## Code

### ChooseMimic.py

This script analyzes the distribution of '0's and '1's in ciphertext sequences to identify statistical similarities between different encryption algorithms to help find the mimic class of every cryptographic algorithm. 


###   Serial Test.py, Binary Matrix Rank Test.py, Non-overlapping Template Matching Test.py
The three scripts are the examples of the NIST-15 features extraction.  

**Serial Test**: examines whether the frequency of all m-bit subsequences in the sequence is consistent with that of a true random sequence. Uneven frequency distributions may indicate non-randomness.

**Binary Matrix Rank Test**: constructs matrices from the sequence and evaluates the linear dependence of fixed-length subsequences. Strong linear dependence suggests non-randomness.

**Non-overlapping Template Matching Test**: counts the occurrences of specific non-overlapping patterns in the sequence and compares them to the expected distribution in a true random sequence. Large deviations suggest non-randomness.

### HybridSearch.py
This script implements the core **Bit-wise Hybrid Greedy Search** algorithm to generate the adversarial perturbation mask. It aims to mislead the cryptographic identification model by systematically identifying and flipping the most critical bits in the ciphertext.

1.  **Importance Ranking (Heuristic Initialization)**
    Instead of a brute-force search, the algorithm first calculates an "**Importance Score**" for each bit position based on a hybrid metric:
    * **Inter-class Difference:** How different this bit is from the target "similar" algorithm.
    * **Intra-class Stability:** How consistent this bit is within the original algorithm's samples.
    * **Result:** A prioritized list of bit positions to attack.

2.  **Iterative Greedy Optimization**
    The algorithm iterates through the ranked positions and performs a "**Flip-and-Test**" operation:
    * **Flip:** Invert the bit at the current position ($0 \to 1$ or $1 \to 0$).
    * **Query:** Send the perturbed samples to the black-box DNN model.
    * **Test:** Check if the model's confidence in the true label decreases.

3.  **Decision & Update**
    * **Keep:** If the confidence drops, the bit flip is permanently added to the adversarial mask.
    * **Revert:** If the confidence rises or stays the same, the flip is discarded.

4.  **Final Output**
    The script outputs a binary **Adversarial Mask** ($m$) of length $L$. Legitimate receivers can use $m$ to restore the original ciphertext ($c = c' \oplus m$) before decryption.


The current implementation uses the NIST Non-overlapping Template Matching Test (m=6) as the core feature extractor to drive the identification model. However, the FeatureExtractor class (in adversarial_attack.py) is designed as a modular interface.Researchers can easily extend this tool by implementing other statistical tests, such as NIST-15 features.

