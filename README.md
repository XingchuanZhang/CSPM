# CSPM: Class-Specific Perturbation Mask Generation

> **Adversarial Perturbations for Defeating Cryptographic Algorithm Identification**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)

## 📖 Introduction

Recent advances in machine learning have enabled highly effective ciphertext-based cryptographic algorithm identification, posing a potential threat to encrypted communication. 

**CSPM (Class-Specific Perturbation Mask Generation)** is a novel adversarial-defense framework designed to safeguard encrypted communications. It constructs lightweight, reversible bit-level perturbations that alter statistical ciphertext features to mislead machine-learning-based classifiers without affecting legitimate decryption.

**Key Features:**
* **Mimicry-based Perturbing:** Steers ciphertexts toward statistically similar cipher classes.
* **Distortion-based Perturbing:** Disrupts distinctive statistical traits.
* **High Efficacy:** Reduces algorithm-identification accuracy by over **25%** across widely used cryptographic algorithms.

---

## 📂 Data

The dataset used in this research is sourced from the **OANC** (Open American National Corpus).
* **Source:** [https://anc.org/](https://anc.org/)

---

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
    cd your-repo
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Modules & Scripts

### 1. Preprocessing
| Script | Description |
| :--- | :--- |
| `ChooseMimic.py` | Analyzes the distribution of '0's and '1's in ciphertext sequences. It identifies statistical similarities between encryption algorithms to determine the optimal **"mimic class"** (target label) for the attack. |

### 2. NIST Feature Extraction
These scripts implement key components of the NIST-15 statistical test suite to extract features from ciphertext.

* **`Serial Test.py`**
    Examines whether the frequency of all $m$-bit subsequences is consistent with a true random sequence. Uneven distributions indicate non-randomness.
* **`Binary Matrix Rank Test.py`**
    Constructs matrices from the sequence to evaluate linear dependence. Strong linear dependence suggests non-randomness.
* **`Non-overlapping Template Matching Test.py`**
    Counts occurrences of specific non-overlapping patterns. Large deviations from expected counts suggest non-randomness.

> **💡 Extensibility:** The current implementation uses the *Non-overlapping Template Matching Test (m=6)* as the core feature extractor. However, the `FeatureExtractor` class is designed as a modular interface. Researchers can easily extend this tool by implementing other NIST-15 tests.

---

## 🧠 Core Algorithm: HybridSearch.py

This script implements the **Bit-wise Hybrid Greedy Search** algorithm. It aims to mislead the identification model by systematically identifying and flipping the most critical bits in the ciphertext.

### Workflow

#### Step 1: Importance Ranking (Heuristic Initialization)
Instead of a brute-force search, we calculate an **"Importance Score"** for each bit position based on a hybrid metric:
* **Inter-class Difference:** How different this bit is from the target "similar" algorithm.
* **Intra-class Stability:** How consistent this bit is within the original algorithm's samples.
* **Result:** A prioritized list of bit positions to attack.

#### Step 2: Iterative Greedy Optimization
The algorithm iterates through the ranked positions and performs a **"Flip-and-Test"** operation:
* **Flip:** Invert the bit at the current position ($0 \to 1$ or $1 \to 0$).
* **Query:** Send the perturbed samples to the black-box DNN model.
* **Test:** Check if the model's confidence in the true label decreases.

#### Step 3: Decision & Update
* **Keep:** If the confidence drops, the bit flip is permanently added to the adversarial mask.
* **Revert:** If the confidence rises or stays the same, the flip is discarded.

#### Step 4: Final Output
The script generates a binary **Adversarial Mask** ($m$) of length $L$.
* **Sender:** Computes perturbed ciphertext $c' = c \oplus m$.
* **Receiver:** Restores original ciphertext $c = c' \oplus m$ before decryption.

