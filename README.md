# RNA BPTT Reinforcement Learning Simulator

A differentiable reinforcement learning framework for RNA sequence generation, targeting motif placement, GC-balance, sequence entropy, and structural accuracy via a ViennaRNA oracle.

---

##  Purpose
This project addresses the challenge of designing RNA sequences with specific structural and sequence properties using a reinforcement learning (RL) approach enhanced with differentiable components. It models RNA sequence generation as a sequential decision process where an RL agent constructs one nucleotide at a time.

---

##  Core Components

### 1. **Environment: `DifferentiableRNAEnvironment`**
- Generates state transitions conditioned on the agent's output (next base).
- Integrates reward signals from:
  - GC-content proximity.
  - Motif presence at randomized locations.
  - Sequence entropy (diversity).
  - Repetition penalty.
  - ViennaRNA structure reward.

### 2. **Agent: `Brain`**
- A feedforward network producing RNA base logits and optional memory update vectors.
- Supports multiple memory modules: `No_memory`, `Minimal_GRU`, `Full_GRU`, `Full_LSTM`, `CARU`.

### 3. **Training**
- Trained using REINFORCE-style policy gradient.
- Reward is computed over a full trajectory of generated nucleotides.
- Randomizes GC and motif targets per batch for generalization.

---

## Reward Functions and Objectives

This simulator trains an agent to generate RNA sequences through reinforcement learning. The reward signal is crafted as a combination of biologically inspired objectives:

1. **GC Content Reward**
   - Encourages the generated sequence to match a randomized GC ratio target (e.g., 40–60%).
   - Implemented via a soft reward: `1.0 - abs(gc_content - target_gc) * 2`.

2. **Motif Presence Reward**
   - A randomly selected nucleotide (e.g., A/C/G/U) is targeted to appear at a randomized position in the sequence.
   - This introduces position-specific reward shaping.

3. **Structural Surrogate Reward**
   - Based on entropy of softmax logits as a proxy for folding complexity.
   - Encourages structured diversity in the generated sequence.

4. **ViennaRNA Oracle Reward**
   - At the final timestep, the full sequence is evaluated using `RNAfold` to compute its free energy.
   - The reward is the negative free energy (lower energy → higher reward).

5. **Repetition Penalty**
   - Discourages excessive repetition of a single nucleotide.
   - Penalizes if one nucleotide dominates the sequence.

6. **Entropy Bonus and KL Regularization**
   - Adds entropy of the action distribution to encourage diversity.
   - Applies KL-divergence against a uniform prior to prevent mode collapse.

These rewards are designed to simulate realistic biological pressures in RNA structure, sequence diversity, and function. By combining soft and oracle-based rewards, the agent is capable of learning diverse, biologically plausible RNA sequences across different memory-augmented policy architectures.

---

##  Visualizations
- **Training Curves**: Loss and reward progression.
- **GC Histograms**: GC content of generated sequences.
- **Sampled Sequences**: Representative decoded RNAs.

---

##  Memory Model Comparison
Experiments benchmark:
- Final reward across architectures.
- GC content distribution.
- Structural feedback (from ViennaRNA).
- Diversity of sampled sequences.

---

##  Status
- [x] Differentiable environment
- [x] Motif and GC targeting
- [x] Memory unit exploration
- [x] Entropy & repetition regularization
- [x] ViennaRNA folding feedback
- [x] Batchwise variation in reward targets
- [x] Full pipeline and visual reporting

---

##  Future Directions
- Incorporate PPO for more stable learning
- Add support for target secondary structure patterns (dot-bracket)
- Enable interactive motif definition
- Visualize RNA structures from ViennaRNA

---

##  Example Output
```
Iteration 950 | Loss: -58.11 | Avg Reward: 58.11
Sampled Sequences:
ACGUGUAAUGCAUGCAUGCAUGCAUGCAUG
...
```

---

##  Dependencies
- TensorFlow 2.x
- NumPy
- Matplotlib
- ViennaRNA (CLI tools like RNAfold)

---

##  How to Run
```bash
python rna_bptt_rl_simulator.py
```

Results and plots will be saved in the working directory.

---

##  Citation
Coming soon. Please contact the author for draft manuscript or project use.

---

##  Contributors
- Mahrad Pisheh Var – Modeling, architecture, implementation

---

