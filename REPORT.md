# Formalizing Latent Visual Reasoning: Token Dynamics Study

## 1. Executive Summary
This study asks whether a formal dynamical-systems view of latent visual reasoning tokens can explain strengths/weaknesses and motivate principled modifications. Using a minimal token dynamics model inspired by Slot Attention, we find that adding inertia and repulsion modestly improves clustering quality (ARI/NMI) but increases convergence time; the strongest gain appears in low-noise, matched-cardinality settings. The practical implication is that dynamics-inspired regularizers can stabilize token updates, but they must be tuned to avoid slower convergence without clear gains in harder regimes.

## 2. Goal
**Hypothesis**: A formal mathematical framework for token dynamics will reveal weaknesses (e.g., collapse or instability) and suggest principled modifications (e.g., inertia, repulsion) that improve behavior.

This matters because latent visual reasoning relies on iterative token updates, yet those dynamics are rarely analyzed formally. A clean dynamical view can expose failure modes and guide architecture or training updates.

## 3. Data Construction

### Dataset Description
The core experiments use synthetic Gaussian clusters as a controlled proxy for object-centric feature sets. Each dataset instance contains K clusters in D dimensions with known assignments.

- Source: synthetic generator in `src/token_dynamics.py`
- Size per condition: 30 seeds x 3 noise levels x 3 cluster counts = 270 runs per method
- Known biases: assumes isotropic Gaussian clusters and ignores image-level structure

We also validated the locally available dataset samples to confirm fields and structure for future extensions:
- `datasets/sample_clevr/samples.json`
- `datasets/clevrer_counterfactual/samples.json`

### Example Samples
Synthetic example (conceptual):
```
Cluster 1 center: [ 1.2, -3.4, ...]
Cluster 2 center: [-2.1,  0.7, ...]
Cluster 3 center: [ 4.5,  2.2, ...]
```

### Data Quality
- Missing values: 0% (synthetic)
- Outliers: controlled by Gaussian noise parameter
- Class distribution: balanced by construction
- Data validation: dataset sample checks recorded in `results/data_checks.json`

### Preprocessing Steps
1. Generate clusters with fixed seed for reproducibility.
2. Standardize experiment settings across methods (same data per seed).

### Train/Val/Test Splits
Not applicable; this is an analysis of dynamics on controlled synthetic data rather than supervised learning.

## 4. Experiment Description

### Methodology

#### High-Level Approach
Model token updates as a discrete-time dynamical system with attention-based attraction to features and optional inertia/repulsion terms. Evaluate stability and clustering quality across noise and cardinality.

#### Why This Method?
It directly mirrors Slot Attention-like update rules while remaining analytically interpretable. Alternatives (full Slot Attention training or MLLM perception-token pipelines) would introduce training confounds and require heavier compute.

### Implementation Details

#### Tools and Libraries
- numpy 2.4.0
- pandas 2.3.3
- scipy 1.16.3
- scikit-learn 1.8.0
- matplotlib 3.10.8

#### Algorithms/Models
Token dynamics (per step):
- Soft assignment weights based on squared Euclidean distance.
- Token update as weighted mean (baseline).
- Optional inertia term (momentum) and repulsion term (Coulomb-like).

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| n_tokens | 4 | fixed across runs |
| n_steps | 30 | fixed across runs |
| temperature | 1.0 | fixed |
| momentum | 0.5 | fixed (ablation) |
| repulsion | 0.05 | fixed (ablation) |
| noise std | [0.3, 0.6, 0.9] | controlled variation |

#### Training Procedure or Analysis Pipeline
1. Generate synthetic clusters for each seed and condition.
2. Run token dynamics for each method variant.
3. Compute ARI/NMI, SSE, steps to convergence, and collapse rate.
4. Run paired t-tests for ARI differences across methods.

### Experimental Protocol

#### Reproducibility Information
- Seeds: 0-29 (30 runs per condition)
- Hardware: CPU-only
- Execution time: ~10 seconds for full suite

#### Evaluation Metrics
- **ARI**: clustering agreement with ground truth.
- **NMI**: mutual information agreement.
- **Convergence steps**: iterations until max update delta < 1e-3.
- **Collapse rate**: heuristic for token collapse (unused clusters).

### Raw Results

#### Tables
Average across all conditions:

| Method | ARI (mean) | NMI (mean) | Steps (mean) | Collapse rate |
|--------|------------|------------|--------------|---------------|
| baseline | 0.791 | 0.882 | 11.91 | 0.00 |
| inertia | 0.799 | 0.887 | 24.78 | 0.00 |
| repulsion | 0.791 | 0.882 | 17.15 | 0.00 |
| inertia+repulsion | 0.810 | 0.893 | 25.93 | 0.00 |

Low-noise matched-cardinality example (noise=0.3, K=4):

| Method | ARI (mean) | ARI (std) | Steps (mean) |
|--------|------------|-----------|--------------|
| baseline | 0.800 | 0.215 | 4.47 |
| inertia | 0.813 | 0.216 | 24.10 |
| repulsion | 0.800 | 0.215 | 17.00 |
| inertia+repulsion | 0.866 | 0.179 | 26.50 |

#### Visualizations
- ARI vs noise: `results/plots/ari_mean.png`
- Convergence steps vs noise: `results/plots/steps_mean.png`
- Collapse rate vs noise: `results/plots/collapsed_rate.png`

#### Output Locations
- Runs: `results/token_dynamics_runs.csv`
- Summary: `results/token_dynamics_summary.csv`
- Statistical tests: `results/stat_tests.json`
- Plots: `results/plots/`

## 5. Result Analysis

### Key Findings
1. Inertia+repulsion yields the highest average ARI/NMI across conditions, but increases convergence steps substantially.
2. Repulsion alone provides little improvement over baseline under these synthetic settings.
3. Collapse was not observed under these settings, suggesting the current noise/cardinality ranges are not stressful enough to trigger it.

### Hypothesis Testing Results
- Null (H0): Modified dynamics do not improve ARI over baseline.
- Alternative (H1): Modified dynamics improve ARI.

Paired t-tests across conditions show one significant improvement for inertia+repulsion at noise=0.3, K=4 (p=0.017, d=0.46). No other comparisons are significant at alpha=0.05. With multiple-comparisons correction, this single effect may not hold, so the evidence is suggestive but not conclusive.

### Comparison to Baselines
- Average ARI improved from 0.791 (baseline) to 0.810 (inertia+repulsion), a +1.9% absolute gain, with a noticeable convergence-time cost.
- Inertia alone improves ARI slightly (+0.8%) but doubles convergence steps.

### Surprises and Insights
- The collapse heuristic never triggered, indicating the baseline dynamics are stable in this synthetic regime.
- The strongest improvement appears only in low-noise, matched-cardinality conditions.

### Error Analysis
- Under high noise (0.9), all methods degrade similarly; modifications do not rescue performance.
- Assignments become diffuse with high variance, indicating the attraction term dominates behavior.

### Limitations
- Synthetic data lacks image structure; findings may not transfer directly to real visual tokens.
- No learned perception modules are included; dynamics are evaluated in isolation.
- Fixed hyperparameters may hide regimes where repulsion is more beneficial.

## 6. Conclusions

### Summary
A dynamical-systems formalization captures token update behavior and suggests that inertia and repulsion can modestly improve clustering quality, but at the cost of slower convergence and limited gains in noisy regimes. The evidence supports the hypothesis qualitatively, but quantitative gains are modest and condition-specific.

### Implications
- Theoretical: Token dynamics can be viewed as energy-minimizing systems with optional regularizers.
- Practical: Adding controlled inertia may stabilize updates, but requires careful tuning to avoid slow convergence.

### Confidence in Findings
Moderate. The synthetic setup is controlled and reproducible, but the absence of collapse suggests we need more challenging settings or real visual features to fully stress token dynamics.

## 7. Next Steps

### Immediate Follow-ups
1. Evaluate on real object-centric features (e.g., Slot Attention encoder outputs) to test whether collapse occurs.
2. Sweep repulsion strength and temperature to identify stable/high-performing regimes.

### Alternative Approaches
- Use variational refinement dynamics (IODINE-style) as a stronger baseline.
- Introduce physics-inspired constraints for temporal token consistency.

### Broader Extensions
- Connect token dynamics to energy-based models for reasoning under uncertainty.
- Apply the framework to perception tokens in multimodal LLMs.

### Open Questions
- When do token dynamics collapse in practice?
- What is the optimal trade-off between convergence speed and clustering quality?

## References
- See `literature_review.md` and `resources.md` for full paper list and links.
