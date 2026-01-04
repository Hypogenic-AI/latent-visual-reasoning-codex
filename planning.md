## Research Question
Can a formal dynamical-systems view of latent visual reasoning tokens (e.g., Slot Attention-style updates and perception tokens) explain strengths/weaknesses of token mechanisms and motivate principled architectural modifications?

## Background and Motivation
Latent visual reasoning relies on token dynamics that iteratively bind scene features to object-like slots. While empirical work shows strong reasoning gains, there is limited formal analysis of how these token update rules behave (stability, collapse, cardinality generalization). A lightweight theoretical framework grounded in dynamical systems can help identify failure modes and suggest modifications such as repulsion or inertia.

## Hypothesis Decomposition
1. Token updates can be formalized as a discrete-time dynamical system with identifiable energy terms (attraction to features, competition/normalization, and optional repulsion).
2. This formalization predicts measurable behaviors: convergence speed, collapse likelihood, and sensitivity to object count.
3. Adding principled terms (e.g., inertia or repulsion) improves stability and assignment quality under noise and cardinality shifts.

## Proposed Methodology

### Approach
Develop a minimal mathematical model for token updates (soft-attention clustering) and test its predictions with controlled synthetic experiments. This keeps the experimental loop tight while connecting directly to the latent-token dynamics described in the literature (Slot Attention, IODINE, perception tokens).

### Experimental Steps
1. **Formalization**: Define token updates as a discrete dynamical system with attention-based attraction and optional repulsion/inertia terms; derive expected behaviors (e.g., collapse when normalization is sharp).
2. **Synthetic data generator**: Create simple object-feature clusters with known ground truth to measure clustering/assignment quality.
3. **Baselines**: Implement standard soft k-means/Slot-Attention-like updates as baseline dynamics.
4. **Proposed modifications**: Add (a) inertia/momentum and (b) inter-token repulsion; test each independently and together.
5. **Evaluation**: Measure assignment quality (ARI/NMI), convergence steps, and collapse rate across seeds and object counts.
6. **Statistics**: Compare variants with paired tests and effect sizes to assess robustness.

### Baselines
- Soft k-means / Slot-Attention-style update (attention-based weighted mean).
- No-repulsion/no-inertia dynamics.

### Evaluation Metrics
- **Adjusted Rand Index (ARI)**: agreement with ground-truth cluster assignments.
- **NMI**: alternative clustering agreement metric.
- **Convergence steps**: iterations until update delta < threshold.
- **Collapse rate**: fraction of runs where multiple tokens converge to same cluster.

### Statistical Analysis Plan
- Paired t-tests (or Wilcoxon if non-normal) across seeds for ARI and convergence steps.
- Effect size (Cohen’s d) and 95% confidence intervals.
- Significance level: α = 0.05 with FDR correction for multiple comparisons.

## Expected Outcomes
- Baseline dynamics show faster collapse under high noise or cardinality mismatch.
- Repulsion reduces collapse and improves ARI/NMI, possibly at a small convergence-speed cost.
- Inertia stabilizes updates and reduces oscillations in high-noise settings.

## Timeline and Milestones
- Phase 1 (planning): 20–30 min
- Phase 2 (setup/data checks): 20 min
- Phase 3 (implementation): 60–90 min
- Phase 4 (experiments): 60–90 min
- Phase 5 (analysis): 30–45 min
- Phase 6 (documentation): 30–45 min

## Potential Challenges
- Synthetic data may not fully capture image-level complexity; note limitation.
- Hyperparameter sensitivity (repulsion strength, temperature) may affect conclusions.
- Small sample size; mitigate with multiple seeds and confidence intervals.

## Success Criteria
- A clear formalization of token dynamics with explicit update equations.
- Empirical evidence (plots + statistics) showing how modifications affect stability/quality.
- Reported strengths/weaknesses aligned with literature and theoretical expectations.
