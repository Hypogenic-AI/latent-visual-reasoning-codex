# Latent Visual Reasoning Token Dynamics

This project formalizes latent token updates as a dynamical system and evaluates stability/quality on synthetic object-feature clusters. The goal is to identify strengths and weaknesses of common token dynamics and test principled modifications (inertia, repulsion).

Key findings:
- Inertia+repulsion yields the highest average ARI/NMI but slows convergence.
- Repulsion alone offers minimal gains under the tested synthetic conditions.
- Collapse was not observed at the tested noise/cardinality ranges.

## Reproduce
```bash
# From workspace root
uv venv
source .venv/bin/activate
uv add numpy pandas matplotlib scipy scikit-learn

python src/log_env.py
python src/data_checks.py
python src/run_experiments.py
python src/analyze_results.py
```

## File Structure
- `src/token_dynamics.py`: token dynamics model and utilities
- `src/run_experiments.py`: experiment runner
- `src/analyze_results.py`: statistics + plots
- `results/`: CSVs, plots, and statistical tests
- `REPORT.md`: full research report

See `REPORT.md` for full methodology, tables, and analysis.
