# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories.

### Papers
Total papers downloaded: 9

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Perception Tokens Enhance Visual Reasoning in Multimodal Language Models | Bigverdi et al. | 2024 | papers/2412.03548_perception_tokens_visual_reasoning_mllms.pdf | Perception tokens for MLLM reasoning |
| Introducing Visual Perception Token into Multimodal Large Language Model | Yu et al. | 2025 | papers/2502.17425_visual_perception_token_mllm.pdf | Controllable perception tokens |
| Reasoning-Enhanced Object-Centric Learning for Videos | Li et al. | 2024 | papers/2403.15245_reasoning_enhanced_object_centric_videos.pdf | Slot-based video + reasoning |
| Systematic Visual Reasoning through Object-Centric Relational Abstraction | Webb et al. | 2023 | papers/2306.02500_object_centric_relational_abstraction.pdf | Relational abstraction over objects |
| Compositional Physical Reasoning of Objects and Events from Videos | Chen et al. | 2024 | papers/2408.02687_compositional_physical_reasoning_videos.pdf | Physical property inference |
| Attention Normalization Impacts Cardinality Generalization in Slot Attention | Krimmel et al. | 2024 | papers/2407.04170_slot_attention_cardinality_generalization.pdf | Slot normalization analysis |
| Object-Centric Learning with Slot Attention | Locatello et al. | 2020 | papers/2006.15055_slot_attention.pdf | Slot Attention core model |
| MONet: Unsupervised Scene Decomposition and Representation | Burgess et al. | 2019 | papers/1901.11390_monet_scene_decomposition.pdf | Scene decomposition with attention |
| Multi-Object Representation Learning with Iterative Variational Inference | Greff et al. | 2019 | papers/1903.00450_iodine_iterative_variational_inference.pdf | Iterative variational refinement |

See `papers/README.md` for detailed descriptions.

### Datasets
Total datasets downloaded: 2 (sample files)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| sample_clevr | HuggingFace (ssampa17/sample_clevr) | sample only | VQA / compositional reasoning | datasets/sample_clevr/ | Streamed 10-sample JSON |
| CLEVRER counterfactual | HuggingFace (zechen-nlp/clevrer) | sample only | Physical reasoning QA | datasets/clevrer_counterfactual/ | Streamed 10-sample JSON |

See `datasets/README.md` for full download instructions.

### Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| relational-networks | https://github.com/kimhc6028/relational-networks | Relational reasoning baseline | code/relational-networks/ | Includes Sort-of-CLEVR generator |
| clevr-dataset-gen | https://github.com/facebookresearch/clevr-dataset-gen | CLEVR dataset generation | code/clevr-dataset-gen/ | Official CLEVR tools |
| multi_object_datasets | https://github.com/deepmind/multi_object_datasets | Multi-object dataset utilities | code/multi_object_datasets/ | Includes segmentation metrics |
| slot-attention-pytorch | https://github.com/evelinehong/slot-attention-pytorch | Slot Attention implementation | code/slot-attention-pytorch/ | Community implementation |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- arXiv keyword search for latent visual reasoning, perception tokens, and object-centric reasoning.
- Focused on object-centric learning foundations and recent perception-token works.
- Used HuggingFace dataset search for CLEVR/CLEVRER resources.

### Selection Criteria
- Direct relevance to latent token mechanisms and object-centric reasoning.
- Coverage of both foundational models and recent perception-token approaches.
- Availability of datasets or code for downstream experimentation.

### Challenges Encountered
- Some datasets are large and not suitable for full download in this phase.
- Several recent papers do not specify datasets in their abstracts.

### Gaps and Workarounds
- Downloaded only small dataset samples; full download instructions provided.
- For missing dataset specifics, the paper PDFs can be consulted later.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: CLEVR-style synthetic datasets for controlled object-centric evaluation; CLEVRER for physical reasoning.
2. **Baseline methods**: Slot Attention, MONet, IODINE, and relational networks.
3. **Evaluation metrics**: Reasoning accuracy, reconstruction fidelity, and generalization to new object counts.
4. **Code to adapt/reuse**: `code/slot-attention-pytorch/` for object-centric tokens; `code/relational-networks/` for relational baselines; dataset generation in `code/clevr-dataset-gen/`.

## Usage in This Study
- Validated sample dataset structures in `datasets/sample_clevr/samples.json` and `datasets/clevrer_counterfactual/samples.json` (see `results/data_checks.json`).
- Ran controlled synthetic experiments to analyze token dynamics; this provides a lightweight proxy for object-centric token behavior while avoiding full model training.
