# Literature Review

## Research Area Overview
Latent visual reasoning focuses on manipulating intermediate visual tokens (object-centric slots, perception tokens, or latent scene variables) to enable systematic reasoning about objects and their relations. The literature spans object-centric representation learning (Slot Attention, MONet, IODINE), newer works that add explicit reasoning modules or relational abstraction, and recent multimodal models that introduce perception tokens to expose structured visual intermediates. Parallel work in physical reasoning from video provides complementary constraints for token dynamics and inductive biases.

## Key Papers

#### Paper 1: Perception Tokens Enhance Visual Reasoning in Multimodal Language Models
- **Authors**: Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, Ethan Shen, Dongping Chen, Linda G. Shapiro, et al.
- **Year**: 2024
- **Source**: arXiv (2412.03548)
- **Key Contribution**: Introduces perception tokens that encode depth or object information and exposes them to the language model to improve reasoning.
- **Methodology**: Adds intermediate perception tokens derived from specialized vision tools; integrates them into MLLM reasoning.
- **Datasets Used**: Not specified in abstract; evaluated on visual reasoning benchmarks requiring depth/instance reasoning.
- **Results**: Improves performance on tasks that benefit from explicit perception signals.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Directly ties latent tokens to reasoning performance and motivates formalizing token dynamics.

#### Paper 2: Introducing Visual Perception Token into Multimodal Large Language Model
- **Authors**: Runpeng Yu, Xinyin Ma, Xinchao Wang
- **Year**: 2025
- **Source**: arXiv (2502.17425)
- **Key Contribution**: Adds controllable visual perception tokens to allow selective re-perception and targeted reasoning.
- **Methodology**: Introduces a perception control mechanism to focus on specific regions or objects.
- **Datasets Used**: Not specified in abstract; reported on MLLM visual reasoning tasks.
- **Results**: Reports stronger spatial reasoning and fine-grained understanding.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Highlights the need for formal token control and dynamics in latent visual reasoning.

#### Paper 3: Reasoning-Enhanced Object-Centric Learning for Videos
- **Authors**: Jian Li, Pu Ren, Yang Liu, Hao Sun
- **Year**: 2024
- **Source**: arXiv (2403.15245)
- **Key Contribution**: Combines slot-based video object decomposition with an explicit reasoning module.
- **Methodology**: Builds object-centric video representations and adds reasoning/prediction heads for physical reasoning.
- **Datasets Used**: Not specified in abstract; focuses on multi-object video reasoning tasks.
- **Results**: Improves reasoning and prediction quality over slot-only baselines.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Supports a framework where latent tokens interface with reasoning modules.

#### Paper 4: Systematic Visual Reasoning through Object-Centric Relational Abstraction
- **Authors**: Taylor W. Webb, Shanka Subhra Mondal, Jonathan D. Cohen
- **Year**: 2023
- **Source**: arXiv (2306.02500)
- **Key Contribution**: Learns relational abstractions over object-centric representations to improve systematic generalization.
- **Methodology**: Extracts object-centric inputs, then applies relational abstraction for reasoning over relations.
- **Datasets Used**: Not specified in abstract; targets systematic generalization tasks.
- **Results**: Improves generalization to novel relational patterns.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Motivates formal relational operators over latent tokens.

#### Paper 5: Compositional Physical Reasoning of Objects and Events from Videos
- **Authors**: Zhenfang Chen, Shilong Dong, Kexin Yi, Yunzhu Li, Mingyu Ding, Antonio Torralba, et al.
- **Year**: 2024
- **Source**: arXiv (2408.02687)
- **Key Contribution**: Infers latent physical properties from video and composes them to predict future dynamics.
- **Methodology**: Learns object-level physical property estimates and uses them for compositional prediction.
- **Datasets Used**: Not specified in abstract; focuses on physics reasoning from video.
- **Results**: Improves prediction of hidden physical properties and dynamics.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Connects latent token properties to differentiable physics dynamics.

#### Paper 6: Attention Normalization Impacts Cardinality Generalization in Slot Attention
- **Authors**: Markus Krimmel, Jan Achterhold, Joerg Stueckler
- **Year**: 2024
- **Source**: arXiv (2407.04170)
- **Key Contribution**: Analyzes how attention normalization affects slot count generalization.
- **Methodology**: Systematic study of normalization variants in Slot Attention with controlled evaluations.
- **Datasets Used**: Not specified in abstract; likely synthetic multi-object datasets.
- **Results**: Shows attention normalization choices influence generalization to different object counts.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Highlights sensitivity of token dynamics to architectural choices.

#### Paper 7: Object-Centric Learning with Slot Attention
- **Authors**: Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, et al.
- **Year**: 2020
- **Source**: arXiv (2006.15055)
- **Key Contribution**: Introduces Slot Attention for object-centric set representations from perceptual features.
- **Methodology**: Iterative attention-based slot updates that compete to explain input features.
- **Datasets Used**: Synthetic multi-object datasets (e.g., CLEVR, Multi-dSprites).
- **Results**: Produces unsupervised object segmentation and improves downstream reasoning.
- **Code Available**: Community implementations (e.g., `code/slot-attention-pytorch`).
- **Relevance to Our Research**: Core latent-token mechanism to formalize and analyze.

#### Paper 8: MONet: Unsupervised Scene Decomposition and Representation
- **Authors**: Christopher P. Burgess, Loic Matthey, Nicholas Watters, Rishabh Kabra, Irina Higgins, Matt Botvinick, et al.
- **Year**: 2019
- **Source**: arXiv (1901.11390)
- **Key Contribution**: Unsupervised scene decomposition with attention-based segmentation and variational components.
- **Methodology**: Sequential attention masks with VAE per object for scene decomposition.
- **Datasets Used**: Synthetic multi-object datasets (e.g., CLEVR, Multi-dSprites).
- **Results**: Learns object factors and supports compositional generalization.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Early latent tokenization of visual scenes, useful for formal dynamics.

#### Paper 9: Multi-Object Representation Learning with Iterative Variational Inference (IODINE)
- **Authors**: Klaus Greff, Raphaël Lopez Kaufman, Rishabh Kabra, Nick Watters, Chris Burgess, Daniel Zoran, et al.
- **Year**: 2019
- **Source**: arXiv (1903.00450)
- **Key Contribution**: Iterative inference to discover objects and their attributes in an unsupervised manner.
- **Methodology**: Variational inference loop with refinement network over object slots.
- **Datasets Used**: Synthetic multi-object datasets (e.g., CLEVR, Multi-dSprites).
- **Results**: Produces object-level representations without supervision.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Provides an alternative token update rule to compare with Slot Attention.

### Common Methodologies
- Object-centric tokenization via iterative attention or variational inference (Slot Attention, MONet, IODINE).
- Relational abstraction layers to reason over object slots (Webb et al.).
- Perception token interfaces to expose intermediate vision outputs in MLLMs.

### Standard Baselines
- Slot Attention and Slot-Attention-style variants for object tokenization.
- MONet and IODINE as classic object-centric baselines.
- Relational Networks for reasoning over object feature sets.

### Evaluation Metrics
- Reconstruction metrics (e.g., MSE/SSIM) for object-centric decomposition.
- Reasoning accuracy on visual QA or physical reasoning tasks.
- Generalization metrics across object count and compositional splits.

### Datasets in the Literature
- CLEVR and Multi-dSprites for synthetic object-centric learning.
- CLEVRER-style physical reasoning datasets for event-based reasoning.
- Multi-object video datasets (e.g., MOVi-style) for dynamics and tracking.

### Gaps and Opportunities
- Limited formal analysis of slot/token dynamics beyond empirical evaluations.
- Few bridges between perception-token mechanisms in MLLMs and object-centric dynamics theory.
- Need principled links between token updates and physical reasoning constraints.

### Recommendations for Our Experiment
- **Recommended datasets**: CLEVR-style object-centric scenes and CLEVRER-style physical reasoning videos for controlled evaluation.
- **Recommended baselines**: Slot Attention, MONet, IODINE, and relational abstraction models.
- **Recommended metrics**: Reasoning accuracy, reconstruction fidelity, and generalization across object counts.
- **Methodological considerations**: Compare token update dynamics (attention vs variational refinement) and test robustness to perceptual noise.
