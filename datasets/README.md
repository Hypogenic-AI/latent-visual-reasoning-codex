# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Only small JSON samples are stored.

## Dataset 1: sample_clevr (ssampa17/sample_clevr)

### Overview
- **Source**: https://huggingface.co/datasets/ssampa17/sample_clevr
- **Size**: small sample; full dataset size not specified on HF card
- **Format**: HuggingFace Dataset with images and QA pairs
- **Task**: visual question answering / compositional reasoning
- **Splits**: train/validation/test (see HF card)
- **License**: see HF card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("ssampa17/sample_clevr")
dataset.save_to_disk("datasets/sample_clevr_full")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/sample_clevr_full")
```

### Sample Data
See `datasets/sample_clevr/samples.json` for 10 streamed examples with image
metadata (image contents not stored).

### Notes
- Useful for fast iteration on CLEVR-style reasoning.
- Image fields are represented by size/mode in the sample file.

## Dataset 2: CLEVRER Counterfactual (zechen-nlp/clevrer)

### Overview
- **Source**: https://huggingface.co/datasets/zechen-nlp/clevrer
- **Size**: large; use streaming or selective download
- **Format**: HuggingFace Dataset with video references and QA programs
- **Task**: physical reasoning and counterfactual QA
- **Splits**: train/validation/test (see HF card)
- **License**: see HF card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("zechen-nlp/clevrer", "counterfactual")
dataset.save_to_disk("datasets/clevrer_counterfactual_full")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/clevrer_counterfactual_full")
```

### Sample Data
See `datasets/clevrer_counterfactual/samples.json` for 10 streamed examples with
video references and programs.

### Notes
- The dataset includes video references and structured programs for reasoning.
- Streaming is recommended to avoid large downloads.
