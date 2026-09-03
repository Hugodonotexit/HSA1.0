# HSA: Hierarchical Softmax Attention

A reference implementation of **Hierarchical Softmax Attention (HSA)**, a `transformers`-compatible
attention mechanism that achieves O(log N) sequence-axis dependency depth while maintaining full
fp16 training.

**Pre-trained weights:** [Hugodonotexit/HSA1.0_base](https://huggingface.co/Hugodonotexit/HSA1.0_base) on the Hugging Face Hub.

## Model Architecture

**Key Innovation:** Exact local softmax attention over the last `window_size` tokens, plus an
O(log N) read over a causal dyadic tree of RMS-normalized, log-mass-weighted key/value summaries
built via recursive soft-assignment pooling ("slot attention").

### Core Properties

- **Attention Complexity:** O(log N) dependency depth (vs O(N) for SSM/linear-attention or unbounded for flat KV cache)
- **Numerical Precision:** Full fp16 training (no fp32 accumulators needed)
- **Training:** Compatible with int4 quantization, gradient accumulation, and DDP
- **Inference:** Efficient incremental decode with bounded KV cache growth

### Architecture Details

- **Local Attention Window:** O(1) exact softmax over recent tokens
- **Frontier Attention:** Hierarchical log-mass-weighted key/value summaries
- **Consistency Guarantee:** Prefill (training) and decode paths use identical insertion primitives
- **Acceleration Support:**
  - xformers: Memory-efficient attention via dense bias
  - Liger kernels: Fused RMSNorm, SwiGLU MLP, Rotary embeddings, Cross-entropy loss

### Model Configuration

- **Parameters:** 1,052M (1B)
- **Hidden Size:** 2048
- **Intermediate Size (MLP):** 8192
- **Attention Heads:** 16
- **Head Dimension:** 128
- **Window Size:** 128 (local attention window)

## Technical Details

### Attention Mechanism

The HSA attention layer computes:
1. **Local branch:** Exact softmax over the last `window_size` tokens (default 128)
2. **Frontier branch:** Log-mass-weighted reads from a tree of hierarchical summaries

The two branches are concatenated and fed through a single softmax, making it a consistent
multiplicative estimator of full softmax attention.

### Dependency Depth

- Full attention: O(N) gradient flow steps for any token to influence token 1
- Linear/SSM attention: O(N) cumulative numerical operations (despite O(1) token-to-token)
- **HSA:** O(log N) true dependency depth (causal tree of height log(N))

### Precision Stability

Unlike SSM/linear-attention methods that accumulate over sequences and require fp32, HSA's local
window (O(1) tokens) can accumulate in fp16 without underflow, enabling full fp16 training without
expensive fp32 accumulators.

## Usage

### Installation

```bash
pip install transformers torch
```

### Loading the Model

Weights and tokenizer are hosted on the Hugging Face Hub at
[`Hugodonotexit/HSA1.0_base`](https://huggingface.co/Hugodonotexit/HSA1.0_base). Since HSA is a
custom architecture, loading it requires `trust_remote_code=True`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Hugodonotexit/HSA1.0_base"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
```

### Inference

```python
import torch

inputs = tokenizer("The quick brown fox", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits
```

### Generation (via HuggingFace generate)

```python
input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
output_ids = model.generate(
    input_ids,
    max_length=200,
    temperature=0.7,
    top_p=0.9,
)
generated_text = tokenizer.decode(output_ids[0])
```

## Tokenizer

- **Type:** BytePair Encoding (BPE)
- **Vocabulary Size:** 32,000
- **Special Tokens:**
  - `<bos>`: Beginning of sequence
  - `<eos>`: End of sequence
  - `<pad>`: Padding
  - `<reserved_0>` - `<reserved_28>`: Reserved for future use
- **Max Length:** 1024 (can be adjusted during fine-tuning)

## File Structure

```
hsa/
├── modeling_hsa.py          # Core model implementation
├── configuration_hsa.py      # Config class
├── __init__.py              # Package initialization
├── tokenizer/
│   ├── tokenizer.json       # BPE vocabulary
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── generation_config.json   # Default generation settings
└── README.md                # This file
```

## License

This model and implementation are provided as a reference for research purposes.

## Citation

If you use HSA in your research, please cite:

```bibtex
@model{hsa2024,
  title={HSA: Hierarchical Softmax Attention for Long-Context Modeling},
  year={2024}
}
```

## Contact & Feedback

For questions, issues, or contributions, please reach out through the GitHub repository.

---

**Model Status:** Research prototype - reference implementation
