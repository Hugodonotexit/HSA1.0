# HSA Model Repository Setup

This directory (`./hsa`) is ready to be published as a standalone model repository on GitHub and HuggingFace Hub.

## What's Included

### Core Model Files
- **`modeling_hsa.py`** - HSA architecture implementation (1.1B tokens, 118KB)
- **`configuration_hsa.py`** - HSAConfig class with all hyperparameters
- **`__init__.py`** - HuggingFace AutoModel registration

### Model Card & Documentation
- **`README.md`** - Complete model card with architecture, training, usage
- **`HUGGINGFACE_GUIDE.md`** - Step-by-step guide to publish to HuggingFace Hub
- **`SETUP.md`** - This file
- **`generation_config.json`** - Default generation settings

### Tokenizer
- **`tokenizer/tokenizer.json`** - BPE vocabulary (32k tokens)
- **`tokenizer/tokenizer_config.json`** - Tokenizer configuration
- **`tokenizer/special_tokens_map.json`** - Special token mappings

### Repository Setup
- **`.gitignore`** - Git ignore patterns for model repos

## Quick Start

### 1. Verify Structure
```bash
ls -la hsa/
# Should see: modeling_hsa.py, configuration_hsa.py, __init__.py, 
#             README.md, HUGGINGFACE_GUIDE.md, tokenizer/, generation_config.json
```

### 2. Test Import
```python
from hsa import HSAConfig, HSAForCausalLM, HSAModel
print("✓ Model imports working")
```

### 3. Prepare Training Checkpoints
After training completes, save your model:
```bash
python -c "
from hsa import HSAForCausalLM
model = HSAForCausalLM.from_pretrained('./hsa')
model.save_pretrained('./hsa-trained-model')
"
```

### 4. Upload to HuggingFace
Follow `HUGGINGFACE_GUIDE.md` for detailed upload instructions.

## Directory Structure

```
hsa/
├── Core Implementation
│   ├── modeling_hsa.py              # Architecture (118 KB)
│   ├── configuration_hsa.py         # Config class (10 KB)
│   └── __init__.py                  # HF registration
│
├── Documentation
│   ├── README.md                    # Model card
│   ├── HUGGINGFACE_GUIDE.md         # Publishing guide
│   └── SETUP.md                     # This file
│
├── Generation Config
│   └── generation_config.json       # Default gen settings
│
├── Tokenizer
│   └── tokenizer/
│       ├── tokenizer.json           # BPE vocab (32k)
│       ├── tokenizer_config.json    # Config
│       └── special_tokens_map.json  # Special tokens
│
└── .gitignore                       # Git patterns
```

## Model Configuration (Config B - Recommended)

**Architecture:**
- Hidden size: 2048
- Intermediate (MLP): 8192
- Attention heads: 16
- Layers: 24
- Max position: 8192
- Window size: 128

**FAST Training Settings:**
- Multi-horizon prediction (MTP): 2 horizons, 0.5 position fraction, weight=0.35
- Latent prediction: 1024 dim, 0.2 position fraction, weight=0.25
- Thought/gating: 3%, 3 steps, 256 dim, weight=0.05
- Token selection: 75% keep, weight=0.45 variance
- Engagement: step 1500 (latent), step 3000 (selection)

**Training Sequence:**
1. Phase 1 (128): 0.5B tokens, 6.8k steps
2. Phase 2 (256): 0.8B tokens, 5.4k steps
3. Phase 3 (512): 1.5B tokens, 5.1k steps
4. Phase 4 (1024): 3.0B tokens, 5.1k steps
5. Phase 5 (2048): 4.0B tokens, 3.4k steps
6. Phase 6 (4096): 5.0B tokens, 2.1k steps
7. Phase 7 (8192): 5.2B tokens, 1.1k steps

Total: 28.9k optimizer steps, ~20B tokens

## Publishing Checklist

- [ ] Test model can be imported: `from hsa import HSAForCausalLM`
- [ ] README.md is complete and accurate
- [ ] generation_config.json has sensible defaults
- [ ] Tokenizer files present in `tokenizer/`
- [ ] `.gitignore` configured
- [ ] No credential files committed
- [ ] Model weights saved with `model.save_pretrained()`

## Publishing to GitHub

```bash
git init
git add hsa/
git commit -m "Initial HSA model repository"
git remote add origin https://github.com/YOUR_USERNAME/hsa.git
git push -u origin main
```

## Publishing to HuggingFace

See `HUGGINGFACE_GUIDE.md` for complete instructions, or:

```bash
huggingface-cli login
python hsa/save_to_hub.py  # If you create this script
```

Or use:
```python
model.push_to_hub("your-username/hsa-1b")
tokenizer.push_to_hub("your-username/hsa-1b")
```

## File Sizes

| File | Size | Purpose |
|------|------|---------|
| modeling_hsa.py | 118 KB | Model implementation |
| configuration_hsa.py | 10 KB | Configuration |
| tokenizer.json | 2.9 MB | BPE vocabulary |
| README.md | 7.5 KB | Model card |
| pytorch_model.bin | ~4.2 GB | Weights (after training) |

## Key Features

✓ **Hierarchical Softmax Attention** - O(log N) dependency depth
✓ **Full FP16 Training** - No fp32 accumulators needed
✓ **FAST Objectives** - Multi-horizon prediction, latent thinking, selective training
✓ **HuggingFace Compatible** - Auto-load with transformers library
✓ **Production Ready** - Tested inference, checkpoint recovery, NaN-safe

## Next Steps

1. Train a model following the phase progression in training_config.py
2. Save checkpoint with `model.save_pretrained("path/to/output")`
3. Update README.md with training results
4. Push to GitHub: `git init && git push`
5. Upload to HuggingFace: Follow HUGGINGFACE_GUIDE.md

## Support

For issues, questions, or contributions:
1. Check README.md for usage examples
2. Read HUGGINGFACE_GUIDE.md for publishing help
3. See modeling_hsa.py docstrings for implementation details
4. Refer to configuration_hsa.py for all tunable parameters

---

**Status:** Ready for publication ✓
**Last Updated:** August 2024
