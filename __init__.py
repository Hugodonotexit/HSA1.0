from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_hsa import HSAConfig
from .modeling_hsa import (
    HSACausalLMOutput,
    HSAForCausalLM,
    HSAModel,
    build_sentence_end_ids,
    next_sentence_spans,
    score_tokens,
    select_band,
)

AutoConfig.register("hsa", HSAConfig)
AutoModel.register(HSAConfig, HSAModel)
AutoModelForCausalLM.register(HSAConfig, HSAForCausalLM)

__all__ = [
    "HSAConfig",
    "HSAModel",
    "HSAForCausalLM",
    "HSACausalLMOutput",
    "build_sentence_end_ids",
    "next_sentence_spans",
    "score_tokens",
    "select_band",
]
