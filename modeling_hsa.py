"""HSA (Hierarchical Softmax Attention) model, transformers-compatible.

Reference implementation of the architecture described in the accompanying
design note: exact local softmax attention over the last `window_size`
tokens, plus an O(log N) read over a causal dyadic tree of RMS-normalized,
log-mass-weighted key/value summaries built by recursive soft-assignment
pooling ("slot attention"). A frontier slot with log-mass ell enters the
softmax as if it were exp(ell) repeated identical tokens, so the frontier
read is a consistent multiplicative estimator of full softmax attention
(see HSAPool docstring). This keeps the sequence-axis dependency depth at
O(log N) instead of O(N) (SSM/linear-attention recurrences, which need
fp32 accumulators) or O(1)-growth-unbounded (a flat KV cache), which is
what lets the whole model run in fp16.

This is a from-scratch reference/prototype: correctness and architectural
fidelity are prioritized over kernel-level performance. The parallel
(prefill/training) path and the incremental (decode) cache path are built
from the *same* insertion primitive (`HSAAttention._insert_chunk`) so they
are guaranteed to agree -- see `scripts/check_prefill_decode_equivalence.py`.

By default this runs as plain PyTorch ops (dense matmul + softmax, no fused
kernel). Two optional, independently-toggleable accelerations are available
when the corresponding package is installed:

  - `config.use_xformers=True`: the combined local+frontier softmax
    attention (still a single softmax over the concatenation of both --
    see HSAAttention._prefill) is computed via
    `xformers.ops.memory_efficient_attention` with a dense additive bias
    (encoding both the causal/window mask and the per-slot log-mass) instead
    of an explicit einsum+softmax+einsum. Verified against the plain-PyTorch
    path in `scripts/check_xformers_liger.py`. Requires the key dimension
    to be padded to a multiple of 8 (xformers' cutlass kernel's alignment
    requirement); handled internally. Only wired into the parallel prefill
    path -- decode's single-query (Mq=1) attention is cheap enough that a
    fused kernel isn't worth the dispatch overhead, so it stays plain.
  - `config.use_liger_kernel=True`: swaps in Triton kernels from
    `liger_kernel` for the pieces of this architecture that have a direct
    equivalent -- `LigerRMSNorm` (every RMSNorm, including inside
    `HSAPool`), `LigerSwiGLUMLP` (the MLP block), `liger_rotary_pos_emb`
    (the paired Q/K rotary application; single-tensor K-only rotation in
    decode has no Liger equivalent and stays custom), and
    `LigerFusedLinearCrossEntropyLoss` (fuses the lm_head projection with
    the loss so the full (N, vocab_size) logits tensor is never
    materialized during training -- `logits` comes back `None` in that case,
    the same trade-off Liger's own reference model patches make). Liger
    ships several other kernels (LayerNorm, GeGLU, Multi-Token Attention,
    Sparsemax, mHC/hyper-connections, a fused Embedding, a generic matmul)
    that don't apply here because this architecture doesn't have a LayerNorm,
    a GeGLU MLP, or those other mechanisms at all -- swapping them in would
    mean *adding* architecture, not accelerating what's already specified.

    A real caveat worth knowing before trusting logit-level parity: on a
    freshly-initialized (untrained) model, `use_liger_kernel=True` can
    produce forward-pass logits that differ from the plain-PyTorch path by
    several percent once the tree is deep enough to actually get read
    (shallow trees -- where the frontier is empty and the tree's output is
    architecturally unused -- agree to fp32 rounding, ~1e-7). Direct
    kernel-vs-kernel comparison shows each individual Liger op agrees with
    its plain-PyTorch equivalent to ~1e-7, so this is not a wrong-computation
    bug; it's HSAPool's norm_k/norm_v being re-applied recursively through
    every tree merge level (the same weight-tied module, several times per
    forward pass), on a model with no gradient-descent-induced smoothness
    protecting it from near-tied softmax logits -- a tiny per-kernel
    difference can flip which key "wins" a close softmax and show up as a
    materially larger output difference. Confirmed by two facts: (1) the
    divergence shrinks roughly linearly with `initializer_range` (smaller
    initial logit magnitude -> fewer near-ties -> less amplification -- see
    `scripts/check_xformers_liger.py`), and (2) `use_xformers=True` shows no
    such effect at any depth (it's a numerically-transparent substitute for
    the same softmax(QK^T*scale+bias)@V computation, not a repeated
    normalization). A model with actual trained weights should not exhibit
    this to nearly the same degree; regardless, don't assume bit-identical
    logits between the accelerated and reference paths -- check convergence/
    eval metrics on your own checkpoint if that matters for your use case.

If requested but the package isn't importable, the model logs a warning and
falls back to plain PyTorch rather than failing to construct.

FAST: FORESIGHT-AUGMENTED SELECTIVE TRAINING
--------------------------------------------------------------------
Everything above is the architecture. The rest of this file is what the model
is trained ON, and it is not optional -- there is no configuration of
`HSAForCausalLM` that trains a plain next-token objective. Four mechanisms,
all always on, changing what signal each token carries:

  1. MULTI-HORIZON PREDICTION. Each position predicts not just token t+1 (the
     ordinary causal-LM loss) but t+2 .. t+1+`mtp_horizons` through small
     auxiliary heads that reuse the tied `lm_head` rather than adding a
     vocab-sized matrix apiece, AND a latent embedding of the *next sentence* --
     the "predict the meaning, not the pixels" (JEPA) version of foresight,
     which rewards planning over surface memorization. Every token now carries
     several training signals instead of one. Multi-token prediction is the
     proven part (Gloeckle et al. 2024; DeepSeek-V3's MTP head).

  2. UNCERTAINTY-GATED LATENT THINKING. Where predictive entropy spikes -- the
     model is genuinely unsure what comes next -- it unrolls `thought_steps`
     extra recurrent latent steps before committing, and the decision to think
     is reinforced only where the thought measurably improved prediction of the
     coming span. Quiet-STaR's idea made cheap: the thought stays a vector
     rather than generated text, and only fires at hard spots, the way a person
     pauses only at the confusing sentence. It fires at generation time too, not
     only in training -- see `_logits_with_thought`.

     SCOPE NOTE, so nobody reads more into this than is here: the recurrence
     operates on the FINAL hidden state (readout-level latent depth), not by
     re-running the trunk. Full-depth recurrence would both cost a multiple of
     the step and break the cache invariants that
     `scripts/check_prefill_decode_equivalence.py` exists to protect. Causality
     is preserved for free, since the cell reads only h_t and its own state --
     no mask surgery, and nothing to get wrong.

  3. LEARNABILITY-GATED TOKEN SELECTION. The loss is applied only to a band of
     tokens that are learnable but not yet learned, skipping both what the model
     already knows and the noise it never will (RHO-1-style selective language
     modeling). See `select_band` for what stands in for RHO-1's reference model
     here, and how to drop a real one in.

TWO NUMERICAL FACTS THE FAST CODE IS BUILT AROUND, both load-bearing:

  - liger's `LigerFusedLinearCrossEntropyLoss` supports `reduction="none"`,
    which yields per-token losses without ever materializing the
    (N, vocab_size) logits tensor -- but its BACKWARD for that reduction is
    explicitly unimplemented (see the comment at
    `liger_kernel/ops/fused_linear_cross_entropy.py:247`). Every per-token
    scoring call here is therefore under `torch.no_grad()`, and every call that
    must carry gradient uses `reduction="mean"`. This is not a stylistic
    choice; mixing them up produces silently wrong gradients, not an error.

  - The fused loss offers no per-token weight -- `ce_weight` is per-CLASS. The
    only per-token lever it has is `ignore_index`. So token selection is applied
    by writing -100 into a COPY of the labels, and `reduction="mean"` then
    normalizes by the selected count, which is exactly the SLM objective.

STATIC SHAPES. `scripts/train_ddp_balanced.py` runs under
`torch.compile(mode="max-autotune-no-cudagraphs")`. Every selection here is a
fixed-count `topk`, never a threshold, so no tensor shape depends on the data.
That also keeps each rank's selected-token count exactly proportional to its
local batch size, which is what lets that script's existing
`loss_weight = my_batch_size / mean_batch_size` correction for heterogeneous
per-rank batches stay valid with no extra collective.

Two things this file deliberately does NOT attempt, consistent with the
design note's own caveats:
  - a single fused kernel spanning the *entire* local+frontier+tree-build
    computation (xformers, when enabled, fuses the final attention op only;
    tree construction is still plain PyTorch);
  - a fully-general batched implementation of the certified best-first
    descent (`HSAAttention.certified_retrieve` runs one query at a time).
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts, noop_context_fn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_hsa import HSAConfig

logger = logging.get_logger(__name__)

try:
    import xformers.ops as xops
    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss, LigerRMSNorm, LigerSwiGLUMLP  # noqa: F401
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Selective activation checkpointing (SAC): recompute cheap, memory-heavy ops (norms,
# softmax, elementwise activations, dropout) instead of the WHOLE layer, but keep the
# outputs of matmul-like/attention ops saved -- those are what's expensive to redo but
# comparatively small to hold onto. Same idea as Megatron's "Reducing Activation
# Recomputation in Large Transformer Models" (Korthikanti et al. 2022): most of a
# transformer layer's activation memory comes from ops that are cheap to recompute, so
# full-layer recompute (plain --grad-checkpointing) pays for FLOPs-heavy matmuls it didn't
# need to redo, for memory it could have kept.
#
# CAVEAT: this policy operates at the ATen op level via dispatch tracing, so it can only
# see/control standard ops. liger_kernel's RMSNorm/fused-loss are opaque custom
# autograd.Functions from this policy's point of view -- whatever they save internally for
# their own backward is fixed by liger itself, unaffected by MUST_SAVE/PREFER_RECOMPUTE
# either way. Fine in practice: their footprint is small relative to attention/MLP
# activations, which are the ops this policy actually controls.
_SAC_SAVE_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
}


def _sac_policy(ctx, op, *args, **kwargs):
    if op in _SAC_SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _sac_context_fn():
    return create_selective_checkpoint_contexts(_sac_policy)


@torch._dynamo.disable  # see call site in HSAForCausalLM.forward for why
def _call_fused_lce(fused_lce, weight, hidden, labels):
    return fused_lce(weight, hidden, labels)


def _resolve_accel_flags(config: HSAConfig):
    use_xformers = bool(config.use_xformers)
    if use_xformers and not _XFORMERS_AVAILABLE:
        logger.warning_once("config.use_xformers=True but `xformers` is not importable; falling back to plain PyTorch attention.")
        use_xformers = False
    use_liger = bool(config.use_liger_kernel)
    if use_liger and not _LIGER_AVAILABLE:
        logger.warning_once("config.use_liger_kernel=True but `liger_kernel` is not importable; falling back to plain PyTorch ops.")
        use_liger = False
    return use_xformers, use_liger


_MASK_VALUE = -1e4  # fixed (not dtype-dependent) so masked values compound safely under fp16


# ---------------------------------------------------------------------------
# Causal dyadic decomposition (pure Python, static given a length -> cacheable)
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return max(p, 1)


def _dyadic_decomposition(n: int) -> List[Tuple[int, int]]:
    """Canonical cover of [0, n) by maximal aligned dyadic blocks.

    Block (level, index) covers [index * 2**level, (index+1) * 2**level).
    Returned coarsest-block-first. len(result) == popcount(n).
    """
    if n <= 0:
        return []
    blocks = []
    levels_set = []
    x, lvl = n, 0
    while x > 0:
        if x & 1:
            levels_set.append(lvl)
        x >>= 1
        lvl += 1
    offset = 0
    for lvl in reversed(levels_set):
        size = 1 << lvl
        blocks.append((lvl, offset // size))
        offset += size
    return blocks


def _chunk_frontier(chunk_index: int) -> List[Tuple[int, int]]:
    """Frontier of tree nodes visible to queries in `chunk_index`: the dyadic
    cover of chunks [0, chunk_index - 1), i.e. all complete chunks *strictly
    before* the immediately-preceding chunk (which is covered exactly by
    local attention instead)."""
    return _dyadic_decomposition(max(chunk_index - 1, 0))


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class HSARMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        # Cast the *final* result back to the input dtype, not xf before the
        # multiply: self.weight is a plain fp32 Parameter, and elementwise
        # multiply isn't an autocast-eligible op, so `self.weight * xf.to(dtype)`
        # would silently undo the cast via normal type promotion (fp16 * fp32
        # -> fp32) and leak fp32 out of every RMSNorm call under autocast --
        # invisible wherever the output happens to feed into the next
        # autocast-eligible nn.Linear (which re-casts anyway), but a real bug
        # anywhere it doesn't (e.g. HSAPool's norm_k/norm_v output feeding
        # directly into concatenation/dtype-sensitive ops downstream).
        return (self.weight * xf).to(dtype)


def _make_rmsnorm(dim: int, config: HSAConfig) -> nn.Module:
    """`LigerRMSNorm` when `config.use_liger_kernel` resolves to available,
    else the plain `HSARMSNorm` above. Used for every RMSNorm in the model
    (per-layer norms, the final norm, and HSAPool's per-slot norm_k/norm_v)
    so the flag is a single, consistent switch."""
    _, use_liger = _resolve_accel_flags(config)
    if use_liger:
        return LigerRMSNorm(dim, eps=config.rms_norm_eps, in_place=False)
    return HSARMSNorm(dim, eps=config.rms_norm_eps)


class HSARotaryEmbedding(nn.Module):
    def __init__(self, config: HSAConfig):
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # persistent=True (was False): a non-persistent buffer is never part of state_dict(),
        # so from_pretrained() -- used when loading a saved checkpoint back for eval/inference,
        # never by training itself -- left this holding uninitialized GPU memory instead of
        # the real geometric decay computed above. Confirmed via direct A/B test
        # (scripts/generate_from_checkpoint.py): HSAForCausalLM(config), what training always
        # does, computes this correctly every time; HSAForCausalLM.from_pretrained(ckpt_dir)
        # did not. Corrupted inv_freq poisons every attention score via apply_rotary_single,
        # producing NaN logits nondeterministically depending on whatever was in that memory.
        # This does NOT affect any already-trained weights or require retraining -- inv_freq
        # is a fixed function of config (rope_theta, head_dim), not a learned parameter, and
        # was never written to disk either way.
        #
        # CORRECTION, measured 2026-08-22 against transformers 5.15.0. This comment used to
        # claim the value is "correct regardless of key presence" -- that a checkpoint saved
        # before this fix would report a harmless missing key and keep __init__'s value. That
        # is FALSE. A/B on the same model, from_pretrained() either way:
        #     inv_freq key present -> max|err| 0.0        (correct)
        #     inv_freq key ABSENT  -> max|err| 4.5e+07    (garbage)
        # and the garbage is FINITE, so it does not announce itself as NaN -- it quietly
        # poisons every attention score through apply_rotary_single. persistent=True fixes
        # this for checkpoints saved from here on, and every checkpoint under ai/ckpt-512 now
        # carries the key, but anything older still needs the explicit recompute in
        # scripts/generate_from_checkpoint.py's fix_rotary_inv_freq().
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        # position_ids: (B, T) long
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)  # (B, T, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_single(x, cos, sin):
    # x: (B,H,T,D), cos/sin: (B,T,D)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary(q, k, cos, sin):
    # q: (B,H,T,D), k: (B,Hkv,T,D), cos/sin: (B,T,D)
    return apply_rotary_single(q, cos, sin), apply_rotary_single(k, cos, sin)


def repeat_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, Hkv, ..., D) -> (B, Hkv*n_rep, ..., D), interleaved to match
    HSAConfig's grouped-query convention (group g uses kv-head g // n_rep)."""
    if n_rep == 1:
        return x
    b, hkv = x.shape[0], x.shape[1]
    rest = x.shape[2:]
    x = x.unsqueeze(2).expand(b, hkv, n_rep, *rest)
    return x.reshape(b, hkv * n_rep, *rest)


# ---------------------------------------------------------------------------
# Soft-assignment pooling: the single primitive used both to summarize a
# chunk of raw tokens into m slots (leaf level) and to merge two child
# nodes' 2m slots into m parent slots (every level above the leaf). Weight
# tied across all merge levels: `HSAAttention.merge_pool` is applied
# recursively regardless of tree depth.
# ---------------------------------------------------------------------------

class HSAPool(nn.Module):
    """Competitive soft-assignment pooling, mass- and RMS-normalized.

    Given E weighted "entries" (raw tokens, or child tree slots) each with a
    log-mass log(mass_e) (mass=1 / log-mass=0 for raw tokens), produces
    `num_slots` output slots (k, v, log_mass, radius):

      log_w[e, q]      = log_softmax_q(  <query_q, k_e> / (sqrt(D)*T) + log_mass_e  )   (competition among slots, per entry)
      log_mass[q]      = logsumexp_e( log_w[e, q] + log_mass_e )                          (stable; avoids ever forming raw mass ~ N)
      p[e, q]          = exp( log_w[e,q] + log_mass_e - log_mass[q] )                     (in [0,1], sums to 1 over e)
      k[q], v[q]       = sum_e p[e,q] * k_e ,  sum_e p[e,q] * v_e                          (mass-weighted centroid)
      radius[q]        = sum_e p[e,q] * ( ||k_e - k[q]|| + radius_e )                      (expected-distance proxy; see note below)

    k[q], v[q] are then RMS-normalized, which is what keeps every node's
    representation at RMS ~= 1 regardless of tree depth or token count --
    the property that makes fp16 error grow as O(log N) instead of O(N)
    (see module docstring).

    Note on `radius`: an admissible A* bound needs max_i ||k_i - k[q]||
    (worst case), not the mass-weighted mean used here. The mean is used
    because it is smooth/trainable; it makes the resulting bound in
    `HSAAttention.certified_retrieve` a *heuristic* rather than a strictly
    provable one unless training pushes cluster radii tight enough that the
    mean and max nearly coincide (which the clustering objective
    encourages, but does not guarantee). This mirrors the honest caveat in
    the design note that admissibility here is trained-for, not proven.
    """

    def __init__(self, config: HSAConfig, num_heads: int):
        super().__init__()
        self.num_slots = config.tree_slots
        self.head_dim = config.head_dim
        self.temperature = config.pool_temperature
        self.query = nn.Parameter(torch.empty(num_heads, self.num_slots, self.head_dim))
        self.norm_k = _make_rmsnorm(self.head_dim, config)
        self.norm_v = _make_rmsnorm(self.head_dim, config)

    def forward(self, entry_k, entry_v, entry_log_mass=None, entry_radius=None, entry_valid=None):
        """
        entry_k, entry_v : (B, H, N, E, D)
        entry_log_mass   : (B, H, N, E) or None (=> 0, i.e. unit mass)
        entry_radius     : (B, H, N, E) or None (=> 0)
        entry_valid      : (B, N, E) bool or None (=> all valid)
        returns slot_k, slot_v : (B, H, N, Q, D); slot_log_mass, slot_radius : (B, H, N, Q)
        """
        scale = math.sqrt(self.head_dim) * self.temperature
        # Scale folded into `query` BEFORE the contraction, not applied to the product
        # after it. Dividing afterwards materializes the raw dot product in the autocast
        # dtype first, and in fp16 that overflows at |q.k| > 65504 -- with head_dim=160
        # that is only element magnitude ~20 (20^2 * 160 = 64000), well inside the range
        # trained q/k outliers reach. The resulting inf survives log_softmax as NaN and
        # poisons hidden for every position in the chunk. Reproduced directly: at
        # magnitude 25, einsum(q,k)/scale is inf while einsum(q/scale,k) is finite.
        # `query` is (H,Q,D) -- a few thousand elements -- so pre-dividing it is free,
        # and it buys `scale` (12.6x here) of headroom over the old form.
        #
        # This is not hypothetical: two runs died this way at step ~1008 and ~873, both
        # in the FORWARD pass at loss_scale 4096 (visible in the no_grad `ce_all` probe,
        # which the loss scaler cannot reach), which is why neither the lower LR nor the
        # longer warmup in the second run changed anything.
        # `.to(entry_k.dtype)` for the same reason as level_embed in _prefill: `query` is a
        # plain fp32 Parameter and `/` is not an autocast-eligible op, so the divided query
        # stays fp32 while entry_k is fp16. einsum under autocast would cast both anyway,
        # but outside autocast that mismatch is a hard error -- matching dtype explicitly
        # keeps this callable in either context. The divide itself still happens in fp32.
        logits = torch.einsum("hqd,bhned->bhnqe", (self.query / scale).to(entry_k.dtype), entry_k)  # (B,H,N,Q,E)
        if entry_log_mass is not None:
            logits = logits + entry_log_mass.unsqueeze(-2)
        if entry_valid is not None:
            valid = entry_valid[:, None, :, None, :]
            logits = logits.masked_fill(~valid, _MASK_VALUE)

        log_w = torch.log_softmax(logits, dim=-2)  # competition over slots Q, per entry e
        log_effective = log_w if entry_log_mass is None else (log_w + entry_log_mass.unsqueeze(-2))
        if entry_valid is not None:
            log_effective = log_effective.masked_fill(~valid, _MASK_VALUE)

        slot_log_mass = torch.logsumexp(log_effective, dim=-1)  # (B,H,N,Q)
        p = torch.exp(log_effective - slot_log_mass.unsqueeze(-1))
        p = torch.nan_to_num(p, nan=0.0)

        slot_k = torch.einsum("bhnqe,bhned->bhnqd", p, entry_k)
        slot_v = torch.einsum("bhnqe,bhned->bhnqd", p, entry_v)

        dist = torch.linalg.vector_norm(entry_k.unsqueeze(-3) - slot_k.unsqueeze(-2), dim=-1)  # (B,H,N,Q,E)
        if entry_radius is not None:
            dist = dist + entry_radius.unsqueeze(-2)
        slot_radius = (p * dist).sum(dim=-1)

        slot_k = self.norm_k(slot_k)
        slot_v = self.norm_v(slot_v)
        return slot_k, slot_v, slot_log_mass, slot_radius


@dataclass
class _TreeNode:
    k: torch.Tensor  # (B, Hkv, m, D)
    v: torch.Tensor
    log_mass: torch.Tensor  # (B, Hkv, m)
    radius: torch.Tensor  # (B, Hkv, m)


class HSACache:
    """Per-layer incremental state: a rolling local-window buffer plus an
    append-only causal dyadic tree, built with amortized O(1) work per
    completed chunk (binary-counter / Fenwick-style cascading merge -- see
    `HSAAttention._insert_chunk`).

    Not a `transformers.Cache` subclass: HSA's state isn't a flat KV
    tensor, so it doesn't fit that interface. `HSAForCausalLM` drives it
    directly through `prepare_inputs_for_generation` / `forward`.
    """

    def __init__(self, num_layers: int):
        self.total_tokens = 0
        self.prev_chunk: List[Optional[_TreeNode]] = [None] * num_layers
        self.curr_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self.curr_v: List[Optional[torch.Tensor]] = [None] * num_layers
        self.curr_len = 0
        # tree_levels[layer][level] maps *canonical* index (idx = chunk_idx >> level,
        # matching _chunk_frontier's dyadic decomposition) -> node. This dict is
        # intentionally sparse: e.g. an odd-indexed level-0 chunk that immediately
        # cascaded into a level-1 merge never gets its own level-0 entry, because
        # _chunk_frontier (a canonical prefix decomposition starting at offset 0)
        # provably never asks for it -- only even-indexed "left children" can ever
        # appear alone. See `_insert_chunk`.
        self.tree_levels: List[List[dict]] = [[] for _ in range(num_layers)]
        self.tree_carry: List[List[Optional[_TreeNode]]] = [[] for _ in range(num_layers)]
        self.next_chunk_idx: List[int] = [0] * num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.total_tokens


# ---------------------------------------------------------------------------
# Attention: local exact window + O(log N) frontier read
# ---------------------------------------------------------------------------

class HSAAttention(nn.Module):
    def __init__(self, config: HSAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.w = config.window_size
        self.m = config.tree_slots
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.leaf_pool = HSAPool(config, self.num_kv_heads)
        self.merge_pool = HSAPool(config, self.num_kv_heads)  # weight-tied across all levels >= 1

        self.level_embed = nn.Parameter(torch.empty(config.max_frontier_levels, self.num_kv_heads, self.head_dim))

        self.use_xformers, use_liger = _resolve_accel_flags(config)
        # Paired Q/K rotation (the hot-path call, in both _prefill and
        # _decode_step's new-token rotation) can use Liger's fused kernel.
        # Single-tensor rotation (apply_rotary_single, used to rotate the
        # cache's buffered prev/curr chunk keys with no associated query) has
        # no Liger equivalent and always uses the plain implementation.
        self._rope = liger_rotary_pos_emb if use_liger else apply_rotary

    # -- shared merge primitive: used by both the parallel tree build and the
    # -- incremental cache insertion, so the two paths are one code path. --
    def _merge(self, left: _TreeNode, right: _TreeNode) -> _TreeNode:
        entry_k = torch.cat([left.k, right.k], dim=-2).unsqueeze(2)  # (B,Hkv,1,2m,D)
        entry_v = torch.cat([left.v, right.v], dim=-2).unsqueeze(2)
        entry_lm = torch.cat([left.log_mass, right.log_mass], dim=-1).unsqueeze(2)  # (B,Hkv,1,2m)
        entry_r = torch.cat([left.radius, right.radius], dim=-1).unsqueeze(2)
        k, v, lm, r = self.merge_pool(entry_k, entry_v, entry_lm, entry_r, entry_valid=None)
        return _TreeNode(k.squeeze(2), v.squeeze(2), lm.squeeze(2), r.squeeze(2))

    def _leaf(self, chunk_k, chunk_v, chunk_valid=None) -> _TreeNode:
        # chunk_k, chunk_v: (B, Hkv, w, D); chunk_valid: (B, w) bool or None
        entry_k = chunk_k.unsqueeze(2)  # (B,Hkv,1,w,D)
        entry_v = chunk_v.unsqueeze(2)
        entry_valid = None if chunk_valid is None else chunk_valid.unsqueeze(1)  # (B,1,w)
        k, v, lm, r = self.leaf_pool(entry_k, entry_v, entry_valid=entry_valid)
        return _TreeNode(k.squeeze(2), v.squeeze(2), lm.squeeze(2), r.squeeze(2))

    def _insert_chunk(self, cache: HSACache, chunk_k, chunk_v, chunk_valid=None):
        """Amortized-O(1) incremental insertion of one complete chunk into
        the layer's tree (binary-counter / Fenwick cascading merge).

        Chunks must be inserted in increasing chunk-index order (both
        `_init_cache_from_prefill` and decode-time insertion guarantee this).
        Every node the cascade passes through gets stored under its
        *canonical* index idx = chunk_idx >> level (matching
        `_chunk_frontier`'s decomposition) -- not just the level where the
        cascade finally stops. Storing intermediate cascade nodes too (not
        only the final one) is still amortized O(1) per insertion, by the
        same accounting as a binary counter: sum of trailing-1-bit-counts
        over N increments is O(N), not O(N log N). Without this, "odd"
        (right-side) nodes are never persisted anywhere -- they're computed
        and immediately consumed within a single cascade -- which silently
        makes half the tree unreachable by `certified_retrieve`'s descent
        even though regular attention (`_chunk_frontier`) never needed them.
        """
        chunk_idx = cache.next_chunk_idx[self.layer_idx]
        cache.next_chunk_idx[self.layer_idx] += 1

        node = self._leaf(chunk_k, chunk_v, chunk_valid)
        levels = cache.tree_levels[self.layer_idx]
        carry = cache.tree_carry[self.layer_idx]
        level = 0
        while True:
            if level == len(levels):
                levels.append({})
                carry.append(None)
            levels[level][chunk_idx >> level] = node
            if carry[level] is None:
                carry[level] = node
                return
            node = self._merge(carry[level], node)
            carry[level] = None
            level += 1

    # ------------------------------------------------------------------
    def forward(self, hidden_states, position_ids, attention_mask, rotary_emb, cache, use_cache, is_decode, pos, curr_len):
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if is_decode:
            out = self._decode_step(q, k, v, cache, rotary_emb, pos, curr_len)
        else:
            out = self._prefill(q, k, v, position_ids, attention_mask, rotary_emb, cache, use_cache)

        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)

    # ------------------------------------------------------------------
    def _prefill(self, q, k, v, position_ids, attention_mask, rotary_emb, cache, use_cache):
        B, H, T, D = q.shape
        Hkv = self.num_kv_heads
        w, m = self.w, self.m
        device, dtype = q.device, q.dtype

        cos, sin = rotary_emb(position_ids, dtype)
        q_rot, k_rot = self._rope(q, k, cos, sin)

        valid = torch.ones(B, T, dtype=torch.bool, device=device) if attention_mask is None \
            else attention_mask.bool()

        C = max(1, (T + w - 1) // w)
        C_pad = _next_pow2(C)
        T_pad = C_pad * w
        pad = T_pad - T

        def pad_seq(x):
            # x: (B, H, T, D) -- pad the T dim (dim=2)
            return F.pad(x, (0, 0, 0, pad, 0, 0)) if pad else x

        q_c = pad_seq(q_rot).reshape(B, H, C_pad, w, D)
        k_rot_c = pad_seq(k_rot).reshape(B, Hkv, C_pad, w, D)
        k_raw_c = pad_seq(k).reshape(B, Hkv, C_pad, w, D)
        v_c = pad_seq(v).reshape(B, Hkv, C_pad, w, D)
        valid_pad = F.pad(valid, (0, pad)) if pad else valid
        valid_c = valid_pad.reshape(B, C_pad, w)

        # ---- local: chunk c attends to (chunk c-1) ++ (chunk c), causal ----
        zeros_k = torch.zeros(B, Hkv, 1, w, D, dtype=dtype, device=device)
        zeros_v = torch.zeros_like(zeros_k)
        prev_k = torch.cat([zeros_k, k_rot_c[:, :, :-1]], dim=2)
        prev_v = torch.cat([zeros_v, v_c[:, :, :-1]], dim=2)
        prev_valid = torch.cat(
            [torch.zeros(B, 1, w, dtype=torch.bool, device=device), valid_c[:, :-1]], dim=1
        )
        local_k = torch.cat([prev_k, k_rot_c], dim=3)  # (B,Hkv,C_pad,2w,D)
        local_v = torch.cat([prev_v, v_c], dim=3)
        local_valid = torch.cat([prev_valid, valid_c], dim=2)  # (B,C_pad,2w)

        local_k = repeat_heads(local_k, self.n_rep)
        local_v = repeat_heads(local_v, self.n_rep)

        # `local_bias` and (below) `frontier_bias` are built once and shared by
        # both the plain-PyTorch path (added to the einsum'd logits before
        # softmax) and the xformers path (passed as memory_efficient_attention's
        # additive attn_bias, which computes softmax(QK^T*scale + bias)@V
        # internally) -- one source of truth for the masking/log-mass logic.
        causal = torch.arange(w, device=device).unsqueeze(1) + w >= torch.arange(2 * w, device=device).unsqueeze(0)
        mask = causal[None, None, None] & local_valid[:, None, :, None, :]  # (B,1,C_pad,w,2w)
        local_bias = torch.zeros(B, 1, C_pad, w, 2 * w, dtype=dtype, device=device).masked_fill(~mask, _MASK_VALUE)

        # ---- tree: build leaf summaries for every chunk, then merge levels ----
        levels = [self._leaf(k_raw_c[:, :, c], v_c[:, :, c], valid_c[:, c]) for c in range(C_pad)]
        levels = [_TreeNode(
            torch.stack([n.k for n in levels], dim=2),
            torch.stack([n.v for n in levels], dim=2),
            torch.stack([n.log_mass for n in levels], dim=2),
            torch.stack([n.radius for n in levels], dim=2),
        )]
        n_nodes = C_pad
        while n_nodes > 1:
            prev = levels[-1]
            half = n_nodes // 2
            ek = prev.k.reshape(B, Hkv, half, 2 * m, D)
            ev = prev.v.reshape(B, Hkv, half, 2 * m, D)
            elm = prev.log_mass.reshape(B, Hkv, half, 2 * m)
            er = prev.radius.reshape(B, Hkv, half, 2 * m)
            sk, sv, slm, sr = self.merge_pool(ek, ev, elm, er, entry_valid=None)
            levels.append(_TreeNode(sk, sv, slm, sr))
            n_nodes = half

        n_levels = len(levels)
        frontiers = [_chunk_frontier(c) for c in range(C_pad)]
        L_max = max((len(f) for f in frontiers), default=0)

        if L_max == 0:
            frontier_bias = torch.empty(B, H, C_pad, w, 0, device=device, dtype=dtype)
            frontier_k_flat = torch.empty(B, H, C_pad, 0, D, device=device, dtype=dtype)
            frontier_v_flat = torch.empty(B, H, C_pad, 0, D, device=device, dtype=dtype)
        else:
            # pad every level's node axis up to C_pad so all levels share one index space
            # (radius is only needed for certified_retrieve's admissible bound,
            # which reads directly from HSACache.tree_levels, not from this
            # transient prefill-time `levels` list -- no need to gather it here)
            big_k = torch.zeros(B, Hkv, n_levels, C_pad, m, D, dtype=dtype, device=device)
            big_v = torch.zeros_like(big_k)
            big_lm = torch.full((B, Hkv, n_levels, C_pad, m), _MASK_VALUE, dtype=dtype, device=device)
            for l, node in enumerate(levels):
                c_l = node.k.shape[2]
                big_k[:, :, l, :c_l] = node.k
                big_v[:, :, l, :c_l] = node.v
                big_lm[:, :, l, :c_l] = node.log_mass

            flat_index = torch.zeros(C_pad, L_max, dtype=torch.long)
            frontier_valid = torch.zeros(C_pad, L_max, dtype=torch.bool)
            frontier_level = torch.zeros(C_pad, L_max, dtype=torch.long)
            for c, blocks in enumerate(frontiers):
                for j, (lvl, idx) in enumerate(blocks):
                    flat_index[c, j] = lvl * C_pad + idx
                    frontier_valid[c, j] = True
                    frontier_level[c, j] = lvl
            flat_index = flat_index.to(device)
            frontier_valid = frontier_valid.to(device)
            frontier_level = frontier_level.to(device)

            big_k_flat = big_k.reshape(B, Hkv, n_levels * C_pad, m, D)
            big_v_flat = big_v.reshape(B, Hkv, n_levels * C_pad, m, D)
            big_lm_flat = big_lm.reshape(B, Hkv, n_levels * C_pad, m)

            f_k = big_k_flat[:, :, flat_index]  # (B,Hkv,C_pad,L_max,m,D)
            f_v = big_v_flat[:, :, flat_index]
            f_lm = big_lm_flat[:, :, flat_index]  # (B,Hkv,C_pad,L_max,m)

            lvl_emb = self.level_embed[frontier_level]  # (C_pad,L_max,Hkv,D)
            lvl_emb = lvl_emb.permute(2, 0, 1, 3).unsqueeze(0).unsqueeze(-2)  # (1,Hkv,C_pad,L_max,1,D)
            # level_embed is a plain fp32 Parameter; elementwise add isn't an
            # autocast-eligible op, so under autocast this would otherwise
            # silently promote f_k to fp32 while q/v stay fp16 -- xformers
            # hard-rejects mismatched Q/K/V dtypes, and even on the plain
            # path it'd mean doing part of the attention math in fp32 without
            # anyone asking for that. Match f_k's (possibly autocast) dtype
            # explicitly instead of relying on default type promotion.
            f_k = f_k + lvl_emb.to(f_k.dtype)

            f_k = repeat_heads(f_k, self.n_rep)  # (B,H,C_pad,L_max,m,D)
            f_v = repeat_heads(f_v, self.n_rep)
            f_lm = repeat_heads(f_lm, self.n_rep)

            # bias only -- no q dependence, so it's shared by both compute paths
            # (same trick as local_bias: the einsum(q,k) term is added
            # separately in the plain path, and left to xformers' internal
            # QK^T in the fused path).
            fvalid = frontier_valid[None, None, :, None, :, None]  # (1,1,C_pad,1,L_max,1)
            frontier_bias = f_lm.unsqueeze(3).expand(B, H, C_pad, w, L_max, m)
            frontier_bias = frontier_bias.masked_fill(~fvalid, _MASK_VALUE).reshape(B, H, C_pad, w, L_max * m)
            frontier_k_flat = f_k.reshape(B, H, C_pad, L_max * m, D)
            frontier_v_flat = f_v.reshape(B, H, C_pad, L_max * m, D)

        n_frontier = frontier_bias.shape[-1]

        if self.use_xformers:
            out = self._xformers_combined_attention(
                q_c, local_k, local_v, local_bias, frontier_k_flat, frontier_v_flat, frontier_bias
            )
        else:
            # Pre-scaled once, for the same fp16 overflow reason as HSAPool.forward above
            # (see the comment there). The xformers path does not need this -- it hands
            # `scale` to the kernel, which applies it inside the fused QK^T -- so only this
            # fallback path was exposed.
            q_s = q_c / self.scale
            local_logits = torch.einsum("bhcwd,bhckd->bhcwk", q_s, local_k) + local_bias
            frontier_logits = (
                torch.einsum("bhcwd,bhcfd->bhcwf", q_s, frontier_k_flat) + frontier_bias
                if n_frontier > 0
                else frontier_bias
            )
            all_logits = torch.cat([local_logits, frontier_logits], dim=-1)
            all_w = torch.softmax(all_logits.float(), dim=-1).to(dtype)
            local_w, frontier_w = all_w.split([2 * w, n_frontier], dim=-1)

            out = torch.einsum("bhcwk,bhckd->bhcwd", local_w, local_v)
            if n_frontier > 0:
                out = out + torch.einsum("bhcwf,bhcfd->bhcwd", frontier_w, frontier_v_flat)

        out = out.reshape(B, H, T_pad, D)[:, :, :T]

        if use_cache and cache is not None:
            self._init_cache_from_prefill(cache, k, v, valid, T)

        return out

    def _xformers_combined_attention(self, q_c, local_k, local_v, local_bias, frontier_k, frontier_v, frontier_bias):
        """The same single-softmax-over-(local ++ frontier) attention as the
        plain path, computed via `xformers.ops.memory_efficient_attention`
        with a dense additive bias instead of explicit einsum+softmax+einsum.

        q_c: (B,H,C,w,D); local_k/v: (B,H,C,2w,D); frontier_k/v: (B,H,C,F,D)
        (F may be 0); local_bias: (B,1,C,w,2w) broadcastable over H;
        frontier_bias: (B,H,C,w,F). Returns (B,H,C,w,D).
        """
        B, H, C, w, D = q_c.shape
        F_ = frontier_k.shape[3]
        N = 2 * w + F_
        N_pad = ((N + 7) // 8) * 8  # cutlass kernel requires attn_bias's key-dim stride % 8 == 0
        pad_n = N_pad - N

        combined_k = torch.cat([local_k, frontier_k], dim=3)  # (B,H,C,N,D)
        combined_v = torch.cat([local_v, frontier_v], dim=3)
        combined_bias = torch.cat([local_bias.expand(B, H, C, w, 2 * w), frontier_bias], dim=-1)  # (B,H,C,w,N)
        if pad_n:
            combined_k = F.pad(combined_k, (0, 0, 0, pad_n))
            combined_v = F.pad(combined_v, (0, 0, 0, pad_n))
            combined_bias = F.pad(combined_bias, (0, pad_n), value=_MASK_VALUE)

        # xformers wants (batch, seq, heads, dim); treat each of the B*C
        # chunks as an independent batch element for this fused call.
        q_x = q_c.permute(0, 2, 3, 1, 4).reshape(B * C, w, H, D)
        k_x = combined_k.permute(0, 2, 3, 1, 4).reshape(B * C, N_pad, H, D)
        v_x = combined_v.permute(0, 2, 3, 1, 4).reshape(B * C, N_pad, H, D)
        bias_x = combined_bias.permute(0, 2, 1, 3, 4).reshape(B * C, H, w, N_pad)

        out_x = xops.memory_efficient_attention(q_x, k_x, v_x, attn_bias=bias_x, scale=1.0 / self.scale)
        return out_x.reshape(B, C, w, H, D).permute(0, 3, 1, 2, 4)  # (B,H,C,w,D)

    def _init_cache_from_prefill(self, cache: HSACache, k_raw, v_raw, valid, T):
        """Replays the same `_insert_chunk` primitive used at decode time over
        every chunk that is already complete and not the immediate
        predecessor of the current position, so cache state after prefill is
        indistinguishable from cache state built by T decode steps.

        Uses floor division (C_full = T // w), not ceiling: if T is an exact
        multiple of w, the trailing chunk is *complete*, not partial, and
        must become `prev_chunk` with a fresh, empty `curr` -- exactly the
        state a genuine decode-time transition would leave behind. Using
        ceiling division here would instead leave `curr_len == w`, violating
        the invariant (curr_len must be in [0, w)) that `_decode_step`'s
        transition check (`new_len == w`) relies on to ever fire.
        """
        w = self.w
        C_full = T // w  # number of chunks that are fully complete
        # chunks [0, C_full-1) go into the tree; chunk C_full-1 (if it
        # exists) stays as `prev_chunk`; chunk C_full is the (possibly
        # empty) partial `curr` buffer.
        for c in range(max(C_full - 1, 0)):
            sl = slice(c * w, (c + 1) * w)
            self._insert_chunk(cache, k_raw[:, :, sl], v_raw[:, :, sl], valid[:, sl])

        if C_full >= 1:
            prev_sl = slice((C_full - 1) * w, C_full * w)
            cache.prev_chunk[self.layer_idx] = _RawChunk(k_raw[:, :, prev_sl], v_raw[:, :, prev_sl])
        else:
            cache.prev_chunk[self.layer_idx] = None

        last_start = C_full * w
        curr_len = T - last_start
        pad_needed = w - curr_len
        ck = k_raw[:, :, last_start:T]
        cv = v_raw[:, :, last_start:T]
        if pad_needed > 0:
            ck = F.pad(ck, (0, 0, 0, pad_needed))
            cv = F.pad(cv, (0, 0, 0, pad_needed))
        cache.curr_k[self.layer_idx] = ck
        cache.curr_v[self.layer_idx] = cv
        cache.curr_len = curr_len
        if self.layer_idx == 0:
            cache.total_tokens = T

    # ------------------------------------------------------------------
    def _decode_step(self, q, k, v, cache: HSACache, rotary_emb, pos: int, curr_len: int):
        # q,k,v: (B,H or Hkv,1,D) for the single new token
        B, H, _, D = q.shape
        Hkv = self.num_kv_heads
        w = self.w
        device, dtype = q.device, q.dtype
        c = pos // w

        position_ids = torch.full((B, 1), pos, dtype=torch.long, device=device)
        cos, sin = rotary_emb(position_ids, dtype)
        q_rot, k_new_rot = self._rope(q, k, cos, sin)

        key_parts, val_parts = [], []
        prev = cache.prev_chunk[self.layer_idx]
        if prev is not None:
            prev_start = (c - 1) * w
            prev_pos = torch.arange(prev_start, prev_start + w, device=device).unsqueeze(0).expand(B, -1)
            pcos, psin = rotary_emb(prev_pos, dtype)
            prev_k_rot = apply_rotary_single(prev.k, pcos, psin)
            key_parts.append(prev_k_rot)
            val_parts.append(prev.v)
        if curr_len > 0:
            curr_k = cache.curr_k[self.layer_idx][:, :, :curr_len]
            curr_v = cache.curr_v[self.layer_idx][:, :, :curr_len]
            curr_start = c * w
            curr_pos = torch.arange(curr_start, curr_start + curr_len, device=device).unsqueeze(0).expand(B, -1)
            ccos, csin = rotary_emb(curr_pos, dtype)
            curr_k_rot = apply_rotary_single(curr_k, ccos, csin)
            key_parts.append(curr_k_rot)
            val_parts.append(curr_v)
        key_parts.append(k_new_rot)
        val_parts.append(v)

        local_k = repeat_heads(torch.cat(key_parts, dim=2), self.n_rep)  # (B,H,Tloc,D)
        local_v = repeat_heads(torch.cat(val_parts, dim=2), self.n_rep)
        # Pre-scaled for the fp16 overflow reason documented in HSAPool.forward. The decode
        # path has no fused equivalent to fall back on, so both einsums here need it.
        q_s = q_rot / self.scale
        local_logits = torch.einsum("bhqd,bhkd->bhqk", q_s, local_k)  # (B,H,1,Tloc)

        blocks = _chunk_frontier(c)
        if blocks:
            nodes = [cache.tree_levels[self.layer_idx][lvl][idx] for lvl, idx in blocks]
            f_k = torch.stack([n.k for n in nodes], dim=2)  # (B,Hkv,L,m,D)
            f_v = torch.stack([n.v for n in nodes], dim=2)
            f_lm = torch.stack([n.log_mass for n in nodes], dim=2)  # (B,Hkv,L,m)
            lvl_idx = torch.tensor([lvl for lvl, _ in blocks], device=device)
            lvl_emb = self.level_embed[lvl_idx]  # (L,Hkv,D)
            # see the matching comment in _prefill: level_embed is fp32 and
            # elementwise add doesn't autocast, so cast explicitly.
            f_k = f_k + lvl_emb.permute(1, 0, 2)[None, :, :, None, :].to(f_k.dtype)
            L = len(blocks)
            f_k = repeat_heads(f_k, self.n_rep).reshape(B, H, L * self.m, D)
            f_v = repeat_heads(f_v, self.n_rep).reshape(B, H, L * self.m, D)
            f_lm = repeat_heads(f_lm, self.n_rep).reshape(B, H, L * self.m)
            f_logits = torch.einsum("bhqd,bhfd->bhqf", q_s, f_k) + f_lm.unsqueeze(2)
        else:
            f_logits = torch.empty(B, H, 1, 0, device=device, dtype=dtype)
            f_v = torch.empty(B, H, 0, D, device=device, dtype=dtype)

        all_logits = torch.cat([local_logits, f_logits], dim=-1)
        all_w = torch.softmax(all_logits.float(), dim=-1).to(dtype)
        local_w, frontier_w = all_w.split([local_logits.shape[-1], f_logits.shape[-1]], dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", local_w, local_v)
        if frontier_w.shape[-1] > 0:
            out = out + torch.einsum("bhqf,bhfd->bhqd", frontier_w, f_v)

        # -- advance this layer's own cache state (tree / prev_chunk / curr buffer). --
        # `pos`/`curr_len` are read-only snapshots taken once by HSAModel.forward
        # before any layer ran, and cache.total_tokens / cache.curr_len (shared
        # across layers) are advanced once there too, after every layer has
        # finished -- NOT here, since layers execute sequentially within one
        # forward call and would otherwise observe each other's already-advanced
        # state for what is supposed to be the same token position.
        new_len = curr_len + 1
        if cache.curr_k[self.layer_idx] is None:
            cache.curr_k[self.layer_idx] = torch.zeros(B, Hkv, w, D, dtype=dtype, device=device)
            cache.curr_v[self.layer_idx] = torch.zeros(B, Hkv, w, D, dtype=dtype, device=device)
        cache.curr_k[self.layer_idx][:, :, curr_len] = k[:, :, 0]
        cache.curr_v[self.layer_idx][:, :, curr_len] = v[:, :, 0]

        if new_len == w:
            if prev is not None:
                self._insert_chunk(cache, prev.k, prev.v)
            cache.prev_chunk[self.layer_idx] = _RawChunk(cache.curr_k[self.layer_idx], cache.curr_v[self.layer_idx])
            cache.curr_k[self.layer_idx] = None
            cache.curr_v[self.layer_idx] = None

        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def certified_retrieve(self, query: torch.Tensor, cache: HSACache, top_k: int, budget: int):
        """Best-first (A*-style) descent for exact top-k retrieval, single
        (batch=1, head=1) query. `query`: (D,) raw (unrotated) vector in this
        layer's key space -- since tree slots carry no rotary phase, this
        deliberately compares content only, not position.

        Expands the frontier node with the highest admissible bound
        `<q, k_slot> + ||q||*radius_slot` (an upper bound on any exact
        token's true logit under that slot -- see HSAPool docstring for the
        caveat that this bound is trained-for, not strictly proven), until
        either `budget` expansions are exhausted or the best remaining bound
        is below the current top-k'th realized exact score (in which case
        the result is certified exact for the tokens visible to this
        query's frontier, i.e. everything before the local window).

        Returns (scores, positions, certified: bool).
        """
        import heapq

        q = query.to(torch.float32)
        q_norm = q.norm()

        heap = []  # max-heap via negated bound
        counter = 0
        levels = cache.tree_levels[self.layer_idx]

        def bound_of(node: _TreeNode, slot: int) -> float:
            k = node.k[0, 0, slot].float()
            r = node.radius[0, 0, slot].float()
            return (q @ k).item() + (q_norm * r).item()

        # `levels` is a list (one entry per depth) of dicts keyed by *canonical*
        # index (idx = chunk_idx >> level -- see HSACache.tree_levels). Seed
        # from the canonical decomposition of the *entire* cached range (the
        # same construction `_chunk_frontier` uses for a query's causal
        # prefix), not just the single highest level -- the cached range is
        # in general covered by several top-level blocks, not one.
        n_chunks = cache.next_chunk_idx[self.layer_idx]
        for lvl, idx in _dyadic_decomposition(n_chunks):
            node = levels[lvl][idx]
            for s in range(self.m):
                counter += 1
                heapq.heappush(heap, (-bound_of(node, s), counter, lvl, idx, s))

        best = []  # list of (score, position)
        expanded_nodes = set()  # (level, idx) already pushed as children -- a node
        # has `self.m` slots on the heap, and popping any one of them must not
        # re-push its children `self.m` times over. `budget` counts *nodes*
        # visited (matching the docstring), not raw heap pops -- popping a
        # slot whose node was already expanded via an earlier slot is free.
        expansions = 0
        while heap and expansions < budget:
            neg_bound, _, level, idx, slot = heapq.heappop(heap)
            bound = -neg_bound
            if len(best) >= top_k and bound <= min(s for s, _ in best):
                heapq.heappush(heap, (neg_bound, _, level, idx, slot))
                break
            if level == 0:
                # Leaf tokens are not retained after pooling in this reference
                # cache (only their pooled summary), so exact-token expansion
                # bottoms out at the level-0 chunk summary rather than a raw
                # token -- the certificate below still holds, it is just
                # granular to a `window_size`-token chunk, not a single token.
                expansions += 1
                start = idx * self.w
                node = levels[0][idx]
                score = bound_of(node, slot)
                best.append((score, start))
            elif (level, idx) not in expanded_nodes:
                expansions += 1
                expanded_nodes.add((level, idx))
                for child_idx in (2 * idx, 2 * idx + 1):
                    child = levels[level - 1].get(child_idx)
                    if child is None:
                        continue
                    for s in range(self.m):
                        counter += 1
                        heapq.heappush(heap, (-bound_of(child, s), counter, level - 1, child_idx, s))

        best.sort(key=lambda x: -x[0])
        certified = (not heap) or (expansions < budget and (not best or -heap[0][0] <= min(s for s, _ in best)))
        scores = [s for s, _ in best[:top_k]]
        positions = [p for _, p in best[:top_k]]
        return scores, positions, certified


# ===========================================================================
# FAST: Foresight-Augmented Selective Training
#
# Everything above this line is the HSA architecture. Everything from here to
# HSADecoderLayer is the training objective -- see the FAST section of the module
# docstring for what the four mechanisms are and why each is shaped the way it
# is. The heads themselves live on HSAForCausalLM at the bottom of the file.
#
# The two nn.Linear subclasses below exist so _init_weights can tell these apart
# from every other Linear in the model without a name-matching heuristic. They
# are still plain nn.Linear as far as anything else is concerned -- in particular
# train_ddp_balanced.py's convert_to_int4() picks them up by isinstance and
# offloads them like the rest.
# ===========================================================================

class _ZeroInitLinear(nn.Linear):
    """Zero-initialized output projection.

    This is what makes the main next-token loss at step 0 EXACTLY the plain
    cross-entropy this model would compute with none of the auxiliary machinery
    attached: the thought module and every auxiliary head contribute precisely
    zero until trained. That buys three things -- warm-starting from a
    pre-FAST checkpoint is safe, an A/B against a plain-objective baseline
    starts from a shared origin, and `scripts/smoke_fast.py` gets an exact
    equality it can assert rather than a tolerance it has to guess.
    """


class _GateLinear(nn.Linear):
    """The think-gate's logit projection; bias initialized negative so gating
    starts mostly closed and opens only where thinking is shown to pay."""


# ---------------------------------------------------------------------------
# Sentence boundaries
# ---------------------------------------------------------------------------

_SENTENCE_ENDERS = (".", "!", "?")

# F.normalize's default eps is 1e-12, which is six orders of magnitude below fp16's smallest
# normal (~6e-5). The forward here is fp32 so the default would compute fine, but the gradient
# flows straight back into fp16 activations, where a 1/eps-scale factor is an immediate
# overflow. These floors bound the worst-case gradient to something the loss scaler can
# actually work with, and only differ from the default at all in the degenerate case.
_NORM_EPS = 1e-3
_VAR_EPS = 1e-8


def build_sentence_end_ids(tokenizer_dir: str) -> List[int]:
    """Token ids that end a sentence, read straight out of `tokenizer.json`.

    Reads the vocab as JSON rather than instantiating a tokenizer: this gets
    called from the training script's startup path on every rank, and it needs
    nothing the `tokenizers` package would give it.

    The vocabulary is byte-level BPE, so the raw token strings use GPT-2's
    printable-byte aliases -- 'Ġ' for a space and 'Ċ' for a newline. A token ends
    a sentence if its decoded text ends in '.', '!' or '?', or if it contains a
    newline at all.

    That second rule is not an afterthought. Roughly 40% of this corpus is
    starcoder, where the unit of meaning is a line and sentence punctuation is
    mostly absent -- without it the latent head would have almost no valid target
    on the largest category in the mixture.
    """
    with open(os.path.join(tokenizer_dir, "tokenizer.json")) as fh:
        vocab = json.load(fh)["model"]["vocab"]

    ids = []
    for token, idx in vocab.items():
        if token.startswith("<") and token.endswith(">"):
            continue  # special tokens are handled by the caller (eos)
        text = token.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t").replace("č", "\r")
        if "\n" in text or text.rstrip().endswith(_SENTENCE_ENDERS):
            ids.append(idx)
    return sorted(ids)


def next_sentence_spans(is_end: torch.Tensor):
    """For every position t, the span of the NEXT complete sentence after t.

    `is_end` is (B, T) bool. Returns `(start, end, valid)`, each (B, T) -- the
    sentence occupying the inclusive range [start[t], end[t]] where valid[t].

    Fully vectorized: a reverse cumulative-min turns "is this position a
    boundary" into "index of the first boundary at or after this position" in one
    pass, and one gather chains that to the boundary after THAT one.

    The span is strictly in the future of t by construction -- `nb[t] >= t`, so
    `start = nb[t] + 1 > t`. `scripts/smoke_fast.py` asserts this rather than
    trusting it, because a bug here would be silent label leakage: the latent
    head would be trained to predict something the query position had already
    seen, and every downstream measurement would be inflated with no visible
    symptom.
    """
    B, T = is_end.shape
    ar = torch.arange(T, device=is_end.device).expand(B, T)
    # T is the "no boundary here" sentinel; it is larger than any real index, so
    # the running minimum below propagates real boundaries over it.
    b_pos = torch.where(is_end, ar, torch.full_like(ar, T))

    # Reverse cumulative min: nb[t] = min over positions >= t of b_pos.
    nb = torch.flip(torch.cummin(torch.flip(b_pos, [1]), dim=1).values, [1])

    raw_start = nb + 1
    start = raw_start.clamp(max=T - 1)    # index-safety only; see `valid` below
    end = torch.gather(nb, 1, start)      # boundary that closes the next sentence

    # Windows are random slices of a shard, so they routinely begin and end
    # mid-sentence; a position whose next sentence is truncated by the window
    # edge has no target and must be dropped rather than trained on a partial one.
    #
    # `raw_start < T` is tested rather than the clamped `start`, and that is not a
    # detail. When the only remaining boundary is the final position, clamping
    # folds start back ONTO that boundary, which then reads as a well-formed
    # one-token span -- and at t == T-1 that span contains the query position
    # itself. Validating the clamped value would hand the latent head a target the
    # query had already seen: silent label leakage, no crash, every downstream
    # number quietly inflated. Caught by scripts/smoke_fast.py's brute-force
    # comparison, which is why that test is exhaustive rather than a spot check.
    valid = (nb < T) & (raw_start < T) & (end < T) & (end >= raw_start)
    return start, end, valid


# ---------------------------------------------------------------------------
# Per-token scoring (no_grad): cross-entropy AND entropy in one chunked pass
# ---------------------------------------------------------------------------

@torch._dynamo.disable
def score_tokens(hidden, weight, labels, chunk_size: int = 256):
    """Per-token CE (surprisal) and predictive entropy, without materializing the
    full (N, vocab_size) logit tensor.

    One pass serves both gated components: entropy picks WHERE to think
    (component 2), CE ranks the learnability band (component 3), and the mean CE
    over all tokens is what the training script keeps logging as `loss` so that
    curve does not change meaning the moment selection engages.

    Chunked because the whole thing at once would be 3072 x 42000 x 4B = 516MB in
    fp32 before any temporaries, on cards where `FP32_GRAD_ACCUM` is already off
    for want of ~2GB. At the default 256 rows the transient peak is ~130MB.

    `@torch._dynamo.disable` and `no_grad` are both deliberate: this is pure
    measurement that must never appear in the autograd graph, and a Python loop
    over data-dependent chunk counts is exactly what Dynamo should not try to
    trace. Positions with an ignored label get CE 0 and entropy -inf, so they
    can never win a `topk`.
    """
    N = hidden.shape[0]
    ce = torch.empty(N, dtype=torch.float32, device=hidden.device)
    ent = torch.empty(N, dtype=torch.float32, device=hidden.device)
    with torch.no_grad():
        for i in range(0, N, chunk_size):
            h = hidden[i:i + chunk_size]
            y = labels[i:i + chunk_size]
            logp = F.log_softmax(F.linear(h, weight).float(), dim=-1)
            ce_c = -logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            ent_c = -(logp.exp() * logp).sum(-1)
            ignored = y < 0
            ce[i:i + chunk_size] = ce_c.masked_fill(ignored, 0.0)
            ent[i:i + chunk_size] = ent_c.masked_fill(ignored, float("-inf"))
    return ce, ent


def select_band(ce, valid, drop_top_frac: float, keep_frac: float, reference_ce=None):
    """Indices of the tokens that are learnable but not yet learned.

    RHO-1 as published ranks by EXCESS loss against a small reference model
    trained on high-quality data -- high loss for the current model, low for the
    reference -- and keeps the top slice. Pass `reference_ce` and that is exactly
    what happens.

    With no reference model (the configuration here: scoring 53.5B corpus tokens
    offline is infeasible, and an online reference is a whole prerequisite
    training run), the stand-in ranks by the model's own CE and keeps a MIDDLE
    band: drop the top `drop_top_frac` as noise or not-yet-learnable, keep the
    next `keep_frac`, and let the already-learned tail fall off the bottom. That
    approximates the same intent with one signal instead of two, and loses the
    genuine noise-vs-hard discrimination that only a second model can give. This
    is the one place FAST here is weaker than FAST as described.

    Fixed counts, not thresholds -- see this module's docstring on static shapes.
    """
    N = ce.shape[0]
    score = ce if reference_ce is None else ce - reference_ce
    score = score.masked_fill(~valid, float("-inf"))

    n_drop = int(drop_top_frac * N)
    n_keep = max(1, int(keep_frac * N))
    ranked = torch.topk(score, min(n_drop + n_keep, N)).indices
    return ranked[n_drop:] if reference_ce is None else ranked[:n_keep]


# ---------------------------------------------------------------------------
# Latent thought cell
# ---------------------------------------------------------------------------

class ThoughtCell(nn.Module):
    """A few recurrent latent steps taken at positions the model is unsure about.

    Built from plain `nn.Linear` and bottlenecked through `thought_dim` for two
    concrete reasons. Plain Linear so `convert_to_int4()` offloads these like
    every other weight in the model (an `nn.GRUCell`'s raw `weight_ih`/`weight_hh`
    Parameters would sit resident in fp32 and be invisible to that pass).
    Bottlenecked because a full-width gated recurrence at hidden_size=2560 is
    ~39M parameters for a module that touches 5% of positions; through a 512-d
    state it is ~4.2M for the same job.
    """

    def __init__(self, hidden_size: int, thought_dim: int, steps: int):
        super().__init__()
        self.steps = steps
        self.read = nn.Linear(hidden_size, thought_dim, bias=False)
        self.update = nn.Linear(thought_dim * 2, thought_dim * 2, bias=True)
        self.write = _ZeroInitLinear(thought_dim, hidden_size, bias=False)
        self.gate = _GateLinear(hidden_size, 1, bias=True)

    def forward(self, h):
        """h: (M, hidden). Returns (delta, gate_logit) -- the caller applies them,
        so it can reuse the gate logit for the reinforcement term.

        Note on the very first backward pass: `write` is zero-initialized, so
        `delta` is exactly zero, and therefore `gate` and `update` receive exactly
        zero gradient on step 0 (the gate's other gradient path, the advantage
        term, is also exactly zero then, by the same zero-init). `write` itself
        does get gradient -- `state` is nonzero from `read`/`update`'s random init
        -- so it moves off zero after one optimizer step and the rest of the cell
        starts learning from step 1. An all-zero gradient here on step 0 is the
        design working, not a dead module; scripts/smoke_train.py says so where it
        prints them."""
        x = self.read(h)
        state = torch.zeros_like(x)
        for _ in range(self.steps):
            z, c = self.update(torch.cat([x, state], dim=-1)).chunk(2, dim=-1)
            state = torch.sigmoid(z) * state + (1 - torch.sigmoid(z)) * torch.tanh(c)
        return self.write(state), self.gate(h)


@dataclass
class _RawChunk:
    k: torch.Tensor
    v: torch.Tensor


# ---------------------------------------------------------------------------
# Decoder layer / model / causal LM head
# ---------------------------------------------------------------------------

class HSAMLP(nn.Module):
    def __init__(self, config: HSAConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _OutScaledLinear(nn.Linear):
    """nn.Linear whose fp16 output cannot overflow, at no cost in speed or accuracy.

    Computes W @ (x/s) in the autocast dtype and restores the factor in fp32, so the value
    that has to fit in fp16 is s times smaller while the returned value is unchanged. `s` is
    a power of two, so `x * (1/s)` only decrements an exponent -- exact for every normal
    input, and the fp32 multiply back is exact too. The matmul itself is untouched, which is
    the point: an fp32 down_proj would move ~3.8% of the model's FLOPs off the tensor cores.

    Subclasses nn.Linear rather than wrapping one so that state_dict keys are unchanged
    (`...mlp.down_proj.weight`, not `...mlp.down_proj.inner.weight`) and existing checkpoints
    still load. `out_scale` is a plain float attribute, not a buffer, so it stays out of the
    state_dict too -- it is a property of the config, not of the trained values.
    """

    def __init__(self, in_features, out_features, bias=True, out_scale=1.0, **kw):
        super().__init__(in_features, out_features, bias=bias, **kw)
        self.out_scale = float(out_scale)

    def forward(self, x):
        s = self.out_scale
        if s == 1.0:
            return super().forward(x)
        return F.linear(x * (1.0 / s), self.weight, self.bias).float() * s


def _down_scale_for_layer(config: HSAConfig, layer_idx) -> float:
    """`mlp_down_scale` for the deepest `mlp_down_scale_layers` layers, 1.0 elsewhere.
    getattr for both, so configs written before these fields existed still construct."""
    s = float(getattr(config, "mlp_down_scale", 1.0))
    n = int(getattr(config, "mlp_down_scale_layers", 0))
    if layer_idx is None or s == 1.0 or n <= 0:
        return 1.0
    return s if layer_idx >= config.num_hidden_layers - n else 1.0


def _make_mlp(config: HSAConfig, layer_idx=None) -> nn.Module:
    """`LigerSwiGLUMLP` when available and requested -- it has the exact
    same gate_proj/up_proj/down_proj structure as `HSAMLP` above (so
    checkpoints are interchangeable between the two), just with a fused
    Triton kernel for silu(gate)*up. It only supports hidden_act in
    {"silu","swish"}; falls back to plain HSAMLP (any ACT2FN activation)
    otherwise."""
    _, use_liger = _resolve_accel_flags(config)
    if use_liger and config.hidden_act in ("silu", "swish"):
        return LigerSwiGLUMLP(config)
    if use_liger:
        logger.warning_once(
            f"config.use_liger_kernel=True but hidden_act={config.hidden_act!r} is not supported by "
            "LigerSwiGLUMLP (requires 'silu'/'swish'); using plain HSAMLP."
        )
    return HSAMLP(config)


def _apply_down_scale(mlp: nn.Module, scale: float) -> nn.Module:
    """Swap mlp.down_proj for an _OutScaledLinear sharing the SAME Parameter objects.

    Works for LigerSwiGLUMLP as well as HSAMLP: liger fuses only silu(gate)*up, leaving
    down_proj a distinct nn.Linear that both call as `self.down_proj(...)`.
    """
    if scale == 1.0:
        return mlp
    old = mlp.down_proj
    new = _OutScaledLinear(old.in_features, old.out_features,
                           bias=old.bias is not None, out_scale=scale,
                           device="meta")
    new.weight = old.weight            # share, don't copy -- keeps init//loaded values
    new.bias = old.bias
    mlp.down_proj = new
    return mlp


class HSADecoderLayer(nn.Module):
    def __init__(self, config: HSAConfig, layer_idx: int):
        super().__init__()
        self.self_attn = HSAAttention(config, layer_idx)
        self.mlp = _apply_down_scale(_make_mlp(config, layer_idx),
                                    _down_scale_for_layer(config, layer_idx))
        self.input_layernorm = _make_rmsnorm(config.hidden_size, config)
        self.post_attention_layernorm = _make_rmsnorm(config.hidden_size, config)
        # getattr, not config.fp32_residual: checkpoints written before this field existed
        # deserialize into an HSAConfig without it, and those should still load.
        self.fp32_residual = getattr(config, "fp32_residual", True)

    def forward(self, hidden_states, position_ids, attention_mask, rotary_emb, cache, use_cache, is_decode, pos, curr_len):
        # `.float()` on the RESIDUAL only -- see HSAConfig.fp32_residual for the measurements.
        # The sublayers still run in fp16: every projection inside self_attn and mlp is an
        # autocast-eligible nn.Linear, so handing them an fp32 hidden_states costs nothing but
        # the cast, and the fp32 accumulator is only carrying the sum they add into. Adding
        # fp16 `h` to an fp32 `residual` promotes to fp32 by ordinary type promotion, so the
        # add itself can no longer reach fp16's ceiling.
        #
        # The stream STAYS fp32 across layers rather than being cast back at each boundary:
        # casting down would just move the same overflow onto the cast. HSAModel.forward's
        # closing self.norm() brings it back to O(1) for everything downstream.
        acc = hidden_states.float() if self.fp32_residual else hidden_states

        # The sublayers consume the NORMALIZED tensor, never `acc` itself, so casting each
        # norm's output down to the weight dtype is safe however large `acc` has grown --
        # RMSNorm has already brought it back to O(1). Cast unconditionally rather than
        # relying on autocast to do it: autocast is not always on. scripts/smoke_fp16.py
        # runs this model in pure fp16 with no autocast region, and there an fp32 `acc`
        # reaches an fp16 nn.Linear as a dtype mismatch ("expected mat1 and mat2 to have the
        # same dtype") rather than as a silent upcast.
        wdtype = next(self.mlp.parameters()).dtype

        residual = acc
        h = self.input_layernorm(acc).to(wdtype)
        h = self.self_attn(h, position_ids, attention_mask, rotary_emb, cache, use_cache, is_decode, pos, curr_len)
        acc = residual + h

        residual = acc
        h = self.post_attention_layernorm(acc).to(wdtype)
        h = self.mlp(h)
        acc = residual + h
        return acc


class HSAPreTrainedModel(PreTrainedModel):
    config_class = HSAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HSADecoderLayer"]
    _supports_cache_class = False
    _supports_static_cache = False
    # HSACache isn't a transformers.Cache, and isn't rewindable (needed for
    # assisted generation) -- same category as Mamba's recurrent state.
    _is_stateful = True

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # GenerationMixin's own implementation of this (transformers 5.15.0,
        # generation/utils.py) does NOT consult _is_stateful/_supports_cache_class at all --
        # it just checks the class name against a hardcoded allowlist (reformer, minimax,
        # xlnet, rwkv, xlstm), so "HSAForCausalLM" falls through to True and generate()
        # injects a real DynamicCache into past_key_values regardless of the flags above.
        # HSAModel.forward creates its own HSACache when use_cache=True and none is
        # supplied -- a DynamicCache there crashes the first cache access
        # (`past_key_values.total_tokens`, HSACache-only). Overriding this method directly
        # is the only thing this transformers version actually reads for that decision.
        return False

    def _init_weights(self, module):
        std = self.config.initializer_range
        # The two marker subclasses are checked before the generic nn.Linear
        # branch because they ARE nn.Linear and would otherwise be caught by it.
        if isinstance(module, _ZeroInitLinear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, _GateLinear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.fill_(-2.0)   # start mostly closed
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, HSARMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, HSAPool):
            module.query.data.normal_(mean=0.0, std=std)
        elif isinstance(module, HSAAttention):
            module.level_embed.data.normal_(mean=0.0, std=std)


@dataclass
class HSACausalLMOutput(CausalLMOutputWithPast):
    """`CausalLMOutputWithPast` plus a breakdown of what went into `loss`.

    The training script logs every entry. Under fp16 with a manual loss scaler
    and four summed objectives, an auxiliary term quietly collapsing to zero or
    swamping the main loss is otherwise completely invisible until the run has
    already been wasted.

    Every value is a DETACHED 0-dim CUDA TENSOR, never a Python float, and that
    is a performance contract rather than a style choice. `.item()` synchronizes
    the device: calling it on a dozen metrics inside the forward, for each of 16
    gradient-accumulation microbatches, would be ~190 syncs per optimizer step,
    each one stalling the CPU mid-graph before the backward has even been queued
    and destroying the run-ahead that keeps the GPU fed. The training loop
    accumulates these as tensors and converts once per log interval, folded into
    the cross-rank all_reduce it was already doing.
    """

    metrics: Optional[Dict[str, torch.Tensor]] = None


class HSAModel(HSAPreTrainedModel):
    def __init__(self, config: HSAConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([HSADecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = _make_rmsnorm(config.hidden_size, config)
        self.rotary_emb = HSARotaryEmbedding(config)
        self.gradient_checkpointing = False
        # Only consulted when gradient_checkpointing is also True -- see _sac_policy's
        # docstring above for what this trades relative to full-layer recompute.
        self.selective_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        B, T, _ = inputs_embeds.shape

        if use_cache and past_key_values is None:
            past_key_values = HSACache(len(self.layers))

        past_len = past_key_values.total_tokens if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + T, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)
        # `is_decode`, `pos`, and `curr_len` are snapshotted once, before any
        # layer runs, and NOT re-read from `past_key_values` by individual
        # layers. Layers execute sequentially within one forward call and each
        # layer's decode step mutates its own slice of cache state; if a later
        # layer re-read shared scalars like total_tokens/curr_len from the
        # cache mid-pass, it would see the *next* step's values (already
        # advanced by layer 0) while still processing the current token.
        is_decode = past_key_values is not None and past_len > 0
        curr_len = past_key_values.curr_len if is_decode else 0
        if is_decode:
            assert T == 1, "HSACache decode path only supports one new token per forward call"

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                ckpt_args = (layer, hidden_states, position_ids, attention_mask, self.rotary_emb,
                             past_key_values, use_cache, is_decode, past_len, curr_len)
                if self.selective_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        *ckpt_args, use_reentrant=False, context_fn=_sac_context_fn,
                    )
                else:
                    # Deliberately does NOT pass context_fn at all, rather than passing
                    # torch's own default (noop_context_fn) explicitly: the two are
                    # semantically identical, but Dynamo's checkpoint handler cannot trace ANY
                    # context_fn argument and dies with "checkpoint not implemented for
                    # LazyVariableTracker context_fn" -- so with COMPILE=True, naming the
                    # default was enough to break compilation of the whole model. Omitting it
                    # keeps this the plain form Dynamo does support.
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        *ckpt_args, use_reentrant=False,
                    )
            else:
                hidden_states = layer(
                    hidden_states, position_ids, attention_mask, self.rotary_emb, past_key_values, use_cache,
                    is_decode, past_len, curr_len,
                )
        # Back to the model's nominal dtype. With config.fp32_residual the stream carried
        # fp32 through the layer stack, so self.norm() returns fp32 too -- and lm_head's
        # weights are fp16, which under autocast is invisible but outside it is a dtype
        # mismatch at the very last matmul. Safe to narrow here for the same reason the
        # per-layer casts are: norm output is O(1), whatever the stream reached.
        hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if is_decode and use_cache:
            new_len = curr_len + 1
            past_key_values.total_tokens = past_len + 1
            past_key_values.curr_len = 0 if new_len == self.config.window_size else new_len

        if not return_dict:
            return (hidden_states, past_key_values, all_hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class HSAForCausalLM(HSAPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: HSAConfig):
        super().__init__(config)
        self.model = HSAModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        _, use_liger = _resolve_accel_flags(config)
        self.fused_lce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100) if use_liger else None
        # Per-token scoring with gradient is unsupported by the fused kernel (see
        # the FAST section of this module's docstring), so this instance is only
        # ever used under no_grad.
        self.fused_lce_none = (
            LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="none")
            if use_liger else None
        )
        # reduction="sum", divided by the target count in _ce_mean -- see its docstring for
        # why that is not the same as asking the kernel for "mean".
        self.fused_lce_sum = (
            LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="sum")
            if use_liger else None
        )

        H, D = config.hidden_size, config.latent_dim

        # --- 1. multi-horizon token heads -----------------------------------
        # Each head is a norm plus one square projection feeding the SHARED,
        # already-tied lm_head weight. Giving a head its own vocab-sized output
        # matrix would add 107M parameters apiece to predict a token the main
        # head already has an output space for.
        self.mtp_norms = nn.ModuleList(
            [_make_rmsnorm(H, config) for _ in range(config.mtp_horizons)]
        )
        self.mtp_projs = nn.ModuleList(
            [_ZeroInitLinear(H, H, bias=False) for _ in range(config.mtp_horizons)]
        )

        # --- 1b. latent (next-sentence) head --------------------------------
        self.latent_proj = nn.Linear(H, D, bias=False)
        # Deliberately NOT a _ZeroInitLinear, unlike every other auxiliary output projection
        # here. Its output is L2-normalized, and F.normalize of an exactly-zero vector divides
        # by eps -- so a zero-init predictor does not start "contributing nothing", it starts
        # with a gradient of order 1/eps. Under fp16 that overflows on step 0 and every step
        # after: the loss scaler halves, the optimizer step is skipped, and the weights never
        # move while every logged loss looks perfectly reasonable. Cost one real training run
        # to find. The zero-init invariant that actually matters is on the MAIN next-token
        # loss, and nothing on that path runs through a normalization.
        self.latent_pred = nn.Linear(D, D, bias=False)

        # The EMA target projection is a BUFFER, not a Module, for three reasons
        # that all bite otherwise: convert_to_int4() quantizes every nn.Linear it
        # finds and an int4 EMA target is worthless; the optimizer is constructed
        # over model.parameters() and would try to step it; and DDP would want to
        # reduce it. As a buffer it is none of their business. It stays identical
        # across ranks without any sync because it is a deterministic EMA of
        # already-all-reduced weights -- provided it is SEEDED after
        # broadcast_all(), which is what init_ema_targets() is for.
        self.register_buffer("latent_target_w", torch.zeros(D, H), persistent=True)
        self.register_buffer("ema_initialized", torch.zeros(1, dtype=torch.bool), persistent=True)

        # The entropy above which thinking fires at INFERENCE time. During training the
        # thought budget is a fixed-count top-k (static shapes, see the module docstring), so
        # there is no explicit threshold -- but generation produces one token at a time and a
        # top-5%-of-1 rule would just mean "always", applying the gate in a regime it was
        # never trained in. Tracking the k-th largest entropy as an EMA during training gives
        # inference the same operating point training actually used. Starts at +inf so an
        # untrained model never thinks.
        self.register_buffer("thought_entropy_threshold",
                             torch.full((), float("inf")), persistent=True)

        # --- 2. thought cell -------------------------------------------------
        self.thought = ThoughtCell(H, config.thought_dim, config.thought_steps)

        # --- sentence boundary lookup ---------------------------------------
        table = torch.zeros(config.vocab_size, dtype=torch.bool)
        if config.sentence_end_token_ids:
            table[torch.tensor(config.sentence_end_token_ids, dtype=torch.long)] = True
        if config.eos_token_id is not None:
            table[config.eos_token_id] = True
        self.register_buffer("sentence_end", table, persistent=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    @torch.no_grad()
    def init_ema_targets(self):
        """Seed the EMA target from the online projection.

        MUST be called after `broadcast_all(model, src=0)` in the training script:
        that function broadcasts parameters only, not buffers, so seeding earlier
        would leave every rank with a different target and the latent loss would
        be reducing gradients toward three different objectives.
        """
        self.latent_target_w.copy_(self.latent_proj.weight.detach().float())
        self.ema_initialized.fill_(True)

    @torch.no_grad()
    def update_ema_targets(self):
        """Advance the EMA target one step.

        Called once per OPTIMIZER step by the training script, not from forward():
        forward runs GRAD_ACCUM_STEPS=16 times per optimizer step, so an in-forward
        update would apply the decay 16x per weight change and silently make the
        target track ~16x faster than `latent_ema_decay` says. Must run while
        weights are materialized -- after optimizer.step(), before evict_all().
        """
        if not bool(self.ema_initialized):
            self.init_ema_targets()
            return
        d = self.config.latent_ema_decay
        self.latent_target_w.mul_(d).add_(self.latent_proj.weight.detach().float(), alpha=1.0 - d)

    # -- loss helpers -------------------------------------------------------

    def _ce_mean(self, hidden, labels):
        """Gradient-carrying mean CE through the tied lm_head.

        Uses liger's fused path when available (the (N, vocab) logits never
        materialize); falls back to a plain projection otherwise, which is what
        the CPU smoke tests and the RTX 3060 generation path take.

        Summed and divided by a clamped count rather than asking for
        `reduction="mean"`. Arithmetically identical whenever there is at least
        one target, and returns 0 instead of NaN when there is none. That is not
        hypothetical: an auxiliary horizon whose targets all fall past the end of
        a short sequence, or a random position subset that happens to miss every
        valid one, hands this an all-ignored label vector, and mean's 0/0 NaN
        would propagate into the summed objective and poison the entire backward.
        The count is a tensor, so nothing here synchronizes, and a NaN arriving
        from anywhere else still propagates -- which matters, because the loss
        scaler and NAN_STOP_STEPS both depend on seeing real ones.
        """
        n = (labels >= 0).sum().clamp(min=1)
        if self.fused_lce_sum is not None:
            return _call_fused_lce(self.fused_lce_sum, self.lm_head.weight, hidden, labels) / n
        return F.cross_entropy(
            F.linear(hidden, self.lm_head.weight).float(), labels,
            ignore_index=-100, reduction="sum",
        ) / n

    @torch._dynamo.disable
    def _ce_none_nograd(self, hidden, labels):
        """Per-token CE with NO gradient. `reduction="none"` has no working
        backward in liger (see module docstring), so this is no_grad by
        construction rather than by convention."""
        with torch.no_grad():
            if self.fused_lce_none is not None:
                return self.fused_lce_none(self.lm_head.weight, hidden, labels).float()
            return F.cross_entropy(
                F.linear(hidden, self.lm_head.weight).float(),
                labels, ignore_index=-100, reduction="none",
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        aux_scale: float = 1.0,
        latent_scale: float = 1.0,
        select: bool = True,
        **kwargs,
    ):
        """With `labels`, this computes the FAST objective (see the FAST section of
        this module's docstring), not a plain next-token cross-entropy -- there is
        no argument that selects the latter. Without `labels` it is an ordinary
        causal-LM forward returning logits, with the thought module active.

        `latent_scale` gates the next-sentence objective specifically, and it exists because
        of a measured failure rather than on principle. That objective asks the model to
        predict where the text is going; very early in training consecutive sentences all
        embed to nearly the same place, so there is nothing to predict, and the cheapest way
        to reduce a cosine loss on a near-constant target is to shrink both branches toward a
        constant. That is collapse, and the cosine loss reports it as excellent all the way
        down -- observed directly (see _latent_loss). Holding the term at zero until the
        trunk has learned something means it engages when there is a signal to learn, in the
        same spirit as `select`. Like the others it ramps in once and stays on.

        `aux_scale` ramps the auxiliary objectives in from zero over warmup, so
        the manual fp16 loss scaler is not asked to absorb four new loss terms at
        full strength on step 0. `select` is the training script's "selection has
        not engaged yet" signal -- ranking tokens by the model's own loss is
        meaningless before the model has one worth ranking by. Neither is an
        on/off switch for a component: aux_scale reaches 1.0 and stays there, and
        select engages at a fixed step and stays engaged.
        """
        return_dict = return_dict if return_dict is not None else True

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state

        if labels is None:
            # Inference. The multi-horizon and latent heads are training signal and have
            # nothing to contribute here, but the thought module does: a mechanism that only
            # ever fires during training is scaffolding, not a model that thinks. It runs at
            # the same operating point training used (see thought_entropy_threshold) and it
            # reads only h_t, so it is causal for free and works identically under the
            # incremental decode cache, where T == 1.
            return CausalLMOutputWithPast(
                loss=None,
                logits=self._logits_with_thought(hidden),
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
            )

        loss, metrics = self._fast_losses(hidden, input_ids, labels, aux_scale,
                                          latent_scale, select)

        # `logits` is None only where it genuinely has to be: the fused linear+CE kernel never
        # materializes the (B, T, vocab_size) tensor, which is the entire point of fusing it
        # and is the trade-off the module docstring describes. On the plain-PyTorch path there
        # is no such constraint, and returning them alongside a loss is behavior that predates
        # FAST -- scripts/smoke_forward.py checks their shape and finiteness while also passing
        # labels. Same function the labels=None branch uses, so "the model's logits" means one
        # thing regardless of how it was called.
        logits = None if self.fused_lce is not None else self._logits_with_thought(hidden)

        if not return_dict:
            return (loss, logits, outputs.past_key_values, outputs.hidden_states)
        return HSACausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            metrics=metrics,
        )

    @torch._dynamo.disable
    def _logits_with_thought(self, hidden):
        """lm_head logits, with the extra latent steps taken where the model is unsure.

        Deliberately NOT the training-time fixed-count top-k: generation hands this one token
        per call, and "the top 5% of one token" is just "always", which would apply the gate
        at low-entropy positions it was never trained on. The threshold is the one training
        actually operated at, tracked as an EMA of its own k-th-largest entropy.

        Falls through to plain logits when nothing crosses the threshold, which is every call
        on an untrained model (the threshold starts at +inf).
        """
        logits = self.lm_head(hidden)
        thr = self.thought_entropy_threshold
        if not torch.isfinite(thr):
            return logits

        B, T, _ = hidden.shape
        flat = hidden.reshape(-1, hidden.shape[-1])
        logp = torch.log_softmax(logits.reshape(flat.shape[0], -1).float(), dim=-1)
        ent = -(logp.exp() * logp).sum(-1)
        idx = (ent > thr).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return logits

        h_sel = flat[idx]
        delta, gate_logit = self.thought(h_sel)
        refined = h_sel + torch.sigmoid(gate_logit) * delta
        out = logits.reshape(flat.shape[0], -1).index_copy(
            0, idx, self.lm_head(refined).to(logits.dtype))
        return out.reshape(B, T, -1)

    @torch._dynamo.disable
    def _fast_losses(self, hidden, input_ids, labels, aux_scale, latent_scale, select):
        """All four components. Deliberately opaque to Dynamo.

        The trunk -- where every one of the ~19 TFLOP per microbatch lives -- is
        compiled normally inside self.model(). What is left here is a handful of
        small projections, a fixed-count topk, and calls into liger's custom
        Triton autograd.Functions, which Dynamo already has to break around (see
        `_call_fused_lce`'s comment in modeling_hsa.py for the Inductor crash that
        partial-tracing one of those caused). Taking one clean break for the whole
        block is both faster to compile and far less fragile than letting it try.
        """
        cfg = self.config
        B, T, H = hidden.shape

        # Everything below lives in the SHIFTED flat space: index n = b*(T-1) + t
        # means "position t predicting token t+1 in batch row b".
        h_flat = hidden[:, :-1].reshape(-1, H)
        y_flat = labels[:, 1:].reshape(-1)
        N = h_flat.shape[0]
        metrics = {}

        if N == 0:
            # A length-1 sequence has no next-token target at all, so every objective here
            # is undefined rather than merely small. Returning a zero attached to the graph
            # keeps .backward() legal for a caller that batches ragged inputs; the full metric
            # key set is returned zeroed rather than empty, because the training loop
            # accumulates metrics by key and all_reduces them positionally -- a microbatch
            # that contributed a different key set would misalign that collective across
            # ranks. scripts/smoke_forward.py covers T=1 explicitly.
            zero = hidden.sum() * 0.0
            keys = ("ce_all", "ce_selected", "entropy", "gate", "gate_loss", "latent",
                    "latent_pred_std", "latent_std", "latent_target_std",
                    "latent_valid_frac", "mtp", "selected_frac", "thought_advantage")
            return zero, {k: zero.detach() for k in keys}

        # --- measurement pass: entropy (where to think) + CE (what to learn) --
        ce, ent = score_tokens(
            h_flat.detach(), self.lm_head.weight, y_flat, cfg.score_chunk_size
        )
        valid = y_flat >= 0
        # Reported as `loss` by the training script: the full-token mean CE, which
        # keeps meaning the same thing before and after selection engages and stays
        # comparable to a baseline run's curve. The selected-subset loss is logged
        # separately rather than in its place.
        n_valid_tok = valid.sum().clamp(min=1)
        metrics["ce_all"] = ce.sum() / n_valid_tok
        metrics["entropy"] = ent.masked_fill(~valid, 0.0).sum() / n_valid_tok

        # --- 3. learnability-gated selection ---------------------------------
        if select:
            keep = select_band(ce, valid, cfg.select_drop_top, cfg.select_keep_fraction)
            band = torch.zeros(N, dtype=torch.bool, device=ce.device)
            band[keep] = True
            band &= valid
        else:
            band = valid
        metrics["selected_frac"] = band.float().mean()

        # --- 2. uncertainty-gated thinking -----------------------------------
        # Thought positions are drawn from WITHIN the selected band. Choosing them
        # over all positions instead would regularly spend the thought budget on
        # tokens whose loss is about to be masked out, where the refinement can
        # produce no gradient at all.
        # Clamped to N: thought_fraction of a short sequence rounds to zero, and max(1, ..)
        # would then ask topk for more elements than exist.
        M = min(N, max(1, int(cfg.thought_fraction * N)))
        idx = torch.topk(ent.masked_fill(~band, float("-inf")), M).indices

        if self.training:
            # ent[idx] is the selected top-M, so its minimum IS the k-th largest entropy --
            # the cutoff this step used. No sort needed, and no host sync.
            with torch.no_grad():
                cut = ent[idx].min()
                d = self.config.latent_ema_decay
                if torch.isfinite(self.thought_entropy_threshold):
                    self.thought_entropy_threshold.mul_(d).add_(cut, alpha=1.0 - d)
                else:
                    self.thought_entropy_threshold.copy_(cut)

        h_sel = h_flat[idx]
        delta, gate_logit = self.thought(h_sel)
        h_new = h_sel + torch.sigmoid(gate_logit) * delta
        # Out-of-place: index_copy_ on a tensor that autograd still needs would be
        # an in-place mutation of a graph input. This form keeps both paths alive.
        h_ref = h_flat.index_copy(0, idx, h_new)

        # "Reinforced only if it improved prediction of the coming span": measure
        # the coming span's loss with and without the refinement and reinforce the
        # DECISION to think by that margin. The thought's CONTENT is trained by the
        # ordinary differentiable path through h_ref below. Both measurements are
        # no_grad, which is also what keeps `reduction="none"` legal here.
        y_sel_th = y_flat[idx]
        ce_with = self._ce_none_nograd(h_new.detach(), y_sel_th)
        # `ce[idx]` is the same quantity and is already computed, but it comes from
        # score_tokens' chunked log_softmax while ce_with comes from liger's fused kernel.
        # Differencing two implementations leaves their disagreement -- order 1e-3 in fp16 --
        # sitting in the advantage, which at init is ENTIRELY what the advantage consists of,
        # since a zero-init thought cannot change any prediction. The gate's reinforcement
        # signal would then be pure numerical noise until the thought learned to outrun it.
        # Recomputing both sides through the same function costs one extra fused-loss call
        # over ~5% of positions and makes the difference exactly zero when it should be.
        ce_without = self._ce_none_nograd(h_sel.detach(), y_sel_th)
        advantage = (ce_without - ce_with).clamp(-cfg.advantage_clamp, cfg.advantage_clamp)
        # logsigmoid, not log(sigmoid(.)): the latter underflows to -inf in fp16
        # exactly where the gate is most closed, which is most of them at the start.
        loss_gate = -(advantage * F.logsigmoid(gate_logit.squeeze(-1).float())).mean()
        metrics["thought_advantage"] = advantage.mean()
        metrics["gate"] = torch.sigmoid(gate_logit).detach().mean()

        # --- main next-token loss, on the refined states, over the band -------
        y_sel = y_flat.masked_fill(~band, -100)
        loss_main = self._ce_mean(h_ref, y_sel)
        metrics["ce_selected"] = loss_main.detach()

        # --- 1. multi-horizon token heads ------------------------------------
        # One shared random subset of positions for all horizons: fixed count, so
        # the shape is static, and half the lm_head projections of scoring every
        # position. randperm rather than a mask because the fused loss charges for
        # every row it is handed, ignored or not.
        n_mtp = max(1, int(cfg.mtp_position_fraction * N))
        sub = torch.randperm(N, device=hidden.device)[:n_mtp]
        h_sub = h_flat[sub]

        loss_mtp = hidden.new_zeros(())
        # A horizon reaching past the end of the sequence has no target anywhere in this
        # window -- skipped rather than run, since it is pure waste, and the divisor below
        # must count only the heads that actually contributed. Depends on T and the config,
        # never on the data, so this stays a compile-time constant.
        horizons = [(i, h) for i, h in enumerate(range(2, 2 + cfg.mtp_horizons)) if T - h > 0]
        for i, horizon in horizons:
            # Position t predicts token t+horizon. Columns past T-horizon have no
            # target inside this window and are ignored rather than wrapped.
            y_h = labels.new_full((B, T - 1), -100)
            y_h[:, :T - horizon] = labels[:, horizon:]
            h_h = self.mtp_projs[i](self.mtp_norms[i](h_sub))
            loss_mtp = loss_mtp + self._ce_mean(h_h, y_h.reshape(-1)[sub])
        loss_mtp = loss_mtp / max(len(horizons), 1)
        metrics["mtp"] = loss_mtp.detach()

        # --- 1b. latent next-sentence prediction ------------------------------
        loss_latent, loss_var, lat_metrics = self._latent_loss(hidden, input_ids)
        metrics.update(lat_metrics)

        total = (
            loss_main
            + aux_scale * cfg.w_mtp * loss_mtp
            + aux_scale * latent_scale * cfg.w_latent * loss_latent
            + aux_scale * latent_scale * cfg.w_variance * loss_var
            + aux_scale * cfg.w_gate * loss_gate
        )
        metrics["gate_loss"] = loss_gate.detach()
        return total, metrics

    def _latent_loss(self, hidden, input_ids):
        """Predict an embedding of the next sentence, not its literal words.

        BYOL-shaped: an online projection plus an asymmetric predictor against a
        stop-gradient EMA target. The variance hinge is not optional decoration --
        this objective's characteristic failure is collapse to a constant, which a
        cosine loss reports as excellent the whole way down.

        WATCH THIS ONE ON THE FIRST REAL RUN. Collapse was observed directly, and
        the three logged spreads (`latent_std`, `latent_pred_std`,
        `latent_target_std`, all of which should sit near 1/sqrt(latent_dim)) are
        what makes it visible; `latent` alone will look better and better as it
        happens. What was measured, at global batch 2 / lr 4e-4 -- a deliberately
        abusive toy configuration, not the production one:

          - both branches collapsed together, since latent_proj feeds the
            predictor AND, through the EMA, the target; the target follows the
            prediction down and nothing pulls either back out;
          - it set in during a step where the TRUNK itself went degenerate
            (entropy 1.8 nats at cross-entropy 12.4 -- confidently wrong). The
            trunk recovered; the latent head did not, because once z has no
            spread the EMA target has none either and the hinge is the only force
            left;
          - at production shapes the same objective degraded far more slowly.

        Three knobs, in the order worth reaching for: W_VARIANCE (must exceed
        W_LATENT -- see training_config.py for why that is an inequality, not a
        preference), LATENT_EMA_DECAY (a slower target lags the online net and is
        the standard anti-collapse lever in this family of objectives), and
        W_LATENT itself. None of them can rescue a run whose trunk is diverging,
        which is the first thing to rule out if these numbers slide.
        """
        cfg = self.config
        B, T, H = hidden.shape
        D = cfg.latent_dim

        is_end = self.sentence_end[input_ids]
        start, end, valid = next_sentence_spans(is_end)
        # Query positions live in the same shifted space as everything else.
        start, end, valid = start[:, :-1], end[:, :-1], valid[:, :-1]

        N = B * (T - 1)
        n_lat = max(1, int(cfg.latent_position_fraction * N))
        sub = torch.randperm(N, device=hidden.device)[:n_lat]

        # Segment means by cumulative sum: two gathers instead of a loop over
        # variable-length spans. fp32 because a 512-step fp16 cumsum of activations
        # loses far too much of the tail to be a usable target.
        hf = hidden.detach().float()
        cs = torch.cat([hf.new_zeros(B, 1, H), hf.cumsum(1)], dim=1)   # (B, T+1, H)

        s_flat = start.reshape(-1)[sub]
        # `end` carries the sentinel T where no next boundary exists, and cs has
        # only T+1 columns -- clamp before it reaches the gather. Those rows are
        # masked out of the loss by v_flat so the value read is irrelevant, but the
        # index still has to be in range.
        e_flat = end.reshape(-1)[sub].clamp(max=T - 1)
        v_flat = valid.reshape(-1)[sub]
        row = torch.div(sub, T - 1, rounding_mode="floor")

        seg = cs[row, e_flat + 1] - cs[row, s_flat]
        seg = seg / (e_flat - s_flat + 1).clamp(min=1).unsqueeze(-1).float()

        with torch.no_grad():
            target = F.linear(seg, self.latent_target_w)
            target = F.normalize(target, dim=-1, eps=_NORM_EPS)

        h_sub = hidden[:, :-1].reshape(-1, H)[sub]
        # z is the REPRESENTATION; pred is the predictor's guess at the target. Keeping them
        # separate matters for the variance term below.
        z = F.normalize(self.latent_proj(h_sub).float(), dim=-1, eps=_NORM_EPS)
        pred = F.normalize(self.latent_pred(z).float(), dim=-1, eps=_NORM_EPS)

        cos = (pred * target).sum(-1)
        n_valid = v_flat.sum()
        loss_latent = ((1.0 - cos) * v_flat).sum() / n_valid.clamp(min=1)

        # VICReg-style hinge on the per-dimension std of the predictions.
        #
        w = v_flat.to(pred.dtype).unsqueeze(-1)
        n_w = w.sum().clamp(min=2.0)

        def masked_std(x, floor):
            """Per-dimension std over the valid rows only.

            Written out rather than `x[v_flat].std(0)`: boolean-mask indexing produces a
            data-dependent shape, which both forces a host synchronization to learn the count
            and hands a new shape to every downstream kernel on every step. Same number,
            neither problem."""
            m = (x * w).sum(0) / n_w
            v = (((x - m) ** 2) * w).sum(0) / (n_w - 1.0)
            return v.clamp(min=floor).sqrt()

        # The hinge is applied to z, the representation, NOT to pred.
        #
        # Applying it to pred does not prevent collapse, and a real run showed exactly how it
        # fails: latent_proj feeds the predictor AND, through the EMA, the target, so the
        # cosine term can be driven down by shrinking BOTH branches together. The target then
        # follows the prediction down and the cosine loss reports success the whole way --
        # latent 1.00 -> 0.03 while latent_std fell 0.029 -> 0.0003 and latent_target_std
        # tracked it 0.026 -> 0.0095. Constraining the predictor's output cannot stop that,
        # because the collapse is upstream of the predictor. Constraining z does: z is what
        # feeds both branches, and it is the only one of the two with a gradient path (the
        # target is stop-grad by construction). This is also where VICReg puts it -- on the
        # embeddings, with the predictor as the asymmetry, not on the predictor's output.
        std = masked_std(z, _VAR_EPS)
        # VICReg's hinge targets a std of 1, which is right for the unnormalized embeddings it
        # was defined on. `pred` here is L2-normalized in D dimensions, so its per-dimension
        # std cannot exceed ~1/sqrt(D) even when perfectly spread over the sphere -- against a
        # target of 1 the hinge would be permanently saturated, contributing a constant ~1.0
        # to the loss and a constant gradient that says "spread out more" no matter how well
        # spread out it already is. Targeting 1/sqrt(D) is the same statement scaled to what
        # this geometry can actually reach; dividing by it keeps the term O(1) so W_VARIANCE
        # stays comparable to the other weights.
        target_std = D ** -0.5
        loss_var = F.relu(target_std - std).mean() / target_std
        pred_std = masked_std(pred, _VAR_EPS)

        # Reports the TRUE count, not the clamped one. Dividing by the clamped
        # value here would make "no valid next-sentence target at all" -- which
        # sends loss_latent to a perfectly healthy-looking 0.0 -- display
        # identically to "exactly one", and the latent head could sit dead for a
        # whole run with nothing in the log to say so.
        # The TARGET's spread is logged next to the prediction's because "latent_std collapsed"
        # has two completely different causes with different fixes, and the cosine loss looks
        # equally good under both. If the target still varies and the prediction does not, the
        # predictor has collapsed and the variance hinge is losing to the cosine term -- a
        # weighting problem. If the target has no variance either, nothing has gone wrong with
        # the objective: the trunk is early enough in training that consecutive sentences all
        # embed to nearly the same place, and there is no signal to predict yet. Without this
        # second number the two are indistinguishable from the log.
        with torch.no_grad():
            t_std = masked_std(target, 0.0).mean()
        # All three spreads are logged, because they fail in distinguishable ways and the
        # cosine loss looks equally healthy under every one of them. latent_std is the
        # regularized representation; latent_pred_std is the predictor's output; and
        # latent_target_std is the stop-grad EMA branch. All three should sit near
        # 1/sqrt(latent_dim) (~0.031 at D=1024). All three sliding toward zero together is
        # collapse; only the target sliding means the EMA decay is too fast for the trunk.
        return loss_latent, loss_var, {
            "latent": loss_latent.detach(),
            "latent_std": std.mean().detach(),
            "latent_pred_std": pred_std.mean().detach(),
            "latent_target_std": t_std,
            "latent_valid_frac": n_valid.detach() / max(n_lat, 1),
        }

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # `past_key_values is not None` alone doesn't mean "has state": a
        # caller (or our own auto-created cache) can hand in a fresh, empty
        # HSACache, which must still get the *full* prompt on this call.
        if past_key_values is not None and past_key_values.total_tokens > 0:
            input_ids = input_ids[:, -1:]
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, **kwargs):
        model_kwargs["past_key_values"] = outputs.past_key_values
        if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
            am = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat([am, am.new_ones((am.shape[0], 1))], dim=-1)
        return model_kwargs

    def _get_initial_cache_position(self, seq_length, device, model_kwargs):
        # HSACache tracks its own position via total_tokens; GenerationMixin
        # only needs *some* cache_position tensor to exist in model_kwargs.
        model_kwargs["cache_position"] = torch.arange(seq_length, device=device)
        return model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        raise NotImplementedError("HSACache does not support beam search in this reference implementation.")

    def new_cache(self) -> HSACache:
        return HSACache(self.config.num_hidden_layers)
