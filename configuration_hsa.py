"""HSA (Hierarchical Softmax Attention) model configuration."""

from transformers.configuration_utils import PretrainedConfig


class HSAConfig(PretrainedConfig):
    """Configuration for [`HSAModel`] / [`HSAForCausalLM`].

    HSA replaces the flat KV cache of a transformer with:
      1. exact local (sliding-window) softmax attention over the last
         `window_size` tokens, and
      2. an O(log N) read over a causal dyadic tree of RMS-normalized,
         log-mass-weighted key/value summaries ("frontier" attention),
      3. an optional certified best-first descent into the tree for
         exact top-k retrieval at inference time (see `certified_retrieve`
         in modeling_hsa.py).

    The `mtp_*` / `latent_*` / `thought_*` / `select_*` / `w_*` fields below
    configure FAST (Foresight-Augmented Selective Training), which is how this
    model is trained and what several of its heads are for -- see the FAST
    section of modeling_hsa.py's module docstring. They live here, rather than
    only in scripts/training_config.py, because the horizon count and latent
    width are architecture: a checkpoint that does not record them cannot be
    reloaded. training_config.py stays the single place a human edits them, and
    the training script carries its values in here at construction.

    None of them can switch a component off. `mtp_horizons=0` raises rather than
    quietly disabling multi-horizon prediction.

    Sequence-axis dependency depth is O(log N) instead of O(N) (SSM/linear
    attention) or O(1) growth with unbounded range (full attention KV
    cache), which keeps roundoff bounded (~u * log2(N)) under pure fp16.
    """

    model_type = "hsa"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=42000,
        hidden_size=2560,
        intermediate_size=7680,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=8,
        window_size=256,
        tree_slots=8,
        max_frontier_levels=32,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        pool_temperature=1.0,
        descent_budget=64,
        hidden_act="silu",
        attention_dropout=0.0,
        initializer_range=0.02,
        use_cache=True,
        tie_word_embeddings=True,
        use_xformers=True,
        use_liger_kernel=True,
        fp32_residual=True,
        mlp_down_scale=64.0,
        mlp_down_scale_layers=3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        # --- FAST ---
        mtp_horizons=2,
        mtp_position_fraction=0.5,
        latent_dim=1024,
        latent_position_fraction=0.25,
        latent_ema_decay=0.99,
        thought_fraction=0.05,
        thought_steps=3,
        thought_dim=512,
        select_drop_top=0.15,
        select_keep_fraction=0.60,
        score_chunk_size=256,
        advantage_clamp=2.0,
        w_mtp=0.3,
        w_latent=0.3,
        w_gate=0.05,
        # Must exceed w_latent: the variance hinge reaches 1.0 at total collapse, and
        # collapsing is worth at most w_latent to the cosine term, so w_variance <= w_latent
        # makes collapse a net win. See HSAForCausalLM._latent_loss.
        w_variance=0.5,
        sentence_end_token_ids=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads

        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.window_size = window_size
        self.tree_slots = tree_slots
        self.max_frontier_levels = max_frontier_levels
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.pool_temperature = pool_temperature
        self.descent_budget = descent_budget
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache 
        # Optional fused-kernel acceleration -- both require the respective
        # package to be installed; if requested but unavailable, the model
        # falls back to the plain PyTorch implementation with a warning
        # rather than failing to construct. See modeling_hsa.py docstring
        # for exactly which pieces each flag swaps and which it doesn't
        # (e.g. no LayerNorm/GeGLU/hyper-connections exist in this
        # architecture, so the corresponding Liger kernels don't apply).
        self.use_xformers = use_xformers
        self.use_liger_kernel = use_liger_kernel

        # Carry the residual stream between decoder layers in fp32 while the sublayers keep
        # computing in fp16 under autocast. NOT a precision nicety -- fp16 cannot hold this
        # model's residual stream at depth, and that is what killed every long run so far.
        # Measured at checkpoint-00000749, max|h| leaving each layer:
        #
        #     layer  8   1,740     layer 10  15,632  (23.9% of fp16's 65504)
        #     layer  9   4,152     layer 11  38,400  (58.6%)
        #
        # i.e. 1.71x headroom, 86 steps before that run went non-finite. The injector is the
        # MLP -- down_proj outputs reach 11,936 and 21,648 in layers 10/11 while attention
        # stays under 1,500 -- and the overflow lands on `residual + h` in
        # HSADecoderLayer.forward, the one piece of arithmetic there that is not an nn.Module
        # (which is why the per-module NaN diagnostic could only name the layer, never a
        # child: both operands were finite, their sum was not).
        #
        # Safe downstream because HSAModel.forward ends in self.norm(), so magnitudes are
        # back to O(1) before lm_head and _fast_losses ever see them. bf16 would be the
        # natural fix -- same exponent range as fp32 -- but these are V100s (Volta, no bf16).
        self.fp32_residual = fp32_residual

        # Second fp16 ceiling, one level below the residual add that fp32_residual fixed.
        # down_proj's own OUTPUT overflows before it ever reaches the residual: measured on
        # checkpoint-00000799, layer 11's per-channel max was 36,960 and climbing
        # (22,176 -> 25,280 -> 28,816 -> 36,960 over steps 199/399/599/799 -- accelerating,
        # +3.1k then +3.5k then +8.1k), which extrapolates through 65,504 at ~step 900. The
        # run went non-finite at 908, origin `model.layers.11.mlp.down_proj`.
        #
        # It is an outlier-feature problem, not a broad one: 9 of 2560 output channels
        # exceed 1e4 while the median channel sits at 1,983. So rather than pay for an fp32
        # matmul, scale down_proj's INPUT by a power of two and restore the scale in fp32
        # after. Powers of two only shift the exponent, so the scaling itself is exact -- the
        # only cost is the matmul's own rounding, which was there anyway.
        #
        # LIMITED TO THE DEEPEST LAYERS ON PURPOSE. The scale is safe where activations are
        # large and harmful where they are small, and this model's are wildly depth-dependent:
        # median |x| into down_proj is 0.016 at layer 1 but 17.05 at layer 11. Dividing by 64
        # pushes 32-34% of layers 1-3's activations subnormal, for no benefit at all (their
        # outputs peak near 500). At layers 10-11 the same divisor costs 2.9% and 0.33%.
        self.mlp_down_scale = float(mlp_down_scale)
        self.mlp_down_scale_layers = int(mlp_down_scale_layers)

        # --- FAST ---
        if mtp_horizons < 1:
            raise ValueError("mtp_horizons must be >= 1; FAST has no off switch")
        if thought_steps < 1:
            raise ValueError("thought_steps must be >= 1; FAST has no off switch")
        if not 0.0 < select_keep_fraction <= 1.0:
            raise ValueError("select_keep_fraction must be in (0, 1]")
        if not 0.0 <= select_drop_top < 1.0:
            raise ValueError("select_drop_top must be in [0, 1)")
        if select_drop_top + select_keep_fraction > 1.0:
            raise ValueError("select_drop_top + select_keep_fraction must be <= 1")

        self.mtp_horizons = mtp_horizons
        self.mtp_position_fraction = mtp_position_fraction
        self.latent_dim = latent_dim
        self.latent_position_fraction = latent_position_fraction
        self.latent_ema_decay = latent_ema_decay
        self.thought_fraction = thought_fraction
        self.thought_steps = thought_steps
        self.thought_dim = thought_dim
        self.select_drop_top = select_drop_top
        self.select_keep_fraction = select_keep_fraction
        self.score_chunk_size = score_chunk_size
        self.advantage_clamp = advantage_clamp
        self.w_mtp = w_mtp
        self.w_latent = w_latent
        self.w_gate = w_gate
        self.w_variance = w_variance
        # A plain list so config.json round-trips; materialized into a bool
        # lookup buffer at model construction. Built by build_sentence_end_ids().
        self.sentence_end_token_ids = list(sentence_end_token_ids or [])

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    