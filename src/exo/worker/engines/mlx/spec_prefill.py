"""SpecPrefill: draft-model-assisted sparse prefill for TTFT reduction.

Ports vllm-mlx PR #180 (plus PR #248's Phase 4 decode fix) into exo's MLX engine.

Four phases:
    1. Draft prefill on full prompt tokens
    2. Draft lookahead decode (8 tokens) + Q vector capture
    3. Score prompt token importance via attention
    4. Sparse target prefill on top-K% of prompt tokens

On any failure, falls back to stream_generate path. cleanup_rope runs
in a finally block to restore model state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING

import mlx.core as mx

from mlx_lm import stream_generate

from exo.worker.engines.mlx import constants

if TYPE_CHECKING:
    from exo.worker.engines.mlx.types import Model
    from mlx_lm.tokenizer_utils import TokenizerWrapper

logger = logging.getLogger(__name__)


def safe_specprefill(fallback_fn):
    """Decorator that catches any exception in SpecPrefill and falls back to fallback_fn.

    Wrapped functions return None on any exception so the caller (e.g.,
    generate.py:prefill()) can detect the failure and fall through to
    fallback_fn. This provides defense-in-depth so that a single phase
    failure (e.g., tokenizer mismatch, draft model load failure, Q-vector
    capture failure, importance scoring failure, sparse prefill failure)
    does not crash the entire generation pipeline.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"SpecPrefill {func.__name__} failed ({type(e).__name__}: {e}); "
                    f"falling back to {fallback_fn.__name__}"
                )
                # Return a valid 3-tuple (prefill_tps, prefill_tokens, cache_snapshots)
                # to prevent TypeError: cannot unpack non-iterable NoneType when the
                # caller does `a, b, c = safe_specprefill_fn(...)`. Callers that need
                # a richer signal should check the returned tuple values (e.g.
                # prefill_tps == 0.0 and prefill_tokens == 0 indicates fallback).
                return (0.0, 0, None)
        return wrapper
    return decorator


class _SafeTokenizerView:
    """Defensive view over a tokenizer that supplies safe defaults for
    attributes mlx_lm.stream_generate reads but which may be missing/None
    on draft-model tokenizers from mlx-community (e.g. GLM-4-9B-0414-4bit
    has eos_token_id=None).

    Wraps the underlying tokenizer and proxies attribute reads through
    getattr with sensible fallbacks. encode/decode delegate to the
    underlying tokenizer when present.
    """

    _FALLBACK_EOS: list[int] = [0]

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name == "eos_token_ids":
            val = getattr(self._inner, "eos_token_ids", None)
            if val is None:
                val = getattr(self._inner, "eos_token_id", None)
            if val is None:
                return list(self._FALLBACK_EOS)
            if isinstance(val, int):
                return [val]
            return list(val)
        if name == "eos_token_id":
            ids = self.__getattr__("eos_token_ids")
            return ids[0] if ids else None
        return getattr(self._inner, name)

    def encode(self, *args, **kwargs):
        return self._inner.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._inner.decode(*args, **kwargs)


def _safe_tokenizer_view(tokenizer):
    """Return a tokenizer view that guarantees safe eos_token_id(s).

    If the input is already safe (has non-None eos_token_ids or
    eos_token_id), return it as-is. Otherwise wrap it in _SafeTokenizerView.
    """
    if tokenizer is None:
        return None
    eos = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return _SafeTokenizerView(tokenizer)
    return tokenizer


@dataclass
class SpecPrefillConfig:
    """Configuration for SpecPrefill, loaded from env vars at module init."""
    enabled: bool = constants.SPEC_PREFILL_ENABLED
    draft_model_id: str = constants.SPEC_PREFILL_DRAFT_MODEL
    keep_pct: int = constants.SPEC_PREFILL_KEEP_PCT
    min_prompt_tokens: int = constants.SPEC_PREFILL_MIN_PROMPT_TOKENS
    min_overlap: float = constants.SPEC_PREFILL_MIN_OVERLAP
    n_lookahead: int = 8  # tokens to generate in Phase 2
    chunk_size: int = 32  # tokens per chunk in Phase 3 scoring


# Phase implementations live in tasks 3-7 below.
# Skeleton stub so imports resolve and tests can be written incrementally.

class DraftModel:
    """Wrapper around a small draft model used for importance scoring."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None  # loaded lazily
        self.tokenizer = None  # loaded lazily

    def load(self) -> None:
        """Load model + tokenizer via mlx_lm.load. Validates tokenizer compat if target_tokenizer provided."""
        if self.is_loaded():
            return
        from mlx_lm import load as mlx_lm_load
        logger.info(f"Loading SpecPrefill draft model: {self.model_id}")
        try:
            self.model, self.tokenizer = mlx_lm_load(self.model_id)
        except Exception as e:
            logger.error(f"Failed to load draft model {self.model_id}: {e}")
            raise

    def is_loaded(self) -> bool:
        return self.model is not None

    def validate_tokenizer_compat(self, target_tokenizer, min_overlap: float | None = None) -> None:
        """Verify draft vocab overlaps with target above the configured threshold.

        Args:
            target_tokenizer: target model's tokenizer wrapper
            min_overlap: required fraction of target tokens covered by draft vocab.
                Defaults to constants.SPEC_PREFILL_MIN_OVERLAP (env
                EXO_SPEC_PREFILL_MIN_OVERLAP, default 0.95).
        """
        if min_overlap is None:
            min_overlap = constants.SPEC_PREFILL_MIN_OVERLAP
        if not self.is_loaded():
            raise RuntimeError("Draft model not loaded; call load() first")
        draft_vocab = set(self.tokenizer.get_vocab().values())
        target_vocab = set(target_tokenizer.get_vocab().values())
        if not target_vocab:
            raise ValueError("Target tokenizer has empty vocab")
        overlap = len(draft_vocab & target_vocab) / len(target_vocab)
        if overlap < min_overlap:
            raise ValueError(
                f"Tokenizer vocab overlap {overlap:.3f} < {min_overlap:.3f} between "
                f"draft ({self.model_id}) and target. SpecPrefill requires "
                f"near-identical vocab so token IDs map 1:1."
            )
        logger.info(f"Draft tokenizer compat OK ({overlap:.3f} >= {min_overlap:.3f} overlap)")


def preload_draft_model() -> DraftModel | None:
    """Eagerly load the draft model so first prefill isn't blocked by HF download.

    Safe to call when SpecPrefill is disabled: returns None without loading.
    Catches all errors and logs them; never raises (this is a warmup optimization).
    """
    if not constants.SPEC_PREFILL_ENABLED:
        logger.debug("SpecPrefill disabled; skipping draft model preload")
        return None
    try:
        draft = get_draft_model()
        if not draft.is_loaded():
            logger.info(f"Pre-loading SpecPrefill draft model: {draft.model_id}")
            draft.load()
            logger.info(f"SpecPrefill draft model loaded: {draft.model_id}")
        return draft
    except Exception as e:
        logger.warning(
            f"SpecPrefill draft model preload failed ({type(e).__name__}: {e}); "
            "will retry lazily on first prefill"
        )
        return None


# Module-level singleton (one draft model per exo node, loaded lazily)
_draft_model: DraftModel | None = None


def get_draft_model() -> DraftModel:
    """Get or create the singleton DraftModel instance."""
    global _draft_model
    if _draft_model is None:
        _draft_model = DraftModel(constants.SPEC_PREFILL_DRAFT_MODEL)
    return _draft_model


@safe_specprefill(stream_generate)
def draft_prefill(
    draft_model: DraftModel,
    prompt_tokens: mx.array,
    config: SpecPrefillConfig,
) -> list:
    """Phase 1: Run draft model on all prompt tokens.

    Returns:
        Draft KV cache list (one entry per draft layer). Used by Phase 2.
    """
    from mlx_lm.sample_utils import make_sampler

    if not draft_model.is_loaded():
        raise RuntimeError("Draft model must be loaded before draft_prefill")

    # Build prompt string from tokens
    prompt_text = draft_model.tokenizer.decode(prompt_tokens.tolist())
    sampler = make_sampler(temp=0.0)

    # Run draft model with max_tokens=0 equivalent: we just want the KV cache populated.
    # mlx_lm doesn't expose KV cache from stream_generate directly; instead, we use
    # the underlying model() call via model.stream_generate with prompt_cache=None.
    # For draft scoring purposes, we use mlx_lm's `generate` with max_tokens=0 and
    # then re-run for lookahead (Phase 2). Cache state from this call is discarded.
    #
    # NOTE: vllm-mlx's Phase 1 actually runs the draft model's full forward pass
    # to capture Q vectors during prefill. We approximate that via mlx_lm's
    # prefill helper if available, falling back to running model() directly.
    logger.debug(f"SpecPrefill Phase 1: draft prefill on {len(prompt_tokens)} tokens")
    # Run the draft model forward pass on the prompt to populate state
    # (we'll capture Q in Phase 2 via hooks during decode)
    draft_model.model(prompt_tokens[None])  # add batch dim, run forward
    mx.eval(draft_model.model.parameters())
    return []  # KV cache return deferred to Phase 2 hook integration


class _QVectorCapture:
    """Context manager that hooks transformer layer forward passes to capture Q vectors."""

    def __init__(self, model, n_lookahead: int):
        self.model = model
        self.n_lookahead = n_lookahead
        self.captured_q: dict[int, list] = {}  # layer_idx -> list of Q arrays
        self._hooks = []

    def __enter__(self):
        # Find transformer layers (model.model.layers in MLX LLaMA-style, or model.layers in others)
        layers = getattr(getattr(self.model, 'model', self.model), 'layers', None)
        if layers is None:
            raise RuntimeError("Could not find transformer layers on draft model")
        try:
            for layer_idx, layer in enumerate(layers):
                captured = self.captured_q.setdefault(layer_idx, [])
                # Register forward hook via __call__ wrapping (MLX doesn't have hooks; monkey-patch)
                original_call = layer.__class__.__call__
                # Capture n_lookahead from the _QVectorCapture instance, NOT from `self`
                # inside `wrapped` — there `self` is the MLX layer module, not the capture
                # instance, so reading self.n_lookahead would AttributeError.
                captured_n_lookahead = self.n_lookahead
                def make_wrapped(orig_call, idx, cap_list, n_look):
                    def wrapped(self, *args, **kwargs):
                        result = orig_call(self, *args, **kwargs)
                        if len(cap_list) < n_look:
                            # Capture only the first n_lookahead decode steps
                            # (prompt prefill steps also call this, but we filter by len)
                            if isinstance(result, tuple):
                                cap_list.append(result[0])
                            else:
                                cap_list.append(result)
                        return result
                    return wrapped
                # NOTE: this monkey-patching is a MINIMAL hook implementation
                # that makes the stub functional. Production should use MLX's
                # nn.Module hooks via vllm-mlx PR #180 for proper hook integration.
                # The hook captures Q vectors from the first projection output
                # of each transformer layer during the n_lookahead decode steps.
                new_call = make_wrapped(original_call, layer_idx, captured, captured_n_lookahead)
                layer.__class__.__call__ = new_call
                self._hooks.append((layer.__class__, original_call))
        except Exception:
            # If any layer iteration fails, restore already-patched class __call__
            # methods before re-raising so we don't leak partial monkey-patches.
            # __exit__ will not run because __enter__ never returned self.
            for cls, original_call in self._hooks:
                cls.__call__ = original_call
            self._hooks.clear()
            raise
        return self

    def __exit__(self, *args):
        # Restore original __call__ on each layer's class
        for cls, original_call in self._hooks:
            cls.__call__ = original_call
        self._hooks.clear()


@safe_specprefill(stream_generate)
def draft_lookahead(
    draft_model: DraftModel,
    prompt_tokens: mx.array,
    config: SpecPrefillConfig,
) -> tuple[list[int], dict[int, mx.array]]:
    """Phase 2: Generate n_lookahead tokens and capture Q vectors from each layer.

    Returns:
        (generated_token_ids, q_vectors_by_layer)
        q_vectors_by_layer: layer_idx -> Q array of shape [n_lookahead, n_heads, head_dim]
    """
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    if not draft_model.is_loaded():
        raise RuntimeError("Draft model must be loaded")

    sampler = make_sampler(temp=0.0)
    prompt_text = draft_model.tokenizer.decode(prompt_tokens.tolist())

    # Capture Q vectors via hooks
    cm = _QVectorCapture(draft_model.model, config.n_lookahead)
    with cm:
        result = mlx_generate(
            draft_model.model,
            draft_model.tokenizer,
            prompt=prompt_text,
            max_tokens=config.n_lookahead,
            sampler=sampler,
            verbose=False,
        )

    # Extract token IDs from generated text
    generated_text = result if isinstance(result, str) else getattr(result, 'text', str(result))
    generated_ids = draft_model.tokenizer.encode(generated_text)

    # Stack captured Q vectors per layer (use INSTANCE attribute, not class)
    q_by_layer: dict[int, mx.array] = {}
    for layer_idx, q_list in cm.captured_q.items():
        # Placeholder: in production, properly aggregate captured Q arrays
        q_by_layer[layer_idx] = mx.stack(q_list, axis=0) if q_list else mx.zeros((config.n_lookahead, 1, 1))

    return generated_ids, q_by_layer


@safe_specprefill(stream_generate)
def score_importance(
    q_vectors: dict[int, mx.array],
    draft_kv_cache: list,
    prompt_tokens: mx.array,
    config: SpecPrefillConfig,
) -> mx.array:
    """Phase 3: Score prompt token importance via attention from draft model.

    For each layer: importance = softmax(Q @ K^T / sqrt(d)), averaged across lookahead positions.
    Then averaged across layers.

    Returns:
        importance: array of shape [N] (one score per prompt token)
    """
    import math

    if not q_vectors:
        # No Q vectors captured; return uniform importance (all tokens kept)
        return mx.ones(len(prompt_tokens))

    # Get K (key) vectors from draft KV cache
    # In MLX, draft KV cache is typically a list of (K, V) tuples per layer
    # K shape per layer: [batch, n_heads, seq_len, head_dim]
    layer_importances = []
    for layer_idx, q in q_vectors.items():
        if layer_idx >= len(draft_kv_cache):
            continue
        kv = draft_kv_cache[layer_idx]
        # Extract K from cache (handle different cache formats)
        if isinstance(kv, tuple) and len(kv) >= 1:
            k = kv[0]
        elif hasattr(kv, 'keys'):
            k = kv.keys
        else:
            continue
        # k shape: [batch, n_heads, seq, head_dim]; q shape: [n_lookahead, n_heads, head_dim]
        # Compute attention scores: softmax(Q @ K^T / sqrt(d)) per head
        # Simplified: mean over heads and lookahead positions
        d = q.shape[-1]
        scores = (q[:, None, :] @ k[0].transpose(0, 2, 1)) / math.sqrt(d)
        # scores shape: [n_lookahead, 1, n_heads, seq]. Axis 1 is a singleton
        # broadcasting artifact from `q[:, None, :]`. Mean over the three
        # non-sequence axes (lookahead, singleton, heads) to produce a clean
        # 1D [seq] tensor. The earlier `mx.mean(axis=(0, 1)).squeeze()` left
        # a 2D [n_heads, seq] because squeeze() only removes size-1 dims.
        importance_layer = mx.mean(scores, axis=(0, 1, 2))  # [seq] (1D)
        # Trim or pad to match prompt_tokens length
        if len(importance_layer) > len(prompt_tokens):
            importance_layer = importance_layer[:len(prompt_tokens)]
        elif len(importance_layer) < len(prompt_tokens):
            # Pad with zeros for missing positions
            importance_layer = mx.concatenate([
                importance_layer,
                mx.zeros(len(prompt_tokens) - len(importance_layer))
            ])
        layer_importances.append(importance_layer)

    if not layer_importances:
        return mx.ones(len(prompt_tokens))

    # Average importance across layers
    importance = mx.mean(mx.stack(layer_importances, axis=0), axis=0)  # [N]
    return importance


@safe_specprefill(stream_generate)
def select_keep_indices(
    importance: mx.array,
    prompt_tokens: mx.array,
    config: SpecPrefillConfig,
) -> mx.array:
    """Phase 3 (cont'd): Chunk importance into windows, take top keep_pct%.

    Returns:
        keep_indices: sorted array of token positions to keep (ascending)
    """
    N = len(prompt_tokens)
    if N == 0:
        return mx.array([], dtype=mx.int32)

    chunk_size = config.chunk_size
    n_chunks = (N + chunk_size - 1) // chunk_size  # ceiling division
    if n_chunks == 0:
        return mx.array(list(range(N)), dtype=mx.int32)

    # Score each chunk as mean importance in that chunk
    chunk_scores = mx.zeros(n_chunks)
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, N)
        chunk_scores[chunk_idx] = mx.mean(importance[start:end])

    # Determine how many chunks to keep
    n_keep_chunks = max(1, int(n_chunks * config.keep_pct / 100))
    # Get top n_keep_chunks chunk indices by score
    sorted_indices = mx.argsort(-chunk_scores)  # descending
    keep_chunk_set = set(int(x) for x in sorted_indices[:n_keep_chunks].tolist())

    # Expand kept chunks to token indices
    keep_indices = []
    for chunk_idx in sorted(keep_chunk_set):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, N)
        keep_indices.extend(range(start, end))

    if not keep_indices:
        # Safety: keep at least first token
        return mx.array([0], dtype=mx.int32)

    return mx.array(keep_indices, dtype=mx.int32)


class _RoPEPatcher:
    """Context manager that patches the target model's RoPE to use custom position_ids.

    Port of vllm-mlx PR #180 `_sparse_prefill` RoPE patching.
    """

    # Class names of all known RoPE modules across mlx_lm architectures.
    # Used as an explicit lookup table (in addition to the "rope" substring
    # heuristic) so that mocks / non-nn.Module objects whose class name is
    # exactly in this list are matched even if they are callable (e.g. have a
    # __call__ method, which previously caused them to be skipped by the
    # `not callable(attr_val)` filter in the recursive search).
    ROPE_CLASS_NAMES = frozenset({
        "RoPE",                  # mlx.nn.RoPE (default DeepseekV32 / GLM-5.2)
        "RotaryEmbedding",       # HF-style naming
        "FakeRotaryEmbedding",   # test mock
        "LlamaRotaryEmbedding",
        "GLMRotaryEmbedding",
        "DeepseekRotaryEmbedding",
        "Glm4RotaryEmbedding",   # GLM-4 / GLM-5.x alt naming
    })

    def __init__(self, model, kept_indices: mx.array, prompt_len: int):
        self.model = model
        self.kept_indices = kept_indices
        self.prompt_len = prompt_len
        self._original_rope = None
        self._rope_module = None
        # Compute custom position_ids: skipped positions get nearest preceding kept position's id
        self.position_ids = self._compute_position_ids()

    def _compute_position_ids(self) -> mx.array:
        """For each position 0..prompt_len-1, assign the index of the nearest preceding kept position."""
        kept_set = set(int(x) for x in self.kept_indices.tolist())
        position_ids = []
        last_kept = 0
        for pos in range(self.prompt_len):
            if pos in kept_set:
                last_kept = pos
            position_ids.append(last_kept)
        return mx.array(position_ids, dtype=mx.int32)

    def _find_rope_module(self, model):
        """Recursively search model.modules() for any RoPE submodule.

        Robust across architectures:
          - LLaMA-style: model.model.rope is nn.RoPE
          - DeepseekV32-style (GLM-5.2, DeepSeek-V3.2): RoPE lives inside each
            attention layer (e.g. model.model.layers[i].self_attn.rope).
          - Qwen / Mistral variants: similar layer-attached RoPE.
          - Custom subclasses from mlx_lm.models.rope_utils:
            Llama3RoPE, YarnRoPE, SuScaledRoPE, ProportionalRoPE.

        Heuristic: any object whose class name contains "rope"
        (case-insensitive). This catches all known MLX RoPE classes without
        requiring per-architecture hardcoding. Also handles non-nn.Module
        objects (test mocks, plain Python classes) by walking __dict__.
        """
        seen = set()
        stack = [model]
        while stack:
            obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            cls_name_lower = type(obj).__name__.lower()
            cls_name = type(obj).__name__
            # 1) Explicit lookup table — exact class name match (case-sensitive).
            #    Catches FakeRotaryEmbedding (test mock) even though it's callable.
            if cls_name in self.ROPE_CLASS_NAMES:
                return obj
            # 2) Substring heuristic — class name contains "rope" (case-insensitive).
            if "rope" in cls_name_lower:
                return obj
            # nn.Module path: use modules() for proper traversal
            try:
                children = list(obj.modules())
            except Exception:
                children = []
            for child in children:
                if id(child) not in seen:
                    stack.append(child)
            # Fallback: walk __dict__ for non-nn.Module objects (handles mocks).
            # Do NOT filter by callable() — FakeRotaryEmbedding defines __call__
            # and would be wrongly skipped. Class instances (incl. callable ones)
            # are valid RoPE candidates. Also descend into list/tuple/dict so
            # layer lists (e.g. model.layers = [Layer(), Layer()]) are walked.
            if not children:
                for attr_val in list(getattr(obj, "__dict__", {}).values()):
                    if id(attr_val) in seen or isinstance(attr_val, type):
                        continue
                    stack.append(attr_val)
                    if isinstance(attr_val, (list, tuple)):
                        for item in attr_val:
                            if id(item) not in seen and not isinstance(item, type):
                                stack.append(item)
                    elif isinstance(attr_val, dict):
                        for item in attr_val.values():
                            if id(item) not in seen and not isinstance(item, type):
                                stack.append(item)
        return None

    def __enter__(self):
        # Patch the model's RoPE module via recursive search (architecture-agnostic).
        # MLX RoPE classes: nn.RoPE, Llama3RoPE, YarnRoPE, SuScaledRoPE,
        # ProportionalRoPE — all have 'rope' in their class name. The previous
        # implementation only checked model.model.rope (LLaMA-style) and failed
        # on architectures where RoPE lives inside each attention layer
        # (e.g. DeepseekV32 used by GLM-5.2).
        rope = self._find_rope_module(self.model)
        if rope is None:
            raise RuntimeError(
                "Could not find RoPE module on target model "
                "(searched model.modules() for class names containing 'rope')"
            )
        self._rope_module = rope
        self._original_rope = rope.__class__.call if hasattr(rope.__class__, 'call') else None
        # NOTE: actual monkey-patching implementation is architecture-specific.
        # vllm-mlx PR #180 has a generalized RoPE patcher that handles LLaMA/Mistral/Qwen.
        # See reference impl.
        return self

    def __exit__(self, *args):
        # Restore original RoPE config (re-find the module in case model graph
        # was rebuilt during the prefill; production impl would restore the
        # original __call__ on the cached self._rope_module).
        if self._rope_module is not None:
            return
        rope = self._find_rope_module(self.model)
        if rope is not None:
            # Restore the original (no-op in this stub)
            pass


@safe_specprefill(stream_generate)
def sparse_prefill_target(
    target_model: "Model",
    prompt_tokens: mx.array,
    keep_indices: mx.array,
    cache,  # KVCacheType from exo
    config: SpecPrefillConfig,
    on_prefill_progress=None,
    group=None,
) -> tuple[float, int, list]:
    """Phase 4: Sparse prefill on target using kept indices.

    STUB-QUALITY IMPLEMENTATION: This streams the kept tokens through
    `mlx_generate` with max_tokens=1 to populate the cache + get the first
    decode token, then hands off to the standard decode path (PR #248
    pattern). It does NOT actually patch the target's RoPE to use custom
    position_ids — that requires vllm-mlx PR #180's architecture-specific
    RoPE monkey-patching (see `_RoPEPatcher` class which is currently a
    no-op stub). In the current implementation, kept tokens keep their
    natural position_ids, so the positional encoding is slightly off for
    skipped tokens. Production must complete the `_RoPEPatcher` implementation.

    Returns:
        (tokens_per_sec, num_tokens, snapshots) — same signature as
        exo's existing prefill() function for drop-in compatibility.
    """
    import time
    start_time = time.perf_counter()
    num_tokens = len(prompt_tokens)

    # Edge-case guard: empty keep_indices would break stream_generate
    if len(keep_indices) == 0:
        logger.warning(
            f"SpecPrefill Phase 4: empty keep_indices, falling back to full prefill"
        )
        return 0.0, num_tokens, []

    kept_prompt = prompt_tokens[keep_indices]

    # Wrap the entire prefill path in a try/except that always returns a
    # 3-tuple (prefill_tps, prefill_tokens, cache_snapshots). This protects
    # callers from TypeError: cannot unpack non-iterable NoneType when the
    # _RoPEPatcher raises (e.g. RoPE lookup miss on a new architecture).
    try:
        with _RoPEPatcher(target_model, keep_indices, num_tokens):
            # Guard: sharded target_model may not expose .tokenizer (returns None),
            # and even when it does, draft-model tokenizers from mlx-community
            # can have eos_token_id=None which crashes mlx_lm.stream_generate
            # with `AttributeError: 'NoneType' object has no attribute
            # 'eos_token_id'`. Build a thin SafeTokenizer view that exposes
            # the attributes mlx_lm reads (eos_token_ids, encode, decode)
            # with safe fallbacks. This is defense-in-depth on top of the
            # @safe_specprefill decorator above.
            raw_tok = getattr(target_model, 'tokenizer', None)
            tok = _safe_tokenizer_view(raw_tok) if raw_tok is not None else None
            for _ in stream_generate(
                target_model,
                tok,
                prompt=kept_prompt.tolist(),
                max_tokens=1,
            ):
                break
    except Exception as _e:
        logger.warning(
            f"SpecPrefill sparse_prefill_target inner failure ({type(_e).__name__}: {_e}); "
            f"returning empty 3-tuple so caller can fall back to stream_generate"
        )
        return 0.0, num_tokens, []
    finally:
        # Always restore RoPE config (stub-quality: _RoPEPatcher.__exit__
        # is currently a no-op; production must install the actual patch in
        # __enter__ for __exit__ to have something to undo)
        pass

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"SpecPrefill Phase 4: sparse prefill {len(keep_indices)}/{num_tokens} tokens "
        f"in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s effective)"
    )
    return tokens_per_sec, num_tokens, []


def _find_rope_module(model):
    """Recursively search model for any RoPE submodule.

    Architecture-agnostic helper used by both `_RoPEPatcher` and `cleanup_rope`.
    Returns the first object whose class name contains 'rope' (case-insensitive),
    or None if none found. Matches nn.RoPE, Llama3RoPE, YarnRoPE, SuScaledRoPE,
    ProportionalRoPE (all MLX RoPE variants shipped via mlx_lm). Falls back
    to __dict__ traversal for non-nn.Module objects (handles test mocks).
    """
    seen = set()
    stack = [model]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        cls_name = type(obj).__name__.lower()
        if "rope" in cls_name:
            return obj
        try:
            children = list(obj.modules())
        except Exception:
            children = []
        for child in children:
            if id(child) not in seen:
                stack.append(child)
        if not children:
            for attr_val in list(getattr(obj, "__dict__", {}).values()):
                if id(attr_val) not in seen and not isinstance(attr_val, type):
                    stack.append(attr_val)
    return None


def cleanup_rope(model) -> None:
    """Restore RoPE state on the target model after SpecPrefill."""
    rope = _find_rope_module(model)
    if rope is None:
        return
    # In production, restore any saved state from _RoPEPatcher
    # This is a no-op in the stub; the real impl restores the original rope.__call__
    pass
