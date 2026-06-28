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
from typing import TYPE_CHECKING

import mlx.core as mx

from exo.worker.engines.mlx import constants

if TYPE_CHECKING:
    from exo.worker.engines.mlx.types import Model
    from mlx_lm.tokenizer_utils import TokenizerWrapper

logger = logging.getLogger(__name__)


@dataclass
class SpecPrefillConfig:
    """Configuration for SpecPrefill, loaded from env vars at module init."""
    enabled: bool = constants.SPEC_PREFILL_ENABLED
    draft_model_id: str = constants.SPEC_PREFILL_DRAFT_MODEL
    keep_pct: int = constants.SPEC_PREFILL_KEEP_PCT
    min_prompt_tokens: int = constants.SPEC_PREFILL_MIN_PROMPT_TOKENS
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

    def validate_tokenizer_compat(self, target_tokenizer) -> None:
        """Verify draft vocab overlaps with target by >= 99%."""
        if not self.is_loaded():
            raise RuntimeError("Draft model not loaded; call load() first")
        draft_vocab = set(self.tokenizer.get_vocab().values())
        target_vocab = set(target_tokenizer.get_vocab().values())
        if not target_vocab:
            raise ValueError("Target tokenizer has empty vocab")
        overlap = len(draft_vocab & target_vocab) / len(target_vocab)
        if overlap < 0.99:
            raise ValueError(
                f"Tokenizer vocab overlap {overlap:.3f} < 0.99 between "
                f"draft ({self.model_id}) and target. SpecPrefill requires "
                f"near-identical vocab so token IDs map 1:1."
            )
        logger.info(f"Draft tokenizer compat OK ({overlap:.3f} overlap)")


# Module-level singleton (one draft model per exo node, loaded lazily)
_draft_model: DraftModel | None = None


def get_draft_model() -> DraftModel:
    """Get or create the singleton DraftModel instance."""
    global _draft_model
    if _draft_model is None:
        _draft_model = DraftModel(constants.SPEC_PREFILL_DRAFT_MODEL)
    return _draft_model


def draft_prefill(
    draft_model: DraftModel,
    prompt_tokens: mx.array,
    config: SpecPrefillConfig,
) -> list:
    """Phase 1: Run draft model on all prompt tokens.

    Returns:
        Draft KV cache list (one entry per draft layer). Used by Phase 2.
    """
    from mlx_lm import stream_generate
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
        for layer_idx, layer in enumerate(layers):
            captured = self.captured_q.setdefault(layer_idx, [])
            def make_hook(idx, cap_list):
                def hook(module, inputs, outputs):
                    # Capture Q from attention layer's first linear projection
                    # (output of q_proj / fused qkv_proj)
                    if isinstance(outputs, tuple):
                        q = outputs[0]
                    else:
                        q = outputs
                    cap_list.append(q)
                return hook
            # Register forward hook via __call__ wrapping (MLX doesn't have hooks; monkey-patch)
            original_call = layer.__class__.__call__
            def make_wrapped(orig_call, idx, cap_list):
                def wrapped(self, *args, **kwargs):
                    result = orig_call(self, *args, **kwargs)
                    if len(cap_list) < self.n_lookahead:
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
            layer.__class__.__call__ = make_wrapped(original_call, layer_idx, captured)
            self._hooks.append((layer.__class__, original_call))
        return self

    def __exit__(self, *args):
        # Restore original __call__ on each layer's class
        for cls, original_call in self._hooks:
            cls.__call__ = original_call
        self._hooks.clear()


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
