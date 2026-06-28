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
        """Load model + tokenizer. Tokenizer compat validated vs target."""
        raise NotImplementedError  # implemented in Task 3

    def is_loaded(self) -> bool:
        return self.model is not None

    def validate_tokenizer_compat(self, target_tokenizer) -> None:
        """Verify draft vocab overlaps with target by >= 99%. Raise ValueError otherwise."""
        raise NotImplementedError  # implemented in Task 3


# Module-level singleton (one draft model per exo node, loaded lazily)
_draft_model: DraftModel | None = None


def get_draft_model() -> DraftModel:
    """Get or create the singleton DraftModel instance."""
    global _draft_model
    if _draft_model is None:
        _draft_model = DraftModel(constants.SPEC_PREFILL_DRAFT_MODEL)
    return _draft_model
