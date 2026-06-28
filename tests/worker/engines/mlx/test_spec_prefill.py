"""Tests for SpecPrefill."""

import pytest
from unittest.mock import MagicMock

from src.exo.worker.engines.mlx.spec_prefill import (
    DraftModel,
    SpecPrefillConfig,
    draft_prefill,
    get_draft_model,
)


def test_config_defaults():
    cfg = SpecPrefillConfig()
    assert cfg.enabled is False  # default off
    assert cfg.keep_pct == 20
    assert cfg.min_prompt_tokens == 4096
    assert cfg.n_lookahead == 8
    assert cfg.chunk_size == 32


def test_draft_model_singleton():
    """get_draft_model() returns the same instance on repeated calls."""
    # Reset singleton for test
    import src.exo.worker.engines.mlx.spec_prefill as sp
    sp._draft_model = None
    a = get_draft_model()
    b = get_draft_model()
    assert a is b


def test_validate_tokenizer_compat_high_overlap_passes():
    """Tokenizers with >=99% vocab overlap pass validation."""
    dm = DraftModel("fake-model")
    dm.model = MagicMock()
    dm.tokenizer = MagicMock()
    target_tok = MagicMock()
    # 1000 target tokens, 1000 draft tokens, 995 overlap
    target_vocab = {i for i in range(1000)}
    draft_vocab = target_vocab | {1000, 1001, 1002, 1003, 1004}  # 5 extra
    dm.tokenizer.get_vocab.return_value = {f"t{i}": i for i in draft_vocab}
    target_tok.get_vocab.return_value = {f"t{i}": i for i in target_vocab}
    # 995/1000 = 0.995 overlap
    dm.validate_tokenizer_compat(target_tok)  # should not raise


def test_validate_tokenizer_compat_low_overlap_raises():
    """Tokenizers with <99% vocab overlap raise ValueError."""
    dm = DraftModel("fake-model")
    dm.model = MagicMock()
    dm.tokenizer = MagicMock()
    target_tok = MagicMock()
    target_vocab = {i for i in range(1000)}
    # Only 500 of the target tokens are in draft vocab
    draft_vocab = {i for i in range(500, 1500)}  # 0-499 missing
    dm.tokenizer.get_vocab.return_value = {f"d{i}": i for i in draft_vocab}
    target_tok.get_vocab.return_value = {f"t{i}": i for i in target_vocab}
    with pytest.raises(ValueError, match="overlap"):
        dm.validate_tokenizer_compat(target_tok)


def test_draft_prefill_requires_loaded_model():
    """draft_prefill raises RuntimeError if draft model not loaded."""
    dm = DraftModel("fake")
    cfg = SpecPrefillConfig()
    import mlx.core as mx
    with pytest.raises(RuntimeError, match="must be loaded"):
        draft_prefill(dm, mx.array([1, 2, 3]), cfg)
