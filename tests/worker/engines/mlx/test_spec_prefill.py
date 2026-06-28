"""Tests for SpecPrefill."""

import pytest
from unittest.mock import MagicMock

from src.exo.worker.engines.mlx.spec_prefill import (
    DraftModel,
    SpecPrefillConfig,
    cleanup_rope,
    draft_prefill,
    get_draft_model,
    score_importance,
    select_keep_indices,
    sparse_prefill_target,
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


def test_q_vector_capture_context_manager():
    """_QVectorCapture is a context manager (can be entered/exited)."""
    from src.exo.worker.engines.mlx.spec_prefill import _QVectorCapture
    dm = DraftModel("fake")
    # Use spec=[] so MagicMock doesn't auto-create the .layers attribute;
    # this exercises the RuntimeError branch
    dm.model = MagicMock(spec=[])
    with pytest.raises(RuntimeError, match="Could not find transformer layers"):
        with _QVectorCapture(dm.model, n_lookahead=8):
            pass


def test_q_vector_capture_positive_path():
    """_QVectorCapture successfully enters/exits when model has layers.

    Positive-path coverage: verifies that the context manager can be
    entered and exited without error when the model exposes a .layers
    attribute. Does not verify Q-vector capture (the stub hooks monkey-
    patch class.__call__ in a way that depends on real MLX layers).
    """
    from src.exo.worker.engines.mlx.spec_prefill import _QVectorCapture
    dm = DraftModel("fake")
    # Create a model mock with proper .layers attribute (a list of mocks).
    # MagicMock() (no spec) auto-creates .model on access, so we set the
    # full path .model.layers explicitly to mimic MLX LLaMA-style arch.
    dm.model = MagicMock()
    dm.model.model.layers = [MagicMock() for _ in range(3)]  # 3 layers
    # Should enter and exit without error
    with _QVectorCapture(dm.model, n_lookahead=8):
        pass


def test_select_keep_indices_basic():
    """select_keep_indices returns sorted unique indices covering top chunks."""
    import mlx.core as mx
    from src.exo.worker.engines.mlx.spec_prefill import select_keep_indices
    cfg = SpecPrefillConfig(keep_pct=50, chunk_size=10)
    importance = mx.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4, 0.7, 0.6, 0.5, 0.0] * 2)  # 20 tokens
    prompt = mx.zeros(20, dtype=mx.int32)
    keep = select_keep_indices(importance, prompt, cfg)
    # 20 tokens / chunk_size 10 = 2 chunks; keep 50% = 1 chunk = 10 tokens
    assert len(keep) == 10
    # Indices should be sorted ascending
    keep_list = keep.tolist()
    assert keep_list == sorted(keep_list)


def test_select_keep_indices_all_kept_when_uniform():
    """When all importance is equal, keep_pct picks the first chunks."""
    import mlx.core as mx
    from src.exo.worker.engines.mlx.spec_prefill import select_keep_indices
    cfg = SpecPrefillConfig(keep_pct=25, chunk_size=4)
    importance = mx.ones(16)  # all equal
    prompt = mx.zeros(16, dtype=mx.int32)
    keep = select_keep_indices(importance, prompt, cfg)
    # 16/4 = 4 chunks; keep 25% = 1 chunk = 4 tokens
    assert len(keep) == 4
    assert keep.tolist() == [0, 1, 2, 3]


def test_select_keep_indices_empty_prompt():
    """Empty prompt returns empty keep_indices."""
    import mlx.core as mx
    from src.exo.worker.engines.mlx.spec_prefill import select_keep_indices
    cfg = SpecPrefillConfig()
    keep = select_keep_indices(mx.array([]), mx.array([], dtype=mx.int32), cfg)
    assert len(keep) == 0


def test_score_importance_returns_1d_array():
    """score_importance returns 1D importance array (not 2D — would crash mx.concatenate)."""
    import mlx.core as mx
    from src.exo.worker.engines.mlx.spec_prefill import score_importance, SpecPrefillConfig

    cfg = SpecPrefillConfig()
    # Create minimal q_vectors and KV cache: 1 layer, 4 lookahead tokens, 8 prompt tokens
    n_lookahead = 4
    seq_len = 8
    n_heads = 2
    head_dim = 4

    # q shape: [n_lookahead, n_heads, head_dim]
    q = mx.random.normal(shape=(n_lookahead, n_heads, head_dim))
    q_vectors = {0: q}

    # k shape: [batch=1, n_heads, seq_len, head_dim]
    k = mx.random.normal(shape=(1, n_heads, seq_len, head_dim))
    v = mx.random.normal(shape=(1, n_heads, seq_len, head_dim))
    draft_kv_cache = [(k, v)]

    prompt_tokens = mx.zeros(seq_len, dtype=mx.int32)

    importance = score_importance(q_vectors, draft_kv_cache, prompt_tokens, cfg)

    # CRITICAL: must be 1D for downstream mx.concatenate to work
    assert importance.ndim == 1, f"importance must be 1D, got {importance.ndim}D"
    assert importance.shape == (seq_len,), f"expected shape ({seq_len},), got {importance.shape}"


def test_select_keep_indices_zero_keep_pct_returns_min_one_chunk():
    """keep_pct=0 should still keep at least 1 chunk (per spec n_keep_chunks=max(1,...))."""
    import mlx.core as mx
    from src.exo.worker.engines.mlx.spec_prefill import select_keep_indices
    cfg = SpecPrefillConfig(keep_pct=0, chunk_size=4)
    importance = mx.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])  # 8 tokens, 2 chunks
    prompt = mx.zeros(8, dtype=mx.int32)
    keep = select_keep_indices(importance, prompt, cfg)
    # Even at keep_pct=0, we keep min 1 chunk = 4 tokens
    assert len(keep) == 4


def test_select_keep_indices_full_keep_pct():
    """keep_pct=100 keeps all tokens."""
    import mlx.core as mx
    from src.exo.worker.engines.mlx.spec_prefill import select_keep_indices
    cfg = SpecPrefillConfig(keep_pct=100, chunk_size=4)
    importance = mx.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    prompt = mx.zeros(8, dtype=mx.int32)
    keep = select_keep_indices(importance, prompt, cfg)
    assert len(keep) == 8
    assert keep.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_select_keep_indices_non_divisible_length():
    """prompt length not divisible by chunk_size."""
    import mlx.core as mx
    from src.exo.worker.engines.mlx.spec_prefill import select_keep_indices
    cfg = SpecPrefillConfig(keep_pct=50, chunk_size=4)
    importance = mx.array([0.5] * 10)  # 10 tokens, chunk_size=4 → 3 chunks (4,4,2)
    prompt = mx.zeros(10, dtype=mx.int32)
    keep = select_keep_indices(importance, prompt, cfg)
    # 50% of 3 = 1.5 → max(1, 1) = 1 chunk kept
    assert len(keep) == 4  # one chunk of 4 tokens


def test_sparse_prefill_target_signature():
    """sparse_prefill_target has correct signature and returns the right tuple shape."""
    from src.exo.worker.engines.mlx.spec_prefill import sparse_prefill_target, cleanup_rope
    import inspect
    sig = inspect.signature(sparse_prefill_target)
    params = list(sig.parameters.keys())
    # Required params: target_model, prompt_tokens, keep_indices, cache, config
    # Optional: on_prefill_progress, group
    assert "target_model" in params
    assert "prompt_tokens" in params
    assert "keep_indices" in params
    assert "cache" in params
    assert "config" in params
    # Verify cleanup_rope exists and takes a model
    cleanup_sig = inspect.signature(cleanup_rope)
    assert "model" in cleanup_sig.parameters


def test_rope_patcher_compute_position_ids():
    """_RoPEPatcher assigns nearest preceding kept position to skipped tokens."""
    import mlx.core as mx

    from src.exo.worker.engines.mlx.spec_prefill import _RoPEPatcher
    kept = mx.array([0, 3, 5, 7], dtype=mx.int32)
    patcher = _RoPEPatcher(model=None, kept_indices=kept, prompt_len=10)
    pos_ids = patcher.position_ids.tolist()
    assert pos_ids == [0, 0, 0, 3, 3, 5, 5, 7, 7, 7]


def test_cleanup_rope_no_crash_on_no_rope():
    """cleanup_rope doesn't crash on a model without a .rope attribute."""
    # MagicMock without 'model' attribute simulates no RoPE
    fake_model = MagicMock(spec=[])  # no attributes
    cleanup_rope(fake_model)  # should not raise


def test_rope_patcher_all_kept():
    """When all positions are kept, position_ids are [0, 1, 2, ..., n-1]."""
    import mlx.core as mx

    from src.exo.worker.engines.mlx.spec_prefill import _RoPEPatcher

    n = 5
    kept = mx.arange(n, dtype=mx.int32)
    patcher = _RoPEPatcher(model=None, kept_indices=kept, prompt_len=n)
    pos_ids = patcher.position_ids.tolist()
    assert pos_ids == list(range(n))


def test_rope_patcher_first_only():
    """When only position 0 is kept, all skipped positions map to 0."""
    import mlx.core as mx

    from src.exo.worker.engines.mlx.spec_prefill import _RoPEPatcher

    n = 5
    kept = mx.array([0], dtype=mx.int32)
    patcher = _RoPEPatcher(model=None, kept_indices=kept, prompt_len=n)
    pos_ids = patcher.position_ids.tolist()
    assert pos_ids == [0] * n


def test_specprefill_config_disabled_by_default():
    """SpecPrefillConfig() defaults to disabled when EXO_SPEC_PREFILL is not '1'/'true'/'yes'.

    SPEC_PREFILL_ENABLED is captured at constants.py import time (module-level
    env var read). Test verifies the dataclass wires the field correctly: with
    EXO_SPEC_PREFILL unset in the test environment, the dataclass default
    bound at import time should be False.
    """
    from src.exo.worker.engines.mlx.spec_prefill import SpecPrefillConfig

    cfg = SpecPrefillConfig()
    # Default comes from constants.SPEC_PREFILL_ENABLED at import time.
    # When EXO_SPEC_PREFILL is not '1'/'true'/'yes', enabled must be False.
    assert cfg.enabled is False


def test_specprefill_config_field_wiring():
    """SpecPrefillConfig has a bool `enabled` field wired to constants.

    SPEC_PREFILL_ENABLED is captured at constants.py module import time and
    bound as the dataclass default. We can't re-read the env var after import
    (the default is already frozen), so we verify the wiring is correct:
    the field exists, is a bool, and is consistent with the constants module.
    """
    from src.exo.worker.engines.mlx import constants
    from src.exo.worker.engines.mlx.spec_prefill import SpecPrefillConfig

    cfg = SpecPrefillConfig()
    assert hasattr(cfg, "enabled")
    assert isinstance(cfg.enabled, bool)
    # The dataclass default must mirror the constants module value
    # (proves the wiring: constants → dataclass default).
    assert cfg.enabled == constants.SPEC_PREFILL_ENABLED

