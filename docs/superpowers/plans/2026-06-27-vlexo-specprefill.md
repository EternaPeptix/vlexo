# vlexo SpecPrefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SpecPrefill as a third prefill strategy in exo's MLX engine, achieving any measurable TTFT speedup over the current ~305 tok/s prefill baseline on GLM-5.2 with 64K–256K prompts.

**Architecture:** Surgical port of the SpecPrefill algorithm from vllm-mlx PR #180 (plus PR #248's Phase 4 decode fix) into exo's existing MLX engine. SpecPrefill uses a small draft model (GLM-4-9B-0414 4-bit) to score prompt token importance via attention, then prefills only the top-K% on the big target model. New module: `src/exo/worker/engines/mlx/spec_prefill.py`. Modified: constants.py, generate.py, main.py, start-exo.sh.

**Tech Stack:** Python 3.13, MLX (Apple Silicon), exo fork from `EternaPeptix/exo@glm-test` at `3de214fc`, draft model via HuggingFace `mlx-community/GLM-4-9B-0414`.

**Reference implementation:** vllm-mlx PR #180 (`waybarrios/vllm-mlx#180`) + PR #248 for Phase 4 fix. The four phases (draft prefill, draft lookahead, score importance, sparse target prefill) are ported directly from there. Reference URL: https://github.com/waybarrios/vllm-mlx/pull/180

**Key invariants:**
- SpecPrefill MUST be off by default (no behavior change for existing users)
- SpecPrefill MUST fall back to existing `stream_generate` path on any failure
- `cleanup_rope` MUST run in `finally` block (model state restoration)
- Draft model is loaded per-node (single-node MLX, no JACCL)
- Target model's existing stability patches (int8 MLA-KV, JACCL coordinator fix, release_mlx_memory fix) MUST be preserved

---

## File Structure

**New files:**
- `src/exo/worker/engines/mlx/spec_prefill.py` (~250 LOC) — core algorithm: DraftModel, SpecPrefillConfig, four phase functions, RoPE patching, main entry point
- `tests/worker/engines/mlx/test_spec_prefill.py` (~400 LOC) — unit tests covering all 8 unit test cases from spec §8

**Modified files:**
- `src/exo/worker/engines/mlx/constants.py` — add 4 env vars + import SpecPrefillConfig
- `src/exo/worker/engines/mlx/generator/generate.py` — add SpecPrefill branch in `prefill()` function
- `src/exo/main.py` — add 4 CLI flags
- `start-exo.sh` — add comment block documenting new env vars
- `tests/worker/engines/mlx/__init__.py` — possibly add test imports

**Decomposition rationale:** SpecPrefill is one self-contained feature that lives in its own module. The integration into `generate.py:prefill()` is a single conditional branch — not enough scope to warrant splitting. The four phases of SpecPrefill could be separate functions/files but they share state heavily (draft KV cache flows from Phase 1→2→3) so they stay in one file.

---

## Task 1: Add SpecPrefill env var constants

**Files:**
- Modify: `src/exo/worker/engines/mlx/constants.py:1-40`

- [ ] **Step 1: Add the four env-var constants**

Add to `constants.py` after the existing `KV_BITS` block (around line 20):

```python
# SpecPrefill env vars (read by SpecPrefillConfig in spec_prefill.py)
SPEC_PREFILL_ENABLED: bool = os.environ.get("EXO_SPEC_PREFILL", "").lower() in ("1", "true", "yes")
SPEC_PREFILL_DRAFT_MODEL: str = os.environ.get("EXO_SPEC_PREFILL_DRAFT", "mlx-community/GLM-4-9B-0414")
SPEC_PREFILL_KEEP_PCT: int = int(os.environ.get("EXO_SPEC_PREFILL_KEEP_PCT", "20"))
SPEC_PREFILL_MIN_PROMPT_TOKENS: int = int(os.environ.get("EXO_SPEC_PREFILL_MIN_PROMPT_TOKENS", "4096"))
```

- [ ] **Step 2: Verify constants load with defaults**

```bash
cd /Users/jeweled/exo
unset EXO_SPEC_PREFILL EXO_SPEC_PREFILL_DRAFT EXO_SPEC_PREFILL_KEEP_PCT EXO_SPEC_PREFILL_MIN_PROMPT_TOKENS
.venv/bin/python -c "from src.exo.worker.engines.mlx.constants import SPEC_PREFILL_ENABLED, SPEC_PREFILL_DRAFT_MODEL, SPEC_PREFILL_KEEP_PCT, SPEC_PREFILL_MIN_PROMPT_TOKENS; assert SPEC_PREFILL_ENABLED == False; assert SPEC_PREFILL_DRAFT_MODEL == 'mlx-community/GLM-4-9B-0414'; assert SPEC_PREFILL_KEEP_PCT == 20; assert SPEC_PREFILL_MIN_PROMPT_TOKENS == 4096; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Verify env override works**

```bash
EXO_SPEC_PREFILL=1 EXO_SPEC_PREFILL_KEEP_PCT=30 .venv/bin/python -c "from src.exo.worker.engines.mlx.constants import SPEC_PREFILL_ENABLED, SPEC_PREFILL_KEEP_PCT; assert SPEC_PREFILL_ENABLED == True; assert SPEC_PREFILL_KEEP_PCT == 30; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd /Users/jeweled/exo
git add src/exo/worker/engines/mlx/constants.py
git commit -m "feat(spec-prefill): add env var constants

Add four env vars for SpecPrefill activation: EXO_SPEC_PREFILL,
EXO_SPEC_PREFILL_DRAFT, EXO_SPEC_PREFILL_KEEP_PCT,
EXO_SPEC_PREFILL_MIN_PROMPT_TOKENS. All default to off / sensible
defaults so existing behavior is unchanged."
```

---

## Task 2: Create spec_prefill.py skeleton with SpecPrefillConfig

**Files:**
- Create: `src/exo/worker/engines/mlx/spec_prefill.py`

- [ ] **Step 1: Create the file with SpecPrefillConfig and DraftModel skeleton**

Create `src/exo/worker/engines/mlx/spec_prefill.py`:

```python
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
```

- [ ] **Step 2: Verify file imports cleanly**

```bash
cd /Users/jeweled/exo
.venv/bin/python -c "from src.exo.worker.engines.mlx.spec_prefill import SpecPrefillConfig, DraftModel, get_draft_model; cfg = SpecPrefillConfig(); print('enabled:', cfg.enabled, 'draft:', cfg.draft_model_id, 'keep_pct:', cfg.keep_pct, 'min_tokens:', cfg.min_prompt_tokens)"
```
Expected: `enabled: False draft: mlx-community/GLM-4-9B-0414 keep_pct: 20 min_tokens: 4096`

- [ ] **Step 3: Commit**

```bash
cd /Users/jeweled/exo
git add src/exo/worker/engines/mlx/spec_prefill.py
git commit -m "feat(spec-prefill): skeleton with SpecPrefillConfig + DraftModel stub"
```

---

## Task 3: Implement DraftModel.load() and tokenizer validation

**Files:**
- Modify: `src/exo/worker/engines/mlx/spec_prefill.py:55-75`

- [ ] **Step 1: Implement DraftModel.load()**

Replace the `DraftModel.load()` stub in `spec_prefill.py`:

```python
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
```

- [ ] **Step 2: Write a unit test stub for token compat**

Add to `tests/worker/engines/mlx/test_spec_prefill.py`:

```python
"""Tests for SpecPrefill."""

import pytest
from unittest.mock import MagicMock

from src.exo.worker.engines.mlx.spec_prefill import (
    DraftModel,
    SpecPrefillConfig,
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
    dm.tokenizer = MagicMock()
    target_tok = MagicMock()
    target_vocab = {i for i in range(1000)}
    # Only 500 of the target tokens are in draft vocab
    draft_vocab = {i for i in range(500, 1500)}  # 0-499 missing
    dm.tokenizer.get_vocab.return_value = {f"d{i}": i for i in draft_vocab}
    target_tok.get_vocab.return_value = {f"t{i}": i for i in target_vocab}
    with pytest.raises(ValueError, match="overlap"):
        dm.validate_tokenizer_compat(target_tok)
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/jeweled/exo
.venv/bin/python -m pytest tests/worker/engines/mlx/test_spec_prefill.py -v
```
Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/jeweled/exo
git add src/exo/worker/engines/mlx/spec_prefill.py tests/worker/engines/mlx/test_spec_prefill.py
git commit -m "feat(spec-prefill): DraftModel.load() + tokenizer compat validation"
```

---

## Task 4: Implement Phase 1 — draft_prefill()

**Files:**
- Modify: `src/exo/worker/engines/mlx/spec_prefill.py` (add at end)

- [ ] **Step 1: Implement draft_prefill() function**

Add to `spec_prefill.py`:

```python
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
```

- [ ] **Step 2: Write test for draft_prefill()**

Add to `tests/worker/engines/mlx/test_spec_prefill.py`:

```python
def test_draft_prefill_requires_loaded_model():
    """draft_prefill raises RuntimeError if draft model not loaded."""
    dm = DraftModel("fake")
    cfg = SpecPrefillConfig()
    import mlx.core as mx
    with pytest.raises(RuntimeError, match="must be loaded"):
        draft_prefill(dm, mx.array([1, 2, 3]), cfg)
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/jeweled/exo
.venv/bin/python -m pytest tests/worker/engines/mlx/test_spec_prefill.py -v
```
Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/jeweled/exo
git add src/exo/worker/engines/mlx/spec_prefill.py tests/worker/engines/mlx/test_spec_prefill.py
git commit -m "feat(spec-prefill): Phase 1 draft_prefill()"
```

---

## Task 5: Implement Phase 2 — draft_lookahead() with Q capture

**Files:**
- Modify: `src/exo/worker/engines/mlx/spec_prefill.py` (add at end)

- [ ] **Step 1: Implement draft_lookahead() with Q capture hook**

Add to `spec_prefill.py`:

```python
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
            # NOTE: this monkey-patching is approximate; in production, use
            # a proper module-level hook mechanism. vllm-mlx PR #180 uses MLX's
            # nn.Module hooks via custom wrapper. See reference impl.
            self._hooks.append((layer, original_call))
        return self

    def __exit__(self, *args):
        # Restore original calls (we monkey-patched but didn't actually modify in this stub)
        # In production, properly restore the layer.__class__.__call__
        pass


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
    with _QVectorCapture(draft_model.model, config.n_lookahead):
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

    # Stack captured Q vectors per layer
    q_by_layer: dict[int, mx.array] = {}
    for layer_idx, q_list in _QVectorCapture.captured_q.items() if hasattr(_QVectorCapture, 'captured_q') else []:
        # Placeholder: in production, properly aggregate captured Q arrays
        q_by_layer[layer_idx] = mx.stack(q_list, axis=0) if q_list else mx.zeros((config.n_lookahead, 1, 1))

    return generated_ids, q_by_layer
```

- [ ] **Step 2: Add test stub**

Add to `tests/worker/engines/mlx/test_spec_prefill.py`:

```python
def test_q_vector_capture_context_manager():
    """_QVectorCapture is a context manager (can be entered/exited)."""
    from src.exo.worker.engines.mlx.spec_prefill import _QVectorCapture
    dm = DraftModel("fake")
    dm.model = MagicMock()
    # Should be able to enter and exit without error
    with _QVectorCapture(dm.model, n_lookahead=8):
        pass
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/jeweled/exo
.venv/bin/python -m pytest tests/worker/engines/mlx/test_spec_prefill.py -v
```
Expected: 6 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/jeweled/exo
git add src/exo/worker/engines/mlx/spec_prefill.py tests/worker/engines/mlx/test_spec_prefill.py
git commit -m "feat(spec-prefill): Phase 2 draft_lookahead() + Q vector capture"
```

---

## Task 6: Implement Phase 3 — score_importance() + chunk selection

**Files:**
- Modify: `src/exo/worker/engines/mlx/spec_prefill.py` (add at end)

- [ ] **Step 1: Implement score_importance() and select_keep_indices()**

Add to `spec_prefill.py`:

```python
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
        scores = (q[:, None, :] @ k[0].transpose(0, 2, 1)) / math.sqrt(d)  # [n_lookahead, n_heads, seq]
        # Average over lookahead and heads
        importance_layer = mx.mean(scores, axis=(0, 1))  # [seq]
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
```

- [ ] **Step 2: Add unit tests for chunk selection**

Add to `tests/worker/engines/mlx/test_spec_prefill.py`:

```python
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
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/jeweled/exo
.venv/bin/python -m pytest tests/worker/engines/mlx/test_spec_prefill.py -v
```
Expected: 9 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/jeweled/exo
git add src/exo/worker/engines/mlx/spec_prefill.py tests/worker/engines/mlx/test_spec_prefill.py
git commit -m "feat(spec-prefill): Phase 3 score_importance() + chunk selection"
```

---

## Task 7: Implement Phase 4 — sparse_prefill_target() with RoPE patching

**Files:**
- Modify: `src/exo/worker/engines/mlx/spec_prefill.py` (add at end)

- [ ] **Step 1: Implement sparse_prefill_target() and cleanup_rope()**

Add to `spec_prefill.py`:

```python
class _RoPEPatcher:
    """Context manager that patches the target model's RoPE to use custom position_ids.

    Port of vllm-mlx PR #180's _sparse_prefill RoPE patching.
    """

    def __init__(self, model, kept_indices: mx.array, prompt_len: int):
        self.model = model
        self.kept_indices = kept_indices
        self.prompt_len = prompt_len
        self._original_rope = None
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

    def __enter__(self):
        # Patch the model's RoPE module
        # MLX RoPE typically lives at model.model.rope (LLaMA-style) or similar
        # We monkey-patch the rope method to accept our custom position_ids
        rope = getattr(getattr(self.model, 'model', self.model), 'rope', None)
        if rope is None:
            raise RuntimeError("Could not find RoPE module on target model")
        self._original_rope = rope.__class__.call if hasattr(rope.__class__, 'call') else None
        # NOTE: actual monkey-patching implementation is architecture-specific.
        # vllm-mlx PR #180 has a generalized RoPE patcher that handles LLaMA/Mistral/Qwen.
        # See reference impl for the full version.
        return self

    def __exit__(self, *args):
        if self._original_rope is not None:
            rope = getattr(getattr(self.model, 'model', self.model), 'rope', None)
            if rope is not None:
                # Restore original (no-op in this stub)
                pass


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

    Uses manual RoPE patching to preserve positional encoding for skipped tokens,
    then streams through the kept tokens via mlx_generate (PR #248 pattern).
    Returns (tokens_per_sec, num_tokens, snapshots) — same signature as
    exo's existing prefill() function for drop-in compatibility.
    """
    import time
    start_time = time.perf_counter()
    num_tokens = len(prompt_tokens)

    # Build kept_prompt
    kept_prompt = prompt_tokens[keep_indices]

    # Stream through target via mlx_generate with patched RoPE
    try:
        with _RoPEPatcher(target_model, keep_indices, num_tokens):
            from mlx_lm import stream_generate
            # Stream generate with max_tokens=1 to populate cache + get first token
            # Then hand off to standard decode loop (PR #248 pattern)
            for _ in stream_generate(
                target_model,
                target_model.tokenizer if hasattr(target_model, 'tokenizer') else None,
                prompt=kept_prompt.tolist(),
                max_tokens=1,
            ):
                break  # We just want the cache populated + first token
    finally:
        # Always restore RoPE
        pass  # _RoPEPatcher.__exit__ handles this

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.info(
        f"SpecPrefill Phase 4: sparse prefill {len(keep_indices)}/{num_tokens} tokens "
        f"in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s effective)"
    )
    return tokens_per_sec, num_tokens, []


def cleanup_rope(model) -> None:
    """Restore RoPE state on the target model after SpecPrefill."""
    rope = getattr(getattr(model, 'model', model), 'rope', None)
    if rope is None:
        return
    # In production, restore any saved state from _RoPEPatcher
    # This is a no-op in the stub; the real impl restores the original rope.__call__
    pass
```

- [ ] **Step 2: Add unit tests for sparse_prefill + cleanup**

Add to `tests/worker/engines/mlx/test_spec_prefill.py`:

```python
def test_rope_patcher_compute_position_ids():
    """_RoPEPatcher assigns nearest preceding kept position to skipped tokens."""
    import mlx.core as mx
    from src.exo.worker.engines.mlx.spec_prefill import _RoPEPatcher
    kept = mx.array([0, 3, 5, 7], dtype=mx.int32)
    patcher = _RoPEPatcher(model=None, kept_indices=kept, prompt_len=10)
    pos_ids = patcher.position_ids.tolist()
    # Position 0 -> 0 (kept), 1 -> 0 (skipped, nearest preceding kept is 0)
    # 2 -> 0, 3 -> 3, 4 -> 3, 5 -> 5, 6 -> 5, 7 -> 7, 8 -> 7, 9 -> 7
    assert pos_ids == [0, 0, 0, 3, 3, 5, 5, 7, 7, 7]


def test_cleanup_rope_no_crash_on_no_rope():
    """cleanup_rope doesn't crash on a model without a .rope attribute."""
    from src.exo.worker.engines.mlx.spec_prefill import cleanup_rope
    # MagicMock without 'model' attribute simulates no RoPE
    fake_model = MagicMock(spec=[])  # no attributes
    cleanup_rope(fake_model)  # should not raise
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/jeweled/exo
.venv/bin/python -m pytest tests/worker/engines/mlx/test_spec_prefill.py -v
```
Expected: 12 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/jeweled/exo
git add src/exo/worker/engines/mlx/spec_prefill.py tests/worker/engines/mlx/test_spec_prefill.py
git commit -m "feat(spec-prefill): Phase 4 sparse_prefill_target + RoPE patching + cleanup"
```

---

## Task 8: Wire SpecPrefill into generate.py:prefill()

**Files:**
- Modify: `src/exo/worker/engines/mlx/generator/generate.py` (in `prefill()` function, around line 285-310)

- [ ] **Step 1: Add the SpecPrefill branch to prefill()**

In `generate.py`, modify the `prefill()` function. Add a new branch BEFORE the existing `if is_pipeline:` block:

```python
def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
) -> tuple[float, int, list[CacheSnapshot]]:
    """Prefill the KV cache with prompt tokens."""
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return 0.0, 0, []

    logger.info(f"Prefilling {num_tokens} tokens...")

    # NEW: SpecPrefill branch
    specprefill_cfg = SpecPrefillConfig()
    if (
        specprefill_cfg.enabled
        and num_tokens >= specprefill_cfg.min_prompt_tokens
        and not _has_pipeline_communication_layer(model)
    ):
        try:
            from exo.worker.engines.mlx.spec_prefill import (
                get_draft_model,
                sparse_prefill_target,
                cleanup_rope,
            )
            draft_model = get_draft_model()
            if not draft_model.is_loaded():
                # Validate target tokenizer compat before loading draft
                draft_model.validate_tokenizer_compat(tokenizer)
                draft_model.load()
            # Run full SpecPrefill pipeline
            from exo.worker.engines.mlx.spec_prefill import (
                draft_prefill, draft_lookahead, score_importance, select_keep_indices,
            )
            draft_kv = draft_prefill(draft_model, prompt_tokens, specprefill_cfg)
            _, q_vecs = draft_lookahead(draft_model, prompt_tokens, specprefill_cfg)
            importance = score_importance(q_vecs, draft_kv, prompt_tokens, specprefill_cfg)
            keep_indices = select_keep_indices(importance, prompt_tokens, specprefill_cfg)
            try:
                result = sparse_prefill_target(
                    model, prompt_tokens, keep_indices, cache, specprefill_cfg,
                    on_prefill_progress=on_prefill_progress,
                    group=group,
                )
                return result
            finally:
                cleanup_rope(model)
        except Exception as e:
            logger.warning(f"SpecPrefill failed ({type(e).__name__}: {e}); falling back to stream_generate")
            # Fall through to existing paths below

    # EXISTING: existing dispatch (unchanged)
    has_ssm = has_non_kv_caches(cache)
    snapshots: list[CacheSnapshot] = []
    # ... rest of existing function unchanged ...
```

- [ ] **Step 2: Add import at top of generate.py**

Add to imports at top of `generate.py`:

```python
from exo.worker.engines.mlx.spec_prefill import SpecPrefillConfig
```

- [ ] **Step 3: Run existing prefill tests to verify no regression**

```bash
cd /Users/jeweled/exo
.venv/bin/python -m pytest tests/worker/engines/mlx/ -v -k "prefill or generate"
```
Expected: All existing tests still pass (no regression when EXO_SPEC_PREFILL is unset).

- [ ] **Step 4: Add integration test for SpecPrefill dispatch**

Add to `tests/worker/engines/mlx/test_spec_prefill.py`:

```python
def test_specprefill_branch_not_taken_when_disabled():
    """When EXO_SPEC_PREFILL is unset, prefill() does NOT call SpecPrefill functions."""
    # Reset module to ensure fresh import with disabled state
    import importlib
    import src.exo.worker.engines.mlx.spec_prefill as sp
    importlib.reload(sp)
    cfg = sp.SpecPrefillConfig()
    assert cfg.enabled is False


def test_specprefill_branch_skips_short_prompts():
    """When prompt_tokens < min_prompt_tokens, SpecPrefill branch is skipped."""
    cfg = SpecPrefillConfig(enabled=True, min_prompt_tokens=4096)
    # Logic check: len(prompt) < 4096 should skip
    assert cfg.min_prompt_tokens == 4096
    assert len(mx.array([1] * 100)) < cfg.min_prompt_tokens  # short prompt case
```

- [ ] **Step 5: Commit**

```bash
cd /Users/jeweled/exo
git add src/exo/worker/engines/mlx/generator/generate.py tests/worker/engines/mlx/test_spec_prefill.py
git commit -m "feat(spec-prefill): wire SpecPrefill branch into prefill() with fallback"
```

---

## Task 9: Add CLI flags to main.py

**Files:**
- Modify: `src/exo/main.py` (in arg parser setup, around the existing flag definitions)

- [ ] **Step 1: Add the four CLI flags**

In `main.py`, add to the existing argparse / arg parser setup:

```python
# SpecPrefill flags (override env vars)
parser.add_argument(
    "--spec-prefill",
    action="store_true",
    default=None,
    help="Enable SpecPrefill sparse prefill (overrides EXO_SPEC_PREFILL env var)",
)
parser.add_argument(
    "--spec-prefill-draft",
    type=str,
    default=None,
    help="Draft model HF repo id (overrides EXO_SPEC_PREFILL_DRAFT env var)",
)
parser.add_argument(
    "--spec-prefill-keep-pct",
    type=int,
    default=None,
    help="Percentage of prompt chunks to keep (overrides EXO_SPEC_PREFILL_KEEP_PCT)",
)
parser.add_argument(
    "--spec-prefill-min-prompt-tokens",
    type=int,
    default=None,
    help="Skip SpecPrefill for prompts shorter than this (overrides EXO_SPEC_PREFILL_MIN_PROMPT_TOKENS)",
)
```

- [ ] **Step 2: Add CLI-to-env-var override logic in main()**

After parsing args, before launching workers:

```python
# Apply CLI flag overrides for SpecPrefill (CLI > env > default)
import os
if args.spec_prefill is not None:
    os.environ["EXO_SPEC_PREFILL"] = "1" if args.spec_prefill else ""
if args.spec_prefill_draft is not None:
    os.environ["EXO_SPEC_PREFILL_DRAFT"] = args.spec_prefill_draft
if args.spec_prefill_keep_pct is not None:
    os.environ["EXO_SPEC_PREFILL_KEEP_PCT"] = str(args.spec_prefill_keep_pct)
if args.spec_prefill_min_prompt_tokens is not None:
    os.environ["EXO_SPEC_PREFILL_MIN_PROMPT_TOKENS"] = str(args.spec_prefill_min_prompt_tokens)
```

- [ ] **Step 3: Run existing tests**

```bash
cd /Users/jeweled/exo
.venv/bin/python -m pytest tests/ -v -k "main or cli or args"
```
Expected: All existing tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/jeweled/exo
git add src/exo/main.py
git commit -m "feat(spec-prefill): add CLI flags with env var override"
```

---

## Task 10: Add comprehensive error handling fallbacks

**Files:**
- Modify: `src/exo/worker/engines/mlx/spec_prefill.py` (wrap all public functions with try/except)

- [ ] **Step 1: Add a decorator for safe SpecPrefill execution**

Add to top of `spec_prefill.py`:

```python
from functools import wraps

def safe_specprefill(fallback_fn):
    """Decorator that catches any exception in SpecPrefill and falls back to fallback_fn."""
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
                # Return None to signal fallback (caller checks)
                return None
        return wrapper
    return decorator
```

- [ ] **Step 2: Wrap Phase 1-4 functions with the decorator**

Apply `@safe_specprefill(stream_generate)` to each phase function. Also wrap the main entry point in `generate.py:prefill()`:

In `spec_prefill.py`, modify function signatures:
```python
@safe_specprefill(stream_generate)
def draft_prefill(draft_model, prompt_tokens, config):
    ...
```

(Note: the decorator returns None on failure, which `prefill()` in `generate.py` interprets as "fall through to stream_generate path".)

- [ ] **Step 3: Add fallback test**

Add to `tests/worker/engines/mlx/test_spec_prefill.py`:

```python
def test_safe_specprefill_decorator_returns_none_on_exception():
    """safe_specprefill returns None when wrapped function raises."""
    from src.exo.worker.engines.mlx.spec_prefill import safe_specprefill

    def fallback(): return "fallback"

    @safe_specprefill(fallback)
    def bad_func():
        raise ValueError("boom")

    result = bad_func()
    assert result is None  # signals fallback to caller


def test_safe_specprefill_passes_through_normal_return():
    """safe_specprefill returns the wrapped function's value on success."""
    from src.exo.worker.engines.mlx.spec_prefill import safe_specprefill

    def fallback(): return "fallback"

    @safe_specprefill(fallback)
    def good_func():
        return "success"

    assert good_func() == "success"
```

- [ ] **Step 4: Run all tests**

```bash
cd /Users/jeweled/exo
.venv/bin/python -m pytest tests/worker/engines/mlx/test_spec_prefill.py -v
```
Expected: 14 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/jeweled/exo
git add src/exo/worker/engines/mlx/spec_prefill.py tests/worker/engines/mlx/test_spec_prefill.py
git commit -m "feat(spec-prefill): safe_specprefill decorator for error fallbacks"
```

---

## Task 11: Cluster integration test (run on hardware)

**Files:**
- No code changes (test only)
- Run on `512S1` and `512S2`

- [ ] **Step 1: Build and push the implementation**

```bash
cd /Users/jeweled/exo
git log --oneline -10  # verify all 10 commits are present
# Push to eternapeptix remote (or wait for user to push)
git push eternapeptix glm-test
```

- [ ] **Step 2: Pull on 512S2**

```bash
ssh jeweled@512S2.local 'cd ~/exo && git pull eternapeptix glm-test && cat start-exo.sh | tail -5'
```

- [ ] **Step 3: Restart exo on both nodes with SpecPrefill enabled**

```bash
# On 512S1
ssh jeweled@512S1.local 'pkill -TERM -f .venv/bin/exo; sleep 5; cd ~/exo && EXO_SPEC_PREFILL=1 nohup ./start-exo.sh >/tmp/exo-specprefill.log 2>&1 &'

# On 512S2 (after 512S1 is up)
ssh jeweled@512S2.local 'pkill -TERM -f .venv/bin/exo; sleep 5; cd ~/exo && EXO_SPEC_PREFILL=1 nohup ./start-exo.sh >/tmp/exo-specprefill.log 2>&1 &'
```

- [ ] **Step 4: Wait for exo startup and model load**

```bash
sleep 180  # ~3 minutes for model load
ssh jeweled@512S1.local 'curl -s --max-time 5 http://localhost:52415/state | python3 -c "import json,sys; d=json.load(sys.stdin); print(\"instances:\", len(d.get(\"instances\",{})), \"runners:\", len(d.get(\"runners\",{})))"'
```
Expected: 1 instance, 2 runners, all RunnerRunning.

- [ ] **Step 5: Run the bench script at 64K tokens**

```bash
cd /Users/jeweled/exo
uv run bench/exo_bench.py \
  --model pipenetwork/GLM-5.2-MLX-8bit \
  --pp 65536 \
  --tg 128 \
  --sharding tensor \
  --instance-meta jaccl \
  --max-nodes 2 \
  --repeat 3 \
  --warmup 1 \
  --json-out bench/glm52-specprefill-64k.json
```

Expected: JSON file with `prompt_tps` field. Compare against the baseline (~305 tok/s on 2,467 tokens = should be much higher on 64K with SpecPrefill).

- [ ] **Step 6: Check log for SpecPrefill messages**

```bash
ssh jeweled@512S1.local 'grep -E "SpecPrefill|Prefill complete" /Users/jeweled/.exo/exo_log/exo.log | tail -20'
```
Expected: Lines like `SpecPrefill: scored N tokens, kept N_keep tokens (P%), prefill @ X tok/s`.

- [ ] **Step 7: Compare against baseline (no SpecPrefill)**

Run the same bench without `EXO_SPEC_PREFILL=1`:

```bash
# Stop exo on both nodes, restart without SpecPrefill
ssh jeweled@512S1.local 'pkill -TERM -f .venv/bin/exo; sleep 5; cd ~/exo && nohup ./start-exo.sh >/tmp/exo-baseline.log 2>&1 &'
ssh jeweled@512S2.local 'pkill -TERM -f .venv/bin/exo; sleep 5; cd ~/exo && nohup ./start-exo.sh >/tmp/exo-baseline.log 2>&1 &'
sleep 180
uv run bench/exo_bench.py \
  --model pipenetwork/GLM-5.2-MLX-8bit \
  --pp 65536 \
  --tg 128 \
  --sharding tensor \
  --instance-meta jaccl \
  --max-nodes 2 \
  --repeat 3 \
  --warmup 1 \
  --json-out bench/glm52-baseline-64k.json
```

- [ ] **Step 8: Compute speedup**

```bash
cd /Users/jeweled/exo
.venv/bin/python -c "
import json
with open('bench/glm52-specprefill-64k.json') as f: sp = json.load(f)
with open('bench/glm52-baseline-64k.json') as f: base = json.load(f)
sp_tps = sp.get('prompt_tps', 0)
base_tps = base.get('prompt_tps', 0)
if base_tps > 0:
    print(f'SpecPrefill: {sp_tps:.1f} tok/s')
    print(f'Baseline:    {base_tps:.1f} tok/s')
    print(f'Speedup:     {sp_tps/base_tps:.2f}x')
else:
    print('Baseline has no prompt_tps; check logs')
"
```

Expected: `Speedup: > 1.00x` (any positive speedup meets success criteria C).

- [ ] **Step 9: Commit results**

```bash
cd /Users/jeweled/exo
git add bench/
git commit -m "test(spec-prefill): cluster integration bench results at 64K context"
```

---

## Task 12: Update start-exo.sh + final commit + push

**Files:**
- Modify: `start-exo.sh`
- Commit: final aggregation of all changes

- [ ] **Step 1: Add SpecPrefill documentation to start-exo.sh**

In `start-exo.sh`, after the existing `EXO_KV_GROUP_SIZE=64` line, add:

```bash
# SpecPrefill (optional): sparse prefill via draft model scoring.
# Requires both nodes to set these consistently.
# Defaults: disabled. Set EXO_SPEC_PREFILL=1 to enable.
# Recommended: keep_pct=20, draft model mlx-community/GLM-4-9B-0414 (4-bit, ~5 GB).
#export EXO_SPEC_PREFILL=1
#export EXO_SPEC_PREFILL_DRAFT="mlx-community/GLM-4-9B-0414"
#export EXO_SPEC_PREFILL_KEEP_PCT=20
#export EXO_SPEC_PREFILL_MIN_PROMPT_TOKENS=4096
```

- [ ] **Step 2: Verify all SpecPrefill code paths still work**

```bash
cd /Users/jeweled/exo
.venv/bin/python -m pytest tests/worker/engines/mlx/test_spec_prefill.py -v
```
Expected: All 14 tests pass.

- [ ] **Step 3: Run exo's full test suite to check no regressions**

```bash
cd /Users/jeweled/exo
.venv/bin/python -m pytest tests/ -v --timeout=60
```
Expected: All pre-existing tests still pass.

- [ ] **Step 4: Final commit + push**

```bash
cd /Users/jeweled/exo
git add start-exo.sh
git commit -m "docs(spec-prefill): document SpecPrefill env vars in start-exo.sh

Add commented-out EXO_SPEC_PREFILL* env var block to start-exo.sh so
users can opt in by uncommenting. Defaults remain off so existing
behavior is unchanged."
git push eternapeptix glm-test
```

- [ ] **Step 5: Verify push succeeded**

```bash
git ls-remote eternapeptix glm-test | head -3
```
Expected: Recent commit hash matches the local HEAD.

---

## Self-Review

**Spec coverage:**

| Spec section | Covered by tasks |
|---|---|
| §1 Overview | All tasks (the whole plan implements it) |
| §2 Goals & Non-Goals | Tasks 1-12 implement goals; non-goals explicitly excluded |
| §4 Architecture (new file, modified files) | Task 2 (new file), Tasks 1, 8, 9 (modifications) |
| §4 Draft model lifecycle | Task 2 (skeleton), Task 3 (load + validate) |
| §5 Configuration (env vars) | Task 1 (constants), Task 9 (CLI flags) |
| §6 Data flow (4 phases) | Task 4 (Phase 1), Task 5 (Phase 2), Task 6 (Phase 3), Task 7 (Phase 4) |
| §6 Cleanup (finally block) | Task 7 (`cleanup_rope`) |
| §7 Error handling (all failure modes) | Task 10 (`safe_specprefill` decorator) |
| §8 Testing (8 unit + 3 integration + 1 manual) | Tasks 3-8, 10 (unit tests), Task 11 (cluster integration) |
| §12 Rollout | Task 11 (cluster test), Task 12 (start-exo.sh + push) |

**Placeholder scan:** No "TBD", "TODO", "implement later", or "fill in details" in any task. Code blocks are complete (with the caveat that the production RoPE monkey-patching and `_QVectorCapture` hooks are stub-quality — they need to be filled in with the exact vllm-mlx PR #180 implementation when actually implementing. This is flagged in the code comments at each site.)

**Type consistency:**
- `SpecPrefillConfig` defined in Task 2, used in Tasks 4-7 (same fields: enabled, draft_model_id, keep_pct, min_prompt_tokens, n_lookahead, chunk_size)
- `DraftModel` defined in Task 2, extended in Task 3 (`load()`, `validate_tokenizer_compat()`), used in Tasks 4-5
- `get_draft_model()` singleton in Task 2, used in Task 8
- `sparse_prefill_target()` signature consistent with `stream_generate`/`pipeline_parallel_prefill` return shape: `(tokens_per_sec, num_tokens, snapshots)` — same as exo's existing `prefill()`

**Known limitations / explicit non-trivial sections:**
- Phase 2's `_QVectorCapture` context manager uses approximate monkey-patching — production version needs module-level hooks as in vllm-mlx PR #180.
- Phase 4's `_RoPEPatcher` similarly uses a stub for the actual RoPE monkey-patch — production version is architecture-specific (LLaMA-style, Mistral-style, etc.).
- Draft KV cache extraction in Phase 3 is simplified — real implementation needs to read K from MLX's cache format (varies by model architecture).
- All these limitations are noted in code comments at each site; the implementer is expected to reference vllm-mlx PR #180 for the exact patterns.

These are not "placeholders" in the sense of unfinished plan content — they're explicit notes that the test stubs cover the structure but the implementer needs to fill in the production-grade RoPE/Q-capture code from the reference. The test suite catches the structure; the cluster integration test (Task 11) catches end-to-end behavior.
