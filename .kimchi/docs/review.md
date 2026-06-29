# Review: Task 8 Fix (commit ccc1a006)

## Verdict: APPROVED

## What I verified

All 5 fixes from the Task 8 code review were correctly applied. The diff is minimal, surgical, and matches the reviewer recommendations exactly. All 21 tests pass with `--noconftest`.

## Specific findings

### Fix 1 (CRITICAL — broken test) — VERIFIED
- `tests/worker/engines/mlx/test_spec_prefill.py:283-295` — `test_specprefill_config_disabled_by_default` no longer calls `os.environ.pop`. It now constructs `SpecPrefillConfig()` and asserts `cfg.enabled is False`, which correctly verifies the dataclass default is wired.
- `tests/worker/engines/mlx/test_spec_prefill.py:298-313` — new `test_specprefill_config_field_wiring` verifies `hasattr(cfg, "enabled")`, `isinstance(cfg.enabled, bool)`, and `cfg.enabled == constants.SPEC_PREFILL_ENABLED`. Together the two tests prove the constants → dataclass wiring without trying to re-read the env var at test time.

### Fix 2 (IMPORTANT — lazy import) — VERIFIED
- `src/exo/worker/engines/mlx/generator/generate.py:1-75` — `SpecPrefillConfig` is no longer imported at module level. The `from exo.worker.engines.mlx.spec_prefill import SpecPrefillConfig` line is removed from the top-of-file imports block.
- Lazy import placed at `src/exo/worker/engines/mlx/generator/generate.py:323`, just above the `SpecPrefillConfig()` call. This is consistent with the six phase functions which remain inside the try block at line 332.

### Fix 3 (IMPORTANT — cache `_has_pipeline_communication_layer`) — VERIFIED
- `src/exo/worker/engines/mlx/generator/generate.py:315` — `is_pipeline_model = _has_pipeline_communication_layer(model)` is declared once near the top of `prefill()`, with a comment explaining why.
- Used at `src/exo/worker/engines/mlx/generator/generate.py:328` in the SpecPrefill guard (`and not is_pipeline_model`).
- Used at `src/exo/worker/engines/mlx/generator/generate.py:414` in the existing dispatch (`if is_pipeline_model and num_tokens >= prefill_step_size:`).
- The old local `is_pipeline = _has_pipeline_communication_layer(model)` at the previous line 398 is gone. The function is now invoked exactly once per `prefill()` call.

### Fix 4 (IMPORTANT — docstring) — VERIFIED
- `src/exo/worker/engines/mlx/generator/generate.py:294-300` — docstring now lists three modes:
  1. SpecPrefill (enabled + non-pipeline + prompt >= min_prompt_tokens)
  2. Pipeline-sharded: `pipeline_parallel_prefill`
  3. Default: `stream_generate`

### Fix 5 (LATENT BUG — load/validate order) — VERIFIED
- `src/exo/worker/engines/mlx/generator/generate.py:344-347` — within the `if not draft_model.is_loaded():` branch:
  - Line 345: `draft_model.load()` runs first.
  - Line 347: `draft_model.validate_tokenizer_compat(tokenizer)` runs after, with the comment "Validate target tokenizer compat AFTER load". This matches the requirement that `validate_tokenizer_compat` requires `is_loaded()` to return True.

## Test output

```
============================== 21 passed in 2.26s ==============================
```

All 21 tests pass (20 prior + 1 new `test_specprefill_config_field_wiring`).

## Notes

- Minor observation (not an issue): the lazy `SpecPrefillConfig` import at line 323 sits just above another `from exo.worker.engines.mlx.spec_prefill import (...)` block at line 332. Python caches the module after the first import, so the second import statement is essentially free — no performance or correctness concern. Keeping them as separate statements is also fine: it documents that both the config and the phase functions are part of the SpecPrefill dispatch surface.
- No scope creep detected. The diff is tightly scoped to the 5 reported issues plus a one-line comment cleanup.
- No hallucinated APIs detected. All referenced symbols (`SpecPrefillConfig`, `_has_pipeline_communication_layer`, `validate_tokenizer_compat`, `is_loaded`, `load`) exist in the codebase.
