# vlexo: SpecPrefill surgical port into exo

**Date:** 2026-06-27
**Status:** Design approved, awaiting user review of written spec
**Base:** fork from `EternaPeptix/exo@glm-test` at `3de214fc`

---

## 1. Overview

vlexo is a thin fork of exo that adds **SpecPrefill** as a third prefill strategy. SpecPrefill uses a small "draft" model to score every prompt token's importance via attention, then prefills only the top-K% on the big target model, skipping low-importance tokens. vllm-mlx's PR #180 reports 3.71–5.45× TTFT speedup on long-context MoE targets.

The algorithm is ported from vllm-mlx (`waybarrios/vllm-mlx` PR #180, merged 2026-03-21, Apache-2.0) into exo's existing MLX engine. We also port PR #248's Phase 4 decode-loop fix to avoid the 100× decode regression that PR #180 left behind.

Nothing else changes: JACCL tensor/pipeline parallel over Thunderbolt 5 RDMA, int8 MLA-KV cache, prefix cache, BatchGenerator, and the four existing stability patches from `EternaPeptix/exo@glm-test` all stay intact.

---

## 2. Goals & Non-Goals

### Goals (v1)
1. Reduce TTFT on 64K–256K prompts for GLM-5.2 (or any compatible target) on a 2-node M3 Ultra cluster with JACCL tensor parallel.
2. Preserve all existing exo behavior when SpecPrefill is disabled (default).
3. Fall back to existing `stream_generate` path cleanly when SpecPrefill cannot run (draft load failure, tokenizer mismatch, scoring error, short prompts).
4. Make the feature opt-in via env vars, with CLI flag overrides.

### Non-Goals (deferred)
- SpecPrefill for pipeline-parallel sharding (TP-only in v1).
- Speculative decoding (separate from sparse prefill; distinct feature).
- MTP / native multi-token prediction (GLM has no native MTP).
- SpecPrefill for vision/multimodal models.
- Auto-tuning `keep_pct` per prompt.
- Distributed draft scoring across both nodes (draft scoring runs single-node; JACCL only sees the target).

---

## 3. Design Decisions

Six questions were answered during brainstorming:

| # | Question | Answer |
|---|---|---|
| 1 | Integration approach | **Surgical port** of the SpecPrefill algorithm from vllm-mlx into exo's existing MLX engine |
| 2 | Draft model | `mlx-community/GLM-4-9B-0414` (4-bit, ~5 GB) |
| 3 | Success bar | Any measurable improvement over the current ~305 tok/s prefill baseline |
| 4 | Scope of v1 | Port both PR #180 (SpecPrefill) **and** PR #248's Phase 4 decode fix |
| 5 | Activation | Env vars + CLI flags (`EXO_SPEC_PREFILL`, `EXO_SPEC_PREFILL_DRAFT`, `EXO_SPEC_PREFILL_KEEP_PCT`, `EXO_SPEC_PREFILL_MIN_PROMPT_TOKENS`) |
| 6 | Base | Fork from `EternaPeptix/exo@glm-test` at HEAD `3de214fc` |

---

## 4. Architecture

### Components

**New file:**
- `src/exo/worker/engines/mlx/spec_prefill.py` (~250 lines) — the four-phase SpecPrefill algorithm, draft model wrapper, RoPE patching utilities.

**Modified files:**
- `src/exo/worker/engines/mlx/constants.py` — add four env-var constants and a `SpecPrefillConfig` dataclass.
- `src/exo/worker/engines/mlx/generator/generate.py` — add one new branch in `prefill()` that calls the SpecPrefill path before falling through to the existing `stream_generate` call.
- `src/exo/main.py` — add four CLI flags mirroring the env vars (env var precedence: CLI > env > default).
- `start-exo.sh` — add a comment block documenting the new env vars. No behavior change.

### Where SpecPrefill fits in the prefill dispatch

`prefill()` in `generate.py` currently has two paths:
1. `pipeline_parallel_prefill(...)` — when target is sharded via pipeline parallel.
2. `stream_generate(...)` — default, tensor-parallel target.

We add a third path, selected before either:

```
prefill(model, prompt_tokens, cache, ...):

    if (SPEC_PREFILL_ENABLED
        and not _has_pipeline_communication_layer(model)
        and len(prompt_tokens) >= SPEC_PREFILL_MIN_PROMPT_TOKENS
        and draft model is loaded):
        # NEW PATH
        sparse_prefill_target(model, prompt_tokens, cache, draft_model, keep_pct)
        return (tokens_per_sec, num_tokens, snapshots)

    # existing dispatch
    if is_pipeline:
        pipeline_parallel_prefill(...)
    else:
        stream_generate(...)
```

### Draft model lifecycle

- **Loaded lazily** on the first SpecPrefill request (not at exo startup).
- **Per-node**: each exo node loads its own copy into local memory. Draft scoring is a single-node MLX operation; it does not go through JACCL.
- **Pinned in memory** across requests. 5 GB resident per node (10 GB total for 2 nodes) — trivial on 512 GB nodes.
- **Validated at load time**: tokenizer vocabulary of draft must match target within a 99% overlap threshold. Mismatch fails fast with a clear error.
- **Failure modes fall back**: load failure, scoring failure, NaN/Inf importance scores → all log a warning and fall through to `stream_generate`.

---

## 5. Configuration

### Env vars (read in `constants.py`)

| Variable | Type | Default | Meaning |
|---|---|---|---|
| `EXO_SPEC_PREFILL` | `bool` | `False` | Master switch. When `True`, SpecPrefill path is considered. |
| `EXO_SPEC_PREFILL_DRAFT` | `str` | `"mlx-community/GLM-4-9B-0414"` | HuggingFace repo id for the draft model. Must be an MLX-converted variant (i.e., from the `mlx-community` namespace). exo calls `mlx_lm.load()` on this directly — no conversion step. |
| `EXO_SPEC_PREFILL_KEEP_PCT` | `int` | `20` | Percentage of prompt chunks (32-token chunks) to keep after importance scoring. 20 = top 20%. |
| `EXO_SPEC_PREFILL_MIN_PROMPT_TOKENS` | `int` | `4096` | Skip SpecPrefill for prompts shorter than this. Avoids draft overhead exceeding savings. |

### CLI flags (read in `main.py`, override env vars)

```
--spec-prefill                        # bool flag, equivalent to EXO_SPEC_PREFILL=1
--spec-prefill-draft <repo_id>         # equivalent to EXO_SPEC_PREFILL_DRAFT
--spec-prefill-keep-pct <int>          # equivalent to EXO_SPEC_PREFILL_KEEP_PCT
--spec-prefill-min-prompt-tokens <int> # equivalent to EXO_SPEC_PREFILL_MIN_PROMPT_TOKENS
```

Precedence: CLI flag > env var > default.

---

## 6. Data Flow (Four Phases)

Given `prompt_tokens` of length `N` (where `N >= SPEC_PREFILL_MIN_PROMPT_TOKENS`):

### Phase 1: Draft prefill on full prompt
- Run the draft model (single-node MLX) on all `N` prompt tokens. Produces draft KV cache.
- No JACCL involvement. Target model is untouched.

### Phase 2: Draft lookahead decode
- Generate `n_lookahead = 8` tokens from the draft via standard `mlx_generate`.
- During decode, attach a forward hook to each transformer layer that captures the query vectors `Q` from the final decode step.
- Output: `Q_lookahead` shape `[num_layers, num_heads, 8, head_dim]`.

### Phase 3: Score importance
- For each layer: `importance_per_layer = softmax(Q_lookahead @ K_prompt^T / sqrt(d))`, averaged across lookahead positions.
- Average across layers → `importance` shape `[N]`.
- Chunk `importance` into 32-token non-overlapping windows; score each chunk as `mean(importance in chunk)`.
- Take top `K%` chunks by score → `keep_chunk_indices`.
- Expand `keep_chunk_indices` to token-level `keep_indices` (a kept chunk keeps all its tokens). Sort ascending.

### Phase 4: Sparse target prefill
- Build `kept_prompt = prompt_tokens[keep_indices]`.
- Apply **manual RoPE** to the target model via a context-manager pattern (port from vllm-mlx PR #180 `_sparse_prefill`):
  - Enter: patch the model's RoPE module to accept a custom `position_ids` array. Kept positions get their original `position_ids`; skipped positions get the `position_ids` of the nearest preceding kept position (so relative-position encoding remains consistent).
  - The patched RoPE is the **only** state mutation. No weight changes.
- Stream `kept_prompt` through target via `mlx_generate(max_tokens=1)` with the patched RoPE in effect.
- **Hand off** to standard decode path via PR #248's pipelined `_generate_step` loop — NOT PR #180's manual `model() + mx.eval(y)` loop (which has the 100× decode regression, see issue #247).
- Output: filled target cache + first decode token.

### Cleanup
- In a `finally` block, call `cleanup_rope(model)` to restore original RoPE config (so future requests without SpecPrefill work normally).

### Logging
- `prefill_tps` returned from the function counts **original** `N` tokens (not kept count), so logs reflect effective tokens/s against the full prompt size for honest comparison to baseline.
- A new log line marks SpecPrefill usage: `"SpecPrefill: scored N tokens, kept N_keep tokens (P%), prefill @ X tok/s"`.

---

## 7. Error Handling

| Failure | Behavior |
|---|---|
| Draft model load fails (OOM, network, HF rate limit) | Log warning, fall back to `stream_generate`, continue |
| Tokenizer mismatch (draft vocab vs target vocab) | Fail fast at load time with clear error message; do not start serving |
| Draft inference produces NaN/Inf importance scores | Log warning, fall back to `stream_generate` |
| Importance scores uniformly zero (model is confused) | Use full prefill (keep all tokens) |
| Keep-set is empty after scoring | Use full prefill (keep all tokens) |
| Keep-set is full (all chunks kept) | Continue normally — equivalent to non-SpecPrefill path |
| Any exception during Phases 1–4 | Caught, logged with stack, fall through to `stream_generate` |
| Cleanup (RoPE restore) | Runs in `finally` block, always restores even on failure |
| Memory pressure during draft load (< 50 GB available) | Log warning, attempt load anyway; if OOM, fall back to `stream_generate` |

All fallbacks log at WARN level with enough context for the user to diagnose.

---

## 8. Testing Strategy

### Unit tests (in `tests/spec_prefill/`)
1. **Equivalence**: 100-token prompt produces text equivalent to non-SpecPrefill path (modulo small numerical drift).
2. **Keep = 100%**: `keep_pct=100` produces output identical to non-SpecPrefill path (no chunks skipped).
3. **Keep = low%**: `keep_pct=5` produces a valid (if lower-quality) response.
4. **Short prompt skip**: prompts < `min_threshold` skip SpecPrefill and use regular path.
5. **Tokenizer mismatch**: simulating vocab mismatch raises a clear error at load time.
6. **NaN/Inf scores**: scoring returns NaN → falls back to `stream_generate`.
7. **Empty keep-set**: scoring returns all-zero importance → use full prefill.
8. **Cleanup on failure**: exception in Phase 4 → `cleanup_rope` still runs.

### Integration tests
9. **1K prompt speedup**: any positive prefill speedup over baseline (success criteria C).
10. **64K cluster bench**: `bench/exo_bench.py --pp 65536 --tg 128 --sharding tensor --instance-meta jaccl --max-nodes 2`. Compare prompt_tps against current baseline (~305 tok/s on 2,467-token real prompt).
11. **Regression — `release_mlx_memory` shutdown**: existing test still passes (we just fixed this in commit 3de214fc).
12. **Smoke — SpecPrefill disabled**: with `EXO_SPEC_PREFILL=` (unset), all existing tests pass unchanged.

### Manual verification
13. End-to-end chat completion through exo's OpenAI-compatible API on a 64K prompt. Compare subjective output quality vs. non-SpecPrefill path. Expect slight quality degradation (3–10% of tokens skipped) but mostly coherent output.

---

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| GLM-4-9B-0414 tokenizer incompatible with GLM-5.2 | Medium | Verify at load time; fail fast; user can switch draft model via env var |
| Draft scoring overhead exceeds savings for short prompts | Low (mitigated by `min_threshold=4096`) | Default to skipping SpecPrefill for prompts under threshold; user-tunable |
| Memory pressure on a node (5 GB draft + ~80 GB target) | Low (well under 512 GB) | Log warning if available memory < 50 GB when loading draft |
| RoPE patching leaves model in bad state for next request | Medium | `cleanup_rope` in `finally` block; unit-tested with exception injection |
| vllm-mlx API drift between PR #180 and current main | Low (we pin to PR #180 logic) | Reference implementation locked at the merge point; document exact commit hash in code comment |
| Generation quality loss from skipped tokens | Medium (expected) | Document expected behavior; show in logs; user can tune `keep_pct` |

---

## 10. Out of Scope (deferred to future versions)

- SpecPrefill for pipeline-parallel sharding.
- Speculative decoding (separate from sparse prefill).
- MTP / multi-token prediction (GLM has none).
- SpecPrefill for vision/multimodal models.
- Auto-tuning `keep_pct` per prompt.
- Distributed draft scoring across both nodes.
- Continuous batching interaction with SpecPrefill.
- Speculative prefill chunk size tuning.

---

## 11. Success Criteria

Per brainstorming decision: **any measurable improvement** over the current ~305 tok/s prefill baseline.

**Verification:**
- Run `bench/exo_bench.py --pp 65536 --sharding tensor --instance-meta jaccl --max-nodes 2` with and without `EXO_SPEC_PREFILL=1`.
- If `prompt_tps_with_specprefill > prompt_tps_baseline`, ship it.
- If equal or worse, debug before declaring v1 done.

**Non-blocking signals:**
- Log lines clearly show SpecPrefill was used and how many tokens were skipped.
- No new crashes vs. baseline.
- Cleanup works (verified by unit test 8).

---

## 12. Rollout

1. Land the spec + implementation on `EternaPeptix/vlexo@main`.
2. Push to both nodes (512S1, 512S2). Restart exo on each.
3. Run a small smoke chat completion (1K tokens) on GLM-5.2 with SpecPrefill enabled. Verify logs and output.
4. Run the bench script at 64K. Confirm any speedup > 0.
5. Document the env vars in `start-exo.sh` comments and README.

---

## 13. References

- vllm-mlx PR #180 — original SpecPrefill implementation: https://github.com/waybarrios/vllm-mlx/pull/180
- vllm-mlx PR #248 — Phase 4 decode-loop fix (100× regression): https://github.com/waybarrios/vllm-mlx/pull/248
- vllm-mlx issue #247 — root cause analysis of the decode regression: https://github.com/waybarrios/vllm-mlx/issues/247
- SpecPrefill paper (arXiv 2502.02789): https://arxiv.org/abs/2502.02789
- vllm-mlx project: https://github.com/waybarrios/vllm-mlx
- vllm-mlx model-registry docs (draft model examples): https://github.com/waybarrios/vllm-mlx/blob/main/docs/guides/model-registry.md
- exo MLX engine (`src/exo/worker/engines/mlx/generator/generate.py`): the file where the SpecPrefill branch will be inserted.
- exo base commit (`3de214fc`): https://github.com/EternaPeptix/exo/commit/3de214fc

---

## Self-Review

Completed during writing:

- **Placeholder scan**: One placeholder found and removed — the original draft ended this section with an empty "to fill in after writing" checklist. Replaced with this self-review block.
- **Internal consistency**: Architecture (§4) names `_has_pipeline_communication_layer` as the pipeline-detection helper; §6 references the same helper via §4. Component list (§4) matches the modified-files list. Env var list (§5) matches the rollout mention (§12). PR #180 and PR #248 are referenced consistently across §1, §6, §9, and §13.
- **Scope check**: One feature (SpecPrefill surgical port), one new file, four modified files, ≤ ~350 LOC total change. Fits a single implementation plan. No decomposition needed.
- **Ambiguity check**: RoPE patching mechanism clarified (§6 Phase 4) — context-manager pattern, position_ids remapping rule stated explicitly, only state mutation is RoPE. Draft model load semantics clarified — must be MLX-converted from `mlx-community` namespace, `mlx_lm.load()` directly, no conversion step. Tokenizer validation threshold stated (99% vocab overlap). Pipeline-detection helper named explicitly (`_has_pipeline_communication_layer`) instead of a placeholder.
