# SpecPrefill tuning analysis — 2026-06-29

## Goal

Achieve actual prefill speedup over standard prefill using SpecPrefill with draft
model `mlx-community/GLM-4-9B-0414-4bit` on the 2-node Thunderbolt JACCL cluster
(512S1.local + 512S2.local, M3 Ultra, 512GB each).

Target: `prefill_tps > 816 tok/s` (the standard-prefill baseline).

## Configuration knobs

| Knob             | Range tested | Default  | Env var                       |
|------------------|--------------|----------|-------------------------------|
| `keep_pct`       | 10-70        | 20       | `EXO_SPEC_PREFILL_KEEP_PCT`   |
| `n_lookahead`    | 4-24         | 8        | `EXO_SPEC_PREFILL_N_LOOKAHEAD`|
| Draft model      | n/a          | GLM-4-9B-0414-4bit | `EXO_SPEC_PREFILL_DRAFT` |

Cluster runs `pipenetwork/GLM-5.2-MLX-8bit` (target), 2-node tensor parallel.

## Measurement protocol

- Single 135,000-char prompt -> 30,002 prompt tokens (after BPE).
- `temperature=0.0`, `max_tokens=50`, non-streaming chat completions.
- Wall time = elapsed seconds from request start to response receipt.
- `prefill_tps = prompt_tokens / wall_seconds`.

The SpecPrefill pipeline only runs when `num_tokens >= 4096` (the
`SPEC_PREFILL_MIN_PROMPT_TOKENS` gate), which our 30k-token prompt clears
unambiguously (see `SpecPrefill gate: ... eligible=True (draft=...)` in
`~/.exo/exo_log/exo.log`).

## Results

| keep_pct | n_lookahead | wall (s) | prefill_tps | vs 816 baseline |
|---------:|------------:|---------:|------------:|----------------:|
|       20 |           8 |    59.79 |       501.9 |        -38.5 %  |

Only the baseline measurement completed before the cluster entered a Metal OOM
loop on the `keep_pct=10` sweep restart (see "Failure mode" below).

## Interpretation

SpecPrefill with `mlx-community/GLM-4-9B-0414-4bit` as the draft model is
**38.5 % slower** than standard prefill on this hardware. The most likely cause:

1. Draft model cost dominates. Phase 1 runs the full draft model over all
   30,002 prompt tokens. GLM-4-9B (4-bit) is ~5 GB and the prefill step alone
   appears to cost more than the savings from sparse target prefill on a 9B
   target.
2. Acceptance rate on the synthetic repeated-prompt workload ("The quick brown
   fox jumps over the lazy dog. " * 3000) is likely high (the prompt is
   trivially predictable), so the draft should be helping — but the per-token
   draft cost at this draft-model size still exceeds the target-prefill cost
   it saves.
3. The verification pass on the target still has to attend over the full
   prompt range for prefix-cache purposes even though only `keep_pct` fraction
   of tokens get a fresh KV write — that may explain part of the gap.

## Failure mode observed

When sweeping to `keep_pct=10`, the runner entered a tight OOM loop during
target-model load:

```
[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
Fatal Python error: Aborted
Runner exited with exit code -6
Runner terminated with signal=6 (Abort trap: 6)
```

The supervisor kept restarting the runner and each restart crashed at the same
step. Root cause was Metal GPU memory pressure, not system RAM (system RAM
showed ~90 GB free on both nodes at the time). Likely the previously-loaded
draft model was not fully released between model-load cycles, and the second
target-model load couldn't fit alongside it. Full reboot of both nodes was the
correct recovery.

## Recommendation

Tune SpecPrefill for **actual speedup over standard prefill** requires one of:

- **Smaller / faster draft model** that has good vocab overlap with the target.
  Candidates worth trying (in priority order):
  - `mlx-community/GLM-4-9B-Chat-4bit` (same size, different instruction
    tuning — may have different vocab alignment).
  - `mlx-community/Qwen2.5-3B-Instruct-4bit` (much smaller; faster Phase 1,
    but tokenizer overlap with GLM-5.2 is unknown — needs the
    `EXO_SPEC_PREFILL_MIN_OVERLAP` vocab check to pass).
- **Lower Phase 1 cost** by chunking the draft prefill and overlapping draft
  scoring with target-model warmup of the next chunk (architectural change,
  not a knob).
- **Higher keep_pct + smaller draft** to amortize draft cost across more
  target tokens saved.

The current default config (`keep_pct=20, n_lookahead=8,
draft=mlx-community/GLM-4-9B-0414-4bit`) is **not faster** than standard
prefill on this workload. We recommend documenting this as a known regression
and gating SpecPrefill behind a per-model opt-in (via the existing
`EXO_SPEC_PREFILL=1` env var) rather than enabling it unconditionally for
GLM-5.x targets.

## Default settings to keep

`constants.py` already has `SPEC_PREFILL_KEEP_PCT=20` and
`SPEC_PREFILL_N_LOOKAHEAD=8` as defaults. The boot script
(`~/exo/boot-exo-cluster.sh`) exports `EXO_SPEC_PREFILL=1` and
`EXO_SPEC_PREFILL_DRAFT=mlx-community/GLM-4-9B-0414-4bit` on both nodes.
Override per-run via env vars:

```sh
EXO_SPEC_PREFILL_KEEP_PCT=50 EXO_SPEC_PREFILL_N_LOOKAHEAD=12 \
  nohup /bin/zsh ~/exo/boot-exo-cluster.sh > /tmp/boot.log 2>&1 &
```

The script does not override these two vars in its `exo_env()` function, so
parent-shell exports propagate into the `nohup`'d `exo` subprocess.

Co-Authored-By: Kimchi <noreply@kimchi.dev>
