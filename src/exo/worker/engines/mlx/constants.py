import os

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?


def _parse_optional_int(env_var: str) -> int | None:
    raw = os.environ.get(env_var)
    if raw is None or raw == "":
        return None
    return int(raw)


KV_GROUP_SIZE: int = int(os.environ.get("EXO_KV_GROUP_SIZE", "64"))
# int8 MLA latent KV (see mlx-lm MLACacheList.to_quantized). Indexer cache stays fp16.
KV_BITS: int | None = _parse_optional_int("EXO_KV_BITS")

# SpecPrefill env vars (read by SpecPrefillConfig in spec_prefill.py)
SPEC_PREFILL_ENABLED: bool = os.environ.get("EXO_SPEC_PREFILL", "").lower() in ("1", "true", "yes")
SPEC_PREFILL_DRAFT_MODEL: str = os.environ.get("EXO_SPEC_PREFILL_DRAFT", "mlx-community/GLM-4-9B-0414")
SPEC_PREFILL_KEEP_PCT: int = int(os.environ.get("EXO_SPEC_PREFILL_KEEP_PCT", "20"))
SPEC_PREFILL_MIN_PROMPT_TOKENS: int = int(os.environ.get("EXO_SPEC_PREFILL_MIN_PROMPT_TOKENS", "4096"))

ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
# Used only for non-MLA models without make_cache(); MLA models use KV_BITS instead.
KV_CACHE_BITS: int | None = _parse_optional_int("EXO_KV_CACHE_BITS")

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
