from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_MISTRAL_REGEX_MODEL_HINTS = ("kimi", "mistral")
_MISTRAL_REGEX_WARNING = "incorrect regex pattern"
_BROKEN_MISTRAL_REGEX_FRAGMENT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
_MISTRAL_REGEX_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def should_fix_mistral_regex(tokenizer_name: str | Path | None) -> bool:
    if tokenizer_name is None:
        return False
    normalized_name = str(tokenizer_name).lower()
    return any(model_hint in normalized_name for model_hint in _MISTRAL_REGEX_MODEL_HINTS)


def _build_hf_tokenizer_kwargs(
    tokenizer_name: str | Path | None, trust_remote_code: bool | None = None
) -> dict[str, Any]:
    del tokenizer_name
    kwargs: dict[str, Any] = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return kwargs


def _get_backend_tokenizer(tokenizer: Any) -> Any:
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None:
        return backend_tokenizer

    backend_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if backend_tokenizer is not None:
        return backend_tokenizer

    raise TypeError(f"Tokenizer {type(tokenizer).__name__} does not expose a backend tokenizer")


def _get_pretokenizer_repr(tokenizer: Any) -> str:
    return repr(_get_backend_tokenizer(tokenizer).pre_tokenizer)


def _has_broken_mistral_regex(tokenizer: Any) -> bool:
    return _BROKEN_MISTRAL_REGEX_FRAGMENT in _get_pretokenizer_repr(tokenizer)


def _has_fixed_mistral_regex(tokenizer: Any) -> bool:
    return _MISTRAL_REGEX_PATTERN in _get_pretokenizer_repr(tokenizer)


@contextmanager
def _suppress_mistral_regex_warning(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    tokenizer_logger = logging.getLogger("transformers.tokenization_utils_tokenizers")

    class _SuppressMistralRegexWarning(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return _MISTRAL_REGEX_WARNING not in record.getMessage()

    warning_filter = _SuppressMistralRegexWarning()
    tokenizer_logger.addFilter(warning_filter)
    try:
        yield
    finally:
        tokenizer_logger.removeFilter(warning_filter)


def apply_mistral_regex_fix(tokenizer: Any) -> Any:
    import tokenizers

    if _has_fixed_mistral_regex(tokenizer):
        tokenizer.fix_mistral_regex = True
        return tokenizer

    backend_tokenizer = _get_backend_tokenizer(tokenizer)
    current_pretokenizer = backend_tokenizer.pre_tokenizer
    split_pretokenizer = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(_MISTRAL_REGEX_PATTERN),
        behavior="isolated",
    )

    if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Sequence):
        current_pretokenizer[0] = split_pretokenizer
    else:
        if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Metaspace):
            current_pretokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)

        backend_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
            [split_pretokenizer, current_pretokenizer]
        )
    tokenizer.fix_mistral_regex = True
    return tokenizer


def load_hf_tokenizer(
    tokenizer_name: str | Path,
    *,
    trust_remote_code: bool | None = None,
    **kwargs: Any,
) -> Any:
    from transformers import AutoTokenizer

    with _suppress_mistral_regex_warning(should_fix_mistral_regex(tokenizer_name)):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            **_build_hf_tokenizer_kwargs(tokenizer_name, trust_remote_code),
            **kwargs,
        )
    try:
        if _has_broken_mistral_regex(tokenizer):
            apply_mistral_regex_fix(tokenizer)
        elif _has_fixed_mistral_regex(tokenizer):
            tokenizer.fix_mistral_regex = True
    except TypeError:
        pass
    return tokenizer
