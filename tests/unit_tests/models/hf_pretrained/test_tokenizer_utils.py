import tokenizers

from megatron.bridge.models.hf_pretrained.tokenizer_utils import (
    apply_mistral_regex_fix,
    should_fix_mistral_regex,
)


def _make_dummy_backend() -> tokenizers.Tokenizer:
    backend = tokenizers.Tokenizer(tokenizers.models.BPE())
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [
            tokenizers.pre_tokenizers.Split(
                pattern=tokenizers.Regex(
                    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                ),
                behavior="isolated",
            ),
            tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    return backend


class _DummyTokenizer:
    def __init__(self, *, backend_tokenizer=None, private_tokenizer=None):
        if backend_tokenizer is not None:
            self.backend_tokenizer = backend_tokenizer
        if private_tokenizer is not None:
            self._tokenizer = private_tokenizer


def test_should_fix_mistral_regex_for_kimi_and_mistral_names():
    assert should_fix_mistral_regex("/local/models/Kimi-K2.5")
    assert should_fix_mistral_regex("moonshotai/Kimi-K2.5")
    assert should_fix_mistral_regex("mistralai/Mistral-7B-Instruct-v0.3")


def test_should_not_fix_mistral_regex_for_other_tokenizers():
    assert not should_fix_mistral_regex("Qwen/Qwen3-4B")
    assert not should_fix_mistral_regex("meta-llama/Llama-3.1-8B-Instruct")


def test_apply_mistral_regex_fix_rewrites_sequence_pretokenizer():
    tokenizer = _DummyTokenizer(backend_tokenizer=_make_dummy_backend())

    apply_mistral_regex_fix(tokenizer)

    pretokenizer_repr = repr(tokenizer.backend_tokenizer.pre_tokenizer)
    assert tokenizer.fix_mistral_regex is True
    assert "ByteLevel(" in pretokenizer_repr
    assert "(?i:'s|'t|'re|'ve|'m|'ll|'d)" not in pretokenizer_repr
    assert r"[^\r\n\p{L}\p{N}]?" in pretokenizer_repr


def test_apply_mistral_regex_fix_is_idempotent():
    tokenizer = _DummyTokenizer(backend_tokenizer=_make_dummy_backend())

    apply_mistral_regex_fix(tokenizer)
    first_repr = repr(tokenizer.backend_tokenizer.pre_tokenizer)

    apply_mistral_regex_fix(tokenizer)

    assert repr(tokenizer.backend_tokenizer.pre_tokenizer) == first_repr


def test_apply_mistral_regex_fix_supports_private_tokenizer_attr():
    tokenizer = _DummyTokenizer(private_tokenizer=_make_dummy_backend())

    apply_mistral_regex_fix(tokenizer)

    assert tokenizer.fix_mistral_regex is True
    assert "ByteLevel(" in repr(tokenizer._tokenizer.pre_tokenizer)
