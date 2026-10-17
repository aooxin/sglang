import json
import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


MIMO_TOKEN_SUFFIX = "_36a8b94076c558de0938f7e4f3cd4f18c5490ca0"


def mimo_token(token: str) -> str:
    """Return the internal spelling of one MiMo added token (idempotently).

    Only template/control tokens use this namespace. Request content must stay
    untouched so literal public spellings are encoded as ordinary text.
    """
    ending = "|>" if token.endswith("|>") else ">" if token.endswith(">") else ""
    name = token[: -len(ending)] if ending else token
    if not name.endswith(MIMO_TOKEN_SUFFIX):
        name += MIMO_TOKEN_SUFFIX
    return name + ending


def _replace_mimo_token_text(value, replacements, *, replace_substrings=False):
    from transformers import AddedToken

    if isinstance(value, AddedToken):
        state = value.__getstate__()
        state["content"] = replacements.get(value.content, value.content)
        return AddedToken(**state)
    if isinstance(value, str):
        if not replace_substrings:
            return replacements.get(value, value)
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, dict):
        return {
            replacements.get(key, key): _replace_mimo_token_text(
                item, replacements, replace_substrings=replace_substrings
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_mimo_token_text(
                item, replacements, replace_substrings=replace_substrings
            )
            for item in value
        ]
    return value


def replace_mimo_chat_template(tokenizer, template):
    """Also used after an explicit Jinja template overrides the tokenizer template."""
    replacements = getattr(tokenizer, "_sglang_mimo_token_replacements", None)
    if not replacements or template is None:
        return template
    return _replace_mimo_token_text(template, replacements, replace_substrings=True)


def patch_mimo_tokenizer(tokenizer, is_mimo=False):
    """Rename MiMo added tokens in memory without allocating new token IDs."""
    from transformers import AddedToken

    if getattr(tokenizer, "_sglang_mimo_token_replacements", None):
        tokenizer.chat_template = replace_mimo_chat_template(
            tokenizer, tokenizer.chat_template
        )
        return tokenizer

    added_tokens = tokenizer.added_tokens_decoder
    # MiMo's own audio/video markers identify anonymous paths such as /preset-models/.
    if not is_mimo and not any(
        token.content.startswith("<|mimo_") for token in added_tokens.values()
    ):
        return tokenizer

    replacements = {}
    for token in added_tokens.values():
        content = token.content
        ending = (
            "|>" if content.endswith("|>") else ">" if content.endswith(">") else ""
        )
        name = content[: -len(ending)] if ending else content
        if name.endswith(MIMO_TOKEN_SUFFIX):
            replacements[name[: -len(MIMO_TOKEN_SUFFIX)] + ending] = content
        else:
            replacements[content] = mimo_token(content)

    original_size = len(tokenizer)
    vocab = tokenizer.get_vocab()
    # fastokens returns _AddedTokenInfo rather than transformers.AddedToken.
    # Rebuild from the shared fields so both backends rename content and retain
    # all flags; _replace_mimo_token_text would leave _AddedTokenInfo unchanged.
    renamed_tokens = {
        token_id: AddedToken(
            replacements.get(token.content, token.content),
            single_word=token.single_word,
            lstrip=token.lstrip,
            rstrip=token.rstrip,
            normalized=token.normalized,
            special=token.special,
        )
        for token_id, token in added_tokens.items()
    }
    for token_id, token in renamed_tokens.items():
        if token.content in vocab and vocab[token.content] != token_id:
            raise ValueError(
                f"MiMo renamed token already has another ID: {token.content}"
            )

    special_tokens = getattr(tokenizer, "special_tokens_map_extended", None)
    if special_tokens is None:
        # Transformers v5 removed special_tokens_map_extended and keeps extra
        # special tokens separately. Read the stored values to preserve AddedToken
        # flags rather than using the string-only special_tokens_map.
        special_tokens = dict(tokenizer._special_tokens_map)
        special_tokens["extra_special_tokens"] = tokenizer._extra_special_tokens
    special_tokens = _replace_mimo_token_text(special_tokens, replacements)
    if tokenizer.is_fast:
        # Updating the Python decoder alone does not update the Rust encoder.
        backend = tokenizer.backend_tokenizer
        # Replace exact token spellings in the backend state, not substrings of
        # unrelated base-vocabulary entries or tokenizer configuration strings.
        state = _replace_mimo_token_text(json.loads(backend.to_str()), replacements)
        tokenizer._tokenizer = type(backend).from_str(json.dumps(state))
    else:
        from transformers.tokenization_utils import Trie

        tokenizer._added_tokens_decoder = renamed_tokens
        tokenizer._added_tokens_encoder = {
            token.content: token_id for token_id, token in renamed_tokens.items()
        }
        # Some Qwen vocabularies also include the added tokens in the base vocab.
        tokenizer.encoder = _replace_mimo_token_text(tokenizer.encoder, replacements)
        tokenizer.decoder = _replace_mimo_token_text(tokenizer.decoder, replacements)
        tokenizer.tokens_trie = Trie()
        tokenizer._update_trie()
        tokenizer._update_total_vocab_size()
        tokenizer.cache.clear()

    for name, value in special_tokens.items():
        setattr(tokenizer, name, value)
    tokenizer.init_kwargs = _replace_mimo_token_text(
        tokenizer.init_kwargs, replacements
    )
    tokenizer._sglang_mimo_token_replacements = replacements
    tokenizer.chat_template = replace_mimo_chat_template(
        tokenizer, tokenizer.chat_template
    )
    if "chat_template" in tokenizer.init_kwargs:
        tokenizer.init_kwargs["chat_template"] = replace_mimo_chat_template(
            tokenizer, tokenizer.init_kwargs["chat_template"]
        )

    if len(tokenizer) != original_size or any(
        tokenizer.convert_tokens_to_ids(token.content) != token_id
        for token_id, token in renamed_tokens.items()
    ):
        raise ValueError("MiMo added-token renaming changed token IDs")
    logger.info("Renamed %d MiMo added tokens in memory", len(renamed_tokens))
    return tokenizer


def patch_tokenizer(tokenizer):
    if not envs.SGLANG_PATCH_TOKENIZER.get():
        return tokenizer

    if _is_kimi_tiktoken_tokenizer(tokenizer):
        logger.info(
            f"Applying special tokens cache patch for Kimi tokenizer: {type(tokenizer)}"
        )
        return _SpecialTokensCachePatcher.patch(tokenizer)

    return tokenizer


def unpatch_tokenizer(tokenizer):
    return _SpecialTokensCachePatcher.unpatch(tokenizer)


def _is_kimi_tiktoken_tokenizer(tokenizer):
    cls = type(tokenizer)
    class_name = cls.__name__
    module_name = cls.__module__ or ""
    return class_name == "TikTokenTokenizer" and "tokenization_kimi" in module_name


class _SpecialTokensCachePatcher:
    _PATCHED_FLAG = "_sglang_special_tokens_patched"
    _CACHED_TOKENS_ATTR = "_sglang_cached_special_tokens"
    _CACHED_IDS_ATTR = "_sglang_cached_special_ids"

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._original_all_special_tokens = (
            tokenizer_cls.all_special_tokens.fget
        )
        tokenizer_cls._original_all_special_ids = tokenizer_cls.all_special_ids.fget
        tokenizer_cls._original_add_special_tokens = tokenizer_cls.add_special_tokens
        tokenizer_cls._original_add_tokens = tokenizer_cls.add_tokens

        patched_all_special_tokens = _make_cached_property(
            cls._CACHED_TOKENS_ATTR, tokenizer_cls._original_all_special_tokens
        )
        patched_all_special_ids = _make_cached_property(
            cls._CACHED_IDS_ATTR, tokenizer_cls._original_all_special_ids
        )

        def patched_add_special_tokens(self, *args, **kwargs):
            assert (
                False
            ), "Cannot modify special tokens after patch. Call unpatch_tokenizer first."

        def patched_add_tokens(self, new_tokens, special_tokens=False):
            assert (
                not special_tokens
            ), "Cannot add special tokens after patch. Call unpatch_tokenizer first."
            return tokenizer_cls._original_add_tokens(
                self, new_tokens, special_tokens=False
            )

        tokenizer_cls.all_special_tokens = patched_all_special_tokens
        tokenizer_cls.all_special_ids = patched_all_special_ids
        tokenizer_cls.add_special_tokens = patched_add_special_tokens
        tokenizer_cls.add_tokens = patched_add_tokens
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)

        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls.all_special_tokens = property(
            tokenizer_cls._original_all_special_tokens
        )
        tokenizer_cls.all_special_ids = property(
            tokenizer_cls._original_all_special_ids
        )
        tokenizer_cls.add_special_tokens = tokenizer_cls._original_add_special_tokens
        tokenizer_cls.add_tokens = tokenizer_cls._original_add_tokens

        del tokenizer_cls._original_all_special_tokens
        del tokenizer_cls._original_all_special_ids
        del tokenizer_cls._original_add_special_tokens
        del tokenizer_cls._original_add_tokens
        delattr(tokenizer_cls, cls._PATCHED_FLAG)

        for attr in [cls._CACHED_TOKENS_ATTR, cls._CACHED_IDS_ATTR]:
            if hasattr(tokenizer, attr):
                delattr(tokenizer, attr)

        logger.info(f"Unpatched special tokens cache for {tokenizer_cls.__name__}")
        return tokenizer


def _make_cached_property(cache_attr, original_fn):
    @property
    def cached_prop(self):
        if getattr(self, cache_attr, None) is None:
            setattr(self, cache_attr, original_fn(self))
        return getattr(self, cache_attr)

    return cached_prop
