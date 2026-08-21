import hashlib
import sys

import pytest

from ncp_olmo_eval.gsm8k_eval import (
    _append_extra_site_packages,
    _apply_prompt_transport,
    _ensure_chat_template,
)


class _ChatTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages == [{"role": "user", "content": "Question: 1 + 1?"}]
        return "<user>Question: 1 + 1?</user><assistant>"


def test_apply_prompt_transport_raw_completion_preserves_prompts():
    prompts = ["Question: 1 + 1?"]

    rendered, template_sha256 = _apply_prompt_transport(
        prompts,
        object(),
        "raw_completion",
    )

    assert rendered == prompts
    assert rendered is not prompts
    assert template_sha256 is None


def test_apply_prompt_transport_chat_template_renders_one_user_message():
    rendered, template_sha256 = _apply_prompt_transport(
        ["Question: 1 + 1?"],
        _ChatTokenizer(),
        "chat_template",
    )

    assert rendered == ["<user>Question: 1 + 1?</user><assistant>"]
    assert template_sha256 == hashlib.sha256(b"{{ messages }}").hexdigest()


def test_apply_prompt_transport_chat_template_requires_template():
    tokenizer = _ChatTokenizer()
    tokenizer.chat_template = None

    with pytest.raises(ValueError, match="tokenizer has no template"):
        _apply_prompt_transport(
            ["Question: 1 + 1?"],
            tokenizer,
            "chat_template",
        )


def test_ensure_chat_template_reads_separate_jinja_file(tmp_path):
    tokenizer = _ChatTokenizer()
    tokenizer.chat_template = None
    template_path = tmp_path / "chat_template.jinja"
    template_path.write_text("{{ messages }}", encoding="utf-8")

    source = _ensure_chat_template(tokenizer, str(tmp_path))

    assert source == str(template_path.resolve())
    assert tokenizer.chat_template == "{{ messages }}"


def test_ensure_chat_template_preserves_loaded_template(tmp_path):
    tokenizer = _ChatTokenizer()

    source = _ensure_chat_template(tokenizer, str(tmp_path))

    assert source == "tokenizer.chat_template"
    assert tokenizer.chat_template == "{{ messages }}"


def test_append_extra_site_packages_appends_existing_directory(tmp_path):
    path = str(tmp_path.resolve())

    appended = _append_extra_site_packages(path)

    assert appended == [path]
    assert sys.path[-1] == path
    sys.path.remove(path)


def test_append_extra_site_packages_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="extra site-packages directory not found"):
        _append_extra_site_packages(str(tmp_path / "missing"))
