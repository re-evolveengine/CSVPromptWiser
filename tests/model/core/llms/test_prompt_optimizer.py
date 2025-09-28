# tests/model/core/utils/test_prompt_optimizer.py
import pytest
import pandas as pd
import types
import streamlit as st
from types import SimpleNamespace
from unittest.mock import patch

# Mock Streamlit secrets
st.secrets = types.SimpleNamespace()
st.secrets.is_local = True

from model.core.llms.prompt_optimizer import PromptOptimizer


@pytest.fixture
def df_example():
    return pd.DataFrame({"A": ["foo"], "B": ["bar"]})


@pytest.fixture
def fake_encoding():
    # Simple fake encoding object
    enc = SimpleNamespace()
    enc.encode = lambda text: list(text)  # token count = string length
    return enc


def test_get_safe_token_limit_matches_prefix(monkeypatch):
    monkeypatch.setattr(
        "model.core.llms.prompt_optimizer.SAFE_PROMPT_LIMITS",
        {"gpt-4": 123, "default": 999}
    )
    opt = PromptOptimizer("gpt-4-32k")
    assert opt._get_safe_token_limit() == 123


def test_get_safe_token_limit_uses_default(monkeypatch):
    monkeypatch.setattr(
        "model.core.llms.prompt_optimizer.SAFE_PROMPT_LIMITS",
        {"default": 555}
    )
    opt = PromptOptimizer("unknown-model")
    assert opt._get_safe_token_limit() == 555


def test_format_row_formats_dataframe_row(df_example):
    row_text = PromptOptimizer._format_row(df_example.iloc[0])
    assert "Row 1:" in row_text
    assert "- A: foo" in row_text
    assert "- B: bar" in row_text


@patch("model.core.llms.prompt_optimizer.tiktoken.encoding_for_model")
def test_find_optimal_row_number_success(mock_encoding_for_model, fake_encoding, df_example, monkeypatch):
    # Always return fake encoding
    mock_encoding_for_model.return_value = fake_encoding
    monkeypatch.setattr(
        "model.core.llms.prompt_optimizer.SAFE_PROMPT_LIMITS",
        {"default": 100}
    )

    opt = PromptOptimizer("anymodel")
    result = opt.find_optimal_row_number("Prompt", df_example, "resp", usage_ratio=1.0)
    assert isinstance(result, int)
    assert 1 <= result <= 100


@patch("model.core.llms.prompt_optimizer.tiktoken.encoding_for_model", side_effect=KeyError)
@patch("model.core.llms.prompt_optimizer.tiktoken.get_encoding")
def test_find_optimal_row_number_falls_back_on_keyerror(mock_get_encoding, mock_enc_for_model, df_example):
    mock_get_encoding.return_value = SimpleNamespace(encode=lambda s: [1, 2])
    opt = PromptOptimizer("missing")
    out = opt.find_optimal_row_number("Prompt", df_example, "resp")
    assert isinstance(out, int)


def test_find_optimal_row_number_tokens_per_row_zero(df_example, monkeypatch):
    opt = PromptOptimizer("model")
    fake_enc = SimpleNamespace(encode=lambda s: [])
    monkeypatch.setattr(
        "model.core.llms.prompt_optimizer.tiktoken.encoding_for_model",
        lambda m: fake_enc
    )
    monkeypatch.setattr(
        "model.core.llms.prompt_optimizer.SAFE_PROMPT_LIMITS",
        {"default": 50}
    )
    result = opt.find_optimal_row_number("Prompt", df_example, "resp")
    assert result == 10


def test_find_optimal_row_number_handles_exception(monkeypatch, df_example):
    opt = PromptOptimizer("model")
    monkeypatch.setattr(opt, "_get_safe_token_limit", lambda: 1/0)
    result = opt.find_optimal_row_number("Prompt", df_example, "resp")
    assert result == 10


@patch("model.core.llms.prompt_optimizer.tiktoken.encoding_for_model")
def test_calculate_used_tokens_basic(mock_encoding_for_model, fake_encoding, df_example):
    mock_encoding_for_model.return_value = fake_encoding
    opt = PromptOptimizer("model")
    result = opt.calculate_used_tokens("prompt", df_example, "resp", num_rows=2)
    assert isinstance(result, int)
    assert result > 0


@patch("model.core.llms.prompt_optimizer.tiktoken.encoding_for_model", side_effect=KeyError)
@patch("model.core.llms.prompt_optimizer.tiktoken.get_encoding")
def test_calculate_used_tokens_fallback(mock_get_encoding, mock_enc_for_model, df_example):
    mock_get_encoding.return_value = SimpleNamespace(encode=lambda s: [1])
    opt = PromptOptimizer("whatever")
    assert opt.calculate_used_tokens("prompt", df_example, "resp", 1) > 0


def test_calculate_used_tokens_handles_exception(df_example):
    opt = PromptOptimizer("model")
    with patch(
        "model.core.llms.prompt_optimizer.tiktoken.encoding_for_model",
        side_effect=Exception("fail")
    ):
        assert opt.calculate_used_tokens("prompt", df_example, "resp", 1) == 0


def test_calculate_max_chunks_with_quota_basic(monkeypatch, df_example):
    opt = PromptOptimizer("model")
    monkeypatch.setattr(opt, "calculate_used_tokens", lambda **kwargs: 5)
    result = opt.calculate_max_chunks_with_quota("p", df_example, "r", 10, token_quota=55)
    assert result == 11  # 55 // 5


def test_calculate_max_chunks_with_quota_zero_tokens(monkeypatch, df_example):
    opt = PromptOptimizer("model")
    monkeypatch.setattr(opt, "calculate_used_tokens", lambda **kwargs: 0)
    assert opt.calculate_max_chunks_with_quota("p", df_example, "r", 10, 100) == 0


def test_calculate_max_chunks_with_quota_handles_exception(monkeypatch, df_example):
    opt = PromptOptimizer("model")
    monkeypatch.setattr(opt, "calculate_used_tokens", lambda **kwargs: 1/0)
    assert opt.calculate_max_chunks_with_quota("p", df_example, "r", 10, 100) == 0
