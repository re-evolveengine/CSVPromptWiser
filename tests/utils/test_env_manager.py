import os
import tempfile
import pytest
from pathlib import Path
import streamlit as st

from env_detection import EnvManager, SecretsWriteError


@pytest.fixture
def fake_secrets_cloud(monkeypatch):
    """Provide a fake st.secrets dict for cloud mode."""
    secrets_dict = {
        "is_local": False,
        "GEMINI_API_KEY": "cloud-gemini"
    }
    monkeypatch.setattr(st, "secrets", secrets_dict)
    return secrets_dict


@pytest.fixture
def fake_secrets_local(monkeypatch):
    """Provide a fake st.secrets dict for local mode."""
    secrets_dict = {"is_local": True}
    monkeypatch.setattr(st, "secrets", secrets_dict)
    return secrets_dict


@pytest.fixture
def temp_env_local(monkeypatch, tmp_path):
    """
    Create a temporary `.env` file in a tmp path and
    patch cwd so EnvManager writes/reads in isolation.
    """
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_API_KEY=local-gemini\n")

    # Patch current working dir to tmp_path
    monkeypatch.chdir(tmp_path)
    return env_path


def test_get_is_local_true(fake_secrets_local, temp_env_local):
    env = EnvManager("CSV PromptWiser")
    assert env.get_is_local() is True


def test_get_is_local_false(fake_secrets_cloud):
    env = EnvManager("CSV PromptWiser")
    assert env.get_is_local() is False


def test_get_api_key_local(fake_secrets_local, temp_env_local):
    env = EnvManager("CSV PromptWiser")
    key = env.get_api_key("GEMINI_API_KEY")
    assert key == "local-gemini"


def test_get_api_key_cloud(fake_secrets_cloud):
    env = EnvManager("CSV PromptWiser")
    key = env.get_api_key("GEMINI_API_KEY")
    assert key == "cloud-gemini"


def test_get_api_key_missing_local(fake_secrets_local, temp_env_local):
    os.environ.pop("MISSING_KEY", None)
    env = EnvManager("CSV PromptWiser")
    with pytest.raises(KeyError):
        env.get_api_key("MISSING_KEY")


def test_get_api_key_missing_cloud(fake_secrets_cloud):
    fake_secrets_cloud.pop("MISSING_KEY", None)
    env = EnvManager("CSV PromptWiser")
    with pytest.raises(KeyError):
        env.get_api_key("MISSING_KEY")


def test_set_api_key_local_updates_env(fake_secrets_local, temp_env_local):
    env = EnvManager("CSV PromptWiser")
    env.set_api_key("NEW_KEY", "value123")

    env_content = temp_env_local.read_text().splitlines()
    assert any(line.startswith("NEW_KEY=value123") for line in env_content)
    assert os.getenv("NEW_KEY") == "value123"


def test_set_api_key_cloud_raises(fake_secrets_cloud):
    env = EnvManager("CSV PromptWiser")
    with pytest.raises(SecretsWriteError):
        env.set_api_key("GEMINI_API_KEY", "should-fail")


def test_get_base_app_dir_local(fake_secrets_local, temp_env_local):
    env = EnvManager("CSV PromptWiser")
    path = env.get_base_app_dir()
    assert path.startswith(os.path.expanduser("~"))
    assert path.endswith("CSV PromptWiser")


def test_get_base_app_dir_cloud(fake_secrets_cloud):
    env = EnvManager("CSV PromptWiser")
    path = env.get_base_app_dir()
    assert path.startswith(tempfile.gettempdir())
    assert path.endswith("CSV PromptWiser")
