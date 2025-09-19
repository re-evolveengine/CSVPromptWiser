import json
import pytest
from pathlib import Path

from model.io.prompt_pref import PromptPreference


# Full path import for the class under test
# Adjust this if your actual package/module path differs


@pytest.fixture
def temp_prefs_file(tmp_path, monkeypatch):
    file_path = tmp_path / "prefs.json"
    monkeypatch.setattr(
        "model.io.prompt_pref.PROMPT_PREF_PATH",
        file_path
    )
    return file_path


def test_save_and_load_prompt(temp_prefs_file):
    pref = PromptPreference()
    pref.save_prompt("Hello!")
    assert json.loads(temp_prefs_file.read_text())["prompt"] == "Hello!"

    loaded = pref.load_prompt()
    assert loaded == "Hello!"


def test_load_prompt_when_no_file(temp_prefs_file):
    pref = PromptPreference()
    assert pref.load_prompt() == ""  # No file yet => default empty string


def test_save_and_load_example_response(temp_prefs_file):
    pref = PromptPreference()
    pref.save_example_response("Example text")
    data = json.loads(temp_prefs_file.read_text())
    assert data["example_response"] == "Example text"
    assert pref.load_example_response() == "Example text"


def test_load_example_response_when_no_file(temp_prefs_file):
    pref = PromptPreference()
    assert pref.load_example_response() == ""


def test__load_all_returns_dict_when_file_exists(temp_prefs_file):
    temp_prefs_file.write_text(json.dumps({"a": 1}))
    pref = PromptPreference()
    result = pref._load_all()
    assert result == {"a": 1}


def test__save_all_creates_file(temp_prefs_file):
    pref = PromptPreference()
    pref._save_all({"foo": "bar"})
    saved_content = json.loads(temp_prefs_file.read_text())
    assert saved_content == {"foo": "bar"}


def test_init_creates_parent_dir(monkeypatch, tmp_path):
    fake_path = tmp_path / "nested" / "prefs.json"
    monkeypatch.setattr("model.io.prompt_pref.PROMPT_PREF_PATH", fake_path)
    PromptPreference()
    assert fake_path.parent.exists()
