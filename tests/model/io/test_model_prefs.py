import os
import tempfile
import shutil
import pytest
from model.io.model_prefs import ModelPreference


class TestModelPreference:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        # Create a temporary directory for the test database
        self.temp_dir = tmp_path / "test_db"
        self.temp_dir.mkdir()
        self.db_path = str(self.temp_dir / "test_db.db")
        self.model_prefs = ModelPreference(db_path=self.db_path)
        yield
        # Cleanup - remove the entire directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_get_total_tokens_used(self):
        # Test saving and retrieving token count
        test_tokens = 1500
        self.model_prefs.remaining_total_tokens = test_tokens
        assert self.model_prefs.remaining_total_tokens == test_tokens

    def test_get_total_tokens_used_defaults_to_zero(self):
        # Test that remaining_total_tokens returns 0 when no value is set
        assert self.model_prefs.remaining_total_tokens == 0

    def test_save_total_tokens_used_overwrites_previous_value(self):
        # Test that setting a new token count overwrites the previous one
        self.model_prefs.remaining_total_tokens = 1000
        self.model_prefs.remaining_total_tokens = 2000
        assert self.model_prefs.remaining_total_tokens == 2000

    def test_token_count_persists_across_instances(self):
        # Test that token count persists when creating a new ModelPreference instance
        test_tokens = 3000
        self.model_prefs.remaining_total_tokens = test_tokens

        # Create a new instance with the same DB path
        new_instance = ModelPreference(db_path=self.db_path)
        assert new_instance.remaining_total_tokens == test_tokens

    def test_model_name_property(self):
        # Test model name getter and setter
        test_model = "test-model-1"
        self.model_prefs.selected_model_name = test_model
        assert self.model_prefs.selected_model_name == test_model

    def test_model_list_property(self):
        # Test model list getter and setter
        test_models = ["model1", "model2", "model3"]
        self.model_prefs.model_list = test_models
        assert self.model_prefs.model_list == test_models

    def test_generation_config_property(self):
        # Test generation config getter and setter
        test_config = {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9
        }
        self.model_prefs.generation_config = test_config
        assert self.model_prefs.generation_config == test_config

    def test_generation_config_defaults(self):
        # Test generation config defaults
        default_config = {
            "temperature": 0.2,
            "top_k": 40,
            "top_p": 1.0
        }
        assert self.model_prefs.generation_config == default_config

    def test_delete_model_name(self):
        # Test deleting a model name
        self.model_prefs.selected_model_name = "test-model"
        del self.model_prefs.selected_model_name
        assert self.model_prefs.selected_model_name == ""

    def test_delete_model_list(self):
        # Test deleting model list
        self.model_prefs.model_list = ["model1", "model2"]
        del self.model_prefs.model_list
        assert self.model_prefs.model_list == []
