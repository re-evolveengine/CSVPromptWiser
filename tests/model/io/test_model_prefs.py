import os
import tempfile
import pytest
from model.io.model_prefs import ModelPreference


class TestModelPreference:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Create a temporary file for testing
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
        self.model_prefs = ModelPreference(db_path=self.temp_db_path)
        yield
        # Cleanup
        os.close(self.temp_db_fd)
        try:
            os.unlink(self.temp_db_path)
        except FileNotFoundError:
            pass

    def test_save_and_get_total_tokens_used(self):
        # Test saving and retrieving token count
        test_tokens = 1500
        self.model_prefs.save_remaining_total_tokens(test_tokens)
        assert self.model_prefs.get_remaining_total_tokens() == test_tokens

    def test_get_total_tokens_used_defaults_to_zero(self):
        # Test that get_total_tokens_used returns 0 when no value is set
        assert self.model_prefs.get_remaining_total_tokens() == 0

    def test_save_total_tokens_used_overwrites_previous_value(self):
        # Test that saving a new token count overwrites the previous one
        self.model_prefs.save_remaining_total_tokens(1000)
        self.model_prefs.save_remaining_total_tokens(2000)
        assert self.model_prefs.get_remaining_total_tokens() == 2000

    def test_token_count_persists_across_instances(self):
        # Test that token count persists when creating a new ModelPreference instance
        test_tokens = 3000
        self.model_prefs.save_remaining_total_tokens(test_tokens)

        # Create a new instance with the same DB path
        new_instance = ModelPreference(db_path=self.temp_db_path)
        assert new_instance.get_remaining_total_tokens() == test_tokens
