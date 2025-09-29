"""Test configuration and fixtures for all tests."""
import sys
import types
import functools
import pytest

# Create a mock cache decorator
def mock_cache(*args, **kwargs):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
    return decorator

# Create a more complete Streamlit mock
class MockStreamlit:
    def __init__(self):
        self.secrets = types.SimpleNamespace(is_local=True)
        self.cache_resource = mock_cache
        self.cache_data = mock_cache
        self.session_state = {}
        
    def __getattr__(self, name):
        # Return a dummy function for any other Streamlit function
        return lambda *args, **kwargs: None

# Mock streamlit before any tests run
sys.modules['streamlit'] = MockStreamlit()
import streamlit as st

@pytest.fixture(autouse=True)
def setup_env():
    """Ensure st is properly set before each test."""
    if not hasattr(st, 'secrets') or not hasattr(st.secrets, 'is_local'):
        st.secrets = types.SimpleNamespace(is_local=True)
    if not hasattr(st, 'cache_resource'):
        st.cache_resource = mock_cache
    if not hasattr(st, 'cache_data'):
        st.cache_data = mock_cache
    if not hasattr(st, 'session_state'):
        st.session_state = {}
    yield
