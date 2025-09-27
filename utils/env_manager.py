import streamlit as st
import tempfile
import os

class EnvManager:
    """
    Environment + API key manager for CSV PromptWiser.
    Uses st.secrets as the sole source of truth.
    """

    def __init__(self, app_name: str):
        self.app_name = app_name

    # -------- Environment --------
    def set_is_local(self, value: bool):
        """
        Set the environment flag in st.secrets.
        Note: In Streamlit, secrets are typically managed via .streamlit/secrets.toml (local)
              or Cloud dashboard, not modified at runtime. This is for local simulation only.
        """
        st.secrets["is_local"] = value

    def get_is_local(self) -> bool:
        """
        Return True if running in local mode.
        """
        return bool(st.secrets.get("is_local", False))

    # -------- API Keys --------
    def set_api_key(self, key_name: str, value: str):
        """
        Set an API key in st.secrets.
        This is only meaningful in local tests with secrets.toml,
        as Cloud secrets are set via the dashboard.
        """
        st.secrets[key_name] = value

    def get_api_key(self, key_name: str) -> str:
        """
        Get an API key from st.secrets.
        Raises KeyError if not present.
        """
        value = st.secrets.get(key_name)
        if not value:
            raise KeyError(
                f"API key '{key_name}' not found in secrets. "
                f"{'Check .streamlit/secrets.toml' if self.get_is_local() else 'Check Streamlit Cloud dashboard'}."
            )
        return value

    # -------- Base directory --------
    def get_base_app_dir(self) -> str:
        """
        Return a base application directory depending on environment.
        Local: ~/Documents/<app_name>
        Cloud: /tmp/<app_name>
        """
        if self.get_is_local():
            return os.path.join(os.path.expanduser("~"), "Documents", self.app_name)
        return os.path.join(tempfile.gettempdir(), self.app_name)
