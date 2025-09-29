import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from platformdirs import user_data_dir


class SecretsWriteError(Exception):
    """Raised when attempting to write secrets in a read-only environment."""
    pass


class EnvManager:
    """
    Manage environment detection and API key retrieval for local and cloud Streamlit apps.

    - Local: Keys stored in `.env`, writeable via set_api_key().
    - Cloud: Keys sourced from st.secrets (read-only).
    - Environment detection based solely on st.secrets["is_local"].
    """

    def __init__(self, app_name: str):
        """
        :param app_name: Name of the application, used for local storage paths.
        """
        self.app_name = app_name
        load_dotenv(override=True)  # Load .env file if present

    def get_is_local(self) -> bool:
        """Return True if running locally, False if on Cloud."""
        try:
            # Try dictionary access first (for real Streamlit secrets)
            if hasattr(st.secrets, "__getitem__"):
                return bool(st.secrets["is_local"])
            # Fall back to attribute access (for SimpleNamespace mocks)
            return bool(st.secrets.is_local)

        except (KeyError, AttributeError):
            raise KeyError(
                "`is_local` not found in secrets. Please add it to `.streamlit/secrets.toml` locally "
                "or in Streamlit Cloud secrets dashboard."
            )

    def get_api_key(self, key_name: str) -> str:
        """Retrieve API key depending on environment."""
        if self.get_is_local():
            value = os.getenv(key_name)
        else:
            value = st.secrets.get(key_name)

        if not value:
            raise KeyError(
                f"API key '{key_name}' not found for environment: "
                f"{'local (.env)' if self.get_is_local() else 'cloud (st.secrets)'}"
            )
        return value

    def set_api_key(self, key_name: str, key_value: str):
        """
        Set API key depending on environment.
        Local: Writes to `.env` file.
        Cloud: Raises SecretsWriteError because secrets are read-only.
        """
        if not self.get_is_local():
            raise SecretsWriteError(
                f"Cannot set '{key_name}' in cloud environment. "
                f"Secrets must be updated via Streamlit Cloud dashboard."
            )

        # Update .env file
        env_path = Path(".env")
        # Preserve existing env variables from file
        env_vars = {}
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        env_vars[k] = v

        env_vars[key_name] = key_value

        with open(env_path, "w", encoding="utf-8") as f:
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")

        os.environ[key_name] = key_value  # Ensure it's in current process env

    def get_base_app_dir(self) -> str:
        """Return base application directory depending on environment."""
        if self.get_is_local():
            return user_data_dir(self.app_name, appauthor=False)
        return os.path.join(tempfile.gettempdir(), self.app_name)
