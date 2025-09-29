import streamlit as st

from utils.constants import APP_NAME
from utils.env_manager import EnvManager, SecretsWriteError


def load_api_key_ui(container) -> str:
    """
    Load the API key via UI, using EnvManager for environment-aware
    get/set operations.
    """
    env_manager = EnvManager(app_name=APP_NAME)
    # is_local = env_manager.get_is_local()
    #
    # # Try to load saved key via EnvManager
    # try:
    #     saved_key = env_manager.get_api_key("GEMINI_API_KEY")
    # except KeyError:
    #     saved_key = ""

    saved_key = env_manager.get_api_key("GEMINI_API_KEY")

    api_key = container.text_input(
        "🔑 Enter your API Key",
        value=saved_key,
        type="password",
        help=(
            "In local mode: Stored in your .env file.\n"
            "In cloud mode: Stored in Streamlit secrets."
        ),
    )

    # Disable saving option in cloud mode
    save_key_checkbox = container.checkbox(
        "💾 Save this key",
        disabled=not env_manager.get_is_local(),
        help="Saving is disabled in cloud mode."
    )

    if api_key and api_key != saved_key:
        if save_key_checkbox:
            try:
                env_manager.set_api_key("GEMINI_API_KEY", api_key)
                container.success("✅ API key saved")
            except SecretsWriteError as e:
                container.error(f"❌ Cannot save key here. {str(e)}")
    elif api_key:
        container.info("✅ Using Saved API key")

    return api_key
