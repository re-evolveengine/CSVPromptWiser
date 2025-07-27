import streamlit as st

from model.core.llms.gemini_model_provider import GeminiModelProvider
from model.io.model_prefs import ModelPreference


@st.cache_data(show_spinner="🔍 Fetching available Gemini models...")
def get_available_models(api_key: str):
    provider = GeminiModelProvider(api_key)
    return provider.get_usable_model_names()

def model_selector_ui(api_key: str) -> str:
    """Streamlit UI version of model selection with saved model support."""
    model_pref = ModelPreference()
    saved_model = model_pref.get_model_name()

    if saved_model:
        st.sidebar.info(f"✅ Using saved model: `{saved_model}`")
        fetch = st.sidebar.checkbox("🔄 Fetch new model list?", value=False)

        if not fetch:
            return saved_model

    # Fetch model list
    model_names = get_available_models(api_key)

    if not model_names:
        st.sidebar.error("❌ No usable models found. Please check your API key.")
        st.stop()

    selected_model = st.sidebar.selectbox("🧠 Select a Gemini model", model_names)

    if selected_model != saved_model:
        model_pref.save_model_name(selected_model)
        st.sidebar.success(f"✅ Model `{selected_model}` saved.")

    return selected_model



import streamlit as st
import os
import dotenv

def load_api_key_ui() -> str:
    """Streamlit version: Load or input Gemini API key."""
    dotenv.load_dotenv()
    saved_key = os.getenv("KEY")

    # Sidebar input with default value (optional)
    api_key = st.sidebar.text_input(
        "🔑 Enter your Gemini API Key",
        value=saved_key if saved_key else "",
        type="password",
        help="Stored in your .env file"
    )

    if api_key and api_key != saved_key:
        # Ask if user wants to save the key
        if st.sidebar.checkbox("💾 Save this key"):
            dotenv.set_key(dotenv.find_dotenv(), "KEY", api_key)
            st.sidebar.success("✅ API key saved")
    elif api_key:
        st.sidebar.info("✅ Using Saved API key")

    return api_key

