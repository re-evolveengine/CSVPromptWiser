import os

import dotenv
import streamlit as st

from model.core.llms.gemini_model_provider import GeminiModelProvider
from model.io.model_prefs import ModelPreference


@st.cache_data(show_spinner="🔍 Fetching available Gemini models...")
def get_available_models(api_key: str):
    provider = GeminiModelProvider(api_key)
    return provider.get_usable_model_names()

def load_api_key_ui(container) -> str:
    dotenv.load_dotenv()
    saved_key = os.getenv("KEY")

    api_key = container.text_input(
        "🔑 Enter your Gemini API Key",
        value=saved_key if saved_key else "",
        type="password",
        help="Stored in your .env file"
    )

    if api_key and api_key != saved_key:
        if container.checkbox("💾 Save this key"):
            dotenv.set_key(dotenv.find_dotenv(), "KEY", api_key)
            container.success("✅ API key saved")
    elif api_key:
        container.info("✅ Using Saved API key")

    return api_key


def model_selector_ui(container, api_key: str) -> str:
    model_pref = ModelPreference()
    saved_model = model_pref.get_model_name()

    if saved_model:
        container.info(f"✅ Using saved model: `{saved_model}`")
        fetch = container.checkbox("🔄 Fetch new model list?", value=False)
        if not fetch:
            return saved_model

    model_names = get_available_models(api_key)

    if not model_names:
        container.error("❌ No usable models found. Please check your API key.")
        st.stop()

    selected_model = container.selectbox("🧠 Select a Gemini model", model_names)

    if selected_model != saved_model:
        model_pref.save_model_name(selected_model)
        container.success(f"✅ Model `{selected_model}` saved.")

    return selected_model
