import os
from pathlib import Path

import dotenv
import streamlit as st

from model.core.llms.gemini_model_provider import GeminiModelProvider
from model.io.model_prefs import ModelPreference
from streamlit_dir.prompt_pref import PromptPreference


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


def prompt_input_ui(container):
    prompt_pref = PromptPreference()
    saved_prompt = prompt_pref.load_prompt()

    prompt = container.text_area("💬 Enter your prompt", value=saved_prompt, height=200)

    if prompt and prompt != saved_prompt:
        if container.button("💾 Save Prompt"):
            prompt_pref.save_prompt(prompt)
            container.success("✅ Prompt saved")

    return prompt


import streamlit as st
from typing import Optional
import pandas as pd

from streamlit_dir.stramlit_dataset_handler import StreamlitDatasetHandler


def handle_dataset_upload_or_load() -> Optional[pd.DataFrame]:
    handler = StreamlitDatasetHandler()

    # Check if a file has already been saved
    saved_filename = st.session_state.get("saved_filename") or handler.get_saved_file_name()

    if saved_filename and not st.session_state.get("upload_new_file"):
        st.success(f"📁 Using saved file: `{saved_filename}`")

        if st.button("📤 Upload a new file?"):
            st.session_state["upload_new_file"] = True
            st.rerun()

        df = handler.load_saved_file(saved_filename)  # <-- FIXED: load from disk
        return df

    # Otherwise show uploader
    uploaded_file = st.file_uploader("📂 Upload CSV or Parquet", type=["csv", "parquet"])
    df = handler.load_from_upload(uploaded_file)

    if df is not None:
        if st.button("💾 Save file to disk"):
            saved_path = handler.save_uploaded_file()
            st.session_state["saved_filename"] = Path(saved_path).name  # store just the name
            st.session_state["upload_new_file"] = False
            st.success(f"✅ File saved: `{saved_path}`")
            st.rerun()

    return df



