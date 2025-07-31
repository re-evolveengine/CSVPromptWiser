import streamlit as st

from model.core.llms.gemini_client import GeminiClient
from model.io.model_prefs import ModelPreference
from model.core.llms.gemini_model_provider import GeminiModelProvider


@st.cache_data(show_spinner="🔍 Fetching available Gemini models...")
def get_available_models(api_key: str):
    provider = GeminiModelProvider(api_key)
    return provider.get_usable_model_names()


def model_selector_ui(container, api_key: str):
    model_pref = ModelPreference()
    saved_models = model_pref.get_model_list()
    saved_selected_model = model_pref.get_selected_model_name()
    selected_model = None

    # Step 1: Check if previously selected model exists and prompt to use it
    if saved_selected_model:
        container.markdown("### 🧠 Model Selection")
        container.info(f"📌 A previously selected model was found: `{saved_selected_model}`")
        btn_key = f"use_saved_model_{saved_selected_model}"
        use_saved = container.button("✅ Use saved model", key=btn_key)

        if use_saved:
            try:
                client = GeminiClient(model=saved_selected_model, api_key=api_key, 
                                   generation_config=model_pref.get_generation_config())
                return saved_selected_model, client, model_pref.get_generation_config()
            except Exception as e:
                container.error(f"❌ Failed to create Gemini client: {e}")
                st.stop()

    # Step 2: Show saved model list if available
    if not selected_model:
        if saved_models and (not saved_selected_model or not use_saved):
            if not container.checkbox("📌 Show previously used models", value=True):
                saved_models = []

        fetch_new = not saved_models or container.checkbox("🔄 Fetch latest model list from API",
                                                           value=not bool(saved_models))

        if fetch_new:
            with container.status("🔍 Fetching available models..."):
                model_names = get_available_models(api_key)
                if not model_names:
                    container.error("❌ No usable models found. Please check your API key.")
                    st.stop()
                model_pref.save_model_list(model_names)
        else:
            model_names = saved_models

        if not model_names:
            container.error("❌ No models available to display.")
            st.stop()

        selected_model = container.selectbox(
            "🧠 Select a Gemini model",
            model_names,
            index=model_names.index(saved_selected_model) if saved_selected_model in model_names else 0
        )

    # --- Model generation config UI ---
    container.markdown("#### ⚙️ Model Generation Configuration")
    gen_config = model_pref.get_generation_config()

    temperature = container.slider("🌡️ Temperature", 0.0, 1.0, gen_config.get("temperature", 0.2), 0.01)
    top_k = container.slider("🔢 Top-K", 1, 100, gen_config.get("top_k", 40), 1)
    top_p = container.slider("🎯 Top-P", 0.0, 1.0, gen_config.get("top_p", 1.0), 0.01)

    updated_config = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p
    }

    # Save config (shared for all models)
    if container.button("💾 Save Generation Settings"):
        model_pref.save_generation_config(updated_config)
        container.success("✅ Generation settings saved.")

    # Save selected model if changed
    if selected_model != saved_selected_model:
        model_pref.save_selected_model_name(selected_model)
        container.success(f"✅ Model `{selected_model}` saved.")

    # ✅ Create GeminiClient here
    try:
        client = GeminiClient(model=selected_model, api_key=api_key, generation_config=updated_config)
    except Exception as e:
        container.error(f"❌ Failed to create Gemini client: {e}")
        st.stop()

    if selected_model:
        return selected_model, client, updated_config
    
    return None, None, None
