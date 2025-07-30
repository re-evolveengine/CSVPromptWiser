import os
import dotenv
import streamlit as st

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
