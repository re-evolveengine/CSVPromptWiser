import streamlit as st
from streamlit_dir.prompt_pref import PromptPreference

def prompt_input_ui(container):
    prompt_pref = PromptPreference()
    saved_prompt = prompt_pref.load_prompt()

    prompt = container.text_area("💬 Enter your prompt", value=saved_prompt, height=200)

    if prompt and prompt != saved_prompt:
        if container.button("💾 Save Prompt"):
            prompt_pref.save_prompt(prompt)
            container.success("✅ Prompt saved")

    return prompt
