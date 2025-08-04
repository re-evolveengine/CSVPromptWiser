import streamlit as st
from streamlit_dir.prompt_pref import PromptPreference

def prompt_input_ui(container):
    prompt_pref = PromptPreference()
    saved_prompt = prompt_pref.load_prompt()
    saved_response = prompt_pref.load_example_response()

    prompt = container.text_area("💬 Enter your prompt", value=saved_prompt, height=200)
    response_example = container.text_area("🧾 Enter example response (one row’s output)", value=saved_response, height=150)

    if (prompt != saved_prompt or response_example != saved_response) and container.button("💾 Save Prompt & Example"):
        prompt_pref.save_prompt(prompt)
        prompt_pref.save_example_response(response_example)
        container.success("✅ Prompt and example response saved")

    return prompt, response_example

