import streamlit as st

from utils.llms_list import LLMProvider

def llm_selector(container):

    llm_choice = container.selectbox(
        "Select a model family",
        [llm.display_name for llm in LLMProvider]
    )

    # Determine which enum value was selected
    selected_llm = next(llm for llm in LLMProvider if llm.display_name == llm_choice)

    if not selected_llm.is_available:
        st.info(f"🚧 {selected_llm.value} is coming soon. Stay tuned!", icon="⏳")
        return None
    else:
        st.success(f"✅ {selected_llm.value} is ready to use!")
        return selected_llm.value

