import streamlit as st
from typing import Callable

# Define the dialog once (do not open it here)
@st.dialog("⚠️ Warning: Existing Data Found!")
def chunking_warning_dialog_body(on_confirm_callback: Callable[[], None]):
    st.error(
        "Your database already contains processed results. Re-chunking will create new unique IDs, and you will lose the ability to link your existing results to the original data."
    )
    st.info("It is highly recommended that you export your current results first before proceeding.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Proceed and Discard Links", type="primary"):
            on_confirm_callback()
            st.session_state.show_chunking_warning = False
            st.rerun()
    with col2:
        if st.button("Cancel"):
            st.session_state.show_chunking_warning = False
            st.rerun()


def show_chunking_warning_dialog(on_confirm_callback: Callable[[], None]):
    """Call this to open the dialog if the session_state flag is set."""
    if st.session_state.get("show_chunking_warning"):
        chunking_warning_dialog_body(on_confirm_callback)
