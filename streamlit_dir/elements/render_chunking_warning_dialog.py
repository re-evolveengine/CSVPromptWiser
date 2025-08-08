import streamlit as st
from typing import Callable

def render_chunking_warning_dialog(on_confirm_callback: Callable[[], None]):
    """
    Displays a generic confirmation dialog.

    If the user confirms, it executes the provided callback function.

    Args:
        on_confirm_callback: A function to run when the user confirms.
    """
    if st.session_state.get('show_chunking_warning'):
        st.error("⚠️ **Warning: Existing Data Found!**")
        st.warning(
            "Your database already contains processed results. Re-chunking will create new unique IDs, and you will lose the ability to link your existing results to the original data."
        )
        st.info("It's highly recommended that you export your current results first before proceeding.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Proceed and Discard Links", type="primary"):
                # Execute the callback function passed as an argument
                on_confirm_callback()
                st.session_state.show_chunking_warning = False
                st.rerun()
        with col2:
            if st.button("Cancel"):
                st.session_state.show_chunking_warning = False
                st.rerun()