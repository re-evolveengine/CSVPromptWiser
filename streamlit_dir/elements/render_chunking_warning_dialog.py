import streamlit as st
from typing import Callable

def render_chunking_warning_dialog(on_confirm_callback: Callable[[], None]):
    """
    Displays a modal confirmation dialog using st.dialog.

    If the user confirms, it executes the provided callback function.

    Args:
        on_confirm_callback: A function to run when the user confirms.
    """
    # The dialog is only created and shown if the session_state flag is True
    if st.session_state.get('show_chunking_warning'):
        # Use st.dialog to create a modal pop-up
        with st.dialog("⚠️ Warning: Existing Data Found!"):
            st.error(
                "Your database already contains processed results. Re-chunking will create new unique IDs, and you will lose the ability to link your existing results to the original data."
            )
            st.info("It is highly recommended that you export your current results first before proceeding.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed and Discard Links", type="primary"):
                    # Execute the callback function passed as an argument
                    on_confirm_callback()
                    # Turn off the flag and rerun to close the dialog
                    st.session_state.show_chunking_warning = False
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    # Just turn off the flag and rerun to close the dialog
                    st.session_state.show_chunking_warning = False
                    st.rerun()