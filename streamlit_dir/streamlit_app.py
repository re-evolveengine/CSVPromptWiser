import streamlit as st
from model.utils.constants import APP_NAME
from streamlit_dir.side_bar import cwp_sidebar

# Place this near the top of your main script (after imports)
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 400px;
            width: 500px;
        }
    </style>
    """,
    unsafe_allow_html=True
)



def main():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🤖",
        initial_sidebar_state="expanded",
        layout="wide")

    st.title(f"🤖 {APP_NAME}")

    # Get sidebar input
    api_key, selected_model, uploaded_file,df,prompt = cwp_sidebar()

    # # Debug info or placeholder content
    # if uploaded_file:
    #     st.success(f"Uploaded file: {uploaded_file.name}")
    # if prompt:
    #     st.info(f"Prompt entered: {prompt[:50]}...")
    # if start:
    #     st.warning("🚧 Processing logic will go here next.")

if __name__ == "__main__":
    main()
