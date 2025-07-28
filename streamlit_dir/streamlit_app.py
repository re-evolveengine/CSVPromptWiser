import streamlit as st
from model.utils.constants import APP_NAME
from streamlit_dir.side_bar import cwp_sidebar

# --- Style: Widen sidebar ---
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 300px;
            width: 350px;
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
        layout="wide"
    )

    st.title(f"🤖 {APP_NAME} Dashboard")

    # --- Sidebar interaction ---
    api_key, model_name, df, chunk_file_path, chunk_summary, prompt = cwp_sidebar()

    # --- Chunk Summary ---
    if chunk_summary:
        st.subheader("📦 Chunk Summary")
        for key, value in chunk_summary.items():
            st.markdown(f"- **{key.replace('_', ' ').capitalize()}**: {value}")

    # --- Dataset Preview ---
    if df is not None:
        st.subheader("📊 Dataset Preview")
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(5), use_container_width=True)

    # --- Prompt Echo ---
    if prompt:
        st.subheader("💬 Your Prompt")
        st.code(prompt, language="markdown")

if __name__ == "__main__":
    main()
