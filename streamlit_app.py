import sys
import os
import streamlit as st
import plotly.graph_objects as go

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from model.utils.constants import APP_NAME
from streamlit_dir.side_bar import cwp_sidebar
from streamlit_dir.ui.chunk_processor_panel import process_chunks_ui


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    initial_sidebar_state="expanded",
    layout="wide"
)

st.markdown(
    """
    <style>
        /* Main app container */
        .stApp {
            max-width: 100% !important;
            overflow-x: hidden !important;
        }

        /* Main content area */
        .main .block-container {
            max-width: 100% !important;
            padding: 1rem 1rem 1rem 1rem !important;
        }

        /* Content wrapper */
        .main > div {
            max-width: 100% !important;
        }

        /* Text and code blocks */
        .stMarkdown, .stText, .stCodeBlock, .stDataFrame {
            max-width: 100% !important;
        }

        /* Code and text areas */
        .stCodeBlock pre, .stTextArea textarea, pre, code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            max-width: 100% !important;
        }

        /* Tables */
        .stDataFrame {
            display: block;
            overflow-x: auto;
            width: 100% !important;
        }

        /* Fix for Streamlit's dynamic content */
        [data-testid="stAppViewContainer"] > .main > div {
            padding: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import plotly.graph_objects as go

def render_token_usage_gauge(percent_used: float):
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=percent_used,
        number={
            'suffix': "%",
            'font': {'size': 40}  # Reduced font size from default
        },
        gauge={
            'shape': 'bullet',
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': "mediumseagreen", 'thickness': 1.0},
            'steps': [
                {'range': [0, 70], 'color': "#d3d3d3"},  # Light gray
                {'range': [70, 90], 'color': "#ffd700"},  # Yellow
                {'range': [90, 100], 'color': "#ff4500"},  # Red
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 1.0,
                'value': 100
            }
        },
        domain={'x': [0.05, 0.95], 'y': [0.2, 0.8]}  # Adjusted x-domain for longer bar
    ))

    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=150,
        width=700  # Increased width from 400 to 600
    )

    st.plotly_chart(fig, use_container_width=False)


def main():
    st.title(f"🤖 {APP_NAME} Dashboard")

    # ✅ Placeholder: Add gauge chart after processing
    st.subheader("🧮 Token Usage Summary")
    percent_used = 32  # Placeholder – replace with actual backend logic later
    render_token_usage_gauge(percent_used)

    # --- Sidebar interaction ---
    api_key, model_name, df, chunk_file_path, chunk_summary, prompt, generation_config, gemini_client = cwp_sidebar()

    # --- Chunk Processing Output ---
    if "start_processing" in st.session_state and st.session_state.get("start_processing"):
        try:
            process_chunks_ui(
                gemini_client,
                prompt,
                chunk_file_path,
                max_chunks=st.session_state.get("num_chunks", 5)
            )
        finally:
            st.session_state["start_processing"] = False

            st.rerun()  # Rerun to refresh UI

    # --- Model Configuration ---
    if model_name and generation_config:
        with st.expander("⚙️ Model Configuration", expanded=True):
            st.metric("Model", model_name)
            st.metric("Temperature", f"{generation_config.get('temperature', 0.2):.2f}")
            st.metric("Top K", generation_config.get('top_k', 40))
            st.metric("Top P", f"{generation_config.get('top_p', 1.0):.2f}")

    # --- Chunk Summary ---
    if chunk_summary:
        with st.expander("📦 Chunk Summary", expanded=True):
            for key, value in chunk_summary.items():
                st.markdown(f"- **{key.replace('_', ' ').capitalize()}**: {value}")

    # --- Dataset Preview ---
    if df is not None:
        with st.expander("📊 Dataset Preview", expanded=True):
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(5), use_container_width=True)

    # --- Prompt Echo ---
    if prompt:
        with st.expander("💬 Your Prompt", expanded=True):
            st.code(prompt, language="markdown")

if __name__ == "__main__":
    main()
