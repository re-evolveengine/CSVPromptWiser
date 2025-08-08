import streamlit as st
import plotly.graph_objects as go
import uuid  # Needed only if you want auto-generated unique keys


def render_token_usage_gauge(percent_used: float, key: str = None):
    """
    Render a gauge chart showing the token usage percentage.

    Args:
        percent_used (float): The percentage of tokens used (0-100)
        key (str, optional): Unique Streamlit key to prevent duplication errors.
    """
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=percent_used,
        number={
            'suffix': "%",
            'font': {'size': 40}
        },
        gauge={
            'shape': 'bullet',
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': "mediumseagreen", 'thickness': 1.0},
            'steps': [
                {'range': [0, 70], 'color': "#d3d3d3"},
                {'range': [70, 90], 'color': "#ffd700"},
                {'range': [90, 100], 'color': "#ff4500"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 1.0,
                'value': 100
            }
        },
        domain={'x': [0.05, 0.95], 'y': [0.2, 0.8]}
    ))

    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=150,
        width=700
    )

    # Use a default UUID-based key if none is provided (avoids duplication)
    key = key or f"token_gauge_{uuid.uuid4()}"
    st.plotly_chart(fig, use_container_width=False, key=key)
