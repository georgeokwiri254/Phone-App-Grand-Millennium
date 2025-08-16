"""
Simple test to identify what's causing the crash
"""
import streamlit as st

st.set_page_config(
    page_title="Test App",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Test Application")
st.write("If you can see this, the basic Streamlit setup works.")

with st.sidebar:
    st.write("Sidebar test")
    selected = st.radio("Test Navigation", ["Option 1", "Option 2"])

if selected == "Option 1":
    st.write("Option 1 selected")
else:
    st.write("Option 2 selected")