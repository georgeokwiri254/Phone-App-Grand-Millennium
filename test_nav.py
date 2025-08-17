import streamlit as st

st.set_page_config(page_title="Navigation Test", layout="wide")

# Test CSS
st.markdown("""
<style>
.test-nav {
    background: #f0f0f0;
    padding: 10px;
    border: 2px solid red;
}
.test-logo {
    width: 15px;
    height: 15px;
    background: blue;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# Test HTML
st.markdown("""
<div class="test-nav">
    <div class="test-logo"></div>
    <span>Test Navigation</span>
</div>
""", unsafe_allow_html=True)

st.write("If you see a blue 15x15px square above with 'Test Navigation', HTML rendering works!")