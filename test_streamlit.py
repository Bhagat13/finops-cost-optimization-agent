import streamlit as st

st.title("Streamlit Test")

name = st.text_input("Enter your name")

if name:
    st.write("Hello", name)