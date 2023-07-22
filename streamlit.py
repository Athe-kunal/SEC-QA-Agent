import streamlit as st
from main import SEC_LLM

st.title("SEC App")


def generate_response(input_text):
    st.info(SEC_LLM(input_text))


with st.form("my_form"):
    text = st.text_area("Enter Text:")
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
