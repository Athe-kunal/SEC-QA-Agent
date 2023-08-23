import streamlit as st
from app.main import SEC_LLM
import time

st.title("SEC App")


def generate_response(input_text,mmr:bool):
    return SEC_LLM(input_text,mmr)


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input():
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Answering..."):
            response,ui_relevant_sentences = generate_response(prompt,mmr=False) 
            st.write(response) 
            expander = st.expander("See relevant Documents")
            expander.text(ui_relevant_sentences)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
