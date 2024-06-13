import streamlit as st
import requests
from os import getenv as env

URL = env("API_URL")

st.title("ArXiv Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = requests.get(URL+prompt,
                              params={'key':'value'})
        response = st.write(stream.json()['answer'])
        #response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})