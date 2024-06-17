import streamlit as st
import requests
from os import getenv as env
from urllib.parse import quote
import time

URL = env("API_URL")


st.set_page_config(
    page_title="ArXiv Chatbot",
    layout="wide",
    menu_items={}
)

st.markdown(
    """
<style>
    .st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)



WELCOME = "Hello! How can I assist you today?"

ARXIV_URL = "https://arxiv.org/abs/"

st.title("ArXiv Chatbot")

def display(stream, wait):
    print(stream)
    for chunk in stream.json():
        if chunk == "output":
            for text in list(stream.json()[chunk]):
                yield(text)
                time.sleep(wait)
        if chunk == "intermediate_steps":
            steps = stream.json()[chunk]
            if len(steps) > 0:
                sources = steps[0][1]['context'] or None
                yield " \n  "
                yield " \n  "
                yield "**Sources**: \n "
                for i in range(len(sources)):
                    metadata = sources[i]['metadata'] or None
                    try:
                        yield f"{i+1}. [{metadata['title']}]({ARXIV_URL}{metadata['source']}) \n "
                    except:
                        pass
                    time.sleep(wait)
        if chunk == "user_id":
            st.session_state.user_id = stream.json()['user_id']
    return stream

def process_query(prompt, user_id):
    # query = quote(prompt, safe='/:')
    with st.spinner(""):
        stream = requests.post(
            URL,
            params={'user_id': user_id, 'message':prompt}
        )
    if stream.status_code != 200:
        stream.raise_for_status()
    stream = display(stream, 0.01)
        
        # region old content

        # counter = 0
        # if type(chunk) == str:
        #     for text in list(chunk):
        #         yield text
        #         time.sleep(0.01)
        #     yield " \n  "
        #     yield " \n  "
        #     yield "**Sources**: \n "
        # if type(chunk) == dict:
        #     try:
        #         yield f"{counter}. [{chunk['title']}]({ARXIV_URL}{chunk['source']}) \n "
        #     except:
        #         pass
        # time.sleep(0.1)
        # counter += 1

        # endregion
    return stream

if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.spinner("Loading..."):
        requests.get(URL[:-6])
    st.session_state.messages.append({"role": "assistant", "content": WELCOME})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input(WELCOME):
    #user_text = f'<div align="right">{prompt}</div>'
    user_text = prompt
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text, unsafe_allow_html=True)

    user_exists = "user_id" in st.session_state
    if user_exists:
        user_id = st.session_state.user_id
    else:
        user_id = 0
    with st.chat_message("assistant"):
        try:
            response = st.write_stream(process_query(prompt, user_id))
        except:
            response = st.write("Sorry, an error occurred. Please refresh the page and try again.")

    st.session_state.messages.append({"role": "assistant", "content": response})

