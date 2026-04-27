import streamlit as st
from emotion_module import emotion_app
from music_module import music_app

if "page" not in st.session_state:
    st.session_state.page = "emotion"

# if emotion is detected and user clicked recommend
if st.session_state.page == "emotion":
    emotion_app.run()

elif st.session_state.page == "music":
    music_app.run()