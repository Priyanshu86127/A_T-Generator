import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import tempfile
import os

st.set_page_config(page_title="Audio To Text")

st.title("Audio to Text")

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

model = load_whisper()

tab1, tab2 = st.tabs(["Upload Audio", "Live Recording"])

with tab1:


    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "m4a"]
    )

    if uploaded_file:

        st.audio(uploaded_file)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        with st.spinner("Converting speech to text..."):

            result = model.transcribe(temp_path)

        st.subheader("Transcript")
        st.write(result["text"])

        st.download_button(
            "Download Transcript",
            result["text"],
            file_name="transcript.txt"
        )

        os.remove(temp_path)

with tab2:

    st.write("Record using microphone")

    audio = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        just_once=True,
        use_container_width=True
    )

    if audio:

        audio_bytes = audio["bytes"]

        st.audio(audio_bytes, format="audio/wav")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp_file:

            tmp_file.write(audio_bytes)
            temp_path = tmp_file.name

        with st.spinner("Transcribing live audio..."):

            result = model.transcribe(temp_path)

        st.subheader("Transcript")
        st.write(result["text"])

        st.download_button(
            "Download Transcript",
            result["text"],
            file_name="live_transcript.txt"
        )

        os.remove(temp_path)