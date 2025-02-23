import streamlit as st
import base64
import time

from transcription import transcribe_audio
from nlp_analysis import (
    extract_keywords,
    perform_sentiment_analysis,
    extract_named_entities,
    generate_summary,
)

st.set_page_config(
    page_title="Spitch AI Speech Analytics Demo",
    layout="wide",
)

def load_svg_as_base64(file_path: str) -> str:
    """Reads an SVG file and returns a base64-encoded string."""
    with open(file_path, "rb") as f:
        svg_data = f.read()
    return base64.b64encode(svg_data).decode("utf-8")

svg_base64 = load_svg_as_base64("spitch_white_new.svg")

custom_css = f"""
<style>
/* Dark background */
body {{
    background-color: #1e1e1e;
    color: #f0f0f0;
    font-family: "Helvetica", sans-serif;
    margin: 0;
}}

/* Remove extra margin around the container */
.block-container {{
    padding: 2rem;
}}

/* Header container to align logo & title horizontally */
.header-container {{
    display: flex;
    align-items: center;
    margin-bottom: 1.5rem;
}}

/* Style the embedded SVG image */
.header-container img {{
    width: 80px;  /* Adjust as needed */
    margin-right: 20px;
}}

/* Title styling */
.header-container h1 {{
    margin: 0;
    font-size: 2rem;
    color: #ffffff;
}}

/* Spinner color override */
.css-2ycy6h {{
    color: #007bff;
}}

/* Sentiment box styling */
.sentiment-box {{
    padding: 10px;
    background-color: #2c2f33;
    border-radius: 5px;
    margin-bottom: 1em;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

header_html = f"""
<div class="header-container">
    <img src="data:image/svg+xml;base64,{svg_base64}" alt="Spitch AI Logo"/>
    <h1>AI Speech Analytics Dashboard</h1>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

st.write(
    "Upload an audio file to analyze call data, listen to the original audio, "
    "and receive actionable insights."
)

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    st.audio(audio_file, format="audio/wav")

    with st.spinner("Transcribing audio..."):
        time.sleep(1)
        transcription_text = transcribe_audio("temp_audio.wav")

    st.subheader("Transcription")
    st.write(transcription_text)

    with st.spinner("Analyzing text..."):
        time.sleep(1)
        keywords = extract_keywords(transcription_text)
        sentiment = perform_sentiment_analysis(transcription_text)
        entities = extract_named_entities(transcription_text)
        summary = generate_summary(transcription_text)

    polarity = sentiment["polarity"]
    subjectivity = sentiment["subjectivity"]
    if polarity > 0.1:
        tone = "Positive 😊"
    elif polarity < -0.1:
        tone = "Negative 😞"
    else:
        tone = "Neutral 😐"

    sentiment_message = f"""
    <div class='sentiment-box'>
        <strong>Overall Tone:</strong> {tone}<br>
        <strong>Polarity:</strong> {polarity:.2f} 
        <span style="font-size: 0.9em; color: #bbb;">
            (scale: -1 negative to +1 positive)
        </span><br>
        <strong>Subjectivity:</strong> {subjectivity:.2f}
        <span style="font-size: 0.9em; color: #bbb;">
            (0 is very objective, 1 is very subjective)
        </span>
    </div>
    """

    st.subheader("Key Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Keywords")
        st.write(keywords[:10])
        st.markdown("### 🏷️ Named Entities")
        st.write(entities)

    with col2:
        st.markdown("### 😊 Sentiment Analysis")
        st.markdown(sentiment_message, unsafe_allow_html=True)
        st.markdown("### 📑 Summary")
        st.write(summary)

    st.success("Analysis complete!")
