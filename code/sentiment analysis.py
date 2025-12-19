import streamlit as st
import whisper
import spacy
from transformers import pipeline
from scipy.io import wavfile
import sounddevice as sd
import tempfile
import numpy as np

# ----------------------------------------------------
# Page Setup
# ----------------------------------------------------
st.set_page_config(page_title="Live Speech Analyzer", layout="centered")
st.title("🎤 Live Speech → Sentiment + Intent + Entities")

# ----------------------------------------------------
# Load models (cached)
# ----------------------------------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_ner():
    return spacy.load("en_core_web_sm")

whisper_model = load_whisper()
sentiment_model = load_sentiment()
ner_model = load_ner()

# ----------------------------------------------------
# Intent detection
# ----------------------------------------------------
def detect_intent(text):
    t = text.lower()
    if any(x in t for x in ["buy", "purchase", "interested", "get a new"]):
        return "Purchase Interest"
    if any(x in t for x in ["price", "cost", "how much", "expensive"]):
        return "Ask for price"
    if any(x in t for x in ["not working", "issue", "problem", "complaint"]):
        return "Complaint"
    if any(x in t for x in ["compare", "specs", "camera", "battery"]):
        return "Compare Product"
    if any(x in t for x in ["offer", "discount", "deal"]):
        return "Ask for offers"
    return "General"

# ----------------------------------------------------
# Record Audio
# ----------------------------------------------------
def record_audio(seconds, fs=44100):
    st.info("🎙 Recording... Speak now.")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    st.success("Recording complete")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wavfile.write(tmp.name, fs, recording)
        return tmp.name

# ----------------------------------------------------
# UI Input
# ----------------------------------------------------
record_seconds = st.number_input("Recording Duration (seconds)", 2, 20, 5)

if st.button("🎤 Start Recording"):
    wav_path = record_audio(record_seconds)
    st.audio(wav_path)

    st.info("🔄 Transcribing with Whisper...")

    # Transcription
    result = whisper_model.transcribe(wav_path, language="en")
    text = result.get("text", "").strip()

    st.subheader("🎤 Transcribed Text")
    st.write(f"**{text}**")

    # ------------------------------------------------
    # Sentiment
    # ------------------------------------------------
    sent = sentiment_model(text)[0]
    sentiment_label = sent["label"]
    sentiment_score = round(float(sent["score"]), 2)

    st.subheader("😊 Sentiment")
    st.write(f"**{sentiment_label} ({sentiment_score})**")

    # ------------------------------------------------
    # Intent
    # ------------------------------------------------
    intent = detect_intent(text)

    st.subheader("🎯 Intent")
    st.write(f"**{intent}**")

    # ------------------------------------------------
    # Entities
    # ------------------------------------------------
    doc = ner_model(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    st.subheader("🏷 Entities")
    st.json(entities)

