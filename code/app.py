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
st.title("🎤 Speech → Sentiment + Intent + Entities + AI Sales Assistant")

# ----------------------------------------------------
# Load models (cached)
# ----------------------------------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")  # very fast

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
# Intent detection function
# ----------------------------------------------------
def detect_intent(text):
    t = text.lower()

    if any(x in t for x in ["buy", "purchase", "interested", "get a new"]):
        return "Purchase Interest"

    if any(x in t for x in ["price", "cost", "how much", "expensive"]):
        return "Ask for Price"

    if any(x in t for x in ["not working", "issue", "problem", "complaint"]):
        return "Complaint"

    if any(x in t for x in ["compare", "specs", "camera", "battery"]):
        return "Product Comparison"

    if any(x in t for x in ["offer", "discount", "deal"]):
        return "Ask for Offers"

    return "General"

# ----------------------------------------------------
# AI Sales Suggestions
# ----------------------------------------------------
def ai_sales_response(intent, sentiment):
    if intent == "Purchase Interest":
        return {
            "next_question": "Would you like to know the available models or pricing?",
            "soft_handling": "Great! Let me help you choose the best option.",
            "recommendation": "Our latest model offers excellent value for buyers."
        }

    if intent == "Ask for Price":
        return {
            "next_question": "Do you have any budget range in mind?",
            "soft_handling": "I’ll help you find the best model in your range.",
            "recommendation": "We have offers and EMI options you might like."
        }

    if intent == "Complaint":
        return {
            "next_question": "Could you explain the issue you're facing?",
            "soft_handling": "I truly apologize for the inconvenience.",
            "recommendation": "We can quickly guide you through a solution."
        }

    if sentiment == "NEGATIVE":
        return {
            "next_question": "What would you like us to improve?",
            "soft_handling": "I understand your concern — I'm here to help.",
            "recommendation": "We can offer alternatives that better fit your needs."
        }

    return {
        "next_question": "How can I assist you further?",
        "soft_handling": "I'm right here to help.",
        "recommendation": "Let me know your preference and I’ll recommend the best option."
    }

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
# Audio Input Section
# ----------------------------------------------------
st.subheader("🎧 Choose Input Type")

input_choice = st.radio(
    "Select audio source:",
    ["🎤 Record Live", "📁 Upload Audio File"],
)

wav_path = None

# -----------------------------------------
# Option A: Record live audio
# -----------------------------------------
if input_choice == "🎤 Record Live":
    record_seconds = st.slider("Recording Duration (seconds)", 2, 20, 5)

    if st.button("🎙 Start Recording", type="primary"):
        wav_path = record_audio(record_seconds)
        st.audio(wav_path)

# -----------------------------------------
# Option B: Upload audio file
# -----------------------------------------
if input_choice == "📁 Upload Audio File":
    uploaded = st.file_uploader("Upload WAV/MP3 file", type=["wav", "mp3", "m4a"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            wav_path = tmp.name
        st.audio(wav_path)

# ----------------------------------------------------
# Run analysis if audio exists
# ----------------------------------------------------
if wav_path:
    st.info("🔄 Transcribing using Whisper...")

    result = whisper_model.transcribe(wav_path, language="en")
    text = result.get("text", "").strip()

    # -----------------------------------------
    # Show transcription
    # -----------------------------------------
    st.subheader("📝 Transcribed Text")
    st.write(f"**{text}**")

    # -----------------------------------------
    # Sentiment
    # -----------------------------------------
    sent = sentiment_model(text)[0]
    sentiment_label = sent["label"]
    sentiment_score = round(float(sent["score"]), 2)

    st.subheader("😊 Sentiment")
    st.write(f"**{sentiment_label} ({sentiment_score})**")

    # -----------------------------------------
    # Intent
    # -----------------------------------------
    intent = detect_intent(text)

    st.subheader("🎯 Intent")
    st.write(f"**{intent}**")

    # -----------------------------------------
    # Entities
    # -----------------------------------------
    doc = ner_model(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    st.subheader("🏷 Entities")
    st.json(entities)

    # -----------------------------------------
    # AI SALES ASSISTANT
    # -----------------------------------------
    ai = ai_sales_response(intent, sentiment_label)

    st.subheader("🤖 AI Sales Assistant")

    st.markdown(f"""
    **🟦 Next Question:**  
    {ai['next_question']}

    **🟩 Soft Handling Response:**  
    {ai['soft_handling']}

    **🟨 Recommendation:**  
    {ai['recommendation']}
    """)

# END
