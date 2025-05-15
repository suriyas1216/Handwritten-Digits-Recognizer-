# streamlit_app.py

import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Simple summarizer: returns the first 3 sentences
def summarize_text(text, max_sentences=3):
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:max_sentences])

# Streamlit app layout
st.set_page_config(page_title="Simple Summarizer", layout="centered")

st.title("Text Summarizer")
st.write("Paste your text below and get a quick summary.")

# User input
text_input = st.text_area("Enter your text:", height=200)

# Process and display summary
if st.button("Summarize"):
    if text_input.strip():
        summary = summarize_text(text_input)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
