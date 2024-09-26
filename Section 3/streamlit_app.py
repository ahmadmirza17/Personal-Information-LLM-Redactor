import streamlit as st
import requests

st.title("PII Redaction Tool")

text_input = st.text_area("Enter text to redact:")

if st.button("Redact"):
    response = requests.post("http://localhost:8000/redact", json={"text": text_input})
    if response.status_code == 200:
        result = response.json()
        st.subheader("Redacted Text:")
        st.write(result["redacted_text"])
        st.subheader("PII Detected:")
        st.write(result["pii_detected"])
    else:
        st.error("An error occurred while processing the request.")