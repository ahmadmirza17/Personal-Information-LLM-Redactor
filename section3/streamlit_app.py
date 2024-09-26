import streamlit as st
import requests
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.title("PII Redaction Tool")

text_input = st.text_area("Enter text to redact:")

if st.button("Redact"):
    logger.debug(f"Sending request to API with text: {text_input}")
    try:
        response = requests.post("http://localhost:8000/redact", json={"text": text_input}, timeout=10)
        logger.debug(f"Received response from API: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            st.subheader("Redacted Text:")
            st.write(result["redacted_text"])
            st.subheader("PII Detected:")
            st.write(result["pii_detected"])
        else:
            st.error(f"An error occurred while processing the request. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to API: {str(e)}")
        st.error(f"An error occurred while connecting to the API: {str(e)}")