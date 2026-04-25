import streamlit as st
import numpy as np
import os
import requests

# Section 1 : Setting up page config.

st.set_page_config(
    page_title='DocuMind',
    page_icon='📄',
    layout='wide'
)

# Section 2

# Locally: defaults to localhost 8000
# In Docker: we set API_URI = http://api:8000 in docker-compose
API_URI = os.getenv('API_URI', 'http://localhost:8000')

# Initialize session state - these run on first page reload
# After that keys exists so these lines are being skipped 
if "messages" not in st.session_state:
    # messages : list of dict.s where each dict has "role" and "content"
    st.session_state.messages = []

if "pdf_name" not in st.session_state:
    # shows the file name so that we can show it in the chat column
    st.session_state.pdf_name = ""

# Section 3 : Header

st.title("📄 DocuMind")
st.markdown("Upload a PDF and ask question about it in Plain English.")
st.divider()

# Section 4 : Two column layout

# left column : 1/3 part
# right column : 2/3
col_left, col_right = st.columns([1,3])

# Section 5 : Left column : PDF Upload

with col_left:
    st.subheader("1. Upload your PDF")
    st.markdown("Your document will be split into chunks and indexed for search.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:

        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                # Send the PDF to FastAPI /upload
                response = requests.post(
                    f"{API_URI}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    }
                )

                if response.status_code == 200:
                    data = response.json()

                    st.success(
                        f"✓ Indexed — {data['chunks_created']} chunks created"
                    )

                    st.session_state.pdf_uploaded = True
                    st.session_state.pdf_name     = uploaded_file.name

                else:
                    st.error(f"Upload failed: {response.text}")
                    st.session_state.pdf_uploaded = False

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is Docker running?")