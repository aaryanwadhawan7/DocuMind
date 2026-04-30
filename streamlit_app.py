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
API_URI = os.getenv('API_URL', 'http://localhost:8000')

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
    
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
st.markdown("Upload a PDF and ask questions about it.")
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

# Session 6 : Right column : Chat Interface

with col_right:
    st.subheader('2. Ask questions')
    
    if st.session_state.pdf_uploaded:
        st.info(f"Active {st.session_state.pdf_name}")
    else:
        st.warning("Pdf not Uploaded!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

    question = st.chat_input(
        "Ask a question about your PDF...",
        disabled=not st.session_state.pdf_uploaded
    )

    # This blocks only run user submit a question
    if question:

        with st.chat_message('user'):
            st.markdown(question)
        
        st.session_state.messages.append({
            "role" : "user",
            "content" : question
        })

    # Call Requests to FastAPI 
    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URI}/ask",
                    json = {"question" : question},
                    timeout = 60 
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data['answer']
                    pages = data['source_pages']
                    latency = data.get("latency_seconds", "N/A")
                
                    st.markdown(answer)

                    st.caption(
                        f"📖 Sources: pages {pages}  ·  ⏱ {latency}s"
                    )

                    # Save answer to history
                    full_answer = (
                       f"{answer}\n\n"
                       f"*📖 Sources: pages {pages} · ⏱ {latency}s*"
                    )
                
                    st.session_state.messages.append({
                       "role":    "assistant",
                       "content": full_answer
                    })
                else:
                    error_msg = f"Error: {response.status_code}: {response.text}"

            except requests.exceptions.Timeout:
                st.error("Timed out after 60s. Try a shorter question.")
            
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is Docker running?")

if st.session_state.messages:
        st.divider()
        if st.button("🗑 Clear chat"):
            st.session_state.messages = []
            st.rerun()