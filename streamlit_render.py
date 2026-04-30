import os
import streamlit as st
import requests

st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide"
)

# On Render this env var is set to your live API URL
# Locally falls back to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

st.title("📄 DocuMind")
st.markdown("Upload a PDF and ask questions about it in plain English.")
st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("1. Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )},
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✓ Indexed — {data['chunks_created']} chunks created")
                    st.session_state.pdf_uploaded = True
                    st.session_state.pdf_name = uploaded_file.name
                else:
                    st.error(f"Upload failed: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Check the API_URL environment variable.")
            except requests.exceptions.Timeout:
                st.error("Upload timed out. Try a smaller PDF.")

with col_right:
    st.subheader("2. Ask questions")

    if st.session_state.pdf_uploaded:
        st.info(f"📄 Active: {st.session_state.pdf_name}")
    else:
        st.warning("Upload a PDF on the left to get started.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your PDF...",
        disabled=not st.session_state.pdf_uploaded
    )

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question},
                        timeout=90
                    )
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        pages = data["source_pages"]
                        latency = data.get("latency_seconds", "N/A")
                        st.markdown(answer)
                        st.caption(f"📖 Sources: pages {pages}  ·  ⏱ {latency}s")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"{answer}\n\n*📖 Pages {pages} · ⏱ {latency}s*"
                        })
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Try a simpler question.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API.")

    if st.session_state.messages:
        st.divider()
        if st.button("🗑 Clear chat"):
            st.session_state.messages = []
            st.rerun()