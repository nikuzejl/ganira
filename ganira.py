import streamlit as st
import os
import numpy as np
from typing import List
import re
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
from PyPDF2 import PdfReader

# === STREAMLIT CONFIG ===
st.set_page_config(page_title="File Q&A Chatbot")
st.title("📄 Chat with your documents")

# === API TOKEN INPUT ===
token = st.text_input("🔑 Enter your API token", type="password")

if token:
    # === CONFIG ===
    embedding_endpoint = "https://models.inference.ai.azure.com"
    embedding_model = "text-embedding-3-small"
    chat_endpoint = "https://models.github.ai/inference"
    chat_model = "openai/gpt-4.1"

    # === CLIENTS ===
    embedding_client = EmbeddingsClient(
        endpoint=embedding_endpoint,
        credential=AzureKeyCredential(token)
    )

    chat_client = OpenAI(
        base_url=chat_endpoint,
        api_key=token
    )

    # === UTILS ===
    def cosine_similarity(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def parse_pdf(file):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()

    def parse_text_file(file):
        return file.read().decode("utf-8").strip()

    def split_into_chunks(text: str, max_tokens: int = 300) -> List[str]:
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_tokens * 4:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def get_embedding(text: str):
        result = embedding_client.embed(model=embedding_model, input=[text])
        return result.data[0].embedding

    # === FILE INPUT ===
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    question = st.text_input("Ask a question based on the uploaded files")

    if uploaded_files and question:
        with st.spinner("Processing files and finding answers..."):
            all_text = ""
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    all_text += parse_pdf(uploaded_file) + "\n"
                elif uploaded_file.type == "text/plain":
                    all_text += parse_text_file(uploaded_file) + "\n"

            chunks = split_into_chunks(all_text)
            chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

            question_embedding = get_embedding(question)
            similarities = [cosine_similarity(question_embedding, emb) for emb in chunk_embeddings]
            top_index = int(np.argmax(similarities))
            best_chunk = chunks[top_index]

            prompt = f"""Context:
{best_chunk}

Question: {question}
"""

            response = chat_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            st.success("Answer:")
            st.write(response.choices[0].message.content)
