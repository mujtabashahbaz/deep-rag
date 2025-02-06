import streamlit as st
import tempfile
import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI

# Set page title and theme
st.set_page_config(page_title="📘 DocuMind AI", layout="wide")

# Groq API Key (Set this in Streamlit Secrets for secure storage)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com")

# Embedding and Vector DB
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=GROQ_API_KEY)
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

# Custom Prompt Template
PROMPT_TEMPLATE = """You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Function to Process PDF
def process_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    with pdfplumber.open(tmp_path) as pdf:
        raw_text = "\n\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    text_processor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_processor.create_documents([raw_text])

    DOCUMENT_VECTOR_DB.add_documents(document_chunks)
    os.remove(tmp_path)  # Clean up temp file

# Function to Find Relevant Docs
def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

# Function to Generate Response using Groq's DeepSeek API
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    prompt = PROMPT_TEMPLATE.format(user_query=user_query, document_context=context_text)
    
    response = client.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=500
    )
    
    return response.choices[0].message["content"]

# Streamlit UI
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")

uploaded_pdf = st.file_uploader("Upload Research Document (PDF)", type="pdf")

if uploaded_pdf:
    st.info("📄 Processing document...")
    process_uploaded_pdf(uploaded_pdf)
    st.success("✅ Document indexed! Ask your questions below.")

    user_input = st.chat_input("Enter your question about the document...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)