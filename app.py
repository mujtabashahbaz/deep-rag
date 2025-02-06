import os
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import InMemoryVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGroq

# Load environment variables
load_dotenv()

# Get API Keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize models with API keys
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = ChatGroq(model="deepseek-chat", groq_api_key=GROQ_API_KEY)

# Streamlit UI Configuration
st.set_page_config(page_title="📘 DocuMind AI", layout="wide")
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload a PDF document",
    type="pdf",
    accept_multiple_files=False,
    help="Upload a PDF file, and the AI will assist you with document insights."
)

if uploaded_pdf:
    # Extract text from PDF
    with pdfplumber.open(uploaded_pdf) as pdf:
        raw_text = "\n\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    # Split text into chunks for processing
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    document_chunks = text_processor.create_documents([raw_text])

    # Store document in vector database
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

    st.success("✅ Document processed successfully! Ask your questions below.")

    # User query input
    user_input = st.chat_input("Enter your question about the document...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Analyzing document..."):
            relevant_docs = DOCUMENT_VECTOR_DB.similarity_search(user_input)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            # AI Prompt Template
            prompt_template = """
            You are an expert research assistant. Use the provided context to answer the query. 
            If unsure, state that you don't know. Be concise and factual (max 3 sentences).
            Query: {user_query} 
            Context: {document_context} 
            Answer:
            """
            conversation_prompt = ChatPromptTemplate.from_template(prompt_template)
            response_chain = conversation_prompt | LANGUAGE_MODEL
            ai_response = response_chain.invoke({"user_query": user_input, "document_context": context_text})

        # Display AI response
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
