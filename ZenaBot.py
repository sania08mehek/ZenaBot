import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# extracting pdf words
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text


# making into chunks
def get_text_chunks(text):
    # Using a more standard chunk size for better performance
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)


# creating the conversational chain
def get_conversational_chain(vector_store):
    prompt_template = """
    Use the following context to answer the question as accurately and completely as possible.
    If the answer cannot be found in the context, say "Answer not found in the document."

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    model = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
    
    # creating the chain
    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain


# UI using streamlit
def custom_css():
    st.markdown("""
        <style>
            .main-title {
                font-size: 36px;
                font-weight: bold;
                color: #3b3b3b;
                text-align: center;
                margin-bottom: 20px;
            }
            .input-box input {
                font-size: 16px;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #ccc;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 16px;
                transition: 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
        </style>
    """, unsafe_allow_html=True)

def set_background_gradient():
    st.markdown("""
        <style>
        .stApp {
            background: radial-gradient(circle at 70% 30%, rgba(168, 85, 247, 0.3), transparent 40%),
                        radial-gradient(circle at 30% 30%, rgba(59, 130, 246, 0.3), transparent 20%),
                        linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            background-attachment: fixed;
            background-size: cover;
        }
        </style>
    """, unsafe_allow_html=True)

def custom_header_with_logo():
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 20px;">
            <img src="https://media1.giphy.com/media/v1.Y2lkPTZjMDliOTUycnA3Mm1kczFrMHExY2V1dGd5MWNzeGhhdjE3dXJsMzJ1NGhlbHM3OCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/0uQv3gEQAGCXhtq6ZQ/giphy.gif" alt="Logo" width="90">
            <h1 style="margin: 0; font-size: 36px;">Hey there ! Chat with ZenaBot</h1>
        </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config("ZenaBot", layout="centered")
    set_background_gradient()
    custom_css()
    custom_header_with_logo()

    # Initialize session state to store the conversation chain
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    col1, col2 = st.columns([5, 2])
    
    with col1:
        st.subheader("What can I help with ❓")
        question = st.text_input("Ask anything:", key="question_input", placeholder="E.g., Summarize the document X")
        
        # Only process question if the chain exists
        if question:
            if st.session_state.conversation_chain:
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation_chain.invoke({"input": question})
                    st.write("Reply: ", response["answer"])
            else:
                st.warning("Please upload and process your documents first!")

    with col2:
        st.subheader("Upload PDF")
        pdf_docs = st.file_uploader("Choose your PDFs", type="pdf", accept_multiple_files=True)
        
        # Processing happens only when this button is clicked
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Analyzing and Vectorizing..."):
                    # 1. Get text and chunks
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    
                    # 2. Create vector store in memory
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                    
                    # 3. Create the chain and save it to the session state
                    st.session_state.conversation_chain = get_conversational_chain(vector_store)
                    
                    st.success("✅ Ready to Chat!")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()