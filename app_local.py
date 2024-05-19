import os
import streamlit as st
import google.generativeai as genai

from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure GenAI with the API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    template = """
    you are a chatbot trained to answer questions related to climate change, and you answer questions related to climate change only.
    Answer the question as detailed as possible only from the provided context.
    If the answer is not available in the context or the pdf is not related to climate change, please state so.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = load_qa_chain(model, prompt=prompt, verbose=True)
    return chain

# Streamlit interface
st.title('Climate Change Q&A')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Option to choose between PDF question answering or climate change chatbot
option = st.radio("Select an option:", ("Climate Change Chatbot", "Answer from PDFs related to Climate change" ))

if option == "Answer from PDFs related to Climate change":
    uploaded_files = st.file_uploader("Choose PDF files (related to climate change)", accept_multiple_files=True)
    if uploaded_files:
        pdf_texts = get_pdf_text(uploaded_files)
        text_chunks = get_text_chunks(pdf_texts)
        get_vector_store(text_chunks)
        st.success('PDFs processed and vector store updated!')

    user_question = st.text_input("Enter your question here:")

    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Bot:", response["output_text"])




elif option == "Climate Change Chatbot":
    # Get user input
    user_question = st.text_input("Ask your question related to climate change:")

    # Generate response upon user input
    if st.button("Ask"):
        if user_question:
            # Prompt template guiding the model to focus on climate change topics
            prompt = f"""I'm a chatbot trained to answer questions related to climate change and related topics. Please ask me anything about climate change. \n\n Q: {user_question}\nA:"""
            # Use Google Generative AI (Gemini) model to generate response
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            # Extract model's response from the generated content
            bot_response = response.text
            st.write("Bot:", bot_response)