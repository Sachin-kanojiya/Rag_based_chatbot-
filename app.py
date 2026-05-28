import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Direct API Key
OPENAI_API_KEY = "open_api_key"

# Streamlit UI
st.header("StudyBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload your Notes PDF here", type="pdf")

# Read PDF
if file is not None:

    pdf_reader = PdfReader(file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    # Create Embeddings
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY
    )

    # Store in FAISS
    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    # User Query
    user_query = st.text_input("Type your query here")

    if user_query:

        # Similarity Search
        matching_chunks = vector_store.similarity_search(
            user_query
        )

        # LLM
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=200
        )

        # Prompt
        customized_prompt = ChatPromptTemplate.from_template(
            """
            Act as my assistant tutor and answer the question
            based only on the provided context.

            If answer is not available in context,
            say "I don't know".

            Context:
            {context}

            Question:
            {question}
            """
        )

        # Chain
        chain = create_stuff_documents_chain(
            llm,
            customized_prompt
        )

        # Generate Response
        output = chain.invoke({
            "context": matching_chunks,
            "question": user_query
        })

        st.write(output)