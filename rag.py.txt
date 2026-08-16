import os
import streamlit as st
import tiktoken

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate


# ============================================================
# 1. API CONFIGURATION
# ============================================================

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()


# ============================================================
# 2. INITIALIZE LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.2
)


# ============================================================
# 3. WORD DOCUMENT PATH
# ============================================================

DOCX_PATH = "PUT_YOUR_WORD_FILE_PATH_HERE.docx"


# ============================================================
# 4. LOAD WORD DOCUMENT
# ============================================================

@st.cache_resource
def create_vector_db():

    loader = Docx2txtLoader(DOCX_PATH)

    documents = loader.load()

    # --------------------------------------------------------
    # Split document into smaller chunks
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    # --------------------------------------------------------
    # Store embeddings in FAISS vector database
    # --------------------------------------------------------

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_db


# ============================================================
# 5. CREATE VECTOR DATABASE
# ============================================================

vector_db = create_vector_db()


# ============================================================
# 6. RAG PROMPT
# ============================================================

prompt_template = """
You are a helpful AI assistant.

Answer the user's question using ONLY the information provided
in the context below.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Do not make up information.

Context:
{context}

User Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)


# ============================================================
# 7. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Document RAG Assistant",
    layout="wide"
)

st.title("📄 Document RAG Assistant")

st.write(
    "Ask questions about the information contained in the Word document."
)


# ============================================================
# 8. USER QUESTION
# ============================================================

question = st.text_input(
    "Ask a question:"
)


# ============================================================
# 9. RUN RAG
# ============================================================

if st.button("Ask"):

    if question:

        with st.spinner("Searching document and generating answer..."):

            try:

                # ------------------------------------------------
                # Retrieve relevant chunks from Vector DB
                # ------------------------------------------------

                retrieved_docs = vector_db.similarity_search(
                    question,
                    k=3
                )

                # ------------------------------------------------
                # Combine retrieved chunks
                # ------------------------------------------------

                context = "\n\n".join(
                    doc.page_content
                    for doc in retrieved_docs
                )

                # ------------------------------------------------
                # Create prompt
                # ------------------------------------------------

                final_prompt = prompt.format(
                    context=context,
                    question=question
                )

                # ------------------------------------------------
                # Send context + question to Gemini
                # ------------------------------------------------

                response = llm.invoke(final_prompt)

                result = response.content


                # =================================================
                # DISPLAY ANSWER
                # =================================================

                st.subheader("Answer")

                st.markdown(result)


                # =================================================
                # DISPLAY SOURCES
                # =================================================

                st.subheader("Retrieved Context")

                for i, doc in enumerate(retrieved_docs):

                    with st.expander(
                        f"Source Chunk {i + 1}"
                    ):

                        st.write(doc.page_content)


            except Exception as e:

                if "429" in str(e):

                    st.error(
                        "Free Tier Quota Exceeded. "
                        "Please wait or switch to a paid API key."
                    )

                elif "404" in str(e):

                    st.error(
                        "Model not found. "
                        "Please check the Gemini model name."
                    )

                else:

                    st.error(
                        f"Error: {e}"
                    )

    else:

        st.warning(
            "Please enter a question."
        )