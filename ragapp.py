import os
import streamlit as st

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG Document Assistant")
st.write(
    "Upload a Word document and ask questions about its contents."
)


# =========================================================
# GOOGLE API KEY
# =========================================================

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error(
        "GOOGLE_API_KEY is missing. "
        "Please add it to Streamlit Secrets."
    )
    st.stop()


# =========================================================
# INITIALIZE GEMINI
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)


# =========================================================
# INITIALIZE EMBEDDING MODEL
# =========================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)


# =========================================================
# SESSION STATE
# =========================================================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None


# =========================================================
# UPLOAD WORD DOCUMENT
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your Word document",
    type=["docx"]
)


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file is not None:

    # Only process the document if it is a new upload
    if uploaded_file.name != st.session_state.uploaded_file_name:

        with st.spinner("Processing document..."):

            # -------------------------------------------------
            # Save uploaded document temporarily
            # -------------------------------------------------

            file_path = "uploaded_document.docx"

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())


            # -------------------------------------------------
            # LOAD DOCUMENT
            # -------------------------------------------------

            loader = Docx2txtLoader(file_path)

            documents = loader.load()


            # -------------------------------------------------
            # SPLIT DOCUMENT INTO CHUNKS
            # -------------------------------------------------

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = text_splitter.split_documents(documents)


            # -------------------------------------------------
            # CREATE VECTOR DATABASE
            # -------------------------------------------------

            vector_db = FAISS.from_documents(
                chunks,
                embeddings
            )


            # -------------------------------------------------
            # STORE IN SESSION
            # -------------------------------------------------

            st.session_state.vector_db = vector_db
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.chunks = chunks


        st.success(
            f"Successfully processed: {uploaded_file.name}"
        )

        st.info(
            f"Document was split into "
            f"{len(chunks)} chunks."
        )


# =========================================================
# ASK QUESTION
# =========================================================

if st.session_state.vector_db is not None:

    st.divider()

    st.subheader("Ask a question")

    question = st.text_input(
        "Enter your question:",
        placeholder="e.g. What are the company's main strengths?"
    )


    if st.button("🔍 Ask", type="primary"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            try:

                # =================================================
                # STEP 1 — VECTOR SEARCH
                # =================================================

                with st.spinner("Searching the document..."):

                    retrieved_docs = (
                        st.session_state.vector_db
                        .similarity_search(
                            question,
                            k=3
                        )
                    )


                # =================================================
                # STEP 2 — CREATE CONTEXT
                # =================================================

                context = "\n\n".join(
                    doc.page_content
                    for doc in retrieved_docs
                )


                # =================================================
                # STEP 3 — CREATE PROMPT
                # =================================================

                prompt = f"""
You are a helpful document assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided document context.

Rules:

1. Do not make up information.
2. Do not use outside knowledge.
3. If the answer cannot be found in the context,
   say exactly:

"I could not find the answer in the document."

4. Give a clear and concise answer.

DOCUMENT CONTEXT:
----------------
{context}
----------------

USER QUESTION:
{question}

ANSWER:
"""


                # =================================================
                # STEP 4 — SEND TO GEMINI
                # =================================================

                with st.spinner(
                    "Generating answer with Gemini..."
                ):

                    response = llm.invoke(prompt)

                    answer = response.content


                # =================================================
                # DISPLAY ANSWER
                # =================================================

                st.subheader("💡 Answer")

                st.markdown(answer)


                # =================================================
                # DISPLAY RETRIEVED SOURCES
                # =================================================

                st.divider()

                st.subheader(
                    "🔎 Retrieved Document Sections"
                )

                for i, doc in enumerate(retrieved_docs):

                    with st.expander(
                        f"Source {i + 1}"
                    ):

                        st.write(
                            doc.page_content
                        )


            # =====================================================
            # ERROR HANDLING
            # =====================================================

            except Exception as e:

                error_message = str(e)

                st.error(
                    "An error occurred while communicating "
                    "with Gemini."
                )

                st.code(error_message)

                if "404" in error_message or "NotFound" in error_message:

                    st.warning(
                        "Gemini returned a 404/NotFound error. "
                        "Please check your Google API key, "
                        "Gemini model access, and installed "
                        "LangChain Google GenAI package."
                    )

                elif "429" in error_message:

                    st.warning(
                        "Gemini API quota/rate limit exceeded. "
                        "Please wait and try again."
                    )

                elif "401" in error_message or "403" in error_message:

                    st.warning(
                        "Your Google API key may be invalid "
                        "or may not have permission to access "
                        "the Gemini API."
                    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📊 RAG Pipeline")

    st.write(
        """
        **1. Upload Document**
        
        ↓
        
        **2. Extract Text**
        
        ↓
        
        **3. Split into Chunks**
        
        ↓
        
        **4. Create Embeddings**
        
        ↓
        
        **5. Store in FAISS**
        
        ↓
        
        **6. Similarity Search**
        
        ↓
        
        **7. Send Context to Gemini**
        
        ↓
        
        **8. Generate Answer**
        """
    )

    st.divider()

    st.write("### Models")

    st.write(
        "**Generation:** Gemini 2.5 Flash"
    )

    st.write(
        "**Embeddings:** Gemini Embedding 001"
    )

    st.write(
        "**Vector DB:** FAISS"
    )
