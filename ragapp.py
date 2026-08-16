import streamlit as st

from google import genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================================
# PAGE CONFIG
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

if "GOOGLE_API_KEY" not in st.secrets:
    st.error(
        "GOOGLE_API_KEY is missing. "
        "Please add it to Streamlit Secrets."
    )
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]


# =========================================================
# GOOGLE GENAI CLIENT
# =========================================================

client = genai.Client(
    api_key=api_key
)


# =========================================================
# EMBEDDING MODEL
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
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your Word document",
    type=["docx"]
)


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file is not None:

    if uploaded_file.name != st.session_state.uploaded_file_name:

        with st.spinner("Processing document..."):

            # -------------------------------------------------
            # Save uploaded file
            # -------------------------------------------------

            file_path = "uploaded_document.docx"

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())


            # -------------------------------------------------
            # LOAD WORD DOCUMENT
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
            f"Document split into {len(chunks)} chunks."
        )


# =========================================================
# QUESTION ANSWERING
# =========================================================

if st.session_state.vector_db is not None:

    st.divider()

    st.subheader("Ask a question")

    question = st.text_input(
        "Enter your question:",
        placeholder="e.g. What are the main strengths?"
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
                # STEP 2 — BUILD CONTEXT
                # =================================================

                context = "\n\n".join(
                    doc.page_content
                    for doc in retrieved_docs
                )


                # =================================================
                # STEP 3 — BUILD PROMPT
                # =================================================

                prompt = f"""
You are a helpful document assistant.

Answer the user's question using ONLY the information
contained in the document context below.

Rules:

1. Do not use outside knowledge.
2. Do not make up information.
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
                # STEP 4 — GEMINI INTERACTIONS API
                # =================================================

                with st.spinner(
                    "Generating answer with Gemini..."
                ):

                    interaction = client.interactions.create(
                        model="gemini-3.6-flash",
                        input=prompt
                    )

                    answer = interaction.output_text


                # =================================================
                # DISPLAY ANSWER
                # =================================================

                st.subheader("💡 Answer")

                st.markdown(answer)


                # =================================================
                # DISPLAY SOURCES
                # =================================================

                st.divider()

                st.subheader(
                    "🔎 Retrieved Document Sections"
                )

                for i, doc in enumerate(retrieved_docs):

                    with st.expander(
                        f"Source {i + 1}"
                    ):

                        st.write(doc.page_content)


            # =====================================================
            # ERROR HANDLING
            # =====================================================

            except Exception as e:

                st.error(
                    "An error occurred while communicating "
                    "with Gemini."
                )

                st.code(str(e))


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📊 RAG Pipeline")

    st.write(
        """
        **Word Document**
        
        ↓
        
        **Text Extraction**
        
        ↓
        
        **Chunking**
        
        ↓
        
        **Gemini Embeddings**
        
        ↓
        
        **FAISS Vector DB**
        
        ↓
        
        **Similarity Search**
        
        ↓
        
        **Retrieved Context**
        
        ↓
        
        **Gemini 3.6 Flash**
        
        ↓
        
        **Answer**
        """
    )

    st.divider()

    st.write("### Models")

    st.write(
        "**LLM:** Gemini 3.6 Flash"
    )

    st.write(
        "**Embedding:** Gemini Embedding 001"
    )

    st.write(
        "**Vector DB:** FAISS"
    )
