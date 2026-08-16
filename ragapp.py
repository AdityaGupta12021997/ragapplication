import streamlit as st

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    layout="wide"
)

st.title("📄 RAG Document Assistant")

st.write(
    "Upload a Word document and ask questions about its contents."
)


# ============================================================
# 2. GOOGLE API KEY
# ============================================================

if "GOOGLE_API_KEY" in st.secrets:

    api_key = st.secrets["GOOGLE_API_KEY"]

else:

    st.error(
        "Please add GOOGLE_API_KEY to Streamlit Secrets."
    )

    st.stop()


# ============================================================
# 3. INITIALIZE GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.2
)


# ============================================================
# 4. UPLOAD WORD DOCUMENT
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your Word document",
    type=["docx"]
)


# ============================================================
# 5. PROCESS DOCUMENT
# ============================================================

if uploaded_file is not None:

    # Save uploaded file temporarily
    with open("uploaded_document.docx", "wb") as f:
        f.write(uploaded_file.getbuffer())


    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    # ========================================================
    # 6. LOAD DOCUMENT
    # ========================================================

    loader = Docx2txtLoader(
        "uploaded_document.docx"
    )

    documents = loader.load()


    # ========================================================
    # 7. SPLIT DOCUMENT INTO CHUNKS
    # ========================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(
        documents
    )


    st.info(
        f"Document split into {len(chunks)} chunks."
    )


    # ========================================================
    # 8. CREATE EMBEDDINGS
    # ========================================================

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )


    # ========================================================
    # 9. CREATE VECTOR DATABASE
    # ========================================================

    with st.spinner(
        "Creating vector database..."
    ):

        vector_db = FAISS.from_documents(
            chunks,
            embeddings
        )


    st.success(
        "Document successfully indexed!"
    )


    # ========================================================
    # 10. ASK QUESTION
    # ========================================================

    question = st.text_input(
        "Ask a question about the document:"
    )


    if st.button("Ask"):

        if question:

            with st.spinner(
                "Searching document and generating answer..."
            ):

                # --------------------------------------------
                # Retrieve relevant chunks
                # --------------------------------------------

                retrieved_docs = vector_db.similarity_search(
                    question,
                    k=3
                )


                # --------------------------------------------
                # Combine retrieved chunks
                # --------------------------------------------

                context = "\n\n".join(
                    doc.page_content
                    for doc in retrieved_docs
                )


                # --------------------------------------------
                # Prompt
                # --------------------------------------------

                prompt = f"""
You are a helpful document assistant.

Answer the user's question using ONLY the
information provided in the context.

If the answer cannot be found in the context,
say:

"I could not find the answer in the document."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


                # --------------------------------------------
                # Send to Gemini
                # --------------------------------------------

                response = llm.invoke(
                    prompt
                )

                answer = response.content


            # =================================================
            # 11. DISPLAY ANSWER
            # =================================================

            st.subheader("Answer")

            st.markdown(answer)


            # =================================================
            # 12. DISPLAY RETRIEVED SOURCES
            # =================================================

            st.subheader(
                "🔎 Retrieved Document Sections"
            )

            for i, doc in enumerate(
                retrieved_docs
            ):

                with st.expander(
                    f"Source {i + 1}"
                ):

                    st.write(
                        doc.page_content
                    )

        else:

            st.warning(
                "Please enter a question."
            )