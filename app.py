import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Remove Streamlit default top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        color: #111827;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 35px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 10px;
        background: #f8fafc;
    }

    /* Question input */
    .question-label {
        font-size: 20px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 8px;
    }

    /* Answer card */
    .answer-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 28px 30px;
        margin-top: 20px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.07);
    }

    .answer-heading {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 18px;
        color: #111827;
    }

    .answer-text {
        font-size: 17px;
        line-height: 1.8;
        color: #374151;
    }

    /* Document processed message */
    .document-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        padding: 16px 20px;
        margin-top: 20px;
        color: #166534;
        font-size: 15px;
    }

    /* Source cards */
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .source-title {
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 6px;
    }

    .source-text {
        color: #475569;
        line-height: 1.6;
        font-size: 14px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }

    /* Hide unnecessary Streamlit elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            text = text.strip()

            if text:

                pages.append(
                    {
                        "page": page_number,
                        "text": text
                    }
                )

    return pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    pages,
    chunk_size=700,
    chunk_overlap=100
):

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = page_data["text"]

        # Clean unnecessary whitespace
        text = " ".join(text.split())

        if not text:
            continue

        words = text.split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]

            chunk_text = " ".join(
                chunk_words
            )

            if chunk_text.strip():

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number
                    }
                )

            if end >= len(words):
                break

            start = end - chunk_overlap

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    # Normalize for cosine similarity
    faiss.normalize_L2(
        embeddings
    )

    return embeddings


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# SEARCH FAISS
# ============================================================

def search_faiss(
    question,
    index,
    chunks,
    top_k=5
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(
        question_embedding
    )

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        results.append(
            {
                "text": chunks[index_id]["text"],
                "page": chunks[index_id]["page"],
                "score": float(score)
            }
        )

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
PAGE: {result['page']}

{result['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# ASK GEMINI
# ============================================================

def ask_gemini(
    client,
    question,
    context
):

    prompt = f"""
You are an intelligent PDF question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

IMPORTANT RULES:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the document, say:
   "I couldn't find the answer in the uploaded document."
4. Give a clear, natural and helpful answer.
5. Do not mention that you are an AI unless necessary.
6. Do not repeat the user's question.
7. If the information comes from a specific page,
   mention the page naturally in the answer.
8. Use paragraphs or bullet points when they improve readability.

DOCUMENT CONTEXT
================

{context}

USER QUESTION
=============

{question}

ANSWER
======
"""

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    return interaction.output_text


# ============================================================
# SESSION STATE
# ============================================================

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 PDF RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload a document and ask questions about its contents.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"],
    label_visibility="visible"
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file:

    # Reprocess only when a different document is uploaded
    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Reading and understanding your document..."
        ):

            # ----------------------------------------
            # Extract PDF text
            # ----------------------------------------

            pages = extract_text_from_pdf(
                uploaded_file
            )

            if not pages:

                st.error(
                    "Unable to extract readable text from this PDF."
                )

                st.stop()

            # ----------------------------------------
            # Create chunks
            # ----------------------------------------

            chunks = create_chunks(
                pages,
                chunk_size=700,
                chunk_overlap=100
            )

            if not chunks:

                st.error(
                    "No usable text was found in the PDF."
                )

                st.stop()

            # ----------------------------------------
            # Create embeddings
            # ----------------------------------------

            embeddings = create_embeddings(
                chunks
            )

            # ----------------------------------------
            # Create FAISS vector database
            # ----------------------------------------

            index = create_faiss_index(
                embeddings
            )

            # ----------------------------------------
            # Save to session
            # ----------------------------------------

            st.session_state.faiss_index = index

            st.session_state.chunks = chunks

            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.last_answer = None

            st.session_state.last_sources = None

        st.markdown(
            f"""
            <div class="document-card">
            ✓ <strong>{uploaded_file.name}</strong>
            is ready. You can now ask questions about it.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# QUESTION AREA
# ============================================================

if st.session_state.faiss_index is not None:

    st.markdown(
        '<div class="question-label">💬 Ask your question</div>',
        unsafe_allow_html=True
    )

    question = st.text_input(
        "Question",
        placeholder="Example: What are the main findings of this document?",
        label_visibility="collapsed"
    )

    ask_button = st.button(
        "🔎 Ask Question",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            client = get_gemini_client()

            if client is None:

                st.error(
                    "Gemini API key is not configured."
                )

                st.info(
                    "Please add GEMINI_API_KEY in Streamlit App Settings → Secrets."
                )

                st.stop()

            # ----------------------------------------
            # Retrieve relevant chunks
            # ----------------------------------------

            with st.spinner(
                "Searching the document..."
            ):

                results = search_faiss(
                    question,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    top_k=5
                )

            # ----------------------------------------
            # Build context
            # ----------------------------------------

            context = build_context(
                results
            )

            # ----------------------------------------
            # Generate answer
            # ----------------------------------------

            with st.spinner(
                "Generating your answer..."
            ):

                try:

                    answer = ask_gemini(
                        client,
                        question,
                        context
                    )

                except Exception as e:

                    st.error(
                        "Something went wrong while generating the answer."
                    )

                    st.exception(e)

                    st.stop()

            # Save result
            st.session_state.last_answer = answer
            st.session_state.last_sources = results


# ============================================================
# DISPLAY ANSWER
# ============================================================

if st.session_state.last_answer:

    st.markdown(
        """
        <div class="answer-card">

        <div class="answer-heading">
        🤖 Answer
        </div>

        <div class="answer-text">
        """,
        unsafe_allow_html=True
    )

    # Display answer as normal Streamlit markdown
    # This intentionally does NOT create a custom copy button.
    st.markdown(
        st.session_state.last_answer
    )

    st.markdown(
        """
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SOURCES
# ============================================================

if st.session_state.last_sources:

    st.markdown("")

    with st.expander(
        "📖 View sources used for this answer"
    ):

        for i, result in enumerate(
            st.session_state.last_sources,
            start=1
        ):

            st.markdown(
                f"""
                <div class="source-card">

                <div class="source-title">
                Source {i} · Page {result['page']}
                </div>

                <div class="source-text">
                {result['text']}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    PDF RAG Assistant · FAISS + Sentence Transformers + Gemini
    </div>
    """,
    unsafe_allow_html=True
)
