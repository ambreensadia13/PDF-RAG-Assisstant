import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7fa;
        border: 1px solid #ddd;
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

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        return None

    client = genai.Client(
        api_key=api_key
    )

    return client


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

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

        # Normalize whitespace
        text = " ".join(text.split())

        if not text:
            continue

        words = text.split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

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

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    return embeddings


# ============================================================
# CREATE FAISS DATABASE
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
# SEARCH VECTOR DATABASE
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

    # Normalize for cosine similarity
    faiss.normalize_L2(
        question_embedding
    )

    scores, indices = index.search(
        question_embedding,
        top_k
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
# CREATE RAG CONTEXT
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
Page: {result['page']}

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
You are a document question-answering assistant.

Your task is to answer the user's question using ONLY
the information provided in the retrieved document context.

Do not invent facts.

If the answer cannot be found in the provided context,
clearly say:

"I couldn't find the answer in the uploaded document."

When possible, mention the page number where the information
was found.

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
    Upload a PDF and ask questions about its contents using
    Retrieval-Augmented Generation.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    chunk_size = st.slider(
        "Chunk size",
        min_value=300,
        max_value=1200,
        value=700,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=50,
        max_value=300,
        value=100,
        step=50
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()

    st.info(
        """
        Embedding model:

        all-MiniLM-L6-v2

        Vector database:

        FAISS

        LLM:

        Gemini 3.8 Flash
        """
    )


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file:

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            # --------------------------------
            # Extract text
            # --------------------------------

            pages = extract_text_from_pdf(
                uploaded_file
            )

            if not pages:

                st.error(
                    "Could not extract text from this PDF."
                )

                st.stop()

            # --------------------------------
            # Create chunks
            # --------------------------------

            chunks = create_chunks(
                pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            if not chunks:

                st.error(
                    "No usable text chunks were created."
                )

                st.stop()

            # --------------------------------
            # Create embeddings
            # --------------------------------

            embeddings = create_embeddings(
                chunks
            )

            # --------------------------------
            # Create FAISS index
            # --------------------------------

            index = create_faiss_index(
                embeddings
            )

            # --------------------------------
            # Store in session state
            # --------------------------------

            st.session_state.faiss_index = index

            st.session_state.chunks = chunks

            st.session_state.document_name = (
                uploaded_file.name
            )

        st.success(
            f"PDF processed successfully! "
            f"Created {len(chunks)} chunks."
        )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.faiss_index is not None:

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Document",
            st.session_state.document_name
        )

    with col2:

        st.metric(
            "Chunks",
            len(st.session_state.chunks)
        )

    with col3:

        st.metric(
            "Vector DB",
            "FAISS"
        )


# ============================================================
# QUESTION SECTION
# ============================================================

if st.session_state.faiss_index is not None:

    st.divider()

    st.subheader(
        "💬 Ask a question"
    )

    question = st.text_input(
        "Enter your question",
        placeholder="What is this document about?"
    )

    ask_button = st.button(
        "🔎 Ask",
        type="primary"
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            client = get_gemini_client()

            if client is None:

                st.error(
                    "Gemini API key is not configured."
                )

                st.info(
                    "Add GEMINI_API_KEY to Streamlit Secrets."
                )

                st.stop()

            # --------------------------------
            # Retrieve relevant chunks
            # --------------------------------

            with st.spinner(
                "Searching the document..."
            ):

                results = search_faiss(
                    question,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    top_k=top_k
                )

            # --------------------------------
            # Build context
            # --------------------------------

            context = build_context(
                results
            )

            # --------------------------------
            # Generate answer
            # --------------------------------

            with st.spinner(
                "Generating answer..."
            ):

                try:

                    answer = ask_gemini(
                        client,
                        question,
                        context
                    )

                except Exception as e:

                    st.error(
                        "Gemini API error:"
                    )

                    st.exception(e)

                    st.stop()

            # --------------------------------
            # Display answer
            # --------------------------------

            st.subheader(
                "🤖 Answer"
            )

            st.markdown(
                f"""
                <div class="answer-box">

                {answer}

                </div>
                """,
                unsafe_allow_html=True
            )

            # --------------------------------
            # Retrieved sources
            # --------------------------------

            with st.expander(
                "📚 View retrieved sources"
            ):

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        f"### Source {i}"
                    )

                    st.write(
                        f"**Page:** {result['page']}"
                    )

                    st.write(
                        f"**Similarity:** "
                        f"{result['score']:.4f}"
                    )

                    st.write(
                        result["text"]
                    )

                    st.divider()
