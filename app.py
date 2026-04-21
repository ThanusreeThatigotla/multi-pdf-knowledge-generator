import streamlit as st
from google import genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# -----------------------------
# 🔐 Put your API Key here
# -----------------------------
API_KEY = ""  # <-- Replace with your real key
client = genai.Client(api_key=API_KEY)

# -----------------------------
# 🖥️ Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Multi-PDF Knowledge Generator", layout="wide")

# -----------------------------
# 🎨 Custom Background and Styling
# -----------------------------
st.markdown(
    """
    <style>
    /* ----------------- */
    /* Bright Gradient Background */
    /* ----------------- */
    .stApp {
        background: linear-gradient(135deg, #a1c4fd, #c2e9fb); /* soft blue gradient */
        position: relative;
        min-height: 100vh;
    }

    /* Optional light overlay for text clarity */
    .stApp::after {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255,255,255,0.1); /* subtle white overlay */
        z-index: 0;
    }

    /* ----------------- */
    /* Main Title Styling */
    /* ----------------- */
    h1 {
        color: #0077b6; /* deep blue title */
        text-shadow: 1px 1px 3px #a1c4fd; /* soft glow */
        font-family: 'Segoe UI', sans-serif;
        position: relative;
        z-index: 1;
    }

    /* ----------------- */
    /* Inputs and Buttons */
    /* ----------------- */
    .stButton button, .stFileUploader, .stTextInput, .stTextArea, .stSelectbox {
        background-color: rgba(255,255,255,0.95);
        border-radius: 12px;
        padding: 12px;
        transition: all 0.2s ease;
        position: relative;
        z-index: 1;
    }

    /* Hover effect for buttons */
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

    /* Hover effect for input fields */
    .stTextInput:hover, .stTextArea:hover, .stFileUploader:hover, .stSelectbox:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    }

    /* ----------------- */
    /* AI Answer Box Styling */
    /* ----------------- */
    .stMarkdown p {
        color: #03045e; /* navy text */
        font-size: 16px;
        background: linear-gradient(135deg, #90e0ef, #caf0f8); /* bright blue gradient */
        border-radius: 14px;
        padding: 15px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        z-index: 1;
    }

    /* Hover effect for AI answer */
    .stMarkdown p:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }

    /* ----------------- */
    /* Headings / Sources Styling */
    /* ----------------- */
    .stMarkdown h2 {
        color: #0077b6; /* matching title color */
        text-shadow: 1px 1px 3px #90e0ef; /* soft glow */
        position: relative;
        z-index: 1;
    }

    /* ----------------- */
    /* Scrollbar Styling */
    /* ----------------- */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.2);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #0077b6, #00b4d8);
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
st.title("📚 Multi-PDF Knowledge Generator")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# -----------------------------
# PDF Upload
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF file(s)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    st.success("✅ PDF successfully uploaded!")
    st.subheader("📄 Uploaded PDF Summary")
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        st.write(f"{file.name} → {len(pdf_reader.pages)} pages")

# -----------------------------
# Extract Text + Metadata
# -----------------------------
def extract_documents(files):
    documents = []
    for file in files:
        pdf_reader = PdfReader(file)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file.name, "page": page_num + 1}
                    )
                )
    return documents

# -----------------------------
# Chunk Documents
# -----------------------------
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# -----------------------------
# Create Vector Store
# -----------------------------
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_store

# -----------------------------
# Process PDFs
# -----------------------------
if uploaded_files and st.button("Process PDFs"):
    with st.spinner("Processing PDFs..."):
        docs = extract_documents(uploaded_files)
        chunks = chunk_documents(docs)
        st.session_state.vector_store = create_vector_store(chunks)
    st.success("✅ PDFs processed successfully!")

# -----------------------------
# Enter User Query + Options
# -----------------------------
query = st.text_input("Enter Topic / Question")

# Output format selection
format_option = st.selectbox(
    "Select Output Format",
    ["Detailed Summary", "Bullet Points", "5-Mark Questions", "10-Mark Questions", "MCQs", "Simple Explanation"]
)

# Optional custom instruction
custom_instruction = st.text_area("Enter extra instructions (optional)")

# -----------------------------
# Generate AI Answer
# -----------------------------
if query and st.session_state.vector_store:
    with st.spinner("🤖 AI is thinking..."):
        relevant_docs = st.session_state.vector_store.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        instruction = f"Format output as: {format_option}"
        if custom_instruction.strip() != "":
            instruction += f". Also, {custom_instruction.strip()}"

        prompt = f"""
You are an intelligent academic assistant.

Use ONLY the provided context.

Context:
{context}

Question/Topic:
{query}

Instruction:
{instruction}
"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.subheader("📄 AI Answer")
            highlighted_text = response.text.replace("AI", "**AI**").replace("PDF", "**PDF**")
            st.markdown(highlighted_text)

            # Word and character count
            st.write(f"**Word Count:** {len(response.text.split())}")
            st.write(f"**Character Count:** {len(response.text)}")

            # Download answer
            st.download_button("📥 Download Answer as Text", response.text, file_name="AI_Answer.txt")

        except Exception as e:
            st.error(f"Error: {e}")

        # Show sources
        st.subheader("📚 Sources")
        for doc in relevant_docs:
            st.write(f"📄 {doc.metadata['source']} | Page {doc.metadata['page']}")