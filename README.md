# 📚 Multi PDF Knowledge Generator

## 📖 Overview

**Multi PDF Knowledge Generator** is an AI-powered document question-answering application that enables users to upload multiple PDF files, extract their contents, and interact with them using natural language. The application leverages semantic search and Google's Gemini AI to generate accurate, context-aware responses based solely on the uploaded documents.

This project is built with **Streamlit**, **Google Gemini API**, **LangChain**, **Chroma Vector Database**, and **Hugging Face Embeddings**, providing an intuitive interface for students, researchers, educators, and professionals to quickly understand large collections of PDF documents.

---

## 🚀 Features

* 📂 Upload and process multiple PDF files simultaneously.
* 📄 Automatically extract text from every page.
* ✂️ Split large documents into semantic chunks.
* 🧠 Generate vector embeddings using Hugging Face.
* 🔍 Perform semantic similarity search with Chroma Vector Store.
* 🤖 Generate AI-powered answers using Google Gemini.
* 📚 Restrict answers to uploaded PDF content only.
* 🎯 Multiple output formats:

  * Detailed Summary
  * Bullet Points
  * Simple Explanation
  * 5-Mark Questions
  * 10-Mark Questions
  * Multiple Choice Questions (MCQs)
* ✍️ Support for custom user instructions.
* 📊 Display word count and character count.
* 📥 Download AI-generated responses as a text file.
* 📖 Display source PDF names and page numbers used to generate answers.
* 🎨 Modern, responsive Streamlit interface with custom CSS styling.

---

## 🛠️ Tech Stack

| Technology              | Purpose                   |
| ----------------------- | ------------------------- |
| Python                  | Core Programming Language |
| Streamlit               | Web Application Framework |
| Google Gemini API       | AI Response Generation    |
| LangChain               | Text Processing Pipeline  |
| Chroma DB               | Vector Database           |
| Hugging Face Embeddings | Text Embeddings           |
| PyPDF                   | PDF Text Extraction       |

---

## 📂 Project Structure

```text
Multi-PDF-Knowledge-Generator/
│
├── app.py                 # Main Streamlit Application
├── requirements.txt       # Project Dependencies
├── README.md              # Project Documentation
└── assets/                # Images or screenshots (optional)
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Multi-PDF-Knowledge-Generator.git
```

```bash
cd Multi-PDF-Knowledge-Generator
```

---

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add Your Gemini API Key

Replace the following line in the code:

```python
API_KEY = ""
```

with:

```python
API_KEY = "YOUR_GEMINI_API_KEY"
```

---

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 📌 How It Works

1. Upload one or more PDF documents.
2. Extract text from every page.
3. Convert extracted text into document chunks.
4. Generate semantic embeddings using Hugging Face.
5. Store embeddings inside Chroma Vector Database.
6. Enter a question or topic.
7. Retrieve the most relevant document chunks.
8. Send the retrieved context to Google Gemini.
9. Display the generated answer along with the document sources.

---

## 🔄 Workflow

```text
Upload PDFs
      │
      ▼
Extract Text
      │
      ▼
Split into Chunks
      │
      ▼
Generate Embeddings
      │
      ▼
Store in Chroma Vector Database
      │
      ▼
User Query
      │
      ▼
Similarity Search
      │
      ▼
Gemini AI
      │
      ▼
Context-Aware Answer
```

---

## 📸 Application Features

* Multi PDF Upload
* Automatic PDF Processing
* AI Question Answering
* Semantic Search
* Multiple Answer Formats
* Download Generated Answers
* Source Page References
* Word and Character Count

---

## 🎯 Supported Output Formats

* Detailed Summary
* Bullet Points
* Simple Explanation
* 5-Mark Questions
* 10-Mark Questions
* Multiple Choice Questions (MCQs)

---

## 💡 Future Enhancements

* Support for DOCX, TXT, and PowerPoint files.
* Chat history and conversation memory.
* Voice-based document querying.
* OCR support for scanned PDF documents.
* PDF highlighting for retrieved answers.
* User authentication and document management.
* Export responses to PDF and Word formats.
* Cloud deployment for online access.

---

## 👨‍💻 Applications

* Academic Research
* Competitive Exam Preparation
* University Study Material
* Company Documentation Search
* Legal Document Analysis
* Technical Manuals
* Healthcare Reports
* Business Documentation

---

## 📈 Key Concepts Used

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Embeddings
* Vector Databases
* Large Language Models (LLMs)
* Natural Language Processing (NLP)
* Document Chunking
* Context-Aware Question Answering

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* Google Gemini API
* Streamlit
* LangChain
* Hugging Face
* Chroma DB
* PyPDF

---

## ⭐ If you found this project useful

Please consider giving the repository a **Star ⭐** on GitHub to support future development.
