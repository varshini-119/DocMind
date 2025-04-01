# DocMind - Smart PDF Assistant

## Overview
DocMind is an AI-powered PDF assistant that allows users to upload and analyze PDF documents. Users can ask questions about the content, and the assistant will retrieve relevant information using embeddings and a language model.

## Features
- 📂 **Upload Multiple PDFs**
- 🔍 **Extract and Process Text**
- 🧠 **Vector Database for Efficient Retrieval**
- 🤖 **AI-Powered Question Answering**
- 🎭 **Interactive Chat Interface**

## Tech Stack
- **Backend**: Python, LangChain, PyPDF2, ChromaDB
- **Frontend**: Streamlit
- **AI Models**:
  - Google Generative AI Embeddings (`models/embedding-001`)
  - ChatGoogleGenerativeAI (`gemini-1.5-flash`)

## Installation

1. Clone the repository:
   git clone https://github.com/your-repo/docmind.git
   cd docmind
   
3. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

4. Install dependencies:
   pip install -r requirements.txt
 
5. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add the following line:
     GEMINI_API_KEY=your_google_gemini_api_key
  
6. Run the application:
   streamlit run DocMind.py

## Usage
1. Upload one or more PDF documents.
2. Click **Process Documents** to analyze the files.
3. Ask questions about the documents via the chat interface.

## File Structure
```
📦 docmind
┣ 📜 DocMind.py               # Main Streamlit application
┣ 📜 requirements.txt     # Dependencies
┣ 📜 .env                 # Environment variables (not included in repo)
┣ 📜 README.md            # Documentation
┣ 📂 chroma_db            # Vector database storage
```

## Troubleshooting
- **Error reading PDF:** Ensure the file is not corrupted.
- **Failed to process documents:** Check if `GEMINI_API_KEY` is correctly set in `.env`.
- **No response to queries:** Make sure documents were processed before asking questions.

## Future Improvements
- Support for additional document formats (DOCX, TXT, etc.)
- Enhanced document visualization
- Integration with cloud storage services

