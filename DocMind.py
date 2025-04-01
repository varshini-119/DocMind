import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st

# Set page config FIRST 
st.set_page_config(
    page_title="DocMind - Smart PDF Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Then load other imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configuration
DB_DIR = "./chroma_db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.makedirs(DB_DIR, exist_ok=True)

# Prompt template
prompt_template = """
Use the following context to answer the question thoroughly.
If the answer isn't in the context, say "I couldn't find this information in the document."

Context: {context}
Question: {question}

Helpful Answer:"""

class GeminiPDFQA:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.uploaded_files = []
        self.qa_prompt = PromptTemplate.from_template(prompt_template)

    def process_pdfs(self, files):
        self.uploaded_files = [file.name for file in files]
        all_texts = []

        for file in files:
            try:
                text = ""
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                all_texts.append(text)
            except Exception as e:
                st.error(f"Error reading {file.name}: {str(e)}")
                continue

        if not all_texts:
            return 0
        
        combined_text = "\n".join(all_texts)
        chunks = self.text_splitter.split_text(combined_text)
        
        try:
            self.vector_db = Chroma.from_texts(
                chunks, 
                embedding=self.embeddings,
                persist_directory=DB_DIR
            )
            return len(chunks)
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            return 0

    def ask(self, question):
        if not self.vector_db:
            return "Please upload PDFs first"
        
        try:
            qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_db.as_retriever(),
                chain_type_kwargs={"prompt": self.qa_prompt}
            )
            return qa.invoke({"query": question})["result"]
        except Exception as e:
            return f"Error answering question: {str(e)}"

def main():
    st.title("📚 DocMind")
    st.caption("Your Smart Document Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processor" not in st.session_state:
        st.session_state.processor = GeminiPDFQA()
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    with st.sidebar:
        st.header("📂 Document Management")
        st.markdown("---")
        
        uploaded_files = st.file_uploader(
            "Upload your PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select multiple PDF files to analyze"
        )

        if uploaded_files:
            uploaded_file_names = [file.name for file in uploaded_files]
            st.session_state.uploaded_files = [file for file in st.session_state.uploaded_files if file in uploaded_file_names]
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(file.name)

        if st.button("Process Documents", use_container_width=True) and uploaded_files:
            with st.spinner("Analyzing documents..."):
                chunk_count = st.session_state.processor.process_pdfs(uploaded_files)
                if chunk_count > 0:
                    st.session_state.processed = True
                    st.session_state.messages = []
                    st.success(f"Processed {len(uploaded_files)} documents successfully")
                else:
                    st.error("Failed to process documents")
        
        st.markdown("---")
        if st.session_state.uploaded_files:
            st.subheader("📚 Uploaded Documents")
            for filename in st.session_state.uploaded_files:
                    st.markdown(f"- {filename}")
        
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("1. Upload PDF documents")
        st.markdown("2. Click 'Process Documents'")
        st.markdown("3. Ask questions in the chat")
        st.markdown("---")
        
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            avatar = "👤" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask anything about your documents..."):
        if not st.session_state.processed:
            st.warning("⚠️ Please upload and process PDFs first")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt, unsafe_allow_html=True)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.processor.ask(prompt)
                        st.markdown(response, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

