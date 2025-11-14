import streamlit as st
import os
import faiss
import tempfile
from PIL import Image
import pytesseract
from ultralytics import YOLO 
import cv2 
import logging # Import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LangChain and Google Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document # Import Document object

# Note: Using langchain_classic imports due to existing code structure, 
# but modifying chain invocation for stability.

from dotenv import load_dotenv

# --- Flan-T5 Imports ---
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded.")

# --- Configuration ---
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"
LLM_TIMEOUT = 300
FAISS_INDEX_PATH = "faiss_index"

# NOTE: Set the Tesseract executable path manually
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    pytesseract.get_tesseract_version()
    logger.info(f"Tesseract version check successful.")
except pytesseract.TesseractNotFoundError:
    st.warning("Tesseract OCR is not found. Please install it or correct the path in the code.")
    logger.error("Tesseract OCR not found. Check installation and path.")


# --- CV Models ---
try:
    DIAGRAM_DETECTION_MODEL = YOLO("yolov8n.pt") 
    logger.info("YOLO model yolov8n.pt loaded successfully.")
except Exception as e:
    st.warning(f"Could not load YOLO model: {e}. Diagram detection will be disabled.")
    logger.error(f"Could not load YOLO model: {e}")
    DIAGRAM_DETECTION_MODEL = None


# Streamlit app title
st.title("🧠 BITS Pilani Smart AI Assistant ")

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.session_state.is_processing = False
    st.session_state.current_mode = "Q&A (Text RAG)"
    st.session_state.temp_result = None 


# ---------------------------------------------
# CORE RAG Setup Functions
# ---------------------------------------------

@st.cache_resource
def initialize_embeddings_model():
    logger.info(f"Initializing embeddings model: {EMBEDDING_MODEL}")
    try:
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Error initializing embeddings model: {e}")
        st.error(f"Error initializing embeddings model: {e}")
        raise

@st.cache_resource
def setup_llm():
    logger.info(f"Initializing LLM: {LLM_MODEL}")
    try:
        return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, timeout=LLM_TIMEOUT)
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        st.error(f"Error initializing LLM: {e}")
        raise

def load_and_merge_pdfs(uploaded_files, embedding_model, faiss_path):
    # ... (remains the same)
    if st.session_state.is_processing:
        logger.warning("Attempted to process while already processing.")
        return

    st.session_state.is_processing = True
    temp_pdf_paths = []
    temp_dir = "temp_pdfs"
    logger.info(f"Starting PDF processing for {len(uploaded_files)} files.")

    try:
        with st.spinner(f"Processing {len(uploaded_files)} PDF(s) and creating/updating embeddings..."):
            
            # 1. Save all uploaded files temporarily
            os.makedirs(temp_dir, exist_ok=True)
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_pdf_paths.append(temp_path)
            logger.info(f"Saved {len(temp_pdf_paths)} files temporarily to {temp_dir}.")

            # --- Core RAG Logic ---
            vectorstore = None
            pdf_paths_to_process = temp_pdf_paths
            
            # Attempt to load existing index
            try:
                vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
                st.info(f"Loaded existing FAISS index. Merging new documents.")
                logger.info(f"Existing FAISS index loaded from {faiss_path}.")

            except Exception:
                logger.warning("FAISS index not found. Will attempt to create a new one.")
                if not temp_pdf_paths:
                    st.warning("No FAISS index found and no PDFs provided to create one.")
                    return

                # Create a new index from the first uploaded PDF
                st.info("FAISS index not found. Creating a new one from the first PDF.")
                loader = PyPDFLoader(temp_pdf_paths[0])
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(data)
                vectorstore = FAISS.from_documents(docs, embedding_model)
                pdf_paths_to_process = temp_pdf_paths[1:]
                logger.info(f"New FAISS index created from {os.path.basename(temp_pdf_paths[0])}.")
                
            # Process remaining PDFs and add documents to the store
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            for pdf_path in pdf_paths_to_process:
                st.info(f"Adding documents from: {os.path.basename(pdf_path)}")
                logger.info(f"Processing and adding documents from: {os.path.basename(pdf_path)}")
                loader = PyPDFLoader(pdf_path)
                data = loader.load()
                docs = text_splitter.split_documents(data)
                vectorstore.add_documents(docs)

            # Save the updated vector store
            vectorstore.save_local(faiss_path)
            st.session_state.vectorstore = vectorstore
            st.success(f"Successfully processed {len(uploaded_files)} PDF(s) and updated the vector store.")
            logger.info(f"FAISS index saved/updated at {faiss_path}.")

    except Exception as e:
        st.error(f"Error during file processing: {e}")
        logger.error(f"Error during file processing: {e}", exc_info=True)
    finally:
        st.session_state.is_processing = False
        # Clean up temporary files
        for temp_path in temp_pdf_paths:
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                logger.warning(f"Could not remove temp directory {temp_dir} fully.")
                pass 

def setup_retriever():
    if st.session_state.vectorstore:
        logger.debug("Setting up vectorstore retriever.")
        return st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    logger.debug("Vectorstore not available, retriever setup skipped.")
    return None

def setup_prompt_template(task="qa"):
    logger.debug(f"Setting up prompt template for task: {task}")
    if task == "qa":
        system_prompt = (
            "You are an assistant for question-answering tasks based on the provided document set. "
            "Use the following retrieved context to answer the question accurately and professionally. "
            "If the context does not contain the answer, politely state that you cannot find the answer in the provided documents. "
            "Keep the answer concise and strictly based on the facts in the context."
            "\n\n"
            "Context: {context}"
        )
    else:
        system_prompt = (
            "You are an assistant. Use the following retrieved context to answer the question. "
            "Context: {context}"
        )

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

# ---------------------------------------------
# Flan-T5 Chatbot Integration
# ---------------------------------------------
@st.cache_resource
def load_flan_t5_chatbot():
    """Loads the Flan-T5 model and tokenizer for fallback."""
    logger.info("Attempting to load Flan-T5 model.")
    try:
        # Hugging Face model repo
        fine_tuned_model_name = "naidu9678/flan_t5_finetuned_education"
        
        # Load tokenizer and model from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_name)
        
        # Create text2text-generation pipeline
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Flan-T5 model loaded successfully. Device: {'CUDA' if device == 0 else 'CPU'}")
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        logger.error(f"Error loading Flan-T5 chatbot: {e}", exc_info=True)
        st.error(f"Error loading Flan-T5 chatbot: {e}")
        return None

flan_t5_chatbot = load_flan_t5_chatbot()

def call_flan_t5(question, context=""):
    """Calls the Flan-T5 chatbot with a question and optional context."""
    logger.info(f"Calling Flan-T5 fallback. Question: {question[:50]}...")
    if flan_t5_chatbot is None:
        logger.error("Flan-T5 pipeline is None.")
        return "Flan-T5 chatbot is not available."

    input_text = f"question: {question} context: {context}"
    try:
        outputs = flan_t5_chatbot(
            input_text,
            max_new_tokens=100,
            do_sample=True,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=0.7
        )
        logger.info("Flan-T5 response generated successfully.")
        return outputs[0]['generated_text']
    except Exception as e:
        logger.error(f"Error generating response from Flan-T5: {e}", exc_info=True)
        return f"Error generating response from fallback model: {e}"

def process_query(query, retriever, llm, prompt):
    """
    Processes a query using RAG. If RAG context is insufficient, falls back to Flan-T5.
    """
    logger.info(f"Starting RAG process for query: {query[:50]}...")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        # First, try to get a response from the RAG chain
        response = rag_chain.invoke({"input": query})
        logger.info("RAG chain invoked successfully.")
        
        # Check if the retrieved context is effectively "blank" or unhelpful
        retrieved_context_docs = response.get("context", [])
        retrieved_context_str = "\n".join([doc.page_content for doc in retrieved_context_docs]).strip()
        
        # Define what constitutes a "blank" response from RAG
        if not retrieved_context_str or "cannot find the answer in the provided documents" in response.get("answer", "").lower():
            logger.warning("RAG context was insufficient or LLM denied answer. Initiating Flan-T5 fallback.")
            st.info("RAG context was insufficient. Falling back to Flan-T5 chatbot...")
            
            # Pass the original query and any meager context to Flan-T5
            flan_t5_answer = call_flan_t5(query, retrieved_context_str)
            
            # Return in the expected dictionary format for chat display
            return {"answer": f"**(Fallback - General Knowledge)**\n\n{flan_t5_answer}"} 
        
        logger.info("RAG provided a satisfactory answer.")
        return response
    except Exception as e:
        logger.error(f"Error during RAG processing: {e}. Falling back to Flan-T5 chatbot.", exc_info=True)
        st.warning(f"Error during RAG processing: {e}. Falling back to Flan-T5 chatbot...")
        # If RAG chain completely fails, just call Flan-T5 with the query
        flan_t5_answer = call_flan_t5(query)
        return {"answer": f"**(Fallback - RAG Error)**\n\n{flan_t5_answer}"}


# ---------------------------------------------
# Multimodal & NLP Logic Functions (Tasks 1, 2, 3, 4)
# ---------------------------------------------

def handle_document_ocr(file_path):
    """
    Handles OCR for both image files and PDF files.
    Returns the extracted text content or an error message.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    logger.info(f"Starting OCR/Text extraction for file: {os.path.basename(file_path)}")
    
    if file_extension in ['.png', '.jpg', '.jpeg']:
        try:
            text = pytesseract.image_to_string(Image.open(file_path))
            logger.info("Image OCR successful.")
            return text, None
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract not accessible for image OCR.")
            return None, "Tesseract is not accessible."
        except Exception as e:
            logger.error(f"Error during image OCR: {e}", exc_info=True)
            return None, f"Error during image OCR: {e}"
            
    elif file_extension == '.pdf':
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text_content = "\n".join([page.page_content for page in pages])
            logger.info("PDF text extraction successful.")
            return text_content, None
        except Exception as e:
            logger.error(f"Error during PDF processing: {e}", exc_info=True)
            return None, f"Error during PDF processing: {e}"
            
    logger.warning(f"Unsupported file type for OCR: {file_extension}")
    return None, "Unsupported file type for OCR."


def process_image_for_grading(script_path, key_path, llm):
    logger.info("Starting CV + LLM Comparative Grading.")
    st.info("Executing Pure CV + LLM Comparative Grading...")
    
    # 1. OCR (CV Step)
    student_text, student_error = handle_document_ocr(script_path)
    if student_error: return f"Could not process student script: {student_error}"
    if not student_text.strip(): return "Could not extract readable text from the student script."
        
    key_text, key_error = handle_document_ocr(key_path)
    if key_error: return f"Could not process answer key: {key_error}"
    if not key_text.strip(): return "Could not extract readable text from the answer key."
    logger.info("OCR phase complete. Extracted text from student script and key.")

    # 2. Setup 
    assignment_question = st.session_state.assignment_question_input
    if not assignment_question: return "Please provide the original assignment question."

    # 3. Generation & Comparison (LLM Step)
    grading_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert academic grader. Compare 'Student Answer' against 'Answer Key'. "
            "Provide a score out of 10 and detailed feedback on key concepts correct vs. missed."
            "\n\nOriginal Question: {question}"
            "\n\n--- Answer Key: ---\n{key_text}"
            "\n\n--- Student Answer: ---\n{student_answer}"
            "\n\n--- Retrieved Context (Ignore this field): {context}" 
        )),
        ("human", "Grade the submission and provide a detailed review.")
    ])
    
    grading_chain = create_stuff_documents_chain(llm, grading_prompt)
    logger.info("Invoking LLM for grading comparison.")
    
    response = grading_chain.invoke({
        "input": "Grade the student's submission.", 
        "student_answer": student_text,
        "question": assignment_question,
        "key_text": key_text,
        "context": "" # Pass empty context to satisfy create_stuff_documents_chain
    })
    logger.info("Grading completed by LLM.")
    
    return f"**📝 Automated Grading Result**\n\n{response}"


def process_image_for_explanation(image_path, llm, retriever):
    logger.info("Starting Diagram/Chart Explanation (CV+RAG).")
    st.info("Executing Diagram/Chart Explanation...")
    
    if not DIAGRAM_DETECTION_MODEL: 
        logger.warning("YOLO model not available.")
        return "YOLO model not loaded. Cannot perform detection."
    if retriever is None: 
        logger.error("Document retriever not initialized for explanation.")
        return "Error: Document retriever is not initialized."

    # 1. OCR (CV Step - Simplified)
    try:
        image_text = pytesseract.image_to_string(Image.open(image_path))
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract not accessible for image OCR during explanation.")
        return "Tesseract is not accessible."
    
    if not image_text: 
        logger.warning("No text extracted from image for context.")
        return "Could not extract any labels or text from the image for context."
    logger.info(f"Extracted OCR text: {image_text[:50]}...")

    # 2. RAG Input (Combination)
    explanation_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert tutor. Explain the chart/diagram below. "
            "Analyze data/labels and use the retrieved context to provide a detailed explanation."
            "\n\nDiagram Text/Labels: {ocr_text}"
            "\n\nContext: {context}"
        )),
        ("human", "Explain the concept shown in this diagram/chart.")
    ])
    
    # 3. Retrieval & Generation
    search_query = f"Explain the concept for diagram with labels: {image_text[:100]}..."
    logger.info(f"RAG Search Query: {search_query}")
    
    question_answer_chain = create_stuff_documents_chain(llm, explanation_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({
        "input": search_query, 
        "ocr_text": image_text 
    })
    logger.info("Diagram explanation completed via RAG.")
    
    return f"**📊 Diagram Explanation Result**\n\n{response['answer']}"


def summarize_content(source_docs, topic, llm):
    """
    (Task 2) RAG-based Educational Content Summarization.
    Accepts List[Document] from either uploaded file or retriever.
    """
    logger.info(f"Starting summarization for topic: {topic}")

    if not source_docs:
        logger.warning("No source documents provided for summarization.")
        return "No source content provided for summarization."

    # 1. Prompt for Summarization
    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an educational assistant. Summarize the following retrieved content "
            "in clear, concise bullet points suitable for a student review. "
            "Focus on key concepts, definitions, and main takeaways only."
            "\n\nContent to Summarize: {context}"
        )),
        ("human", "Generate a summary based on the retrieved content.")
    ])

    # 2. Generation (Stuffing retrieved documents into the prompt)
    summarization_chain = create_stuff_documents_chain(llm, summarization_prompt)
    
    response = summarization_chain.invoke({
        "input": topic, 
        "context": source_docs 
    })
    logger.info("Summarization completed by LLM.")

    return f"**📖 Summary for: {topic}**\n\n{response}"


def generate_questions(source_docs, topic, difficulty, num_questions, llm):
    """
    (Task 3) RAG-based Adaptive Question Generation.
    Accepts List[Document] from either uploaded file or retriever.
    """
    logger.info(f"Starting question generation for topic: {topic}, difficulty: {difficulty}, count: {num_questions}")

    if not source_docs:
        logger.warning("No source documents provided for question generation.")
        return "No source content provided to generate questions from."

    # 1. Prompt for Question Generation
    question_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            f"You are a sophisticated test generator. Based *only* on the context provided, "
            f"generate {num_questions} unique, challenging questions for a student at the {difficulty} level. "
            f"Format the output clearly with the question and the corresponding answer/explanation. "
            f"Ensure the questions test conceptual understanding."
            "\n\nSource Context: {context}"
        )),
        ("human", f"Generate the questions on the topic: {topic}.")
    ])

    # 2. Generation 
    generation_chain = create_stuff_documents_chain(llm, question_prompt)
    
    response = generation_chain.invoke({
        "input": topic,
        "context": source_docs 
    })
    logger.info("Question generation completed by LLM.")

    return f"**❓ Adaptive Questions ({difficulty}, {num_questions}) for: {topic}**\n\n{response}"


# ---------------------------------------------
# Streamlit Interface Logic
# ---------------------------------------------

embeddings_model = initialize_embeddings_model()
llm = setup_llm()

# 1. Sidebar for File Upload and Index Creation
with st.sidebar:
    st.header("1. Document Management")
    
    # Check if existing index is present on disk and attempt to load
    if os.path.exists(FAISS_INDEX_PATH) and st.session_state.vectorstore is None:
        with st.spinner("Loading vector store from disk..."):
            try:
                st.session_state.vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
                st.success("Vector store loaded successfully.")
                logger.info("Vector store loaded successfully from disk during sidebar setup.")
            except Exception as e:
                st.warning(f"Could not load FAISS index: {e}")
                logger.error(f"Could not load FAISS index during startup: {e}", exc_info=True)

    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Choose PDF file(s) to upload or merge", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    # Process button
    if uploaded_files and st.button("Process & Update Vector Store", disabled=st.session_state.is_processing):
        logger.info("User triggered 'Process & Update Vector Store'.")
        load_and_merge_pdfs(uploaded_files, embeddings_model, FAISS_INDEX_PATH)

    st.header("2. Operation Mode")
    
    # Mode Selector - Consolidated modes
    st.session_state.current_mode = st.radio(
        "Select Operation Mode:",
        ("Q&A (Text RAG)", "Assignment Grading (CV+LLM)", "Educational Tools (NLP)", "Diagram/Chart Explanation (CV+RAG)"),
        index=0
    )
    logger.debug(f"Current mode set to: {st.session_state.current_mode}")

    # --- Conditional Inputs for GRADING ---
    if st.session_state.current_mode == "Assignment Grading (CV+LLM)":
        
        student_script_file = st.file_uploader(
            "1. Upload **Student Answer Script** (PDF/Image)", 
            type=["png", "jpg", "jpeg", "pdf"],
            key="student_script_uploader"
        )
        answer_key_file = st.file_uploader(
            "2. Upload **Answer Key / Rubric** (PDF/Image)", 
            type=["png", "jpg", "jpeg", "pdf"],
            key="answer_key_uploader"
        )
        st.text_input(
            "3. Enter the **Original Assignment Question**:", 
            key="assignment_question_input"
        )
        
        if st.button("Start Grading", key="grading_button"):
            logger.info("User triggered 'Start Grading'.")
            if student_script_file and answer_key_file and st.session_state.assignment_question_input:
                st.session_state.grading_script = student_script_file
                st.session_state.grading_key = answer_key_file
                st.rerun()
            else:
                st.warning("Please upload both files and enter the question to start grading.")
                logger.warning("Grading failed: missing inputs.")

    # --- Conditional Inputs for EDUCATIONAL TOOLS (Summarization/Q-Gen) ---
    elif st.session_state.current_mode == "Educational Tools (NLP)":
        
        st.subheader("📚 Source Material Upload (Optional)")
        source_file = st.file_uploader(
            "Upload PDF/TXT for direct summarization/questions:", 
            type=["pdf", "txt"], 
            key="nlp_source_uploader"
        )

        st.subheader("📖 Content Summarization")
        summary_topic = st.text_input("Topic to Summarize :", key="summary_topic")
        if st.button("Get Summary", key="summary_button"):
            logger.info("User triggered 'Get Summary'.")
            if source_file or (summary_topic and st.session_state.vectorstore):
                st.session_state.summary_trigger = True
                st.rerun()
            else:
                st.warning("Provide a topic (RAG) or upload a file (Direct).")

        st.subheader("❓ Adaptive Question Generation")
        question_topic = st.text_input("Topic for Questions:", key="question_topic")
        question_difficulty = st.selectbox("Difficulty Level:", ["Easy", "Medium", "Hard"], key="question_difficulty")
        question_count = st.number_input("Number of Questions:", min_value=1, max_value=10, value=3, key="question_count")
        
        if st.button("Generate Questions", key="qgen_button"):
            logger.info("User triggered 'Generate Questions'.")
            if source_file or (question_topic and st.session_state.vectorstore):
                st.session_state.qgen_trigger = True
                st.session_state.qgen_difficulty = question_difficulty
                st.session_state.qgen_count = question_count
                st.rerun()
            else:
                st.warning("Provide a topic (RAG) or upload a file (Direct).")


    # --- Conditional Inputs for DIAGRAM EXPLANATION ---
    elif st.session_state.current_mode == "Diagram/Chart Explanation (CV+RAG)":
        
        image_file = st.file_uploader(
            "Upload Diagram/Chart Image", 
            type=["png", "jpg", "jpeg"],
            key="diagram_uploader"
        )
        
        if st.button("Explain Diagram", key="explain_button"):
            logger.info("User triggered 'Explain Diagram'.")
            if image_file and st.session_state.vectorstore:
                st.session_state.cv_image_file = image_file
                st.rerun()
            elif image_file and not st.session_state.vectorstore:
                st.warning("Please upload and process PDF documents first for context.")
            else:
                st.warning("Please upload an image to explain.")


# 2. Main Processing Logic
retriever = setup_retriever()

# --- HELPER: Function to load documents from source file (for summarization/QGen) ---
def load_docs_from_file(uploaded_file):
    logger.info(f"Loading docs directly from uploaded file: {uploaded_file.name}")
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f: f.write(uploaded_file.read())

    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(temp_path)
    elif uploaded_file.name.endswith(".txt"):
        # For simplicity, treat TXT as a single Document
        text = open(temp_path, 'r', encoding='utf-8').read()
        logger.debug("Loaded TXT file as a single document.")
        return [Document(page_content=text)], temp_dir

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    logger.info(f"PDF split into {len(split_docs)} chunks.")
    return split_docs, temp_dir


# --- PROCESSING TRIGGERS ---

# --- GRADING Trigger ---
if st.session_state.current_mode == "Assignment Grading (CV+LLM)" and 'grading_script' in st.session_state and 'grading_key' in st.session_state:
    
    script_file = st.session_state.grading_script
    key_file = st.session_state.grading_key
    
    temp_dir = tempfile.mkdtemp()
    temp_script_path = os.path.join(temp_dir, script_file.name)
    temp_key_path = os.path.join(temp_dir, key_file.name)
    logger.info("Saving grading files temporarily.")

    with open(temp_script_path, "wb") as f: f.write(script_file.read())
    with open(temp_key_path, "wb") as f: f.write(key_file.read())
        
    output_container = st.container()

    with st.spinner("Processing documents and grading..."):
        result = process_image_for_grading(temp_script_path, temp_key_path, llm) 

        with output_container:
            st.subheader("📝 Automated Grading Result")
            st.markdown(result)

        del st.session_state.grading_script 
        del st.session_state.grading_key
        for f in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        logger.info("Grading process finished. Temporary files cleaned up.")

# --- SUMMARIZATION Trigger ---
elif st.session_state.current_mode == "Educational Tools (NLP)" and 'summary_trigger' in st.session_state:
    
    topic = st.session_state.summary_topic
    source_file = st.session_state.nlp_source_uploader
    output_container = st.container()
    
    source_docs = []
    temp_dir = None
    
    with st.spinner(f"Generating summary for '{topic if topic else 'uploaded file'}'..."):
        try:
            if source_file:
                source_docs, temp_dir = load_docs_from_file(source_file)
                topic = source_file.name
                logger.info("Summarization source: uploaded file.")
            elif topic and retriever:
                logger.info("Summarization source: RAG retrieval.")
                source_docs = retriever.invoke(f"Summarize comprehensive educational content on: {topic}")
            else:
                raise Exception("Missing source or retriever.")

            result = summarize_content(source_docs, topic, llm)
            
            with output_container:
                st.subheader("📖 Content Summarization Result")
                st.markdown(result)

        except Exception as e:
            logger.error(f"Error during summarization trigger: {e}", exc_info=True)
            with output_container:
                st.error(f"Error during summarization: {e}")

        finally:
            del st.session_state.summary_trigger
            if temp_dir:
                for f in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, f))
                os.rmdir(temp_dir)
                logger.info("Summarization temporary files cleaned up.")

# --- QUESTION GENERATION Trigger ---
elif st.session_state.current_mode == "Educational Tools (NLP)" and 'qgen_trigger' in st.session_state:
    
    topic = st.session_state.question_topic
    source_file = st.session_state.nlp_source_uploader
    difficulty = st.session_state.qgen_difficulty
    count = st.session_state.qgen_count
    output_container = st.container()
    
    source_docs = []
    temp_dir = None

    with st.spinner(f"Generating {count} questions for '{topic if topic else 'uploaded file'}'..."):
        try:
            if source_file:
                source_docs, temp_dir = load_docs_from_file(source_file)
                topic = source_file.name
                logger.info("Question generation source: uploaded file.")
            elif topic and retriever:
                logger.info("Question generation source: RAG retrieval.")
                source_docs = retriever.invoke(f"Source material for generating questions on: {topic}")
            else:
                raise Exception("Missing source or retriever.")

            result = generate_questions(source_docs, topic, difficulty, count, llm)

            with output_container:
                st.subheader("❓ Adaptive Question Generation Result")
                st.markdown(result)

        except Exception as e:
            logger.error(f"Error during question generation trigger: {e}", exc_info=True)
            with output_container:
                st.error(f"Error during question generation: {e}")

        finally:
            del st.session_state.qgen_trigger
            del st.session_state.qgen_difficulty
            del st.session_state.qgen_count
            if temp_dir:
                for f in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, f))
                os.rmdir(temp_dir)
                logger.info("Question generation temporary files cleaned up.")


# --- DIAGRAM EXPLANATION Trigger ---
elif 'cv_image_file' in st.session_state and st.session_state.cv_image_file:
    
    uploaded_image = st.session_state.cv_image_file
    logger.info(f"Starting Diagram Explanation for: {uploaded_image.name}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_image.name)[1]) as tmp_file:
        tmp_file.write(uploaded_image.read())
        temp_image_path = tmp_file.name
    logger.debug(f"Image saved to temp path: {temp_image_path}")

    output_container = st.container()

    with st.spinner(f"Processing image for {st.session_state.current_mode}..."):
        if st.session_state.current_mode == "Diagram/Chart Explanation (CV+RAG)":
            result = process_image_for_explanation(temp_image_path, llm, retriever)
        else:
            result = "Invalid mode selected or vector store not ready."
            
        with output_container:
            st.subheader("📊 Diagram Explanation Result")
            st.markdown(result)
            
        del st.session_state.cv_image_file 
        os.remove(temp_image_path)
        logger.info("Diagram explanation finished. Temporary image cleaned up.")


# --- Standard Q&A/Chat Interface (Only visible if no other trigger fires) ---
else:
    if st.session_state.vectorstore is None and st.session_state.current_mode == "Q&A (Text RAG)":
        st.info("Please upload PDF files on the sidebar and click 'Process & Update Vector Store' to begin chatting.")
    elif st.session_state.current_mode == "Q&A (Text RAG)":
        
        # Display existing chat history (only Q&A messages)
        for chat in st.session_state.chat_history:
            if "user" in chat:
                with st.chat_message("user"):
                    st.write(chat['user'])
            elif "assistant" in chat:
                with st.chat_message("assistant"):
                    st.write(chat['assistant'])
        
        # Process new query
        query = st.chat_input("Ask something about the uploaded document set: ")
        
        if query:
            logger.info(f"New user query received: {query}")
            st.session_state.chat_history.append({"user": query})
            with st.chat_message("user"):
                st.write(query)
            
            with st.spinner("Processing your request..."):
                try:
                    prompt = setup_prompt_template(task="qa")
                    # process_query handles RAG/Flan-T5 fallback
                    response = process_query(query, retriever, llm, prompt) 
                    
                    assistant_response = response["answer"]
                    st.session_state.chat_history.append({"assistant": assistant_response})

                    with st.chat_message("assistant"):
                        st.write(assistant_response)

                except Exception as e:
                    logger.critical(f"Critical error during Q&A chat processing: {e}", exc_info=True)
                    st.error(f"A critical error occurred during query processing. Error: {e}")