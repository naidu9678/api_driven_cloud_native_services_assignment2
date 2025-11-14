import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# --- Configuration ---
# Fix: Updated the embedding model to the current standard model to resolve quota errors.
EMBEDDING_MODEL = "models/text-embedding-004"

def load_pdfs_to_vectorstore(pdf_paths, faiss_index_path):
    """
    Load multiple PDFs and create/update the FAISS vector store.

    :param pdf_paths: List of paths to the PDF files.
    :param faiss_index_path: Path to save/load the FAISS index.
    :return: The updated FAISS vector store.
    """
    # Load environment variables from .env file
    load_dotenv()
    try:
        # 1. Create embeddings using the standard model
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        
        # 2. Attempt to load the existing index
        # If the index does not exist, we will create a new one below.
        try:
            vectorstore = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
            print(f"Loaded existing FAISS index from {faiss_index_path}.")
            
            # If loaded, we check if there are new PDFs to add
            if not pdf_paths:
                print("No new PDF paths provided. Returning existing vector store.")
                return vectorstore

        except Exception as e:
             # If loading failed (e.g., FileNotFoundError), initialize a new one by processing the first PDF
            if not pdf_paths:
                raise Exception("FAISS index not found and no PDFs provided to create one.")
            
            # If no index, create an initial vectorstore from the first PDF
            print("FAISS index not found. Creating a new one from the first PDF.")
            loader = PyPDFLoader(pdf_paths[0])
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(data)
            vectorstore = FAISS.from_documents(docs, embedding_model)
            
            # Start loop from the second PDF if it exists
            pdf_paths_to_process = pdf_paths[1:]
        
        # If an index was loaded, all provided PDFs need to be processed
        else:
             pdf_paths_to_process = pdf_paths


        # 3. Process remaining PDFs and add documents to the store
        for pdf_path in pdf_paths_to_process:
            print(f"Processing and adding documents from: {pdf_path}")
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            data = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(data)

            # Add new documents to the vector store
            vectorstore.add_documents(docs)

        # 4. Save the updated vector store
        vectorstore.save_local(faiss_index_path)
        print(f"Updated FAISS index saved to {faiss_index_path}.")

        return vectorstore
    except Exception as e:
        raise Exception(f"Error loading PDFs into vector store: {e}")

if __name__ == "__main__":
    # Define the folder containing PDF files
    pdfs_folder = "pdfs"  # Path to your PDFs folder
    faiss_index_path = "faiss_index"  # Path to your FAISS index file
    
    # NOTE: You must have a folder named 'pdfs' containing PDF files for this script to run successfully.
    
    # Collect all PDF file paths from the specified folder
    try:
        if not os.path.exists(pdfs_folder):
            print(f"Error: Folder '{pdfs_folder}' not found. Please create it and add your PDF files.")
            exit()
            
        pdf_files = [os.path.join(pdfs_folder, filename) for filename in os.listdir(pdfs_folder) if filename.endswith('.pdf')]
        
        if not pdf_files:
            print(f"Warning: No PDF files found in the '{pdfs_folder}' folder.")

        updated_vectorstore = load_pdfs_to_vectorstore(pdf_files, faiss_index_path)
        print("All specified PDFs successfully loaded/merged into the vector store.")
        
    except Exception as e:
        print(f"An error occurred: {e}")