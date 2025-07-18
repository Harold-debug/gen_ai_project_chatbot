from data_loader import AivancityDataLoader
from rag import RAGSystem
import os

def initialize_rag():
    os.makedirs("data", exist_ok=True)
    
    # Initialize components
    data_loader = AivancityDataLoader()
    rag_system = RAGSystem()
    
    print("Loading PDFs from data directory...")
    documents = data_loader.load_pdfs()
    
    if not documents:
        print("No PDFs found in the data directory. Please add PDF files to the 'data' directory.")
        return None
    
    # Process documents
    print("Processing documents...")
    processed_docs = data_loader.process_documents(documents)
    
    data_loader.save_documents(processed_docs, "data/processed_documents.txt")
    
    print("Indexing documents...")
    rag_system.index_documents(processed_docs)
    
    # Save the index
    rag_system.save_index("data/faiss_index")
    
    print("\nRAG system initialized successfully!")
    print(f"Total documents processed: {len(processed_docs)}")
    print("\nYou can now run the test script with:")
    print("poetry run python src/test_rag.py")
    
    return rag_system

if __name__ == "__main__":
    initialize_rag() 