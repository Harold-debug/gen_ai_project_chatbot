import os
from typing import List
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class AivancityDataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def load_pdfs(self) -> List[Document]:
        """Load documents from PDF files in the data directory."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory at {self.data_dir}")
            return []
            
        # Load PDFs
        documents = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    # Open PDF with PyMuPDF
                    pdf_document = fitz.open(file_path)
                    
                    # Extract text from each page
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        text = page.get_text()
                        
                        # Create a Document for each page
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": file_path,
                                "page": page_num + 1,
                                "filename": filename
                            }
                        )
                        documents.append(doc)
                    
                    print(f"Loaded {len(pdf_document)} pages from {filename}")
                    pdf_document.close()
                    
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        print(f"Total documents loaded: {len(documents)}")
        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and split documents into chunks."""
        processed_docs = self.text_splitter.split_documents(documents)
        print(f"Split into {len(processed_docs)} chunks")
        return processed_docs

    def save_documents(self, documents: List[Document], filename: str):
        """Save processed documents to a file for reference."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(f"Content: {doc.page_content}\n")
                f.write(f"Metadata: {doc.metadata}\n")
                f.write("-" * 80 + "\n") 