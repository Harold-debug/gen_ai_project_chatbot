from typing import List, Dict
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class RAGSystem:
    def __init__(self):
        # Set environment variable to avoid tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = None

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)

    def create_index(self, documents: List[Document], path: str = "data/faiss_index"):
        """Create and save FAISS index from documents."""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store.save_local(path)

    def load_index(self, path: str = "data/faiss_index"):
        """Load FAISS index from disk."""
        self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for a query."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please load or create an index first.")
        return self.vector_store.similarity_search(query, k=k)

    def create_documents(self, data: List[Dict[str, str]]) -> List[Document]:
        documents = []
        for item in data:
            chunks = self.text_splitter.split_text(item["content"])
            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata=item["metadata"]
                    )
                )
        return documents

    def index_documents(self, documents: List[Document]):
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def save_index(self, path: str = "faiss_index"):
        if self.vector_store:
            self.vector_store.save_local(path) 