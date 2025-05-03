# Aivancity RAG Agent

A Retrieval-Augmented Generation (RAG) based conversational agent for Aivancity School, built using LangChain, LangGraph, and FAISS.

## Features

- PDF document processing and indexing
- Semantic search using FAISS vector store
- OpenAI integration for natural language generation
- Interactive chat interface
- Efficient document chunking and retrieval

## Prerequisites

- Python 3.12.9
- Poetry for dependency management
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Harold-debug/gen_ai_project_chatbot.git
cd gen_ai_project_chatbot
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
gen_ai_project_chatbot/
├── data/                  # Directory for PDF documents and indices
├── src/
│   ├── agent.py          # Conversational agent implementation
│   ├── data_loader.py    # PDF document loading and processing
│   ├── rag.py            # RAG system implementation
│   └── initialize.py     # System initialization script
├── pyproject.toml        # Project dependencies
└── .env                  # Environment variables (not tracked in git)
```

## Usage

1. Place your PDF documents in the `data/` directory.

2. Initialize the RAG system:
```bash
poetry run python src/initialize.py
```

3. Test the RAG system:
```bash
poetry run python src/test_rag.py
```

## Dependencies

- langchain (^0.1.0)
- langchain-community (^0.0.38)
- langchain-core (^0.1.27)
- langchain-openai (^0.0.5)
- langgraph (^0.0.26)
- faiss-cpu (^1.7.4)
- sentence-transformers (^2.5.1)
- chainlit (^1.0.0)
- python-dotenv (^1.0.1)
- beautifulsoup4 (^4.12.3)
- requests (^2.31.0)
- pymupdf (^1.23.26)
- duckduckgo-search (^3.9.9)

## Development

The project uses Poetry for dependency management. To add new dependencies:

```bash
poetry add package-name
```

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]