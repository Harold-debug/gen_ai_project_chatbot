# Aivancity RAG Agent

A Retrieval-Augmented Generation (RAG) based conversational agent for Aivancity School, built using LangChain and Chainlit.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/Harold-debug/gen_ai_project_chatbot.git
cd gen_ai_project_chatbot
```

2. Install dependencies (choose one method):

   **Using Poetry:**
   ```bash
   poetry install
   ```

   **Using pip:**
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_api_key_here
```

4. Place your PDF documents in the `data/` directory

5. Initialize the RAG system:
```bash
# With Poetry
poetry run python src/initialize.py

# With pip
python src/initialize.py
```

6. Start the application:
```bash
# With Poetry
poetry run chainlit run src/app.py

# With pip
chainlit run src/app.py
```

7. Open your browser and navigate to `http://localhost:8000`

## Features

- PDF document processing and indexing
- Semantic search using FAISS
- OpenAI integration for natural language generation
- Interactive chat interface
- Modern UI with Chainlit
- Real-time streaming responses

## Project Structure

```
gen_ai_project_chatbot/
├── data/                  # PDF documents and indices
├── src/
│   ├── agent.py          # Conversational agent
│   ├── data_loader.py    # PDF processing
│   ├── rag.py            # RAG system
│   ├── initialize.py     # System setup
│   └── app.py            # Main application
├── chainlit.md           # Welcome page
├── chainlit.yaml         # UI config
└── .env                  # Environment variables
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Aivancity RAG Agent                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Document Processing                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              RAG System                                 │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Conversational Agent                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Web Interface                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Details

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Document Processing                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
│  │   PDF Docs  │───▶│  Text Extr. │───▶│  Chunking   │───▶│Embedding │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                              RAG System                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
│  │  FAISS Index│◀───│  Vector DB  │◀───│  Embeddings │◀───│Chunking  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         Conversational Agent                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
│  │User Query   │───▶│Context      │───▶│  LLM        │───▶│Response  │  │
│  └─────────────┘    │Retrieval    │    │Processing   │    └──────────┘  │
│                     └─────┬───────┘    └─────┬───────┘                   │
│                           │                  │                           │
│                           ▼                  ▼                           │
│                     ┌─────────────┐    ┌─────────────┐                   │
│                     │Web Search   │    │Context     │                   │
│                     │(DuckDuckGo) │    │Merging     │                   │
│                     └─────────────┘    └─────────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           Web Interface                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
│  │  Chainlit   │───▶│  UI/UX      │───▶│  Session    │───▶│  Stream  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Schema

```
┌─────────────┐     ┌─────────────────────────────────┐     ┌─────────────┐
│  User Input │────▶│         Context Retrieval       │────▶│  Response   │
└─────────────┘     └─────────────────────────────────┘     │ Generation  │
                    │                                       └─────────────┘
                    │  ┌─────────────┐     ┌─────────────┐        ▲
                    │  │  FAISS      │     │  Web       │        │
                    └─▶│  Search     │     │  Search    │        │
                       └─────────────┘     └─────────────┘        │
                            │                  │                  │
                            ▼                  ▼                  │
                       ┌─────────────────────────────────┐        │
                       │         LLM Processing          │────────┘
                       └─────────────────────────────────┘
```

### Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Technology Stack                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
│  │  LangChain  │    │   FAISS     │    │   OpenAI    │    │Chainlit  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘  │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
│  │ PyMuPDF     │    │Sentence     │    │DuckDuckGo   │    │Python    │  │
│  │             │    │Transformers │    │Search       │    │3.12.9    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```


## Requirements

- Python 3.12.9
- OpenAI API key
- PDF documents to process

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

## Contributing