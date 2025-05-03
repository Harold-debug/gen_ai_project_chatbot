"""
Aivancity Agent implementation using LangGraph and OpenAI.
This module implements a conversational agent that combines RAG with web search capabilities.
"""

from typing import Dict, List, Tuple, Any, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.chat_models import ChatOllama  # Kept for future reference
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from rag import RAGSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    """
    Type definition for the agent's state.
    
    Attributes:
        input: The current user input
        chat_history: List of previous messages
        context: Retrieved context for the current query
    """
    input: str
    chat_history: List[Annotated[HumanMessage | AIMessage, "chat_history"]]
    context: str

class AivancityAgent:
    """
    Conversational agent for Aivancity School that combines RAG with web search.
    
    This agent uses:
    - OpenAI's GPT model for response generation
    - FAISS-based RAG for document retrieval
    - DuckDuckGo for web search when needed
    """
    
    def __init__(self, rag_system: RAGSystem, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the agent with RAG system and language model.
        
        Args:
            rag_system: The RAG system for document retrieval
            model_name: The OpenAI model to use (default: gpt-3.5-turbo)
        """
        # Original Ollama implementation (commented for future reference)
        # self.llm = ChatOllama(model=model_name)
        
        # OpenAI implementation
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize components
        self.rag = rag_system
        self.search = DuckDuckGoSearchRun()
        self.tool_executor = ToolExecutor([self.search])
        
        # Define the system prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant for Aivancity School for Technology, Business and Society. 
            Your role is to provide accurate and helpful information about the school based on the retrieved context.
            If the context doesn't provide enough information, you can use web search to find additional information.
            Always be professional and maintain a helpful tone.
            If you don't know something or can't find enough information, say so."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

    def create_agent_graph(self):
        """
        Create the LangGraph workflow for the agent.
        
        The workflow consists of:
        1. Retrieve: Get relevant documents and web search results
        2. Generate: Create response using the retrieved context
        
        Returns:
            A compiled LangGraph workflow
        """
        def retrieve(state: AgentState) -> AgentState:
            """Retrieve relevant documents and web search results."""
            query = state["input"]
            docs = self.rag.retrieve(query)
            state["context"] = "\n\n".join(doc.page_content for doc in docs)
            
            # If we don't have enough context, perform web search
            if len(docs) < 2:  # Arbitrary threshold
                search_results = self.search.run(query)
                state["context"] += f"\n\nAdditional information from web search:\n{search_results}"
            
            return state

        def generate(state: AgentState) -> AgentState:
            """Generate response using the retrieved context."""
            messages = self.prompt.format_messages(
                chat_history=state["chat_history"],
                input=state["input"]
            )
            response = self.llm.invoke(messages)
            state["chat_history"].append(AIMessage(content=response.content))
            return state

        # Create the graph with the schema
        workflow = StateGraph(schema=AgentState)

        # Add nodes
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)

        # Add edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Set entry point
        workflow.set_entry_point("retrieve")

        return workflow.compile()

    def get_response(self, message: str, chat_history: List[Tuple[str, str]] = None) -> str:
        """
        Process a message and generate a response.
        
        Args:
            message: The user's message
            chat_history: Optional list of previous messages
            
        Returns:
            The generated response
        """
        if chat_history is None:
            chat_history = []
            
        state: AgentState = {
            "input": message,
            "chat_history": [
                HumanMessage(content=msg) if role == "human" else AIMessage(content=msg)
                for role, msg in chat_history
            ],
            "context": ""
        }
        
        graph = self.create_agent_graph()
        result = graph.invoke(state)
        
        return result["chat_history"][-1].content 