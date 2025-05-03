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
    input: str
    chat_history: List[Annotated[HumanMessage | AIMessage, "chat_history"]]
    context: str

class AivancityAgent:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        # Original Ollama implementation (commented for future reference)
        # self.llm = ChatOllama(model=model_name)
        
        # OpenAI implementation
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.rag = RAGSystem()
        self.search = DuckDuckGoSearchRun()
        self.tool_executor = ToolExecutor([self.search])
        
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
        def retrieve(state: AgentState) -> AgentState:
            query = state["input"]
            docs = self.rag.retrieve(query)
            state["context"] = "\n\n".join(doc.page_content for doc in docs)
            
            # If we don't have enough context, perform web search
            if len(docs) < 2:  # Arbitrary threshold
                search_results = self.search.run(query)
                state["context"] += f"\n\nAdditional information from web search:\n{search_results}"
            
            return state

        def generate(state: AgentState) -> AgentState:
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

    def process_message(self, message: str, chat_history: List[Tuple[str, str]]) -> str:
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