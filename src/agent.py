"""
Aivancity conversational agent – LangGraph version.
Integrates RAG and live web search with real‑time streaming. The reasoning
workflow is orchestrated with LangGraph. Public API (`get_response`) is
unchanged, so `app.py` and the rest of the project keep working.
"""

from __future__ import annotations

import os
import logging
import time
from typing import Dict, Any, List, AsyncGenerator, TypedDict, Optional
from datetime import datetime

from dotenv import load_dotenv
import chainlit as cl

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from tavily import TavilyClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag import RAGSystem

load_dotenv()
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    user_input: str
    session_id: str
    history: ChatMessageHistory
    docs: List[Any]
    web_results: str
    context: str

class AivancityAgent:
    """Handles RAG + web search via LangGraph, streams answers to Chainlit."""

    def __init__(self, rag_system: RAGSystem, model_name: str = "gpt-3.5-turbo"):
        start_time = time.time()
        logger.info("Initializing AivancityAgent...")
        
        self.rag = rag_system
        
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set")
        self.tavily = TavilyClient(api_key=tavily_api_key)

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            streaming=True,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.search_decision_prompt = PromptTemplate.from_template(
            "You are a supervisor deciding if a web search is needed.\n"
            "Question: {question}\n\n"
            "Known context:\n{context}\n\n"
            "Decision rules:\n"
            "1. Respond YES if the question can be answered with:\n"
            "   - Common knowledge\n"
            "   - Basic conversation\n"
            "   - The provided context\n"
            "2. Respond NO only if the question requires:\n"
            "   - Up-to-date information\n"
            "   - Specific facts not in the context and not known by you\n"
            "   - Recent events or data\n\n"
            "Respond with exactly YES or NO."
        )
        self.search_decision_chain = (
            self.search_decision_prompt 
            | self.llm 
            | StrOutputParser()
        )

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are an AI assistant for Aivancity School for Technology, "
                "Business and Society. Answer factually; if unsure, say so. And be friendly."
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
        ])

        self._histories: Dict[str, ChatMessageHistory] = {}
        self._graph = None
        
        logger.info(f"AivancityAgent initialized in {time.time() - start_time:.2f}s")

    @property
    def graph(self):
        if self._graph is None:
            start_time = time.time()
            logger.info("Building LangGraph...")
            self._graph = self._build_graph()
            logger.info(f"LangGraph built in {time.time() - start_time:.2f}s")
        return self._graph

    def _history(self, session_id: str) -> ChatMessageHistory:
        return self._histories.setdefault(session_id, ChatMessageHistory())

    def clear_history(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    def _web_search(self, query: str, k: int = 3) -> str:
        try:
            res = self.tavily.search(query, max_results=k)
            results = []
            for r in res.get("results", []):
                title = r.get("title", "No title")
                content = r.get("content", "No content")
                url = r.get("url", "No URL")
                results.append(f"- {title}: {content} ({url})")
            return "\n".join(results) if results else "No search results found."
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return "Error performing web search."

    def _should_use_search(self, docs: List[Any], question: str) -> str:
        logger.info(f"Evaluating if web search is needed for question: {question[:100]}...")
        context_preview = "\n".join(d.page_content[:300] for d in docs[:3])
        logger.debug(f"Context preview: {context_preview[:200]}...")
        result = self.search_decision_chain.invoke({
            "question": question,
            "context": context_preview
        })
        logger.info(f"Search decision result: {result}")
        return "yes" if "yes" in result.lower() else "no"

    def _build_graph(self):
        g = StateGraph(AgentState)

        async def retrieve(state: AgentState) -> AgentState:
            logger.info("Retrieving from knowledge base...")
            state["docs"] = self.rag.retrieve(state["user_input"])
            logger.info(f"Retrieved {len(state['docs'])} documents")
            return state

        async def supervisor(state: AgentState) -> AgentState:
            status = cl.user_session.get("status_msg")
            status.content = "🤔 Evaluating if web search is needed..."
            await status.update()
            logger.info("Supervisor evaluating search need...")
            return state

        def router(state: AgentState) -> str:
            route = self._should_use_search(state["docs"], state["user_input"])
            logger.info(f"Router decided to: {'generate' if route == 'yes' else 'search'}")
            return "generate" if route == "yes" else "search"

        async def search(state: AgentState) -> AgentState:
            status = cl.user_session.get("status_msg")
            status.content = "🌐 Searching the web for additional information..."
            await status.update()
            logger.info("Performing web search...")
            state["web_results"] = self._web_search(state["user_input"])
            logger.info(f"Web search returned {len(state['web_results'].split('\n'))} results")
            return state

        async def generate(state: AgentState) -> AgentState:
            status = cl.user_session.get("status_msg")
            status.content = "📝 Combining knowledge base and web search results..."
            await status.update()
            logger.info("Generating final response...")
            pieces = ["\n".join(d.page_content for d in state["docs"])]
            if state["web_results"]:
                pieces.append(f"Additional web search:\n{state['web_results']}")
            state["context"] = "\n\n".join(pieces)
            logger.info(f"Combined context length: {len(state['context'])} characters")
            return state

        g.add_node("retrieve", retrieve)
        g.add_node("supervisor", supervisor)
        g.add_node("search", search)
        g.add_node("generate", generate)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "supervisor")
        g.add_conditional_edges(
            "supervisor",
            router,
            {"generate": "generate", "search": "search"},
        )
        g.add_edge("search", "generate")
        g.set_finish_point("generate")
        return g.compile()

    async def get_response(
        self, user_input: str, session_id: str
    ) -> AsyncGenerator[str, None]:
        status = cl.Message("💭 Starting to process your query…", author="Assistant")
        await status.send()
        cl.user_session.set("status_msg", status)

        hist = self._history(session_id)
        hist.add_user_message(user_input)

        init_state: AgentState = {
            "user_input": user_input,
            "session_id": session_id,
            "history": hist,
            "docs": [],
            "web_results": "",
            "context": "",
        }

        status.content = "🔍 Searching through knowledge base…"
        await status.update()
        state: AgentState = await self.graph.ainvoke(init_state)

        status.content = "🤔 Generating response…"
        await status.update()

        messages: List[BaseMessage] = self.prompt.format_messages(
            chat_history=hist.messages, input=user_input
        )
        orig_sys = messages[0]
        messages[0] = SystemMessage(content=f"{orig_sys.content}\n\n{state['context']}")

        full_answer = ""
        first = True
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                token = chunk.content
                full_answer += token
                if first:
                    await status.remove()
                    first = False
                yield token

        status.content = "✅ Response complete!"
        await status.update()
        hist.add_ai_message(full_answer)
