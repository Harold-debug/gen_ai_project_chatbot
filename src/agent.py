"""
Aivancity conversational agent – integrates RAG and web search with real-time streaming.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, AsyncGenerator

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import chainlit as cl
from duckduckgo_search import DDGS

from rag import RAGSystem

load_dotenv()


class AivancityAgent:
    """AivancityAgent integrates RAG and web search capabilities to provide real-time, streaming responses. 
    It leverages a lightweight architecture to efficiently handle user queries and deliver accurate, 
    context-aware answers by utilizing both a retrieval-augmented generation system and live web search."""

    def __init__(self, rag_system: RAGSystem, model_name: str = "gpt-3.5-turbo"):
        self.rag = rag_system

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            streaming=True,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are an AI assistant for Aivancity School for Technology, "
                        "Business and Society. Answer factually; if unsure, say so."
                    )
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="{input}"),
            ]
        )

        # keep one ChatMessageHistory per client (Chainlit session id)
        self._histories: Dict[str, ChatMessageHistory] = {}

    
    def _history(self, session_id: str) -> ChatMessageHistory:
        hist = self._histories.setdefault(session_id, ChatMessageHistory())
        return hist

    def clear_history(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    def _web_search(self, query: str, max_results: int = 3) -> str:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"- {r['title']}: {r['body']} ({r['href']})")
        return "\n".join(results)

    async def get_response(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncGenerator[str, None]:
        """
        Yield assistant tokens for *this* turn.

        Chainlit will display them as they arrive.
        """
        # Create status message for thought process
        status_msg = cl.Message(content="💭 Starting to process your query...", author="Assistant")
        await status_msg.send()
        cl.user_session.set("status_msg", status_msg)

        history = self._history(session_id)
        history.add_user_message(user_input)

        # Update status for RAG retrieval
        status_msg.content = "🔍 Searching through knowledge base..."
        await status_msg.update()
        
        docs = self.rag.retrieve(user_input)
        context = "\n".join(d.page_content for d in docs)

        if len(docs) < 2:
            # Update status for web search
            status_msg.content = "🌐 Performing web search for additional context..."
            await status_msg.update()
            web_results = self._web_search(user_input)
            context += f"\n\nAdditional web search:\n{web_results}"

        # Update status for response generation
        status_msg.content = "🤔 Generating response..."
        await status_msg.update()
        
        messages: List[BaseMessage] = self.prompt.format_messages(
            chat_history=history.messages,
            input=user_input,
        )
        # insert context into the system message
        sys_msg = messages[0]
        messages[0] = SystemMessage(content=f"{sys_msg.content}\n\n{context}")

        
        full_answer = ""
        first_token = True
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                token: str = chunk.content
                full_answer += token
                # Update final status
                status_msg.content = "✅ Response complete!"
                await status_msg.update()
                if first_token:
                    await status_msg.remove()  # Remove status message as soon as the first token is about to be shown
                    first_token = False
                yield token  # Chainlit gets the token immediately

        

        history.add_ai_message(full_answer)