"""
Aivancity conversational agent – integrates RAG and web search with real-time streaming.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, AsyncGenerator

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from rag import RAGSystem

load_dotenv()


class AivancityAgent:
    """AivancityAgent integrates RAG and web search capabilities to provide real-time, streaming responses. 
    It leverages a lightweight architecture to efficiently handle user queries and deliver accurate, 
    context-aware answers by utilizing both a retrieval-augmented generation system and live web search."""

    def __init__(self, rag_system: RAGSystem, model_name: str = "gpt-3.5-turbo"):
        self.rag = rag_system
        self.web_search = DuckDuckGoSearchRun()

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


    async def get_response(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncGenerator[str, None]:
        """
        Yield assistant tokens for *this* turn.

        Chainlit will display them as they arrive.
        """
        history = self._history(session_id)
        history.add_user_message(user_input)

        docs = self.rag.retrieve(user_input)
        context = "\n".join(d.page_content for d in docs)

        # fall back to web search if the RAG docs are scarce
        if len(docs) < 2:
            context += "\n\nAdditional web search:\n" + self.web_search.run(user_input)

        
        messages: List[BaseMessage] = self.prompt.format_messages(
            chat_history=history.messages,
            input=user_input,
        )
        # insert context into the system message
        sys_msg = messages[0]
        messages[0] = SystemMessage(content=f"{sys_msg.content}\n\n{context}")

        
        full_answer = ""
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                token: str = chunk.content
                full_answer += token
                yield token  # Chainlit gets the token immediately

        history.add_ai_message(full_answer)