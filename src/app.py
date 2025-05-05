"""
Main Chainlit application for the Aivancity RAG Agent.
This module handles the web interface and chat interactions.
"""

import uuid
import os
import chainlit as cl
from dotenv import load_dotenv
from rag import RAGSystem
from agent import AivancityAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

rag_system = RAGSystem()

try:
    rag_system.load_index()
except ValueError:
    print("Warning: No FAISS index found. Please run initialize.py first.")

agent = AivancityAgent(rag_system)

@cl.on_chat_start
async def start():
    cl.user_session.set("id", cl.user_session.get("id") or str(uuid.uuid4()))   # ← NEW
    await cl.Message(
        content="Welcome to Aivancity Assistant! How can I help you today?",
        author="Assistant",
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """
    Handle incoming chat messages.
    
    Args:
        message (cl.Message): The incoming message from the user
    
    The function:
    1. Shows a thinking indicator
    2. Processes the message through the agent
    3. Updates the UI with the response
    4. Handles any errors gracefully
    """
    session_id: str = cl.user_session.get("id")
    #logger.info(f"Received message: {message.content}")
    
    # Create a new message for the response
    response = cl.Message(content="", author="Assistant")
    await response.send()
    # logger.info("Created response message")

    try:
        # Get response from agent and stream tokens
        logger.info("Starting token streaming...")
        async for token in agent.get_response(message.content, session_id):
            # logger.info(f"Received token: {token}")
            await response.stream_token(token)
            # logger.info("Token streamed to UI")
            await response.update()
            # logger.info("Message updated in UI")
        
    except Exception as e:
        error_message = f"I apologize, but I encountered an error: {str(e)}"
        logger.error(f"Error in chat: {str(e)}")
        response.content = error_message
        await response.update()

@cl.on_stop
async def on_stop():
    """
    Handle chat session end.
    Called when the user ends the chat session.
    """
    logger.info("Chat session ended") 