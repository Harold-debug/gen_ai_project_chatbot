"""
Main Chainlit application for the Aivancity RAG Agent.
This module handles the web interface and chat interactions.
"""

import os
import chainlit as cl
from dotenv import load_dotenv
from rag import RAGSystem
from agent import AivancityAgent

# Load environment variables from .env file
load_dotenv()

# Initialize RAG system for document retrieval
rag_system = RAGSystem()

# Load the FAISS index for document search
try:
    rag_system.load_index()
except ValueError:
    print("Warning: No FAISS index found. Please run initialize.py first.")

# Initialize the conversational agent with OpenAI
agent = AivancityAgent(rag_system)

@cl.on_chat_start
async def start():
    """
    Initialize a new chat session.
    Sends a welcome message to the user.
    """
    await cl.Message(
        content="Welcome to Aivancity Assistant! How can I help you today?",
        author="Assistant"
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
    # Create a new message for the response
    response = cl.Message(content="", author="Assistant")
    await response.send()

    try:
        # Get response from agent
        agent_response = agent.get_response(message.content)
        
        # Stream the response token by token
        for token in agent_response.split():
            await response.stream_token(token + " ")
        
        # Finalize the message to prevent flickering
        await response.update()
        
    except Exception as e:
        # Handle errors gracefully
        error_message = f"I apologize, but I encountered an error: {str(e)}"
        await response.update(error_message)
        print(f"Error in chat: {str(e)}")

@cl.on_stop
async def on_stop():
    """
    Handle chat session end.
    Called when the user ends the chat session.
    """
    print("Chat session ended") 