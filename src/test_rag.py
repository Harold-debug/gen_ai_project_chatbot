from agent import AivancityAgent
import os

def test_rag():
    # Initialize the agent
    agent = AivancityAgent()
    
    # Load the FAISS index
    agent.rag.load_index("data/faiss_index")
    
    # Test questions
    test_questions = [
        "What programs does Aivancity offer?",
        "Tell me about the research at Aivancity",
        "What is Aivancity's mission?",
        "How can I contact Aivancity?",
    ]
    
    print("Testing RAG system with sample questions...\n")
    
    for question in test_questions:
        print(f"Q: {question}")
        response = agent.process_message(question, [])
        print(f"A: {response}\n")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    test_rag() 