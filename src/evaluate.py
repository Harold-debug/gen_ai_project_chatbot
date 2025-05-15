"""
Evaluation module for the RAG pipeline.
Evaluates retrieval quality, answer relevance, and overall system performance.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag import RAGSystem

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluates RAG pipeline performance using various metrics."""

    def __init__(self, rag_system: RAGSystem, model_name: str = "gpt-3.5-turbo"):
        self.rag = rag_system
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        # Initialize evaluation prompts
        self.relevance_prompt = ChatPromptTemplate.from_template(
            "You are a strict evaluator assessing the relevance of retrieved documents.\n"
            "Question: {question}\n\n"
            "Retrieved documents:\n{documents}\n\n"
            "Rate the relevance of these documents to the question on a scale of 1-5.\n"
            "1: Completely irrelevant or contains no useful information\n"
            "2: Mostly irrelevant, with only minor relevant points\n"
            "3: Partially relevant, contains some useful information\n"
            "4: Mostly relevant, with minor gaps or irrelevant parts\n"
            "5: Perfectly relevant, contains all necessary information\n\n"
            "For each document, provide:\n"
            "1. A rating (1-5)\n"
            "2. A brief explanation of why it received that rating\n"
            "3. What information is missing or irrelevant\n\n"
            "Be critical and specific in your assessment."
        )
        
        self.answer_quality_prompt = ChatPromptTemplate.from_template(
            "You are a strict evaluator assessing the quality of an answer.\n"
            "Question: {question}\n\n"
            "Answer: {answer}\n\n"
            "Rate the answer on the following criteria (1-5):\n"
            "1. Accuracy: Is the information correct and verifiable?\n"
            "   - 1: Contains significant factual errors\n"
            "   - 2: Has some factual errors\n"
            "   - 3: Mostly accurate with minor errors\n"
            "   - 4: Accurate with no significant errors\n"
            "   - 5: Completely accurate and verifiable\n\n"
            "2. Completeness: Does it fully address the question?\n"
            "   - 1: Missing critical information\n"
            "   - 2: Missing important details\n"
            "   - 3: Covers main points but lacks depth\n"
            "   - 4: Comprehensive but could be more detailed\n"
            "   - 5: Fully comprehensive and detailed\n\n"
            "3. Relevance: Is it focused on the question?\n"
            "   - 1: Completely off-topic\n"
            "   - 2: Mostly off-topic\n"
            "   - 3: Somewhat relevant but includes tangents\n"
            "   - 4: Mostly focused with minor digressions\n"
            "   - 5: Completely focused on the question\n\n"
            "4. Clarity: Is it well-explained?\n"
            "   - 1: Unclear and confusing\n"
            "   - 2: Difficult to understand\n"
            "   - 3: Somewhat clear but could be better\n"
            "   - 4: Clear with minor issues\n"
            "   - 5: Exceptionally clear and well-structured\n\n"
            "For each criterion:\n"
            "1. Provide a rating (1-5)\n"
            "2. Explain why it received that rating\n"
            "3. Suggest specific improvements\n\n"
            "Be critical and specific in your assessment."
        )

        self.answer_prompt = ChatPromptTemplate.from_template(
            "You are an AI assistant for Aivancity School for Technology, "
            "Business and Society. Answer factually; if unsure, say so. And be friendly.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        self.relevance_chain = (
            self.relevance_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        self.answer_quality_chain = (
            self.answer_quality_prompt 
            | self.llm 
            | StrOutputParser()
        )

        self.answer_chain = (
            self.answer_prompt
            | self.llm
            | StrOutputParser()
        )

    def evaluate_retrieval(
        self, 
        question: str, 
        k: int = 3
    ) -> Dict[str, Any]:
        """Evaluate the quality of document retrieval."""
        logger.info(f"Evaluating retrieval for question: {question[:100]}...")
        
        # Get retrieved documents
        docs = self.rag.retrieve(question, k=k)
        
        # Format documents for evaluation
        docs_text = "\n\n".join(
            f"Document {i+1}:\n{doc.page_content[:500]}..."
            for i, doc in enumerate(docs)
        )
        
        # Get relevance assessment
        relevance_assessment = self.relevance_chain.invoke({
            "question": question,
            "documents": docs_text
        })
        
        return {
            "question": question,
            "num_docs_retrieved": len(docs),
            "relevance_assessment": relevance_assessment,
            "documents": [doc.page_content for doc in docs]
        }

    def evaluate_answer(
        self, 
        question: str, 
        answer: str
    ) -> Dict[str, Any]:
        """Evaluate the quality of a generated answer."""
        logger.info(f"Evaluating answer quality for question: {question[:100]}...")
        
        quality_assessment = self.answer_quality_chain.invoke({
            "question": question,
            "answer": answer
        })
        
        return {
            "question": question,
            "answer": answer,
            "quality_assessment": quality_assessment
        }

    def evaluate_pipeline(
        self, 
        test_cases: List[Dict[str, str]],
        output_dir: str = "evaluation_results1"
    ) -> Dict[str, Any]:
        """Run full pipeline evaluation on a set of test cases."""
        logger.info(f"Starting pipeline evaluation with {len(test_cases)} test cases...")
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Processing test case {i}/{len(test_cases)}...")
            
            question = test_case["question"]
            expected_answer = test_case.get("expected_answer")
            
            # Evaluate retrieval
            retrieval_results = self.evaluate_retrieval(question)
            
            # Get system's answer using retrieved context
            context = "\n\n".join(retrieval_results["documents"])
            answer = self.answer_chain.invoke({
                "question": question,
                "context": context
            })
            
            # Evaluate answer quality
            answer_results = self.evaluate_answer(question, answer)
            
            # Combine results
            case_results = {
                "test_case": i,
                "question": question,
                "expected_answer": expected_answer,
                "retrieval_evaluation": retrieval_results,
                "answer_evaluation": answer_results
            }
            results.append(case_results)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"evaluation_results_{timestamp}.txt"
        
        with open(results_file, "w") as f:
            f.write("RAG Pipeline Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Test Cases: {len(test_cases)}\n\n")
            
            for case in results:
                f.write(f"Test Case {case['test_case']}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Question: {case['question']}\n")
                if case['expected_answer']:
                    f.write(f"Expected Answer: {case['expected_answer']}\n")
                f.write("\n")
                
                # Write retrieval evaluation
                f.write("Retrieval Evaluation:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Number of Documents Retrieved: {case['retrieval_evaluation']['num_docs_retrieved']}\n")
                f.write("\nRelevance Assessment:\n")
                f.write(case['retrieval_evaluation']['relevance_assessment'])
                f.write("\n\n")
                
                # Write answer evaluation
                f.write("Answer Evaluation:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Generated Answer:\n{case['answer_evaluation']['answer']}\n\n")
                f.write("Quality Assessment:\n")
                f.write(case['answer_evaluation']['quality_assessment'])
                f.write("\n\n")
                
                f.write("=" * 50 + "\n\n")
        
        logger.info(f"Evaluation results saved to {results_file}")
        
        return {
            "num_test_cases": len(test_cases),
            "results_file": str(results_file),
            "detailed_results": results
        }

def create_test_cases() -> List[Dict[str, str]]:
    """Create a set of test cases for evaluation."""
    return [
        {
            "question": "What are the main programs offered at Aivancity?",
            "expected_answer": "Aivancity offers various programs in technology, business, and society."
        },
        {
            "question": "How can I apply to Aivancity?",
            "expected_answer": "Information about the application process at Aivancity."
        },
        {
            "question": "What is the campus location of Aivancity?",
            "expected_answer": "Details about Aivancity's campus location."
        },
        # Add more test cases as needed
    ]

if __name__ == "__main__":
    # Example usage
    from initialize import initialize_rag
    
    # Initialize RAG system
    rag_system = initialize_rag()
    
    # Create evaluator
    evaluator = RAGEvaluator(rag_system)
    
    # Get test cases
    test_cases = create_test_cases()
    
    # Run evaluation
    results = evaluator.evaluate_pipeline(test_cases)
    
    print(f"Evaluation completed. Results saved to {results['results_file']}") 