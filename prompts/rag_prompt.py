"""
RAG Chat Prompt Template

This module contains the prompt template for RAG-based conversation
when PDF documents are loaded and context is available.
"""

# System prompt for the chatbot in RAG mode
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in answering questions based on provided documents.

You have been given context from uploaded PDF documents. Use this context to answer the user's questions accurately.

IMPORTANT GUIDELINES:
1. ONLY use information from the provided context to answer questions
2. If the context doesn't contain relevant information, say "I couldn't find information about that in the uploaded documents"
3. Always cite which document and page the information comes from when possible
4. If asked about something outside the documents, politely explain you can only answer based on the uploaded content
5. Be precise and factual - don't make up information
6. Use markdown formatting for better readability

Remember: Your answers should be grounded in the provided context."""


RAG_CONTEXT_TEMPLATE = """
CONTEXT FROM UPLOADED DOCUMENTS:
{context}

---

Based on the above context, please answer the following question. If the context doesn't contain relevant information, say so clearly.
"""


def get_rag_prompt(user_message: str, context: str, chat_history: list = None) -> list:
    """
    Format the RAG prompt with context, user message, and history.
    
    Args:
        user_message: The current user message
        context: Retrieved context from documents
        chat_history: List of previous messages
        
    Returns:
        List of message dicts for the LLM
    """
    messages = []
    
    # Add system prompt
    messages.append({
        "role": "system",
        "content": RAG_SYSTEM_PROMPT
    })
    
    # Add chat history if provided (limited to prevent context overflow)
    if chat_history:
        # Keep only last 10 messages to manage context length
        recent_history = chat_history[-10:]
        for msg in recent_history:
            messages.append(msg)
    
    # Add context and user question
    augmented_message = RAG_CONTEXT_TEMPLATE.format(context=context) + f"\nUSER QUESTION: {user_message}"
    
    messages.append({
        "role": "user",
        "content": augmented_message
    })
    
    return messages


def get_rag_system_prompt() -> str:
    """Get the RAG system prompt."""
    return RAG_SYSTEM_PROMPT


def format_context_for_display(context: str) -> str:
    """
    Format context for display in the UI (shortened version).
    
    Args:
        context: The full context string
        
    Returns:
        Shortened context for display
    """
    if len(context) > 500:
        return context[:500] + "..."
    return context
