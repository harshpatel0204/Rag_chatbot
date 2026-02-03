"""
Normal Chat Prompt Template

This module contains the prompt template for normal conversation
when no PDF documents are loaded.
"""

# System prompt for the chatbot in normal conversation mode
CHAT_SYSTEM_PROMPT = """You are a helpful, friendly, and knowledgeable AI assistant.

Your role is to have natural conversations with users and help them with their questions.

Guidelines:
- Be conversational and friendly
- Provide accurate and helpful information
- If you don't know something, admit it honestly
- Keep responses concise but comprehensive
- Use markdown formatting when helpful (lists, code blocks, etc.)

Remember: You are having a direct conversation. Respond naturally and helpfully."""


def get_chat_prompt(user_message: str, chat_history: list = None) -> str:
    """
    Format the chat prompt with user message and history.
    
    Args:
        user_message: The current user message
        chat_history: List of previous messages as dicts with 'role' and 'content'
        
    Returns:
        Formatted prompt string
    """
    messages = []
    
    # Add system prompt
    messages.append({
        "role": "system",
        "content": CHAT_SYSTEM_PROMPT
    })
    
    # Add chat history if provided
    if chat_history:
        for msg in chat_history:
            messages.append(msg)
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    return messages


def format_chat_history(messages: list) -> str:
    """
    Format chat history as a readable string.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        
    Returns:
        Formatted string of the conversation
    """
    formatted = []
    for msg in messages:
        role = msg.get('role', 'unknown').capitalize()
        content = msg.get('content', '')
        formatted.append(f"{role}: {content}")
    
    return "\n\n".join(formatted)
