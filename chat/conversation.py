"""
Conversation Module

This module handles the chat functionality using Groq for LLM inference.
It supports both normal chat and RAG-augmented chat.
"""

from typing import List, Dict, Optional, Generator
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import sys
sys.path.append(str(__file__).rsplit('\\', 2)[0])
from config import GROQ_MODEL, GROQ_API_KEY
from prompts.chat_prompt import get_chat_prompt, CHAT_SYSTEM_PROMPT
from prompts.rag_prompt import get_rag_prompt, RAG_SYSTEM_PROMPT
from retriever.retriever import get_retriever

# Set up logging
logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversations with the LLM using Groq.
    
    Features:
    - Normal chat mode
    - RAG-augmented chat mode
    - Automatic mode switching based on document availability
    - Chat history management
    """
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize the conversation manager.
        
        Args:
            model_name: Groq model to use (defaults to config value)
            api_key: Groq API key (defaults to config value)
        """
        self.model_name = model_name or GROQ_MODEL
        self.api_key = api_key or GROQ_API_KEY
        self.llm = None
        self.retriever = get_retriever()
        
    def _get_llm(self) -> ChatGroq:
        """Get or create the LLM instance."""
        if self.llm is None:
            if not self.api_key:
                raise ValueError(
                    "Groq API key not found! Please set GROQ_API_KEY environment variable "
                    "or add it to config.py. Get your free key at: https://console.groq.com/keys"
                )
            logger.info(f"Initializing Groq with model: {self.model_name}")
            self.llm = ChatGroq(
                model=self.model_name,
                groq_api_key=self.api_key,
                temperature=0.7,
            )
        return self.llm
    
    def _convert_to_langchain_messages(self, messages: List[Dict]) -> list:
        """Convert message dicts to LangChain message objects."""
        lc_messages = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            elif role == 'user':
                lc_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                lc_messages.append(AIMessage(content=content))
        
        return lc_messages
    
    def chat(
        self, 
        user_message: str, 
        chat_history: List[Dict] = None
    ) -> str:
        """
        Normal chat without RAG context.
        """
        llm = self._get_llm()
        messages = get_chat_prompt(user_message, chat_history)
        lc_messages = self._convert_to_langchain_messages(messages)
        
        try:
            response = llm.invoke(lc_messages)
            return response.content
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}. Please check your Groq API key."
    
    def rag_chat(
        self, 
        user_message: str, 
        context: str,
        chat_history: List[Dict] = None
    ) -> str:
        """
        RAG-augmented chat with document context.
        """
        llm = self._get_llm()
        messages = get_rag_prompt(user_message, context, chat_history)
        lc_messages = self._convert_to_langchain_messages(messages)
        
        try:
            response = llm.invoke(lc_messages)
            return response.content
        except Exception as e:
            logger.error(f"Error in RAG chat: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}. Please check your Groq API key."
    
    def smart_chat(
        self, 
        user_message: str, 
        chat_history: List[Dict] = None
    ) -> tuple:
        """
        Smart chat that automatically chooses between normal and RAG mode.
        """
        if self.retriever.has_documents():
            context, docs = self.retriever.retrieve_with_context(user_message)
            if context:
                response = self.rag_chat(user_message, context, chat_history)
                return response, 'rag', context
        
        response = self.chat(user_message, chat_history)
        return response, 'normal', ''
    
    def stream_chat(
        self, 
        user_message: str, 
        chat_history: List[Dict] = None
    ) -> Generator[str, None, None]:
        """
        Stream chat responses token by token.
        """
        llm = self._get_llm()
        if self.retriever.has_documents():
            context, docs = self.retriever.retrieve_with_context(user_message)
            if context:
                messages = get_rag_prompt(user_message, context, chat_history)
            else:
                messages = get_chat_prompt(user_message, chat_history)
        else:
            messages = get_chat_prompt(user_message, chat_history)
        
        lc_messages = self._convert_to_langchain_messages(messages)
        
        try:
            for chunk in llm.stream(lc_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            yield f"Error: {str(e)}. Please check your Groq API key."


# Global conversation manager
_manager = None

def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager."""
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager


def chat(user_message: str, chat_history: List[Dict] = None) -> str:
    """Convenience function for normal chat."""
    return get_conversation_manager().chat(user_message, chat_history)


def smart_chat(user_message: str, chat_history: List[Dict] = None) -> tuple:
    """Convenience function for smart chat (auto RAG/normal)."""
    return get_conversation_manager().smart_chat(user_message, chat_history)
