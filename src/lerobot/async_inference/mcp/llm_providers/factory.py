"""
Factory for creating LLM providers based on model names.
"""

import os
from typing import Optional
from .base_provider import LLMProvider


def create_llm_provider(model_name: str, api_key: str = None) -> LLMProvider:
    """
    Create an LLM provider based on the model name.
    
    Args:
        model_name: Model name (e.g., "claude-3-7-sonnet-latest", "gemini-2.5-flash", "gpt", "o-series")
        api_key: Optional API key override
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If model is not supported or API key is missing
    """
    model_lower = model_name.lower()
    
    if "claude" in model_lower:
        from .claude_provider import ClaudeProvider
        return ClaudeProvider(api_key=api_key, model=model_name)
    
    elif "gemini" in model_lower:
        from .gemini_provider import GeminiProvider
        return GeminiProvider(api_key=api_key, model=model_name)
    
    elif "gpt" in model_lower or model_lower.startswith('o1') or model_lower.startswith('o3') or model_lower.startswith('o4'):
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key, model=model_name)
    
    else:
        raise ValueError(f"Unsupported model provider for: {model_name}")
