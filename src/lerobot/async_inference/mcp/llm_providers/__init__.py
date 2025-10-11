"""
LLM Provider Package

Provides native API integrations for Claude (Anthropic) and Gemini (Google GenAI).
Supports streaming, thinking, tool calling, and multimodal capabilities.
"""

from .factory import create_llm_provider
from .base_provider import LLMResponse

__all__ = ['create_llm_provider', 'LLMResponse']

# This file makes the llm_providers directory a Python package. 