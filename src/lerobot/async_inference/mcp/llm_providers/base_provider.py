"""
Base LLM Provider interface and response classes.
"""

import json
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from functools import wraps


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: Optional[str] = None
    thinking: Optional[Any] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    provider: str = ""
    usage: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {}


def retry_llm_call(max_retries: int = 5, initial_delay: float = 1.0):
    """
    Decorator to retry LLM calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Check if this is a retryable error
                    retryable_errors = [
                        'rate limit', 'overload', 'server', 'timeout', 'busy',
                        'service unavailable', 'internal error', 'throttle',
                        'quota', 'capacity', 'resource exhausted', 'too many requests',
                        'connection', 'network', 'temporary', 'unavailable'
                    ]
                    
                    is_retryable = any(keyword in error_msg for keyword in retryable_errors)
                    
                    if not is_retryable or attempt == max_retries:
                        # Don't retry for non-retryable errors or if we've exhausted retries
                        if attempt > 0:
                            print(f"❌ Final attempt failed after {attempt} retries: {str(e)}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = initial_delay * 2 ** (attempt)  # 1s, 2s, 4s, 8s, 16s
                    
                    print(f"⚠️  LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f"🔄 Retrying in {delay}s...")
                    
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @property
    @abstractmethod
    def supports_thinking(self) -> bool:
        """Return whether the provider supports thinking."""
        pass
    
    def format_tools_for_llm(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to LLM format. Override if provider needs special formatting."""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}})
            }
            for tool in tools
        ]
    
    def format_tool_calls_for_execution(self, raw_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw tool calls from LLM for local execution. Override if needed."""
        formatted_tool_calls = []
        for tool_call in raw_tool_calls:
            try:
                arguments = json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}
            
            formatted_tool_calls.append({
                "id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "input": arguments
            })
        
        return formatted_tool_calls
    
    @abstractmethod
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for the provider's API."""
        pass
    
    @abstractmethod
    def format_messages(self, messages: List[Dict[str, Any]], thinking_enabled: bool = False) -> List[Dict[str, Any]]:
        """Format messages for the provider's API."""
        pass
    
    @retry_llm_call(max_retries=5, initial_delay=1.0)
    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        thinking_enabled: bool = False,
        thinking_budget: int = 1024,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Generate a response from the LLM with automatic retry logic."""
        return await self._generate_response_impl(
            messages=messages,
            tools=tools,
            temperature=temperature,
            thinking_enabled=thinking_enabled,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens
        )
    
    @abstractmethod
    async def _generate_response_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        thinking_enabled: bool = False,
        thinking_budget: int = 1024,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Internal implementation of generate_response. Override this method in subclasses."""
        pass
    
    def print_thinking_header(self):
        """Print thinking header."""
        print("🧠 Thinking: ", end="", flush=True)
    
    def print_response_header(self):
        """Print response header."""
        print(f"🤖 {self.provider_name}: ", end="", flush=True)
    
    def format_tool_results_for_conversation(self, tool_calls: List[Dict[str, Any]], tool_outputs: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Format tool results for conversation history. Override if needed."""
        tool_results_with_images = []
        image_parts = []
        
        for tool_call, tool_output_parts in zip(tool_calls, tool_outputs):
            text_parts = []
            current_image_parts = []

            for part in tool_output_parts:
                if part['type'] == 'image':
                    current_image_parts.append(part)
                    image_parts.append(part)
                else:
                    text_parts.append(part.get('text', str(part)))
            
            tool_results_with_images.append({
                "type": "tool_result",
                "tool_use_id": tool_call["id"],
                "content": "\n".join(text_parts) if text_parts else "Tool executed successfully."
            })
            
            for i, image_part in enumerate(current_image_parts, 1):
                tool_results_with_images.append({
                    "type": "text",
                    "text": f"Image {i}:"
                })
                tool_results_with_images.append(image_part)
        
        return tool_results_with_images, image_parts 