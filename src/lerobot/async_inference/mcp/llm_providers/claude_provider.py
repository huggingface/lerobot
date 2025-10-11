"""
Claude LLM Provider using native Anthropic API.
Supports streaming, thinking, tool calling, and multimodal capabilities.
"""

import json
import os
from typing import Dict, List, Any, Optional
import anthropic
from .base_provider import LLMProvider, LLMResponse


class ClaudeProvider(LLMProvider):
    """Claude provider using native Anthropic API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-7-sonnet-latest"):
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables or direct input.")
        
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)
    
    @property
    def provider_name(self) -> str:
        return "Claude"
    
    @property
    def supports_thinking(self) -> bool:
        return True
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for Claude's API."""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool.get("input_schema", {"type": "object", "properties": {}})
            }
            for tool in tools
        ]
    
    def format_messages(self, messages: List[Dict[str, Any]], thinking_enabled: bool = False) -> List[Dict[str, Any]]:
        """Format messages for Claude's API."""
        formatted_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Handle system messages separately
            if role == "system":
                continue
                
            # Handle tool results
            if role == "tool":
                formatted_messages.append({
                    "role": "user",
                    "content": content
                })
                continue
            
            # Handle assistant messages with tool calls
            if role == "assistant" and ("tool_calls" in msg or "thinking" in msg):
                assistant_content = []
                
                # If thinking is enabled for this turn, include the full thinking block
                if thinking_enabled and "thinking" in msg and msg["thinking"]:
                    thinking_block = msg["thinking"]
                    if isinstance(thinking_block, dict):
                        assistant_content.append(thinking_block)
                # If thinking is disabled, convert the thinking block to simple text content
                elif "thinking" in msg and msg["thinking"]:
                    thinking_block = msg["thinking"]
                    if isinstance(thinking_block, dict) and "thinking" in thinking_block:
                         assistant_content.append({"type": "text", "text": f"I previously thought: {thinking_block['thinking']}"})

                # Add text content if present
                if content:
                    assistant_content.append({"type": "text", "text": content})
                
                # Add tool use blocks
                for tool_call in msg["tool_calls"]:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}
                    })
                
                formatted_messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                continue
            
            # Handle regular messages
            formatted_messages.append({
                "role": role,
                "content": content
            })
        
        return formatted_messages
    
    def _extract_system_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract system message from messages list."""
        for msg in messages:
            if msg["role"] == "system":
                return msg["content"]
        return None
    
    async def _generate_response_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        thinking_enabled: bool = False,
        thinking_budget: int = 1024,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Generate response using Claude's native API with streaming."""
        
        # Extract system message
        system_message = self._extract_system_message(messages)
        formatted_messages = self.format_messages(messages, thinking_enabled)
        
        # Build request parameters
        stream_params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": formatted_messages,
            "temperature": temperature,
        }
        
        if system_message:
            stream_params["system"] = system_message
            
        if tools:
            stream_params["tools"] = self.format_tools(tools)
            
        # Add thinking configuration
        if thinking_enabled:
            stream_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }
        
        # Use streaming API
        with self.client.messages.stream(**stream_params) as stream:
            thinking_started = False
            response_started = False
            response_content = []
            thinking_content = []
            tool_calls = []
            usage_info = {}
            
            for event in stream:
                if event.type == "message_start":
                    # Extract usage info from message start
                    if hasattr(event.message, 'usage'):
                        usage_info = {
                            "input_tokens": event.message.usage.input_tokens,
                            "output_tokens": 0,  # Will be updated later
                            "total_tokens": event.message.usage.input_tokens
                        }
                        
                elif event.type == "content_block_start":
                    thinking_started = False
                    response_started = False
                    
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        if not thinking_started:
                            self.print_thinking_header()
                            thinking_started = True
                        print(event.delta.thinking, end="", flush=True)
                        thinking_content.append(event.delta.thinking)
                        
                    elif event.delta.type == "text_delta":
                        if not response_started:
                            if thinking_started:
                                print()  # New line after thinking
                            self.print_response_header()
                            response_started = True
                        print(event.delta.text, end="", flush=True)
                        response_content.append(event.delta.text)
                        
                elif event.type == "content_block_stop":
                    if thinking_started or response_started:
                        print()  # New line after block
            
            # Get final message and extract tool calls and thinking block
            response = stream.get_final_message()
            final_thinking_block = None
            
            # Extract tool calls from response
            for block in response.content:
                if block.type == 'tool_use':
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })
                elif block.type == 'thinking':
                    final_thinking_block = block.model_dump()
            
            # Update usage information with final data
            if hasattr(response, 'usage'):
                usage_info.update({
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                })
                # Add thinking tokens if available
                if hasattr(response.usage, 'thinking_tokens'):
                    usage_info["thinking_tokens"] = response.usage.thinking_tokens
            
            return LLMResponse(
                content="".join(response_content),
                thinking=final_thinking_block,
                tool_calls=tool_calls,
                provider=self.provider_name,
                usage=usage_info
            ) 