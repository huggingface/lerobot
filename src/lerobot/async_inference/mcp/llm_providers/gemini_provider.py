"""
Gemini LLM Provider using native Google GenAI API.
Supports streaming, thinking, tool calling, and multimodal capabilities.
"""

import json
import os
from typing import Dict, List, Any, Optional, AsyncIterator
from google import genai
from google.genai import types
from .base_provider import LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    """Gemini provider using native Google GenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or direct input.")
        
        super().__init__(api_key, model)
        self.client = genai.Client(api_key=api_key)
    
    @property
    def provider_name(self) -> str:
        return "Gemini"
        
    @property
    def supports_thinking(self) -> bool:
        return "2.5" in self.model.lower()
    
    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for Gemini's API."""
        function_declarations = []
        
        for tool in tools:
            function_declarations.append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
            })
        
        return [types.Tool(function_declarations=function_declarations)]
    
    def format_messages(self, messages: List[Dict[str, Any]], thinking_enabled: bool = False) -> List[Dict[str, Any]]:
        """Format messages for Gemini's API."""
        formatted_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Skip system messages (handled separately)
            if role == "system":
                continue
            
            # Handle tool results - convert to function responses for Gemini
            if role == "tool":
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "tool_result":
                                # Convert tool result to function response for Gemini
                                parts.append(
                                    types.Part.from_function_response(
                                        name=part.get("tool_name", "unknown"),
                                        response={"result": part.get("content", "")}
                                    )
                                )
                            elif part.get("type") == "text":
                                parts.append(types.Part(text=part["text"]))
                            elif part.get("type") == "image":
                                # Handle image content
                                source = part.get("source", {})
                                if source.get("type") == "base64":
                                    parts.append(
                                        types.Part.from_bytes(
                                            data=source["data"],
                                            mime_type=source["media_type"]
                                        )
                                    )
                        else:
                            parts.append(types.Part(text=str(part)))
                    
                    if parts:
                        formatted_messages.append(
                            types.Content(role="function", parts=parts)
                        )
                else:
                    # Simple string content
                    formatted_messages.append(
                        types.Content(
                            role="function",
                            parts=[
                                types.Part.from_function_response(
                                    name="unknown",
                                    response={"result": content}
                                )
                            ]
                        )
                    )
                continue
            
            # Handle assistant messages with tool calls
            if role == "assistant" and "tool_calls" in msg:
                parts = []
                
                # Add text content if present
                if content:
                    parts.append(types.Part(text=content))
                
                # Add function calls
                for tool_call in msg["tool_calls"]:
                    parts.append(
                        types.Part.from_function_call(
                            name=tool_call["function"]["name"],
                            args=json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}
                        )
                    )
                
                formatted_messages.append(
                    types.Content(role="model", parts=parts)
                )
                continue
            
            # Handle regular messages
            if isinstance(content, str):
                formatted_messages.append(
                    types.Content(
                        role="user" if role == "user" else "model",
                        parts=[types.Part(text=content)]
                    )
                )
            else:
                # Handle multimodal content
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(types.Part(text=part["text"]))
                        elif part.get("type") == "image":
                            # Handle image content
                            source = part.get("source", {})
                            if source.get("type") == "base64":
                                parts.append(
                                    types.Part.from_bytes(
                                        data=source["data"],
                                        mime_type=source["media_type"]
                                    )
                                )
                    else:
                        parts.append(types.Part(text=str(part)))
                
                formatted_messages.append(
                    types.Content(
                        role="user" if role == "user" else "model",
                        parts=parts
                    )
                )
        
        return formatted_messages
    
    def _extract_system_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract system message from messages list."""
        for msg in messages:
            if msg["role"] == "system":
                return msg["content"]
        return None
    
    def _count_images_in_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count total images in messages."""
        image_count = 0
        
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        image_count += 1
        
        return image_count
    
    def format_tool_results_for_conversation(self, tool_calls: List[Dict[str, Any]], tool_outputs: List[List[Dict[str, Any]]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Format tool results for Gemini conversation history."""
        tool_results_with_images = []
        image_parts = []
        
        for tool_call, tool_output_parts in zip(tool_calls, tool_outputs):
            text_parts = []
            current_image_parts = []

            for part in tool_output_parts:
                if part['type'] == 'image':
                    current_image_parts.append(part)
                    image_parts.append(part)
                elif part['type'] == 'text':
                    # Try to parse as JSON first to format it nicely
                    try:
                        json_data = json.loads(part['text'])
                        text_parts.append(json.dumps(json_data, indent=2))
                    except (json.JSONDecodeError, KeyError):
                        text_parts.append(part.get('text', str(part)))
                else:
                    # Handle other types (like dict objects from MCP)
                    if isinstance(part, dict) and part.get('type') != 'image':
                        # This might be the JSON data directly
                        if any(key in part for key in ['robot_state', 'message', 'status']):
                            text_parts.append(json.dumps(part, indent=2))
                        else:
                            text_parts.append(str(part))
                    else:
                        text_parts.append(str(part))
            
            # Create the tool result in the standard format
            result_content = "\n".join(text_parts) if text_parts else "Tool executed successfully."
            
            # Store tool name for Gemini-specific formatting later
            tool_results_with_images.append({
                "type": "tool_result",
                "tool_use_id": tool_call["id"],
                "tool_name": tool_call["name"],  # Store tool name for Gemini
                "content": result_content
            })
            
            # Add images
            for i, image_part in enumerate(current_image_parts, 1):
                tool_results_with_images.append({
                    "type": "text",
                    "text": f"Image {i} from {tool_call['name']}:"
                })
                tool_results_with_images.append(image_part)
        
        return tool_results_with_images, image_parts

    async def _generate_response_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        thinking_enabled: bool = False,
        thinking_budget: int = 1024,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Generate response using Gemini's native API with streaming."""
        
        # Extract system message
        system_message = self._extract_system_message(messages)
        formatted_messages = self.format_messages(messages)
        
        # Count images in messages
        image_count = self._count_images_in_messages(messages)
        
        # Build configuration
        config_params = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Add thinking configuration if supported
        if thinking_enabled and self.supports_thinking:
            if thinking_budget == -1:
                # Dynamic thinking
                config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)
            elif thinking_budget > 0:
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=True  # Enable thought summaries
                )
        
        # Add tools if provided
        if tools:
            config_params["tools"] = tools
        
        # Add system instruction if present
        if system_message:
            config_params["system_instruction"] = system_message
        
        config = types.GenerateContentConfig(**config_params)
        
        # Use streaming API
        thinking_started = False
        response_started = False
        response_content = []
        thinking_content = []
        tool_calls = []
        
        # Track usage info
        usage_info = {}
        
        # Generate streaming response
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=formatted_messages,
            config=config
        )
        
        for chunk in stream:
            if chunk.candidates and len(chunk.candidates) > 0:
                candidate = chunk.candidates[0]
                
                # Extract usage information from chunks
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage_info = {
                        "input_tokens": getattr(chunk.usage_metadata, 'prompt_token_count', 0),
                        "output_tokens": getattr(chunk.usage_metadata, 'candidates_token_count', 0),
                        "total_tokens": getattr(chunk.usage_metadata, 'total_token_count', 0)
                    }
                    # Add thinking tokens if available
                    if hasattr(chunk.usage_metadata, 'thoughts_token_count'):
                        usage_info["thinking_tokens"] = chunk.usage_metadata.thoughts_token_count
                
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # Handle thinking content
                        if hasattr(part, 'thought') and part.thought:
                            if not thinking_started:
                                self.print_thinking_header()
                                thinking_started = True
                            print(part.text, end="", flush=True)
                            thinking_content.append(part.text)
                        
                        # Handle text content
                        elif hasattr(part, 'text') and part.text and not getattr(part, 'thought', False):
                            if not response_started:
                                if thinking_started:
                                    print()  # New line after thinking
                                self.print_response_header()
                                response_started = True
                            
                            # Clean up control characters and formatting artifacts
                            clean_text = part.text.replace('<ctrl46>', '').replace('**', '')
                            print(clean_text, end="", flush=True)
                            response_content.append(clean_text)
                        
                        # Handle function calls
                        elif hasattr(part, 'function_call') and part.function_call:
                            if not response_started:
                                if thinking_started:
                                    print()  # New line after thinking
                                self.print_response_header()
                                response_started = True
                            
                            # Convert function call to standard format
                            tool_calls.append({
                                "id": f"call_{len(tool_calls)}",  # Generate ID
                                "type": "function",
                                "function": {
                                    "name": part.function_call.name,
                                    "arguments": json.dumps(dict(part.function_call.args))
                                }
                            })
        
        if thinking_started or response_started:
            print()  # Final newline
        
        # Add image count to usage info
        if image_count > 0:
            usage_info["image_count"] = image_count
        
        return LLMResponse(
            content="".join(response_content),
            thinking="".join(thinking_content) if thinking_content else None,
            tool_calls=tool_calls,
            provider=self.provider_name,
            usage=usage_info
        ) 