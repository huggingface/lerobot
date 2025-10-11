"""
LLM provider for OpenAI models (GPT-4, GPT-4o, o1-series, etc.).
Supports both regular GPT models (with streaming and function calling) 
and o-series reasoning models (with thinking but no function calling).
"""
import os
import json
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI
from .base_provider import LLMProvider, LLMResponse

class OpenAIProvider(LLMProvider):
    """LLM Provider for OpenAI models."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        super().__init__(api_key, model)
        
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or direct input.")
        
        self.client = AsyncOpenAI(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "OpenAI"
    
    @property
    def supports_thinking(self) -> bool:
        return self._is_reasoning_model()

    def _is_reasoning_model(self) -> bool:
        """Check if the model is a reasoning model (o-series)."""
        return (self.model.startswith('o1') or 
                self.model.startswith('o3') or 
                self.model.startswith('o4'))

    def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools into OpenAI's expected format."""
        formatted_tools = []
        for tool in tools:
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
                }
            })
        return formatted_tools

    def format_messages(self, messages: List[Dict[str, Any]], thinking_enabled: bool = False) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API with proper tool message handling."""
        formatted_messages = []
        system_content = None
        
        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                if self._is_reasoning_model():
                    # o-series models don't support system messages, save for later
                    system_content = content
                    continue
                else:
                    formatted_messages.append({"role": "system", "content": content})
                
            elif role == "user":
                formatted_message = self._format_message(message)
                
                # For o-series models, inject system content into first user message
                if self._is_reasoning_model() and system_content and len(formatted_messages) == 0:
                    if isinstance(formatted_message["content"], str):
                        formatted_message["content"] = f"{system_content}\n\n{formatted_message['content']}"
                    elif isinstance(formatted_message["content"], list):
                        # Insert system content as first text element
                        formatted_message["content"].insert(0, {"type": "text", "text": system_content + "\n\n"})
                
                formatted_messages.append(formatted_message)
                
            elif role == "assistant":
                # Handle assistant messages with tool calls
                if "tool_calls" in message and message["tool_calls"]:
                    formatted_message = {
                        "role": "assistant",
                        "content": content or None
                    }
                    
                    # Add tool calls in OpenAI format
                    formatted_tool_calls = []
                    for tool_call in message["tool_calls"]:
                        formatted_tool_calls.append({
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"]
                            }
                        })
                    formatted_message["tool_calls"] = formatted_tool_calls
                    formatted_messages.append(formatted_message)
                else:
                    # Regular assistant message
                    formatted_messages.append({"role": "assistant", "content": content})
                    
            elif role == "tool":
                # Handle tool results - separate text and images
                tool_messages = []
                image_parts = []
                
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            # Handle tool result items (with tool_call_id)
                            if item.get("tool_call_id"):
                                tool_messages.append({
                                    "role": "tool",
                                    "tool_call_id": item["tool_call_id"],
                                    "content": str(item.get("content", ""))
                                })
                            # Handle image items (no tool_call_id)
                            elif item.get("type") == "image" and "source" in item:
                                # Convert image to OpenAI format
                                source = item["source"]
                                image_parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{source['media_type']};base64,{source['data']}",
                                        "detail": "low"
                                    }
                                })
                            # Handle text items (descriptions, etc.)
                            elif item.get("type") == "text":
                                # Skip these as they're usually image descriptions
                                pass
                else:
                    # Fallback for simple tool content
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": "unknown",
                        "content": str(content)
                    })
                
                # Add tool messages first
                formatted_messages.extend(tool_messages)
                
                # Add images as separate user message if any exist
                if image_parts:
                    image_content = [{
                        "type": "text",
                        "text": f"Here are {len(image_parts)} images from the robot's cameras:"
                    }]
                    image_content.extend(image_parts)
                    
                    formatted_messages.append({
                        "role": "user",
                        "content": image_content
                    })
        
        return formatted_messages

    def format_tool_calls_for_execution(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format LLM tool calls for local execution."""
        execution_calls = []
        for tool_call in tool_calls:
            if tool_call.get('type') == 'function':
                try:
                    arguments = json.loads(tool_call['function']['arguments'])
                except json.JSONDecodeError:
                    arguments = {} # Handle empty/invalid JSON
                
                execution_calls.append({
                    "id": tool_call['id'],
                    "name": tool_call['function']['name'],
                    "input": arguments
                })
        return execution_calls

    def format_tool_results_for_conversation(self, tool_calls: List[Dict[str, Any]], tool_outputs: List[List[Dict[str, Any]]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Format tool execution results for the conversation history.
        Returns (tool_results_with_images, image_parts) where:
        - tool_results_with_images is a list suitable for the agent to
          place inside one `role: tool` message.  It MUST contain both the
          tool result text (with a `tool_call_id`) **and** the raw image
          parts so the provider can convert them later.
        - image_parts is the raw list of image dicts so the agent can pass
          them to the optional ImageViewer.
        """
        tool_results_with_images: List[Dict[str, Any]] = []
        image_parts: List[Dict[str, Any]] = []

        for tool_call, output_parts in zip(tool_calls, tool_outputs):
            call_id = tool_call["id"]

            # Separate text parts and image parts
            text_parts: List[str] = []
            current_image_parts: List[Dict[str, Any]] = []

            for part in output_parts:
                if part.get("type") == "image":
                    current_image_parts.append(part)
                    image_parts.append(part)
                elif part.get("type") == "text":
                    text_parts.append(part.get("text", str(part)))
                else:
                    # Fallback for any other structure
                    text_parts.append(str(part))

            # Tool result item (text only) – mandatory `tool_call_id`
            tool_results_with_images.append({
                "tool_call_id": call_id,
                "type": "tool_result",
                "content": "\n".join(text_parts) if text_parts else "Tool executed successfully."
            })

            # Append image parts directly after the tool result so they
            # appear in the same `role: tool` message content list.
            # These are raw image dicts (type == "image" with base64 source)
            tool_results_with_images.extend(current_image_parts)

        return tool_results_with_images, image_parts
    
    def _format_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single message for the OpenAI API."""
        role = message["role"]
        content = message["content"]

        if role == "tool":
            # Handle tool messages - content is a list of tool results
            if isinstance(content, list):
                # Return the first tool result (OpenAI expects one tool message per tool call)
                for item in content:
                    if isinstance(item, dict) and item.get("tool_call_id"):
                        return {
                            "role": "tool",
                            "tool_call_id": item["tool_call_id"],
                            "content": item.get("content", ""),
                        }
                # Fallback if no proper tool result found
                return {
                    "role": "tool", 
                    "tool_call_id": "unknown",
                    "content": str(content)
                }
            else:
                # Handle simple string content
                return {
                    "role": "tool",
                    "tool_call_id": "unknown", 
                    "content": str(content)
                }
        
        # Handle multimodal content (text + images)
        if isinstance(content, list):
            formatted_content = []
            for part in content:
                if part.get("type") == "text":
                    formatted_content.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image" and "source" in part:
                    source = part["source"]
                    formatted_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{source['media_type']};base64,{source['data']}",
                            "detail": "low"  # Use low detail to reduce token cost
                        }
                    })
                elif part.get("type") == "image_url":
                    # Already in OpenAI format, ensure low detail for cost control
                    image_part = part.copy()
                    if "image_url" in image_part and isinstance(image_part["image_url"], dict):
                        image_part["image_url"]["detail"] = "low"
                    formatted_content.append(image_part)
            return {"role": role, "content": formatted_content}

        return {"role": role, "content": content}

    async def _generate_response_impl(self,
                                messages: List[Dict[str, Any]],
                                tools: List[Dict[str, Any]] = None,
                                temperature: float = 0.1,
                                thinking_enabled: bool = False,
                                thinking_budget: int = 1024,
                                max_tokens: int = 4096) -> LLMResponse:
        """Generate response from OpenAI with streaming."""
        
        formatted_messages = self.format_messages(messages)
        
        # Count images in the conversation
        image_count = 0
        total_image_size = 0
        for msg in formatted_messages:
            if isinstance(msg.get("content"), list):
                for content_part in msg["content"]:
                    if isinstance(content_part, dict) and content_part.get("type") == "image_url":
                        image_count += 1
                        # Check image size
                        if "image_url" in content_part and "url" in content_part["image_url"]:
                            url = content_part["image_url"]["url"]
                            if url.startswith("data:"):
                                # Extract base64 part and estimate size
                                base64_part = url.split(",", 1)[1] if "," in url else ""
                                estimated_size = len(base64_part) * 3 // 4  # Base64 to bytes approximation
                                total_image_size += estimated_size
        
        if image_count > 0:
            print(f"📊 Sending {image_count} images, total size: ~{total_image_size//1024}KB")
        


        # o-series models don't support streaming
        use_streaming = not self._is_reasoning_model()
        
        request_params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
        }
        
        if use_streaming:
            request_params["stream"] = True
            request_params["stream_options"] = {"include_usage": True}

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"
        
        if thinking_enabled and self._is_reasoning_model():
            # Reserve more tokens for reasoning (docs recommend 25k minimum)
            reasoning_buffer = max(thinking_budget, 25000)
            request_params["max_completion_tokens"] = reasoning_buffer

        if use_streaming:
            # Streaming response for regular GPT models
            stream = await self.client.chat.completions.create(**request_params)

            full_content = ""
            tool_calls = []
            usage_data = None
            
            async for chunk in stream:
                # Check if chunk has choices and handle empty choices
                if not chunk.choices:
                    if chunk.usage:
                        usage_data = chunk.usage.dict()
                    continue
                    
                delta = chunk.choices[0].delta
                
                if delta.content:
                    full_content += delta.content
                    print(delta.content, end="", flush=True)

                if delta.tool_calls:
                    for tool_chunk in delta.tool_calls:
                        if tool_chunk.index == len(tool_calls):
                            # New tool call
                            tool_calls.append({
                                "id": tool_chunk.id,
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        tc = tool_calls[tool_chunk.index]['function']
                        if tool_chunk.function.name:
                            tc['name'] += tool_chunk.function.name
                        if tool_chunk.function.arguments:
                            tc['arguments'] += tool_chunk.function.arguments

                # Capture usage data when available
                if chunk.usage:
                    usage_data = chunk.usage.dict()
        else:
            # Non-streaming response for o-series models
            response = await self.client.chat.completions.create(**request_params)
            
            full_content = response.choices[0].message.content or ""
            tool_calls = []
            
            # Handle tool calls in non-streaming response
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            # Print content for o-series models (no streaming)
            if full_content:
                print(full_content)
            
            usage_data = response.usage.dict() if response.usage else None

        # Process final usage data
        if usage_data:
            input_tokens = usage_data.get("prompt_tokens", 0)
            output_tokens = usage_data.get("completion_tokens", 0)
            total_tokens = usage_data.get("total_tokens", 0)
            
            # Check for reasoning tokens
            thinking_tokens = 0
            if "completion_tokens_details" in usage_data and usage_data["completion_tokens_details"]:
                thinking_tokens = usage_data["completion_tokens_details"].get("reasoning_tokens", 0)
            
            final_usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "thinking_tokens": thinking_tokens,
                "image_count": image_count
            }
        else:
            # Fallback if no usage data
            final_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "thinking_tokens": 0,
                "image_count": image_count
            }
        
        return LLMResponse(
            content=full_content or None, 
            tool_calls=tool_calls if tool_calls else None, 
            usage=final_usage
        ) 