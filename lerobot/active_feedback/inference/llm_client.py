from abc import ABC, abstractmethod
from typing import List, Dict, Any
import openai
from google import genai
from google.genai import types
from openai import OpenAI          # ← unified-API client

class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Send messages to the LLM and return the raw text response."""
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        # Set up the classic chat/complete client
        openai.api_key = api_key
        self.chat_client = openai
        
        # Set up the new unified multimodal client
        self.response_client = OpenAI(api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        resp = self.chat_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1
        )
        return resp.choices[0].message.content.strip()

    def generate_with_images(
        self,
        prompt: str,
        image_urls: List[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate text from a prompt plus one or more image URLs.

        Args:
            prompt: Free-form text prompt.
            image_urls: List of publicly-accessible image URLs.
            model: Model identifier (e.g. "gpt-4o").
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text response.
        """
        # Debug logging
        import logging
        logger = logging.getLogger(self.__class__.__name__)
        logger.info(f"Generating with images: model={model}, num_images={len(image_urls) if image_urls else 0}")
        
        # Build the multimodal input array
        inputs: List[Dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]
        if image_urls:
            for i, url in enumerate(image_urls):
                logger.debug(f"Image {i+1}: {url[:50]}... (truncated)")
                inputs.append({
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": url}
                    ]
                })

        # Call the unified `responses` endpoint
        try:
            resp = self.response_client.responses.create(
                model=model,
                input=inputs,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            logger.info(f"Response received: {len(resp.output_text.strip())} chars")
            return resp.output_text.strip()
        except Exception as e:
            logger.error(f"Error in generate_with_images: {e}")
            raise

class GeminiClient(LLMClient):
    def __init__(self, api_key: str = None):
        # Instantiate underlying SDK client
        self.client = genai.Client(api_key=api_key)

    def completion(self,
                   model: str,
                   prompt: str,
                   temperature: float = 0.7,
                   max_tokens: int = 1024) -> str:
        resp = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return resp.text

    def chat(self,
             messages: List[Dict[str, Any]],
             model: str,
             top_k: int = 40,
             top_p: float = 0.95,
             temperature: float = 0.7,
             max_tokens: int = 1024) -> str:
        # 1) Start a new chat session
        chat_session = self.client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        )

        # 2) Flatten any string or list‐of‐dict content into plain text
        user_snippets: list[str] = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    user_snippets.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            user_snippets.append(part.get("text", ""))
        user_prompt = "\n".join(user_snippets)

        # 3) Send the combined prompt and return the text reply
        response = chat_session.send_message(user_prompt)
        return response.text.strip()
        
    def generate_with_images(
        self,
        prompt: str,
        images: List[str] = None,
        image_bytes: List[bytes] = None,
        image_mime_types: List[str] = None,
        model: str = "gemini-2.5-flash-preview-04-17",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate content with text prompt and one or more images.
        
        Args:
            prompt: Text prompt to send to the model
            images: List of image file paths to upload
            image_bytes: List of image byte arrays to send directly
            image_mime_types: MIME types for the image_bytes (must match length)
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Note:
            - Provide either images (file paths) or image_bytes+image_mime_types
            - If both are provided, all images will be included in the request
        """
        # Initialize contents with the text prompt
        contents = [prompt]
        
        # Process file paths (upload method)
        if images:
            for image_path in images:
                uploaded_file = self.client.files.upload(file=image_path)
                contents.append(uploaded_file)
        
        # Process image bytes (inline method)
        if image_bytes:
            if not image_mime_types or len(image_bytes) != len(image_mime_types):
                raise ValueError("image_mime_types must be provided and match length of image_bytes")
            
            for img_bytes, mime_type in zip(image_bytes, image_mime_types):
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
        
        # Generate content with the prepared contents
        resp = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        
        return resp.text


if __name__ == "__main__":
    import yaml
    # get api key from config
    with open("inference_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    api_key = config["gemini_api_key"]
    client = GeminiClient(api_key=api_key)
    print(client.generate_with_images("Is the blue cube in the box and is the box in the blue bin?", images=["/home/demo/lerobot-beavr/overhead_detections.png"]))