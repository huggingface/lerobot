"""
policy_selector.py
This file contains the LLMPolicySelector class, which is a wrapper around the LLMClient class.

Usage:
    selector = LLMPolicySelector(llm_client=OpenAIClient(api_key="your_api_key"), model="gpt-4o", temperature=0.5, max_tokens=1024, debug=True)
    policy = selector.select_policy(
        detected_objects=["red cube", "blue cylinder"],
        goal="move the red cube to the left bin",
        candidate_policies=["pick_and_place", "push_and_slide", "inspect_and_report"]
    )
"""

import logging
from typing import List, Union
import cv2
from .llm_client import LLMClient

# ------------------------------------------------------------------
# Policy Selector
# ------------------------------------------------------------------

class LLMPolicySelector:
    """
    Wraps LLM client to pick a policy given objects + goal.

    Attributes:
        llm: LLM client instance
        model: model name to use
        temperature: temperature parameter for LLM
        max_tokens: maximum tokens for LLM response
        debug: if True, prints prompt & raw response
    """

    def __init__(self, llm_client: LLMClient, model: str, temperature: float, max_tokens: int, debug: bool=False):
        self.llm = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        self.interactions = []

    def select_policy(
        self,
        detected_objects: dict[str, list[float]],
        goal: str,
        candidate_policies: List[str],
        camera_frames: dict = None,
        return_prompt_response: bool = False,
    ) -> Union[str, tuple[str, str, str]]:
        """
        Ask LLM to choose the best policy.

        Args:
            detected_objects: dictionary mapping object labels to bounding box coordinates
            goal: high-level task description
            candidate_policies: names of your pretrained policies
            camera_frames: dictionary of camera frames to include in the prompt
            return_prompt_response: if True, returns the prompt and response along with the policy

        Returns:
            if return_prompt_response is False: the chosen policy name (must be one of candidate_policies)
            if return_prompt_response is True: tuple of (chosen policy name, prompt, raw response)
        """
        # build a concise prompt
        obj_list = ", ".join([f"{label}: {bbox}" for label, bbox in detected_objects.items()]) or "none"
        policies_list = ", ".join(candidate_policies)
        prompt = (
            f"You are a robotics control assistant.  "
            f"A robot with a gripper on a table sees these objects: {obj_list}.  "
            f"The task is: {goal}.  "
            f"You have the following pretrained policies: {policies_list}.  "
            "Based on the objects and the goal, choose the single best policy name "
            "to accomplish a subtask that will help the robot complete the goal.  "
            "Answer with the policy name as the first token, then add a brief one sentence explanation of the scene and why you chose this policy. "
            "Note: Each detected object is represented by a bounding box [x1, y1, x2, y2], "
            "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."
        )

        if self.debug:
            logging.debug("=== GPT Prompt ===\n%s\n=================", prompt)

        messages = [
            {"role": "system", "content": "You help pick control policies."},
            {"role": "user", "content": prompt}
        ]
        
        # Add camera frames as images if available
        if camera_frames:
            try:
                # Create a new message list with images
                image_messages = [{"role": "system", "content": "You help pick control policies."}]
                
                # Create a content list for the user message with text and images
                user_content = [{"type": "text", "text": prompt}]
                
                # Add each camera frame as an image
                for cam_name, frame in camera_frames.items():
                    # Convert OpenCV BGR to RGB
                    if frame.shape[2] == 3:  # Check if it's a color image
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = frame
                    
                    # Convert to base64
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    import numpy as np
                    
                    pil_img = Image.fromarray(frame_rgb)
                    buffered = BytesIO()
                    pil_img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # Add to content
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}",
                            "detail": "high"
                        }
                    })
                
                # Add the user message with text and images
                image_messages.append({"role": "user", "content": user_content})
                
                # Use the image-enabled messages
                messages = image_messages
                logging.debug("Added %d camera frames to the policy selection prompt", len(camera_frames))
            except Exception as e:
                logging.error("Failed to add images to prompt: %s", e)
                # Fall back to text-only prompt

        try:
            # Use the LLMClient to get the response
            raw_response = self.llm.chat(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            logging.error("LLM API call failed: %s", e)
            raise

        choice = next(
            (p for p in candidate_policies
            if raw_response.startswith(p) or p in raw_response)
        )

        if self.debug:
            logging.debug("=== GPT Raw Response ===\n%s\n=================", raw_response)

        self.interactions.append({"prompt": prompt, "response": raw_response, "chosen_policy": choice})

        if return_prompt_response:
            return choice, prompt, raw_response
        return choice
    
    def get_interactions(self):
        return self.interactions