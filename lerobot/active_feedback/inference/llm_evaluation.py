import logging
from typing import Dict, List, Tuple
import openai
import cv2
from pathlib import Path

from lerobot.active_feedback.detect.object_detector import ObjectDetector
from lerobot.active_feedback.inference.llm_client import LLMClient, OpenAIClient, GeminiClient

class TaskSuccessEvaluator:
    """
    Evaluates task success by sampling frames during inference and analyzing them with object detection
    and LLM reasoning after the task is complete.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        llm_client: LLMClient,
        debug: bool = False,
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.detector = detector
        self.llm_client = llm_client
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # LLM configuration
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Storage for sampled frames and detections
        self.sampled_frames = {}
        self.sampled_detections = {}
        
        # Create directory for saving images if it doesn't exist
        if self.debug:
            self.image_dir = Path("./lerobot/lga/inference/determine_success_images")
            self.image_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Images will be saved to {self.image_dir.absolute()}")
        self.interactions = []

    def add_frame(self, cam_name: str, sample_name: str, frame):
        """
        Add a frame from the observation to the sampled frames
        
        Args:
            cam_name: Name of the camera
            sample_name: Name of the sample point (start, middle, end)
            frame: The camera frame in BGR format (numpy array)
        """
        # Debug logging
        self.logger.debug(f"Adding {sample_name} frame for {cam_name}, shape: {frame.shape}, type: {type(frame)}, dtype: {frame.dtype}")
        
        # Initialize storage for this camera if not exists
        if cam_name not in self.sampled_frames:
            self.sampled_frames[cam_name] = {}
            self.sampled_detections[cam_name] = {}
        
        # Store the frame
        self.sampled_frames[cam_name][sample_name] = frame.copy()  # Make a copy to ensure we don't have reference issues
        
        # Run object detection
        try:
            det = self.detector.detect_and_segment(frame, box_threshold=0.5)
            self.sampled_detections[cam_name][sample_name] = det
            
            # Log detection results
            objects_str = ", ".join(det["labels"])
            self.logger.debug(f"Detected objects in {cam_name} ({sample_name}): {objects_str}")
        except Exception as e:
            self.logger.error(f"Error detecting objects in {cam_name} ({sample_name}): {e}")
            # Create an empty detection result to avoid errors later
            self.sampled_detections[cam_name][sample_name] = {"labels": [], "boxes": []}
        
        # Save the image if enabled
        if self.debug:
            try:
                # Create a timestamp-based unique identifier
                import time
                timestamp = int(time.time())
                
                # Save original frame
                img_path = self.image_dir / f"{timestamp}_{cam_name}_{sample_name}_original.jpg"
                cv2.imwrite(str(img_path), frame)
                
                # Save frame with detection visualization if detection succeeded
                if cam_name in self.sampled_detections and sample_name in self.sampled_detections[cam_name]:
                    det = self.sampled_detections[cam_name][sample_name]
                    vis_frame = frame.copy()
                    for label, box in zip(det["labels"], det["boxes"]):
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    vis_path = self.image_dir / f"{timestamp}_{cam_name}_{sample_name}_detection.jpg"
                    cv2.imwrite(str(vis_path), vis_frame)
                    
                    self.logger.info(f"Saved images to {img_path.name} and {vis_path.name}")
            except Exception as e:
                self.logger.error(f"Error saving debug images for {cam_name} ({sample_name}): {e}")
    
    def evaluate_task_success(self, subtask: str) -> Tuple[bool, str]:
        """
        Evaluate if the subtask was successful based on the sampled frames
        
        Args:
            subtask: The subtask to evaluate
            
        Returns:
            Tuple[bool, str]: (success, explanation)
        """
        # Check if we have any frames
        if not self.sampled_frames:
            self.logger.warning("No frames were sampled during inference")
            return False, "No frames were sampled during inference"
        
        # Check if we have end frames for all cameras
        for cam_name in self.sampled_frames:
            if "end" not in self.sampled_frames[cam_name]:
                self.logger.warning(f"No end frame for camera {cam_name}")
                return False, f"No end frame for camera {cam_name}"
        
        # Prepare the prompt for the LLM
        prompt = self._prepare_evaluation_prompt(subtask)
        
        # Query the LLM using the LLMClient
        self.logger.info("Querying LLM to evaluate subtask success")
        try:
            # Convert frames to base64 images for the LLM
            image_urls = []
            
            # Add frames from each camera and time point
            for cam_name in self.sampled_frames:
                for time_point in ["start", "middle", "end"]:
                    if time_point in self.sampled_frames[cam_name]:
                        frame = self.sampled_frames[cam_name][time_point]
                        
                        # Convert OpenCV BGR to RGB
                        if frame.shape[2] == 3:  # Check if it's a color image
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            frame_rgb = frame
                        
                        # Convert to base64
                        import base64
                        from io import BytesIO
                        from PIL import Image
                        
                        pil_img = Image.fromarray(frame_rgb)
                        buffered = BytesIO()
                        pil_img.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
                        # Add to image_urls for generate_with_images
                        image_url = f"data:image/jpeg;base64,{img_str}"
                        image_urls.append(image_url)
                        
                        # Debug log image size
                        self.logger.debug(f"Added {cam_name} {time_point} image: {len(img_str)} bytes")
            
            self.logger.info(f"Sending {len(image_urls)} images to LLM for task success evaluation")
            
            # Prepare a simplified prompt for generate_with_images
            image_prompt = f"""Evaluate if this robotic subtask was successful: {subtask}
                    
I'm showing you frames from different cameras at the start, middle, and end of the task execution.
Based on these images, determine if the subtask was successfully completed.

Start your response with either "SUCCESS:" or "FAILURE:" followed by your explanation."""
                    
            # Use generate_with_images with the image URLs
            if isinstance(self.llm_client, OpenAIClient):
                response = self.llm_client.generate_with_images(
                    prompt=image_prompt,
                    image_urls=image_urls,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            elif isinstance(self.llm_client, GeminiClient):
                # Strip the "data:image/jpeg;base64," prefix and decode
                image_bytes = [base64.b64decode(u.split(",",1)[1]) for u in image_urls]
                mime_types = ["image/jpeg"] * len(image_bytes)
                response = self.llm_client.generate_with_images(
                    prompt=image_prompt,
                    image_bytes=image_bytes,
                    image_mime_types=mime_types,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            else:
                raise ValueError(f"Unsupported LLM client for images: {type(self.llm_client)}")

        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            return False, f"Error querying LLM: {str(e)}"
        
        # Append interaction
        self.interactions.append({
            "type": "subtask",
            "prompt": prompt,
            "response": response,
        })

        # Parse the response
        success = response.lower().strip().startswith("success")
        
        return success, response
    
    def evaluate_goal_completion(self, overall_goal: str, subtask_history=None) -> Tuple[bool, str]:
        """
        Evaluate if the overall goal has been completed based on the final state
        and the history of subtask evaluations.
        
        Args:
            overall_goal: The overall goal statement
            subtask_history: Optional list of dictionaries containing subtask evaluation results
                Each dict should have: {'subtask': str, 'success': bool, 'explanation': str}
            
        Returns:
            Tuple of (completed: bool, explanation: str)
        """
        if not self.sampled_frames:
            self.logger.warning("No frames were sampled during inference")
            return False, "No frames were sampled during inference"
        
        # Get the final frame and detections from each camera
        final_frames = {}
        final_detections = {}
        image_urls = []
        
        for cam_name in self.sampled_frames:
            if "end" in self.sampled_frames[cam_name]:
                final_frames[cam_name] = self.sampled_frames[cam_name]["end"]
                final_detections[cam_name] = self.sampled_detections[cam_name]["end"]
        
        if not final_frames:
            self.logger.warning("No final frames available to evaluate goal completion")
            return False, "No final frames available to evaluate goal completion"
        
        # Prepare the prompt for the LLM, including subtask history if available
        prompt = self._prepare_goal_completion_prompt(overall_goal, final_detections, subtask_history)
        
        # Query the LLM using the LLMClient
        self.logger.info("Querying LLM to evaluate overall goal completion")
        try:
            # Add final frames from each camera
            for cam_name, frame in final_frames.items():
                # Convert OpenCV BGR to RGB
                if frame.shape[2] == 3:  # Check if it's a color image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                
                # Convert to base64
                import base64
                from io import BytesIO
                from PIL import Image
                
                pil_img = Image.fromarray(frame_rgb)
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Add to image_urls for generate_with_images
                image_urls.append(f"data:image/jpeg;base64,{img_str}")
            
            # Prepare a simplified prompt for generate_with_images
            image_prompt = f"""Evaluate if this robotic task's overall goal has been completed: {overall_goal}
                    
I'm showing you the final state from different camera angles.
Based on these images, determine if the overall goal was completed. You may ask yourself, 
"Did the robot successfully complete the task?"
Start your response with either "COMPLETE:" or "INCOMPLETE:" followed by your explanation."""
                    
            # Use generate_with_images with the image URLs
            if isinstance(self.llm_client, OpenAIClient):
                response = self.llm_client.generate_with_images(
                    prompt=image_prompt,
                    image_urls=image_urls,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            elif isinstance(self.llm_client, GeminiClient):
                # Strip the "data:image/jpeg;base64," prefix and decode
                image_bytes = [base64.b64decode(u.split(",",1)[1]) for u in image_urls]
                mime_types = ["image/jpeg"] * len(image_bytes)
                response = self.llm_client.generate_with_images(
                    prompt=image_prompt,
                    image_bytes=image_bytes,
                    image_mime_types=mime_types,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            else:
                raise ValueError(f"Unsupported LLM client for images: {type(self.llm_client)}")
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            return False, f"Error querying LLM: {str(e)}"
        
        # Append interaction
        self.interactions.append({
            "type": "goal",
            "prompt": prompt,
            "response": response,
        })

        # Parse the response
        completed = response.lower().strip().startswith("complete")
        
        # Save the final frame with detections if debug is enabled
        if self.debug:
            import time
            timestamp = int(time.time())
            
            for cam_name, frame in final_frames.items():
                # Save original frame
                img_path = self.image_dir / f"{timestamp}_{cam_name}_final_goal_completion.jpg"
                cv2.imwrite(str(img_path), frame)
                
                # Save frame with detection visualization
                vis_frame = frame.copy()
                det = final_detections[cam_name]
                for label, box in zip(det["labels"], det["boxes"]):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                vis_path = self.image_dir / f"{timestamp}_{cam_name}_final_goal_completion_detection.jpg"
                cv2.imwrite(str(vis_path), vis_frame)
                
                self.logger.info(f"Saved goal completion images to {img_path.name} and {vis_path.name}")
        
        return completed, response
    
    def _prepare_evaluation_prompt(self, subtask: str) -> str:
        """
        Prepare the prompt for the LLM to evaluate subtask success
        """
        prompt = f"""Based on the following object detections from three time points (start, middle, end) during a robotic task execution, determine if the subtask was successfully achieved.

Subtask: {subtask}

Object detections:
"""
        
        # Add detections from each camera and time point
        for cam_name in self.sampled_detections:
            prompt += f"\n{cam_name.upper()} CAMERA:\n"
            
            for time_point in ["start", "middle", "end"]:
                if time_point in self.sampled_detections[cam_name]:
                    det = self.sampled_detections[cam_name][time_point]
                    objects_str = ", ".join([f"{label}: {box}" for label, box in 
                                           zip(det["labels"], det["boxes"])])
                    prompt += f"  {time_point}: {objects_str}\n"
        
        prompt += """
Based on these object detections in chronological order, would you say this is a successful completion of the subtask?
Respond with either stating "Success" or "Failure" at the beginning, followed by a brief, one sentence summary of why.
"""
        return prompt
    
    def _prepare_goal_completion_prompt(self, overall_goal: str, final_detections: Dict, subtask_history=None) -> str:
        """
        Prepare the prompt for the LLM to evaluate overall goal completion
        
        Args:
            overall_goal: The overall goal statement
            final_detections: Dictionary of object detections from the final state
            subtask_history: Optional list of subtask evaluation results
            
        Returns:
            Prompt string for the LLM
        """
        prompt = f"""Based on the final state of a robotic task execution, determine if the overall goal has been completed.

Overall goal: {overall_goal}

Final state object detections:
"""
        
        # Add detections from each camera
        for cam_name in final_detections:
            prompt += f"\n{cam_name.upper()} CAMERA:\n"
            det = final_detections[cam_name]
            objects_str = ", ".join([f"{label}: {box}" for label, box in 
                                   zip(det["labels"], det["boxes"])])
            prompt += f"  {objects_str}\n"
        
        # Add subtask history if available
        if subtask_history:
            prompt += "\nSubtask execution history:\n"
            for i, result in enumerate(subtask_history):
                subtask = result.get('subtask', 'unknown')
                success = result.get('success', False)
                explanation = result.get('explanation', '')
                prompt += f"{i+1}. Subtask: {subtask}\n"
                prompt += f"   Success: {success}\n"
                prompt += f"   Explanation: {explanation}\n"
        
        prompt += """
Based on these object detections in the final state and the subtask history, would you say the overall goal has been completed?
Respond with either "COMPLETE:" or "INCOMPLETE:" at the beginning, followed by a brief explanation.
"""
        return prompt
    
    def evaluate_task_recoverability(self, subtask: str, failed_explanation: str) -> Tuple[bool, str]:
        """
        Evaluate if a failed subtask is recoverable from the final state using the last frame
        and object detection results.
        
        Args:
            subtask: The subtask goal statement
            failed_explanation: The explanation of why the subtask failed
            
        Returns:
            Tuple of (recoverable: bool, recovery_plan: str)
        """
        if not self.sampled_frames:
            self.logger.warning("No frames were sampled during inference")
            return False, "No frames were sampled to evaluate recoverability"
        
        # Get the final frame and detections from each camera
        final_frames = {}
        final_detections = {}
        
        for cam_name in self.sampled_frames:
            if "end" in self.sampled_frames[cam_name]:
                final_frames[cam_name] = self.sampled_frames[cam_name]["end"]
                final_detections[cam_name] = self.sampled_detections[cam_name]["end"]
        
        if not final_frames:
            self.logger.warning("No final frames available to evaluate recoverability")
            return False, "No final frames available to evaluate recoverability"
        
        # Prepare the prompt for the LLM
        prompt = self._prepare_recoverability_prompt(subtask, failed_explanation, final_detections)
        
        # Query the LLM using the LLMClient
        self.logger.info("Querying LLM to evaluate task recoverability")
        try:
            # Create a message list with text and images
            messages = [
                {"role": "system", "content": "You evaluate if a failed robotic task is recoverable based on the final state and object detections."}
            ]
            
            # Create a content list for the user message
            user_content = [{"type": "text", "text": prompt}]
            
            # Add final frames from each camera
            image_urls = []  # Initialize image_urls list here
            
            for cam_name, frame in final_frames.items():
                # Convert OpenCV BGR to RGB
                if frame.shape[2] == 3:  # Check if it's a color image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                
                # Convert to base64
                import base64
                from io import BytesIO
                from PIL import Image
                
                pil_img = Image.fromarray(frame_rgb)
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Add to image_urls for the Gemini client
                image_url = f"data:image/jpeg;base64,{img_str}"
                image_urls.append(image_url)
                
                # Add to content with a caption
                user_content.append({
                    "type": "text", 
                    "text": f"\n{cam_name.upper()} CAMERA - FINAL STATE:"
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high"
                    }
                })
            
            # Add the user message with text and images
            messages.append({"role": "user", "content": user_content})
            
            # Use the LLMClient to get the response
            if isinstance(self.llm_client, OpenAIClient):
                response = self.llm_client.chat(
                    messages=messages,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            elif isinstance(self.llm_client, GeminiClient):
                # Strip the "data:image/jpeg;base64," prefix and decode
                image_bytes = [base64.b64decode(u.split(",",1)[1]) for u in image_urls]
                mime_types = ["image/jpeg"] * len(image_bytes)
                response = self.llm_client.chat(
                    messages=messages,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            else:
                raise ValueError(f"Unsupported LLM client for images: {type(self.llm_client)}")
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            return False, f"Error querying LLM: {str(e)}"
        
        # Append interaction
        self.interactions.append({
            "type": "recoverability",
            "prompt": prompt,
            "response": response,
        })

        # Parse the response
        recoverable = response.lower().strip().startswith("recoverable")
        
        # Save the final frame with detections if debug is enabled
        if self.debug:
            import time
            timestamp = int(time.time())
            
            for cam_name, frame in final_frames.items():
                # Save original frame
                img_path = self.image_dir / f"{timestamp}_{cam_name}_final_recoverability.jpg"
                cv2.imwrite(str(img_path), frame)
                
                # Save frame with detection visualization
                vis_frame = frame.copy()
                det = final_detections[cam_name]
                for label, box in zip(det["labels"], det["boxes"]):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                vis_path = self.image_dir / f"{timestamp}_{cam_name}_final_recoverability_detection.jpg"
                cv2.imwrite(str(vis_path), vis_frame)
                
                self.logger.info(f"Saved recoverability images to {img_path.name} and {vis_path.name}")
        
        return recoverable, response

    def _prepare_recoverability_prompt(self, subtask: str, failed_explanation: str, final_detections: Dict) -> str:
        """
        Prepare the prompt for the LLM to evaluate task recoverability
        """
        prompt = f"""Based on the final state of a failed robotic subtask execution, determine if the subtask is recoverable.

Subtask: {subtask}

Failure explanation: {failed_explanation}

Final state object detections:
"""
        
        # Add detections from each camera
        for cam_name in final_detections:
            prompt += f"\n{cam_name.upper()} CAMERA:\n"
            det = final_detections[cam_name]
            objects_str = ", ".join([f"{label}: {box}" for label, box in 
                                   zip(det["labels"], det["boxes"])])
            prompt += f"  {objects_str}\n"
        
        prompt += """
Based on these object detections in the final state, would you say this failed subtask is recoverable?
Respond with either stating "Recoverable" or "Not recoverable" at the beginning, followed by:
1. A brief, one sentence explanation of why it is or isn't recoverable
2. Say "Please help me recover the task" if it is not recoverable
"""
        return prompt
    
    def reset_sampled_frames(self):
        """
        Reset the sampled frames and detections for a new policy execution
        """
        self.sampled_frames = {}
        self.sampled_detections = {}
        self.logger.info("Reset sampled frames and detections for new policy execution")
    
    def get_sampled_frame_count(self):
        """
        Get the number of sampled frames
        
        Returns:
            dict: Dictionary with camera names as keys and the number of frames as values
        """
        result = {}
        for cam_name in self.sampled_frames:
            result[cam_name] = len(self.sampled_frames[cam_name])
        return result
    
    def get_interactions(self):
        """Return the logged interactions."""
        return self.interactions

if __name__ == "__main__":
    pass