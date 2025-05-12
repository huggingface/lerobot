import logging
import yaml
import argparse
from typing import Optional, List

import torch
from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.active_feedback.detect.object_detector import ObjectDetector
from lerobot.active_feedback.inference.llm_policy_selector import LLMPolicySelector
from lerobot.active_feedback.inference.llm_client import OpenAIClient, GeminiClient
from lerobot.active_feedback.inference.inference import Inference

class InferenceRunner:
    """
    Runner class for LGA inference.
    """
    
    def __init__(self, debug: bool = False, device: str = "cuda"):
        self.DEBUG = debug
        self.device = device
        
        # build configs
        self.leader_config = DynamixelMotorsBusConfig(
            port="/dev/ttyACM0",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        )

        self.follower_config = DynamixelMotorsBusConfig(
            port="/dev/ttyACM1",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        )
    
        self.leader_arm = DynamixelMotorsBus(self.leader_config)
        self.follower_arm = DynamixelMotorsBus(self.follower_config)

        self.robot_cfg = KochRobotConfig(
            leader_arms={"main": self.leader_config},
            follower_arms={"main": self.follower_config},
            calibration_dir=".cache/calibration/koch",
            cameras={
                "front": OpenCVCameraConfig(0, fps=30, width=640, height=480),
                "overhead": OpenCVCameraConfig(2, fps=30, width=640, height=480),
            },
        )

        # Load config file
        with open("inference_config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        
        # Get API keys
        self.roboflow_api_key = self.config["roboflow_api_key"]
        self.openai_api_key = self.config.get("openai_api_key", "")
        self.gemini_api_key = self.config.get("gemini_api_key", "")

        # Initialize object detector
        self.detector = ObjectDetector(
            debug=self.DEBUG,
            inference_server_url="http://localhost:9001",
            roboflow_api_key=self.roboflow_api_key
        )

        # LLM clients will be initialized when needed
        self.openai_client = None
        self.gemini_client = None
        self.llm_client = None
        
        # LLM configuration for policy selection
        self.policy_model = {
            "openai": "gpt-4o",
            "gemini": "gemini-2.5-flash-preview-04-17"
        }
        self.policy_temperature = 0.5
        self.policy_max_tokens = 2048
        
        # LLM configuration for evaluation
        self.eval_model = {
            "openai": "gpt-4o",
            "gemini": "gemini-2.5-flash-preview-04-17"
        }
        self.eval_temperature = 0.5
        self.eval_max_tokens = 2048

        # Policy selector will be initialized when needed
        self.selector = None

        self.policy_ckpts = {
            "Place cube inside the box": "arclabmit/koch_act_cubebin_model",
            "Place box inside the bin": "arclabmit/koch_act_boxbin_model",
            "Land on the moon": "arclabmit/lunar_lander_act_model",
        }
        
        # Initialize Inference
        self.lga = None
    
    def _initialize_llm_client(self, provider: str = "openai"):
        """Initialize the LLM client based on the provider."""
        if provider.lower() == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not found in config.yaml")
            
            if self.openai_client is None:
                self.openai_client = OpenAIClient(api_key=self.openai_api_key)
            
            self.llm_client = self.openai_client
            return self.policy_model["openai"], self.eval_model["openai"]
            
        elif provider.lower() == "gemini":
            if not self.gemini_api_key:
                raise ValueError("Gemini API key not found in config.yaml")
            
            if self.gemini_client is None:
                self.gemini_client = GeminiClient(api_key=self.gemini_api_key)
            
            self.llm_client = self.gemini_client
            return self.policy_model["gemini"], self.eval_model["gemini"]
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _initialize_lga(self, provider: str = "openai", fps: int = 30):
        """Initialize the LGA inference engine with the specified LLM provider."""
        # Initialize LLM client
        policy_model, eval_model = self._initialize_llm_client(provider)
        
        # Set provider attribute for tracking
        self.llm_client._provider = provider.lower()
        
        # Create policy selector
        self.selector = LLMPolicySelector(
            llm_client=self.llm_client,
            model=policy_model,
            temperature=self.policy_temperature,
            max_tokens=self.policy_max_tokens if provider.lower() == "openai" else None,  # Only use max_tokens for OpenAI
            debug=self.DEBUG
        )
        
        # Initialize LGA
        self.lga = Inference(
            leader_cfg=self.leader_config,
            follower_cfg=self.follower_config,
            robot_cfg=self.robot_cfg,
            detector=self.detector,
            selector=self.selector,
            llm_client=self.llm_client,
            policy_checkpoints=self.policy_ckpts,
            fps=fps,
            device=self.device,
            debug=self.DEBUG,
            eval_model_name=eval_model,
            eval_temperature=self.eval_temperature,
            eval_max_tokens=self.eval_max_tokens if provider.lower() == "openai" else None  # Only use max_tokens for OpenAI
        )
    
    def run_inference(self, goal: str, provider: str = "openai", candidate_policies: Optional[List[str]] = None, fps: int = 30):
        """Run inference with the specified goal and LLM provider."""
        if self.lga is None or self.llm_client is None or provider.lower() != getattr(self.llm_client, '_provider', 'openai'):
            self._initialize_lga(provider=provider, fps=fps)
            
        if not self.lga.robot.is_connected:
            self.lga.robot.connect()
            
        if candidate_policies is None:
            candidate_policies = list(self.policy_ckpts.keys())
        
        try:            
            # Run the LGA inference
            results = self.lga.run(goal=goal, candidate_policies=candidate_policies)
            
            # Log results summary
            if results:
                logging.info(f"Goal achieved: {results.get('goal_complete', False)}")
                logging.info(f"Steps taken: {results.get('steps_taken', 0)}")
                
                # Log detailed step results if available
                step_results = results.get('step_results', [])
                for i, step in enumerate(step_results):
                    logging.info(f"Step {i+1} - Policy: {step.get('chosen_policy', 'unknown')}")
                    logging.info(f"  Subtask success: {step.get('subtask_success', False)}")
                    logging.info(f"  Goal complete: {step.get('goal_complete', False)}")
            
            return results
            
        finally:
            # Ensure robot is properly disconnected
            if self.lga and self.lga.robot and self.lga.robot.is_connected:
                try:
                    self.lga.robot.disconnect()
                except Exception as e:
                    logging.error(f"Error during disconnect: {e}")


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LGA inference with different LLM providers")
    parser.add_argument("--goal", type=str, default="Our goal is to place a cube into a ROBOTIS box then place the box into a bin.",
                        help="Goal description for the robot")
    parser.add_argument("--provider", type=str, choices=["openai", "gemini"], default="gemini",
                        help="LLM provider to use (openai or gemini)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for robot control")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], 
                        help="Device to run inference on")
    
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create runner
    runner = InferenceRunner(debug=args.debug, device=args.device)

    # Run inference
    results = runner.run_inference(
        goal=args.goal,
        provider=args.provider,
        fps=args.fps
    )
    
    # Print final results summary
    if results:
        print("\n=== FINAL RESULTS ===")
        print(f"Goal: {results.get('goal', 'unknown')}")
        print(f"Goal achieved: {results.get('goal_complete', False)}")
        print(f"Steps taken: {results.get('steps_taken', 0)}")
        print(f"LLM Provider: {args.provider}")