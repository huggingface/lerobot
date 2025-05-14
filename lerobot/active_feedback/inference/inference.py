import time
import logging
import imageio
import cv2
import torch
import numpy as np

from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

from lerobot.active_feedback.detect.object_detector import ObjectDetector
from lerobot.active_feedback.inference.llm_policy_selector import LLMPolicySelector
from lerobot.active_feedback.inference.llm_client import LLMClient
from lerobot.scripts.control_robot import busy_wait
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.active_feedback.inference.llm_evaluation import TaskSuccessEvaluator

class Inference:
    """
    Encapsulates the LGA inference pipeline:
     1) object detection
     2) policy selection via ChatGPT
     3) checkpoint loading
     4) action loop
     5) task success evaluation
     6) safe shutdown
    """

    def __init__(
        self,
        # motor buses
        leader_cfg: DynamixelMotorsBusConfig,
        follower_cfg: DynamixelMotorsBusConfig,
        # robot + camera config
        robot_cfg: KochRobotConfig,
        # object detector
        detector: ObjectDetector,
        # selector
        selector: LLMPolicySelector,
        # LLM client for evaluation
        llm_client: LLMClient,
        # mapping from policy_name -> checkpoint path
        policy_checkpoints: dict,
        # runtime params
        fps: int = 30,
        device: str = "cuda",
        debug: bool = False,
        max_steps: int = 3,  # Maximum number of policy steps to try
        # LLM model parameters for evaluation
        eval_model_name: str = "gpt-4o",
        eval_temperature: float = 0.5,
        eval_max_tokens: int = 1024,
    ):
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")
        self.logger.info("Initializing LGAInference pipeline")

        # hardware buses & robot
        self.leader_bus   = DynamixelMotorsBus(leader_cfg)
        self.follower_bus = DynamixelMotorsBus(follower_cfg)
        self.robot        = ManipulatorRobot(robot_cfg)
        self.robot.connect()

        # vision & policy selector
        self.detector = detector
        self.selector = selector
        self.llm_client = llm_client

        # checkpoint mapping
        self.policy_checkpoints = policy_checkpoints

        # runtime params
        self.fps              = fps
        self.device           = device
        self.max_steps        = max_steps

        # Initialize cameras and perform warmup once during initialization
        self._initialize_cameras()
        
        # Create the task success evaluator
        self.task_evaluator = TaskSuccessEvaluator(
            detector=self.detector,
            llm_client=self.llm_client,
            debug=self.debug,
            model_name=eval_model_name,
            temperature=eval_temperature,
            max_tokens=eval_max_tokens
        )

        # Initialize history tracking for multi-step reasoning
        self.action_history = []
        self.sampled_frames_history = {}  # Will store frames from each step

    def _initialize_cameras(self):
        """
        Initialize cameras and perform warmup once during robot initialization.
        """
        if not hasattr(self.robot, 'cameras') or not self.robot.cameras:
            self.logger.warning("No cameras found to initialize")
            return
        
        self.logger.info("Initializing cameras with warmup...")
        warmup_frames = 10
        for _ in range(warmup_frames):
            for cam_name, cam in self.robot.cameras.items():
                try:
                    _ = cam.read()
                    time.sleep(0.05)  # Short delay between warmup frames
                except Exception as e:
                    self.logger.warning(f"Camera {cam_name} warmup error: {e}")
        self.logger.info("Camera warmup complete")

    def detect_objects(self) -> dict[str, list[float]]:
        """
        Runs object detection on each camera, returns the unique set of labels.
        If debug, writes overlay images too.
        """
        all_objects = {}
        for cam_name, cam in self.robot.cameras.items():
            frame = cam.read()
            det   = self.detector.detect_and_segment(frame, box_threshold=0.5)
            self.logger.debug(f"[{cam_name}] detections = {det}")
            all_objects.update({det["labels"][i]: det["boxes"][i] for i in range(len(det["labels"]))})

            if self.debug:
                vis = frame.copy()
                for box, label in zip(det["boxes"], det["labels"]):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(vis, label, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                out_path = f"{cam_name}_detections.png"
                imageio.imwrite(out_path, vis)
                self.logger.info(f"Saved overlay → {out_path}")

        return all_objects

    def load_policy(self, policy_name: str) -> ACTPolicy:
        """
        Loads ACTPolicy from a checkpoint.
        """
        ckpt = self.policy_checkpoints.get(policy_name)
        if ckpt is None:
            raise RuntimeError(f"No checkpoint for policy '{policy_name}'")
        self.logger.info(f"Loading ACTPolicy from {ckpt}")
        return ACTPolicy.from_pretrained(ckpt).to(self.device)

    def run_inference_loop(self, policy: ACTPolicy, goal: str, subtask: str, max_time_s: int = 15, step_count: int = 1):
        """
        Real-time loop: capture → normalize → policy → send_action → wait.
        Also samples frames for task success evaluation.
        
        Args:
            policy: The policy to execute
            goal: The overall goal
            subtask: The current subtask
            max_time_s: Maximum time to run the policy
            step_count: Current step number in the multi-step process
        """
        # Camera warmup removed from here as it's now done once during initialization
        
        total_steps = max_time_s * self.fps
        self.logger.info(f"Entering inference loop for {total_steps} steps")
        
        # Reset the task evaluator's sampled frames
        self.task_evaluator.reset_sampled_frames()
        
        # Calculate the step indices for sampling (start, middle, end)
        sample_points = [0.0, 0.5, 0.9]  # Relative positions to sample
        sample_indices = [int(p * total_steps) for p in sample_points]
        
        # Create a dictionary to store frames for this step
        step_frames = {
            "start": {},
            "middle": {},
            "end": {}
        }
        
        for step in range(total_steps):
            t0 = time.perf_counter()
            
            # Capture observation (this already gets frames from all cameras)
            obs = self.robot.capture_observation()
            
            # Log observation keys for debugging
            if step == 0:
                self.logger.debug(f"Observation keys: {list(obs.keys())}")
            
            # Sample frames at specific points for task success evaluation
            if step in sample_indices:
                sample_idx = sample_indices.index(step)
                sample_name = ["start", "middle", "end"][sample_idx]
                self.logger.debug(f"Sampling {sample_name} frame at step {step}/{total_steps}")
                
                # Process each camera frame
                for key in list(obs.keys()):
                    if "image" in key:
                        cam_name = key.replace("_image", "")
                        
                        # Get the original frame before normalization
                        frame = obs[key]
                        
                        # Convert to numpy if it's a torch tensor
                        if isinstance(frame, torch.Tensor):
                            frame = frame.cpu().numpy()
                        
                        # Make sure it's in the right format (HWC for OpenCV)
                        if len(frame.shape) == 3 and frame.shape[0] == 3:  # CHW format
                            frame = np.transpose(frame, (1, 2, 0))
                        
                        # If normalized to [0,1], convert back to [0,255]
                        if frame.dtype == np.float32 or frame.dtype == np.float64:
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                        
                        # Convert from RGB to BGR for OpenCV
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        self.logger.debug(f"Frame for {cam_name}: shape={frame.shape}, type={type(frame)}, dtype={frame.dtype}")
                        
                        # Add the frame to the task evaluator
                        self.task_evaluator.add_frame(cam_name=cam_name, sample_name=sample_name, frame=frame)
                        
                        # Also store the frame in our step_frames dictionary for history
                        step_frames[sample_name][cam_name] = frame.copy()
            
            # normalize images & move to device for policy
            for k, v in obs.items():
                if "image" in k:
                    v = v.type(torch.float32) / 255.0
                    v = v.permute(2, 0, 1).contiguous()
                obs[k] = v.unsqueeze(0).to(self.device)

            action = policy.select_action(obs).squeeze(0).to("cpu")
            self.robot.send_action(action)
            busy_wait(max(0.0, 1/self.fps - (time.perf_counter() - t0)))

        self.logger.info("Inference loop complete")
        
        # Log the number of frames sampled for debugging
        self.logger.debug(f"Sampled frames: {self.task_evaluator.get_sampled_frame_count()}")
        
        # Evaluate subtask success
        subtask_success, subtask_explanation = self.task_evaluator.evaluate_task_success(subtask)
        self.logger.info(f"Subtask success evaluation: {subtask_success}")
        self.logger.info(f"Subtask reasoning: {subtask_explanation}")
        
        # Create subtask history entry
        subtask_result = {
            'subtask': subtask,
            'success': subtask_success,
            'explanation': subtask_explanation
        }
        
        # Add to subtask history
        if not hasattr(self, 'subtask_history'):
            self.subtask_history = []
        self.subtask_history.append(subtask_result)
        
        # Evaluate overall goal completion with subtask history
        goal_complete, goal_explanation = self.task_evaluator.evaluate_goal_completion(
            goal, 
            subtask_history=self.subtask_history
        )
        self.logger.info(f"Overall goal completion: {goal_complete}")
        self.logger.info(f"Goal reasoning: {goal_explanation}")
        
        # If subtask failed, evaluate recoverability
        recoverability_result = None
        if not subtask_success:
            self.logger.info("Subtask failed, evaluating recoverability...")
            recoverable, recovery_plan = self.task_evaluator.evaluate_task_recoverability(subtask, subtask_explanation)
            self.logger.info(f"Task recoverability: {recoverable}")
            self.logger.info(f"Recovery reasoning: {recovery_plan}")
            recoverability_result = (recoverable, recovery_plan)
        
        # Store the frames from this step in our history
        self.sampled_frames_history[f"step_{step_count}"] = step_frames
        
        return {
            "subtask_success": subtask_success,
            "subtask_explanation": subtask_explanation,
            "goal_complete": goal_complete,
            "goal_explanation": goal_explanation,
            "recoverability_result": recoverability_result
        }

    def run(self, goal: str, candidate_policies: list[str], max_time_s: int = 15):
        """
        Orchestrates:
          1) detect_objects
          2) select_policy
          3) load_policy
          4) run_inference_loop
          5) evaluate task success
          6) evaluate goal completion
          7) if goal not complete, repeat steps 1-6
          8) safe shutdown
        """
        try:
            step_results = []
            goal_complete = False
            step_count = 0
            
            # Initialize subtask history
            self.subtask_history = []
            
            while not goal_complete and step_count < self.max_steps:
                step_count += 1
                self.logger.info(f"Starting step {step_count}/{self.max_steps} to achieve goal: {goal}")
                
                # Detect objects in the scene
                objects = self.detect_objects()
                
                # Capture current frames for policy selection
                camera_frames = {}
                for cam_name, cam in self.robot.cameras.items():
                    frame = cam.read()
                    camera_frames[cam_name] = frame
                
                # Create a history summary for the LLM if we have previous steps
                history_summary = ""
                if step_count > 1:
                    history_summary = "Previous actions:\n"
                    for i, result in enumerate(step_results):
                        step_num = i + 1
                        policy = result["chosen_policy"]
                        success = "succeeded" if result["subtask_success"] else "failed"
                        explanation = result["subtask_explanation"]
                        history_summary += f"Step {step_num}: Used policy '{policy}' which {success}. {explanation}\n"
                
                # Select policy directly using the selector
                # For the first step, we don't have history
                if step_count == 1:
                    chosen_policy, prompt, response = self.selector.select_policy(
                        detected_objects=objects,
                        goal=goal,
                        candidate_policies=candidate_policies,
                        camera_frames=camera_frames,
                        return_prompt_response=True
                    )
                else:
                    # For subsequent steps, include history in the prompt
                    # We'll modify the prompt to include history information
                    history_prompt = (
                        f"Goal: {goal}\n\n"
                        f"{history_summary}\n\n"
                        f"Based on the previous actions and current scene, select the next policy to progress toward the goal."
                    )
                    
                    # Include both current frames and key frames from history
                    # For history frames, we'll use the end frames from the previous step
                    history_frames = {}
                    if step_count > 1:
                        prev_step = f"step_{step_count-1}"
                        if prev_step in self.sampled_frames_history:
                            for cam_name, frame in self.sampled_frames_history[prev_step]["end"].items():
                                history_frames[f"previous_{cam_name}"] = frame
                    
                    # Combine current and history frames
                    all_frames = {**camera_frames, **history_frames}
                    
                    chosen_policy, prompt, response = self.selector.select_policy(
                        detected_objects=objects,
                        goal=history_prompt,  # Use our enhanced prompt with history
                        candidate_policies=candidate_policies,
                        camera_frames=all_frames,  # Include both current and history frames
                        return_prompt_response=True
                    )

                # Log reasoning for policy selection
                self.logger.info(f"Policy selection reasoning: {response}")
                
                # Load the selected policy
                policy = self.load_policy(chosen_policy)
                
                # Extract the subtask from the policy name or use the policy name as the subtask
                subtask = chosen_policy
                
                # Run the inference loop for this policy
                result = self.run_inference_loop(policy, goal, subtask, max_time_s, step_count)
                
                # Add policy selection info to the result
                result["step"] = step_count
                result["chosen_policy"] = chosen_policy
                result["policy_selection_prompt"] = prompt
                result["policy_selection_response"] = response
                
                # Store the result for this step
                step_results.append(result)
                
                # Add to action history
                self.action_history.append({
                    "step": step_count,
                    "policy": chosen_policy,
                    "success": result["subtask_success"],
                    "explanation": result["subtask_explanation"]
                })
                
                # Check if the overall goal is complete
                goal_complete = result["goal_complete"]
                
                if goal_complete:
                    self.logger.info(f"Goal achieved after {step_count} steps!")
                    break
                    
                # If we've reached the maximum number of steps, log a warning
                if step_count >= self.max_steps and not goal_complete:
                    self.logger.warning(f"Reached maximum number of steps ({self.max_steps}) without completing the goal")
                
                # Short pause between steps to allow for system stabilization
                if not goal_complete and step_count < self.max_steps:
                    self.logger.info("Pausing between policy steps...")
                    time.sleep(2.0)
            
            # Compile final results
            final_results = {
                "goal": goal,
                "goal_complete": goal_complete,
                "steps_taken": step_count,
                "step_results": step_results,
                "action_history": self.action_history,
                "subtask_history": self.subtask_history,  # Include the subtask history in results
                "policy_selector_interactions": self.selector.get_interactions(),
                "task_evaluator_interactions": self.task_evaluator.get_interactions()
            }
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # Safe shutdown
            self.logger.info("Shutting down robot...")
            try:
                self.robot.disconnect()
                self.logger.info("Robot disconnected successfully")
            except Exception as e:
                self.logger.error(f"Error disconnecting robot: {e}")