import torch
import numpy as np
import cv2
import os
import yaml

class ObjectDetector:
    """
    Handles object detection and segmentation using a model hosted on a local inference server.
    
    This class communicates with an inference server running a workflow (e.g., "detect-count-and-visualize")
    that returns a list with structured results. The expected format is:
    
    [
      {
        "count_objects": 3,
        "output_image": <binary image data>,
        "predictions": {
          "image": {"width": 640, "height": 480},
          "predictions": [
            {
              "width": 75,
              "height": 74,
              "x": 401.5,
              "y": 99,
              "confidence": 0.97706,
              "class_id": 1,
              "class": "box",
              "detection_id": "8545e92d-ac88-492c-8ba2-5936144017a9",
              "parent_id": "image"
            },
            { ... },
            { ... }
          ]
        }
      }
    ]
    
    The detector extracts the predictions from the response and uses them for further processing.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", debug=False, 
                roboflow_api_key=None, config_path=None, 
                 inference_server_url=None):
        self.device = device
        self.debug = debug
        self.inference_client = None
        
        # Load configuration if available.
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "config.yaml")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                if roboflow_api_key is None and 'roboflow_api_key' in config:
                    roboflow_api_key = config['roboflow_api_key']
                    if debug:
                        print("Loaded Roboflow API key from config")
                if inference_server_url is None and 'inference_server_url' in config:
                    inference_server_url = config['inference_server_url']
                    if debug:
                        print(f"Loaded inference server URL from config: {inference_server_url}")
            except Exception as e:
                if debug:
                    print(f"Error loading config: {e}")
        
        if inference_server_url:
            try:
                from inference_sdk import InferenceHTTPClient
                self.inference_client = InferenceHTTPClient(
                    api_url=inference_server_url,
                    api_key=roboflow_api_key
                )
                if debug:
                    print(f"Connected to inference server at {inference_server_url}")
            except Exception as e:
                raise RuntimeError(f"Error connecting to inference server: {e}")
        else:
            raise ValueError("inference_server_url must be provided to use the inference server.")
    
    def detect_and_segment(self, image, text_prompt=None, box_threshold=0.51):
        """
        Detect and segment objects in an image using the inference server.
        
        Args:
            image (numpy.ndarray or torch.Tensor): Input image.
            text_prompt (str, optional): Text prompt for filtering objects.
            box_threshold (float): Minimum confidence to accept a detection.
            
        Returns:
            dict: Contains lists "boxes", "labels", and "masks" for each detected object.
        """
        if isinstance(image, torch.Tensor):
            from lerobot.active_feedback.detect.frame_processor import FrameProcessor
            image_np = FrameProcessor.tensor_to_numpy(image)
        else:
            image_np = image
        
        if self.inference_client is None:
            raise RuntimeError("Inference client is not initialized.")
        
        # Save image to a temporary file.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
            temp_path = temp.name
            cv2.imwrite(temp_path, image_np)
        
        try:
            # Call the workflow on the inference server.
            results = self.inference_client.run_workflow(
                workspace_name="lga-act",
                workflow_id="detect-count-and-visualize",
                images={"image": temp_path}
            )
        except Exception as e:
            os.unlink(temp_path)
            raise RuntimeError(f"Inference server error: {e}")
        
        os.unlink(temp_path)
        
        # The inference server returns a list; take the first element.
        if isinstance(results, list) and len(results) > 0:
            workflow_result = results[0]
        else:
            workflow_result = results
        
        # Get the nested predictions.
        predictions_data = workflow_result.get("predictions", {})
        predictions_list = predictions_data.get("predictions", [])
        
        boxes = []
        labels = []
        masks = []
        h, w = image_np.shape[:2]
        
        if predictions_list and len(predictions_list) > 0:
            for pred in predictions_list:
                confidence = pred.get("confidence", 0)
                if confidence < box_threshold:
                    continue
                x_center = pred.get("x")
                y_center = pred.get("y")
                width_val = pred.get("width")
                height_val = pred.get("height")
                
                # Determine if coordinates are normalized (<= 1.0) or absolute.
                if x_center <= 1.0 and y_center <= 1.0:
                    x1 = (x_center - width_val / 2) * w
                    y1 = (y_center - height_val / 2) * h
                    x2 = (x_center + width_val / 2) * w
                    y2 = (y_center + height_val / 2) * h
                else:
                    x1 = x_center - width_val / 2
                    y1 = y_center - height_val / 2
                    x2 = x_center + width_val / 2
                    y2 = y_center + height_val / 2
                
                boxes.append([x1, y1, x2, y2])
                labels.append(pred.get("class", "unknown"))
                
                # Create a simple binary mask from the bounding box.
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
                masks.append(mask)
        else:
            if self.debug:
                print("No predictions received from inference server; creating dummy detection.")
            # Fall back to a dummy detection if none found.
            x1, y1, x2, y2 = w / 4, h / 4, 3 * w / 4, 3 * h / 4
            boxes.append([x1, y1, x2, y2])
            labels.append("background")
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
            masks.append(mask)
        
        if text_prompt is not None and boxes:
            boxes, labels, masks = self._filter_by_text_prompt(boxes, labels, masks, text_prompt)
        
        if self.debug:
            print(f"Detected {len(boxes)} objects.")
        return {"masks": masks, "boxes": boxes, "labels": labels}
    
    def _filter_by_text_prompt(self, boxes, labels, masks, text_prompt):
        keywords = [k.strip().lower().rstrip('.') for k in text_prompt.split('.') if k.strip()]
        filtered_boxes = []
        filtered_labels = []
        filtered_masks = []
        for i, label in enumerate(labels):
            if any(keyword in label.lower() for keyword in keywords):
                filtered_boxes.append(boxes[i])
                filtered_labels.append(label)
                filtered_masks.append(masks[i])
        return filtered_boxes, filtered_labels, filtered_masks
