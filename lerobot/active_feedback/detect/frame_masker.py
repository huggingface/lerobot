import cv2
import numpy as np

class FrameMasker:
    """
    Applies masks to frames for visualization or further processing.
    
    This class provides utilities to apply segmentation masks to frames,
    create masked images, and visualize detection results.
    """
    
    def __init__(self, debug=False):
        """
        Initialize the frame masker.
        
        Args:
            debug (bool): Whether to enable debug mode with additional logging
        """
        self.debug = debug
        
    def apply_mask(self, image, mask, color=(0, 255, 0), alpha=0.5):
        """
        Apply a colored mask overlay to an image.
        
        Args:
            image (numpy.ndarray): Input image
            mask (numpy.ndarray): Binary mask
            color (tuple): RGB color for the mask
            alpha (float): Transparency factor (0-1)
            
        Returns:
            numpy.ndarray: Image with colored mask overlay
        """
        # Ensure mask is binary
        if mask.dtype != bool:
            mask = mask > 0.5
            
        # Create colored mask
        colored_mask = np.zeros_like(image)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
            
        # Blend with original image
        return cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    def create_masked_image(self, image, mask):
        """
        Create an image showing only the masked region.
        
        Args:
            image (numpy.ndarray): Input image
            mask (numpy.ndarray): Binary mask
            
        Returns:
            numpy.ndarray: Masked image
        """
        # Ensure mask is binary
        if mask.dtype != bool:
            mask = mask > 0.5
            
        # Apply mask to each channel
        masked_image = image.copy()
        for c in range(3):
            masked_image[:, :, c] = masked_image[:, :, c] * mask
            
        return masked_image
    
    def visualize_detections(self, image, detection_results, show_labels=True):
        """
        Visualize detection results on an image.
        
        Args:
            image (numpy.ndarray): Input image
            detection_results (dict): Results from ObjectDetector
            show_labels (bool): Whether to show labels
            
        Returns:
            numpy.ndarray: Visualization image
        """
        vis_image = image.copy()
        
        # Generate random colors for each mask
        masks = detection_results.get("masks", [])
        num_masks = len(masks)
        colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
                 for _ in range(num_masks)]
        
        # Apply each mask with a different color
        for i, mask in enumerate(masks):
            vis_image = self.apply_mask(vis_image, mask, color=colors[i], alpha=0.4)
            
        # Draw bounding boxes if available
        if "boxes" in detection_results:
            boxes = detection_results["boxes"]
            labels = detection_results.get("labels", [""] * len(boxes))
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), colors[i], 2)
                
                if show_labels and i < len(labels):
                    label = labels[i]
                    cv2.putText(vis_image, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        
        return vis_image