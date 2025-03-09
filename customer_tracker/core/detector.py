"""
Customer detection module using YOLOv8.
"""
from ultralytics import YOLO
import supervision as sv
import numpy as np

class CustomerDetector:
    """
    Handles customer detection using YOLOv8 model.
    """
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the customer detector.
        
        Args:
            model_path (str): Path to the YOLOv8 model file
        """
        self.model = YOLO(model_path)
        self.model.classes = [0]  # Customer class ID
    
    def detect(self, frame):
        """
        Detect customer in a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            sv.Detections: Supervision detections object
        """
        # Run inference
        result = self.model(frame)[0]
        
        # Convert to supervision format
        return sv.Detections.from_ultralytics(result) 