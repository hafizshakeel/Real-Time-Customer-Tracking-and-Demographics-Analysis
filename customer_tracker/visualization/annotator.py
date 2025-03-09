"""
Visualization module for annotating video frames.
"""
import supervision as sv
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

class FrameAnnotator:
    """
    Handles annotation of video frames with detections, tracking, and counting information.
    """
    def __init__(self, video_info):
        """
        Initialize the frame annotator.
        
        Args:
            video_info: Supervision VideoInfo object
        """
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
        
        self.box_annotator = sv.BoxAnnotator(
            thickness=thickness,
            color_lookup=sv.ColorLookup.TRACK
        )

        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_padding=5
        )

        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=30,
            position=sv.Position.CENTER
        )
        
        self.dot_annotator = sv.DotAnnotator(
            radius=5,
            color=sv.Color.BLUE,
            position=sv.Position.CENTER
        )
        
        self.line_counter_annotator = sv.LineZoneAnnotator(
            thickness=thickness,
            text_thickness=thickness,
            text_scale=text_scale
        )
    
    def annotate_frame(self, frame, detections, labels=None, line_counter=None):
        """
        Annotate a frame with detections and tracking information.
        
        Args:
            frame: Input frame
            detections: Supervision detections object
            labels: Optional labels for detections
            line_counter: Optional line counter
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw line counter if provided
        if line_counter:
            annotated_frame = self.line_counter_annotator.annotate(
                frame=annotated_frame,
                line_counter=line_counter
            )
        
        # Draw traces
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        # Draw bounding boxes with labels
        if labels:
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame, 
                    detections=detections, 
                    labels=labels
                )
        
        # Draw center dots
        annotated_frame = self.dot_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        return annotated_frame
    
    def resize_frame(self, frame, scale):
        """
        Resize a frame by a scale factor.
        
        Args:
            frame: Input frame
            scale: Scale factor
            
        Returns:
            np.ndarray: Resized frame
        """
        height, width = frame.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))

class ModernAnnotator:
    """
    Modern style annotator with cleaner visuals.
    """
    def __init__(self, thickness=1, text_scale=0.4, text_thickness=1):
        self.thickness = thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        
        self.halo_annotator = sv.HaloAnnotator(
            color=sv.Color.RED,
            opacity=0.5,
            kernel_size=15,
            color_lookup=sv.ColorLookup.TRACK
        )
        
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=5,
            color=sv.Color.BLACK
        )
        
    def annotate(self, scene: np.ndarray, detections, labels: list = None) -> np.ndarray:
        """
        Annotate a scene with modern style annotations.
        
        Args:
            scene: Input scene
            detections: Supervision detections
            labels: Optional labels for detections
            
        Returns:
            np.ndarray: Annotated scene
        """
        # Add halos
        annotated_scene = self.halo_annotator.annotate(scene=scene.copy(), detections=detections)
        
        # Add labels if provided
        if labels:
            annotated_scene = self.label_annotator.annotate(
                scene=annotated_scene,
                detections=detections,
                labels=labels
            )
        
        return annotated_scene

class HeatmapGenerator:
    """
    Generates and visualizes movement heatmaps.
    """
    def __init__(self, frame_shape, decay_factor=0.99):
        self.heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        self.decay_factor = decay_factor
        
    def update(self, points):
        # Apply decay to existing heatmap
        self.heatmap *= self.decay_factor
        
        # Add new points
        for point in points:
            if np.all(point >= 0):  # Filter invalid points
                x, y = int(point[0]), int(point[1])
                if 0 <= y < self.heatmap.shape[0] and 0 <= x < self.heatmap.shape[1]:
                    self.heatmap[y, x] += 1.0
        
        # Apply Gaussian blur
        self.heatmap = gaussian_filter(self.heatmap, sigma=5)
        
    def get_visualization(self, alpha=0.5):
        # Normalize heatmap
        if np.max(self.heatmap) > 0:
            normalized = self.heatmap / np.max(self.heatmap)
        else:
            normalized = self.heatmap
            
        # Convert to color heatmap
        heatmap_vis = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return heatmap_vis, alpha

class DemographicsAnnotator:
    """
    Annotates frames with demographics information.
    """
    def __init__(self, text_scale=0.8, text_thickness=2, padding=30):
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.padding = padding
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_color = (255, 255, 255)
        
    def add_stats(self, frame, stats_list, position=(10, 30)):
        """
        Add statistics text to the frame.
        
        Args:
            frame: Input frame
            stats_list: List of statistics strings
            position: Top-left position for the first text line
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        x, y = position
        
        # Calculate overlay dimensions
        max_text_width = 0
        for text in stats_list:
            (text_width, text_height), _ = cv2.getTextSize(
                text, self.font, self.text_scale, self.text_thickness
            )
            max_text_width = max(max_text_width, text_width)
        
        overlay_width = max_text_width + 20  # Add padding
        overlay_height = len(stats_list) * self.padding + 10
        
        # Create overlay
        overlay = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
        
        # Calculate safe ROI coordinates
        start_y = max(0, y-20)
        end_y = min(frame.shape[0], y+overlay_height-20)
        start_x = max(0, x-10)
        end_x = min(frame.shape[1], x+overlay_width-10)
        
        # Adjust overlay size to match ROI
        overlay_height = end_y - start_y
        overlay_width = end_x - start_x
        
        if overlay_height > 0 and overlay_width > 0:
            # Create adjusted overlay
            overlay = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
            
            # Add semi-transparent overlay to frame
            alpha = 0.7
            roi = annotated_frame[start_y:end_y, start_x:end_x].copy()
            cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)
            annotated_frame[start_y:end_y, start_x:end_x] = roi
        
        # Add text on top of overlay
        for i, text in enumerate(stats_list):
            cv2.putText(
                annotated_frame,
                text,
                (x, y + i * self.padding),
                self.font,
                self.text_scale,
                self.text_color,
                self.text_thickness,
                cv2.LINE_AA
            )
            
        return annotated_frame
        
    def create_demographics_overlay(self, frame, gender_counts, age_groups, width=300, padding=20):
        """
        Create a semi-transparent overlay with demographics stats.
        
        Args:
            frame: Input frame
            gender_counts: Dictionary of gender counts
            age_groups: Dictionary of age group counts
            width: Width of the overlay
            padding: Padding inside the overlay
            
        Returns:
            np.ndarray: Frame with overlay
        """
        height = 200  # Fixed height
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate total counts
        total = sum(gender_counts.values())
        
        # Draw header
        cv2.putText(overlay, "Demographics", (padding, padding + 15), 
                   self.font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                   
        # Draw gender distribution
        bar_y = padding + 40
        bar_width = width - (padding * 2)
        bar_height = 20
        
        # Male bar
        male_percent = int((gender_counts["Male"] / total) * 100) if total > 0 else 0
        male_width = int((male_percent / 100) * bar_width) if male_percent > 0 else 0
        cv2.rectangle(overlay, (padding, bar_y), (padding + male_width, bar_y + bar_height), 
                      (0, 0, 255), -1)
        cv2.putText(overlay, f"Male: {male_percent}%", (padding + 5, bar_y + 15), 
                   self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                   
        # Female bar
        bar_y += bar_height + 10
        female_percent = int((gender_counts["Female"] / total) * 100) if total > 0 else 0
        female_width = int((female_percent / 100) * bar_width) if female_percent > 0 else 0
        cv2.rectangle(overlay, (padding, bar_y), (padding + female_width, bar_y + bar_height), 
                      (255, 0, 0), -1)
        cv2.putText(overlay, f"Female: {female_percent}%", (padding + 5, bar_y + 15), 
                   self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                   
        # Draw age distribution
        bar_y += bar_height + 20
        cv2.putText(overlay, "Age Groups", (padding, bar_y), 
                   self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                   
        bar_y += 20
        bar_height = 15
        for age_group, count in age_groups.items():
            percent = int((count / total) * 100) if total > 0 else 0
            age_width = int((percent / 100) * bar_width) if percent > 0 else 0
            cv2.rectangle(overlay, (padding, bar_y), (padding + age_width, bar_y + bar_height), 
                         (0, 255, 0), -1)
            cv2.putText(overlay, f"{age_group}: {percent}%", (padding + 5, bar_y + 12), 
                       self.font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            bar_y += bar_height + 5
            
        # Add overlay to frame
        x = frame.shape[1] - width - padding
        y = padding
        
        # Create mask for transparent overlay
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (width, height), 255, -1)
        
        # Add semi-transparent overlay
        alpha = 0.7
        roi = frame[y:y+height, x:x+width].copy()
        cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)
        frame[y:y+height, x:x+width] = roi
        
        return frame 