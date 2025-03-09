"""
Demographics analysis module using InsightFace.
"""
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from collections import defaultdict

class DemographicsAnalyzer:
    """
    Handles demographics analysis (age and gender) using InsightFace.
    """
    def __init__(self, det_size=(640, 640), providers=None):
        """
        Initialize the demographics analyzer.
        
        Args:
            det_size (tuple): Detection size (width, height)
            providers (list): List of providers to use (default: CPU)
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
            
        self.face_analyzer = FaceAnalysis(name="buffalo_l", providers=providers)
        self.face_analyzer.prepare(ctx_id=0, det_size=det_size)
        
        # Store demographics stats
        self.gender_counts = {"Male": 0, "Female": 0}
        self.age_groups = {"0-18": 0, "19-35": 0, "36-60": 0, "60+": 0}
        self.tracked_ids = set()
        
    def analyze_frame(self, frame, detections):
        """
        Analyze demographics for customer in the frame.
        
        Args:
            frame: The input frame
            detections: Supervision detections object
            
        Returns:
            list: List of demographic info (age, gender) for each detection
        """
        demographics = []
        
        # For each detection, crop the customer and analyze faces
        for i, (xyxy, confidence, class_id, tracker_id) in enumerate(
            zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id)
        ):
            if tracker_id is None:
                demographics.append(None)
                continue
                
            # Crop the customer from the frame
            x1, y1, x2, y2 = map(int, xyxy)
            person_crop = frame[y1:y2, x1:x2]
            
            # Skip if crop is too small
            if person_crop.size == 0 or person_crop.shape[0] < 20 or person_crop.shape[1] < 20:
                demographics.append(None)
                continue
            
            # Analyze faces in the cropped region
            faces = self.face_analyzer.get(person_crop)
            
            if len(faces) > 0:
                # Use the first detected face for demographics
                face = faces[0]
                age = int(face.age)
                gender = "Male" if face.gender == 1 else "Female"
                demographics.append({"age": age, "gender": gender})
            else:
                demographics.append(None)
                
        return demographics
        
    def update_stats(self, new_ids, demographics_info, detections):
        """
        Update demographics statistics for new IDs.
        
        Args:
            new_ids (list): List of new tracker IDs
            demographics_info (list): List of demographic info for each detection
            detections: Supervision detections object
        """
        # Create a mapping of tracker_id to demographics_info index
        tracker_to_index = {tracker_id: i for i, tracker_id in enumerate(detections.tracker_id) if tracker_id is not None}
        
        for tracker_id in new_ids:
            if tracker_id in tracker_to_index:
                idx = tracker_to_index[tracker_id]
                
                if idx < len(demographics_info) and demographics_info[idx] is not None:
                    demo = demographics_info[idx]
                    
                    # Add to tracked IDs
                    self.tracked_ids.add(tracker_id)
                    
                    # Update gender counts
                    self.gender_counts[demo['gender']] += 1
                    
                    # Update age group counts
                    age = demo['age']
                    if age <= 18:
                        self.age_groups["0-18"] += 1
                    elif age <= 35:
                        self.age_groups["19-35"] += 1
                    elif age <= 60:
                        self.age_groups["36-60"] += 1
                    else:
                        self.age_groups["60+"] += 1
    
    def get_stats(self):
        """
        Get current demographics statistics.
        
        Returns:
            dict: Dictionary containing demographics statistics
        """
        return {
            "gender_counts": self.gender_counts,
            "age_groups": self.age_groups,
            "total_analyzed": len(self.tracked_ids)
        }
        
    def print_summary(self, total_entered):
        """
        Print a summary of demographics statistics.
        
        Args:
            total_entered (int): Total number of customer who entered
        """
        total_analyzed = len(self.tracked_ids)
        
        # Ensure we don't report analyzing more customer than entered
        if total_analyzed > total_entered and total_entered > 0:
            total_analyzed = total_entered
        
        print("\n----- DEMOGRAPHICS SUMMARY -----")
        print(f"Total customer entered: {total_entered}")
        percent_analyzed = (total_analyzed/total_entered*100) if total_entered > 0 else 0
        percent_analyzed = min(percent_analyzed, 100.0)  # Cap at 100%
        print(f"Customer with demographics analyzed: {total_analyzed} ({percent_analyzed:.1f}%)")
        
        print("\nGender Distribution:")
        if total_analyzed > 0:
            male_percent = (self.gender_counts["Male"] / total_analyzed) * 100
            female_percent = (self.gender_counts["Female"] / total_analyzed) * 100
            print(f"  Male: {self.gender_counts['Male']} ({male_percent:.1f}%)")
            print(f"  Female: {self.gender_counts['Female']} ({female_percent:.1f}%)")
        else:
            print("  No gender data collected")
            
        print("\nAge Distribution:")
        if total_analyzed > 0:
            for age_group, count in self.age_groups.items():
                percent = (count / total_analyzed) * 100
                print(f"  {age_group}: {count} ({percent:.1f}%)")
        else:
            print("  No age data collected")
        print("-------------------------------") 