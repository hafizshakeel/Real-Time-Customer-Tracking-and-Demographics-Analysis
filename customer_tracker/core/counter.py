"""
Customer counting module using tracking.
"""
import numpy as np
import supervision as sv
from collections import defaultdict
from datetime import datetime

class OccupancyCounter:
    """
    Handles real-time occupancy counting and statistics.
    """
    def __init__(self):
        """Initialize occupancy counter."""
        self.total_tracked = 0
        self.current_count = 0
        self.start_time = datetime.now()
        self.track_history = defaultdict(lambda: {"first_seen": None, "last_seen": None})
    
    def update(self, detections):
        """
        Update occupancy counter with new detections.
        
        Args:
            detections (sv.Detections): Detections to process
        """
        current_time = datetime.now()
        current_ids = set(detections.tracker_id)
        
        # Update tracking information
        for tracker_id in current_ids:
            if tracker_id not in self.track_history or self.track_history[tracker_id]["first_seen"] is None:
                self.track_history[tracker_id]["first_seen"] = current_time
                self.total_tracked += 1
            self.track_history[tracker_id]["last_seen"] = current_time
        
        # Update current count
        self.current_count = len(current_ids)
        
        # Clean up old tracks
        for track_id in list(self.track_history.keys()):
            if track_id not in current_ids:
                if (current_time - self.track_history[track_id]["last_seen"]).seconds > 5:
                    del self.track_history[track_id]
    
    def get_stats(self):
        """
        Get current occupancy statistics.
        
        Returns:
            dict: Dictionary containing occupancy statistics
        """
        return {
            "total_tracked": self.total_tracked,
            "current_count": self.current_count,
            "uptime": str(datetime.now() - self.start_time).split('.')[0]
        } 