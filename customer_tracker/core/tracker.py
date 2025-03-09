"""
Customer tracking module using ByteTrack.
"""
import supervision as sv
from collections import defaultdict, deque
from datetime import datetime
import numpy as np

class CustomerTracker:
    def __init__(self):
        self.tracks = defaultdict(lambda: {"first_seen": None, "last_seen": None, "positions": []})
        self.total_count = 0
        self.current_count = 0
        self.start_time = datetime.now()

    def update(self, detections):
        current_time = datetime.now()
        current_ids = set()

        for tracker_id, point in zip(detections.tracker_id, detections.get_anchors_coordinates(anchor=sv.Position.CENTER)):
            current_ids.add(tracker_id)
            
            if tracker_id not in self.tracks or self.tracks[tracker_id]["first_seen"] is None:
                self.tracks[tracker_id]["first_seen"] = current_time
                self.total_count += 1
            
            self.tracks[tracker_id]["last_seen"] = current_time
            self.tracks[tracker_id]["positions"].append(point)

        # Update current count
        self.current_count = len(current_ids)

        # Clean up old tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                if (current_time - self.tracks[track_id]["last_seen"]).seconds > 5:
                    del self.tracks[track_id]

    def get_stats(self):
        return {
            "total_count": self.total_count,
            "current_count": self.current_count,
            "uptime": str(datetime.now() - self.start_time).split('.')[0]
        }

class PathPredictor:
    def __init__(self, history_size=30, future_steps=10):
        self.history_size = history_size
        self.future_steps = future_steps
        
    def predict_paths(self, tracks):
        predictions = {}
        for track_id, track_info in tracks.items():
            positions = track_info["positions"]
            if len(positions) >= 2:
                # Get recent positions
                recent_positions = positions[-min(self.history_size, len(positions)):]
                
                if len(recent_positions) >= 2:
                    # Calculate average velocity
                    velocities = np.diff(recent_positions, axis=0)
                    avg_velocity = np.mean(velocities, axis=0)
                    
                    # Predict future positions
                    last_pos = np.array(recent_positions[-1])
                    future_positions = [last_pos + avg_velocity * (i + 1) for i in range(self.future_steps)]
                    predictions[track_id] = future_positions
        
        return predictions

class TimeAnalytics:
    def __init__(self, interval_minutes=5):
        self.interval_minutes = interval_minutes
        self.time_bins = defaultdict(int)
        self.current_interval = None
        
    def update(self, current_time, count):
        # Convert Unix timestamp to datetime if needed
        if isinstance(current_time, (int, float)):
            current_time = datetime.fromtimestamp(current_time)
            
        interval = current_time.replace(
            minute=(current_time.minute // self.interval_minutes) * self.interval_minutes,
            second=0,
            microsecond=0
        )
        
        if self.current_interval != interval:
            self.current_interval = interval
            self.time_bins[interval] = count
    
    def get_stats(self):
        if not self.time_bins:
            return "No data"
        
        peak_time = max(self.time_bins.items(), key=lambda x: x[1])
        return f"Peak: {peak_time[1]} customer at {peak_time[0].strftime('%H:%M')}" 