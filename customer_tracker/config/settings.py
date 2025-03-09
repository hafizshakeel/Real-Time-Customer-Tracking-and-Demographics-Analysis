"""
Configuration settings for customer tracking.
"""
import numpy as np

# Line zone for counting (Vertical line)
DEFAULT_LINE_START = np.array([433, 33])
DEFAULT_LINE_END = np.array([432, 673])

# Model settings
MODEL_PATH = "yolov8n.pt"
PERSON_CLASS_ID = 0

# Tracking settings
TRACK_CLEANUP_SECONDS = 5
PATH_HISTORY_SIZE = 30
PATH_PREDICTION_STEPS = 10

# Analytics settings
TIME_INTERVAL_MINUTES = 5

# Visualization settings
HEATMAP_DECAY = 0.99
HEATMAP_ALPHA = 0.5
TRACE_LENGTH_SECONDS = 2

# Demographics settings
DEMOGRAPHICS_DET_SIZE = (640, 640)
AGE_GROUPS = {"0-18": 0, "19-35": 0, 
              "36-60": 0, "60+": 0} 