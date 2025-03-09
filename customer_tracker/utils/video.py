"""
Video processing utilities.
"""
import cv2
import supervision as sv

def get_video_info(video_path):
    """
    Get information about a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        tuple: (width, height, fps, frame_count)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return (width, height, fps, frame_count)

def initialize_video_writer(output_path, width, height, fps):
    """
    Initialize a video writer for output.
    
    Args:
        output_path (str): Path to save the output video
        width (int): Width of the output video
        height (int): Height of the output video
        fps (int): Frames per second
        
    Returns:
        cv2.VideoWriter: Initialized video writer
    """
    return cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

def resize_frame_for_display(frame, max_width=1280, max_height=720):
    """
    Resize a frame to fit within max dimensions while maintaining aspect ratio.
    
    Args:
        frame: The input frame
        max_width: Maximum width for the resized frame
        max_height: Maximum height for the resized frame
    
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    # If frame is already smaller than max dimensions, return original
    if width <= max_width and height <= max_height:
        return frame
    
    # Calculate scaling factors for width and height
    width_scale = max_width / width
    height_scale = max_height / height
    
    # Use the smaller scaling factor to ensure both dimensions fit
    scale = min(width_scale, height_scale)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize the frame
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

def setup_display_window(window_name, width, height, x=50, y=50):
    """
    Set up a named window for display.
    
    Args:
        window_name (str): Name of the window
        width (int): Initial window width
        height (int): Initial window height
        x (int): Window x position
        y (int): Window y position
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, x, y)

def get_frame_generator(video_path):
    """
    Create a generator that yields frames from a video.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        generator: Generator that yields frames
    """
    return sv.get_video_frames_generator(video_path)

def get_first_frame(video_path):
    """
    Get the first frame of a video.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        np.ndarray: First frame of the video
    """
    frame_generator = get_frame_generator(video_path)
    return next(frame_generator)

def create_video_writer(output_path, frame_width, frame_height, fps):
    """
    Create a video writer for saving processed frames.
    
    Args:
        output_path (str): Path to save the output video
        frame_width (int): Width of the video frames
        frame_height (int): Height of the video frames
        fps (float): Frames per second
        
    Returns:
        cv2.VideoWriter: Video writer object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height)) 