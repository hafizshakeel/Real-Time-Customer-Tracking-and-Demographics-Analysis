"""
Command-line interface for customer tracking.
"""
import argparse

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Customer detection, tracking, and analytics")
    
    parser.add_argument(
        "--source_video_path", 
        required=True, 
        help="Path to the source video file", 
        type=str
    )
    
    parser.add_argument(
        "--output_video_path", 
        help="Path to save the output video (optional)", 
        type=str,
        default="output.mp4"
    )
    
    parser.add_argument(
        "--confidence_threshold", 
        type=float, 
        default=0.3,
        help="Confidence threshold for detections (0-1)"
    )
    
    parser.add_argument(
        "--show_traces", 
        action="store_true",
        help="Enable movement traces visualization"
    )
    
    parser.add_argument(
        "--show_heatmap",
        action="store_true",
        help="Show movement heatmap"
    )
    
    parser.add_argument(
        "--show_predictions",
        action="store_true",
        help="Show path predictions"
    )
    
    # Add new options for demographics and display
    parser.add_argument(
        "--demographics",
        action="store_true",
        help="Enable demographics analysis (age/gender)"
    )
    
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Disable video display while processing"
    )
    
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Disable saving the output video"
    )
    
    parser.add_argument(
        "--display_width",
        type=int,
        default=1280,
        help="Maximum width for display window"
    )
    
    parser.add_argument(
        "--display_height",
        type=int,
        default=720,
        help="Maximum height for display window"
    )
    
    args = parser.parse_args()
    
    # Set display and save_video flags based on the negative flags
    args.display = not args.no_display
    args.save_video = not args.no_save
    
    return args 