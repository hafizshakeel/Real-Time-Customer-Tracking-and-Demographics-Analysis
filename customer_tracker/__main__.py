"""
Main entry point for the customer tracking application.
"""
import cv2
import time
import numpy as np
import supervision as sv
from collections import deque
from datetime import datetime

from.core.detector import CustomerDetector
from.core.tracker import CustomerTracker
from.core.counter import OccupancyCounter
from.core.demographics import DemographicsAnalyzer
from.utils.cli import parse_arguments
from.utils.video import get_video_info, get_frame_generator, setup_display_window, resize_frame_for_display, initialize_video_writer
from.visualization.annotator import ModernAnnotator, HeatmapGenerator, DemographicsAnnotator
from.config.settings import (
    DEFAULT_LINE_START,
    DEFAULT_LINE_END,
    MODEL_PATH,
    PERSON_CLASS_ID,
    HEATMAP_DECAY,
    TRACE_LENGTH_SECONDS,
    AGE_GROUPS
)

class FPSCounter:
    """Tracks and calculates frames per second."""
    def __init__(self, avg_frames=30):
        self.fps_buffer = deque(maxlen=avg_frames)
        self.last_time = time.time()
    
    def update(self):
        current_time = time.time()
        self.fps_buffer.append(1 / (current_time - self.last_time))
        self.last_time = current_time
    
    def get_fps(self):
        return sum(self.fps_buffer) / len(self.fps_buffer) if self.fps_buffer else 0

def main():
    """Main execution function for customer tracking."""
    # Parse command-line arguments
    args = parse_arguments()
    start_time = datetime.now()

    # Get video information
    width, height, fps, _ = get_video_info(args.source_video_path)
    
    # Create VideoInfo with correct format (width, height, fps, [total_frames])
    video_info = sv.VideoInfo(
        width=width,
        height=height,
        fps=fps
    )

    # Initialize detector and tracker
    detector = CustomerDetector(model_path=MODEL_PATH)
    byte_track = sv.ByteTrack(frame_rate=fps)
    fps_counter = FPSCounter()
    
    # Initialize analytics modules
    customer_tracker = CustomerTracker()
    occupancy_counter = OccupancyCounter()
    
    # Initialize line counter for entrance/exit counting
    line_counter = sv.LineZone(
        start=sv.Point(DEFAULT_LINE_START[0], DEFAULT_LINE_START[1]),
        end=sv.Point(DEFAULT_LINE_END[0], DEFAULT_LINE_END[1]),
        triggering_anchors=[sv.Position.CENTER]
    )
    
    # Initialize demographics analyzer if requested
    demographics_analyzer = None
    if args.demographics:
        demographics_analyzer = DemographicsAnalyzer()
    
    # Initialize visualization components
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(width, height))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(width, height))

    # Initialize annotators
    modern_annotator = ModernAnnotator(
        thickness=thickness,
        text_scale=0.6,
        text_thickness=1
    )
    
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=fps * TRACE_LENGTH_SECONDS,
        position=sv.Position.CENTER,
        color=sv.Color.GREEN
    )
    
    line_counter_annotator = sv.LineZoneAnnotator(
        thickness=thickness,
        text_thickness=thickness,
        text_scale=text_scale,
        color=sv.Color.YELLOW
    )
    
    demographics_annotator = DemographicsAnnotator(
        text_scale=0.6,
        text_thickness=1,
        padding=20
    )

    # Initialize heatmap if requested
    heatmap_gen = None
    if args.show_heatmap:
        heatmap_gen = HeatmapGenerator(
            frame_shape=(height, width),
            decay_factor=HEATMAP_DECAY
        )

    # Initialize video writer if requested
    video_writer = None
    if args.save_video:
        video_writer = initialize_video_writer(
            args.output_video_path,
            width,
            height,
            fps
        )
        
    # Set up display window if showing video
    if args.display:
        setup_display_window(
            "Customer Tracking",
            args.display_width,
            args.display_height
        )
    
    # Create sets to track IDs for demographics
    tracked_ids_for_demographics = set()
    
    # Process video frames
    frame_generator = get_frame_generator(args.source_video_path)
    print("Processing video... Press 'q' to quit")
    
    for frame in frame_generator:
        # Detect customer
        detections = detector.detect(frame)
        
        # Track detections
        detections = byte_track.update_with_detections(detections=detections)
        
        # Get center points of detections
        points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
        
        # Update line counts
        prev_in_count = line_counter.in_count
        
        # Store the IDs that have already been analyzed for demographics
        already_crossed = set()
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is not None and tracker_id in tracked_ids_for_demographics:
                already_crossed.add(tracker_id)
                
        # Trigger the line counter with current detections
        line_counter.trigger(detections=detections)
        
        # Find new IDs that have crossed the line
        new_entries = []
        if line_counter.in_count > prev_in_count:
            # Look through current detections for new entries
            for i, tracker_id in enumerate(detections.tracker_id):
                # Only consider detection if:
                # 1. It has a tracker ID
                # 2. It hasn't been counted for demographics yet
                # 3. It's not already in the already_crossed set
                # 4. The in_count increased this frame (someone entered)
                if (tracker_id is not None and 
                    tracker_id not in tracked_ids_for_demographics and
                    tracker_id not in already_crossed):
                    
                    # Check if this ID is near the line (this is to more precisely determine who crossed)
                    bbox = detections.xyxy[i]
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Calculate distance to line - simplified check for demo purposes
                    # In a production environment, you might want a more sophisticated crossing detection
                    line_start = np.array([DEFAULT_LINE_START[0], DEFAULT_LINE_START[1]])
                    line_end = np.array([DEFAULT_LINE_END[0], DEFAULT_LINE_END[1]])
                    point = np.array([center_x, center_y])
                    
                    # Only add if we haven't already counted too many customer
                    # This prevents counting more customer than have actually entered
                    if len(new_entries) < (line_counter.in_count - prev_in_count):
                        new_entries.append(tracker_id)
                        tracked_ids_for_demographics.add(tracker_id)
        
        # Update analytics components
        customer_tracker.update(detections)
        occupancy_counter.update(detections)
        
        # Update heatmap if enabled
        if args.show_heatmap and heatmap_gen:
            heatmap_gen.update(points)
        
        # Analyze demographics if enabled
        demographics_info = None
        if args.demographics and demographics_analyzer:
            demographics_info = demographics_analyzer.analyze_frame(frame, detections)
            demographics_analyzer.update_stats(new_entries, demographics_info, detections)
        
        # Create labels
        labels = []
        for i, (tracker_id, confidence) in enumerate(zip(detections.tracker_id, detections.confidence)):
            label = f"#{tracker_id} {confidence:.2f}"
            
            # Add demographics info if available
            if args.demographics and demographics_info and i < len(demographics_info) and demographics_info[i]:
                demo = demographics_info[i]
                label += f" {demo['gender']}, {demo['age']}y"
            
            labels.append(label)
        
        # Annotate frame
        annotated_frame = frame.copy()
        
        # Add visualizations based on enabled features
        if args.show_heatmap and heatmap_gen:
            heatmap, alpha = heatmap_gen.get_visualization()
            annotated_frame = cv2.addWeighted(annotated_frame, 1-alpha, heatmap, alpha, 0)
        
        # Draw line counter
        annotated_frame = line_counter_annotator.annotate(frame=annotated_frame, line_counter=line_counter)
        
        # Add traces
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        
        # Add labels and halos
        annotated_frame = modern_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Create stats text
        occupancy_stats = occupancy_counter.get_stats()
        
        # Basic stats without demographics
        stats_text = [
            f"Entered: {line_counter.in_count}",
            f"Exited: {line_counter.out_count}",
            f"Total Tracked: {occupancy_stats['total_tracked']}",
            f"Current Count: {occupancy_stats['current_count']}",
            f"Uptime: {str(datetime.now() - start_time).split('.')[0]}"
        ]
        
        # Add stats text with black background
        annotated_frame = demographics_annotator.add_stats(annotated_frame, stats_text)
        
        # Add demographics statistics if enabled
        if args.demographics and demographics_analyzer:
            demo_stats = demographics_analyzer.get_stats()
            gender_counts = demo_stats["gender_counts"]
            age_groups = demo_stats["age_groups"]
            
            # Create separate demographics stats text
            demo_stats_text = [
                f"Gender: M={gender_counts['Male']}, F={gender_counts['Female']}",
                f"Age: 0-18={age_groups['0-18']}, 19-35={age_groups['19-35']}",
                f"     36-60={age_groups['36-60']}, 60+={age_groups['60+']}"
            ]
            
            # Add demographics stats text below the basic stats
            y_offset = len(stats_text) * demographics_annotator.padding + 10
            annotated_frame = demographics_annotator.add_stats(
                annotated_frame, 
                demo_stats_text, 
                position=(10, 30 + y_offset)
            )
            
            # Add demographics visualization overlay - always show when demographics are enabled
            annotated_frame = demographics_annotator.create_demographics_overlay(
                annotated_frame, gender_counts, age_groups
            )
        
        # Update FPS counter
        fps_counter.update()

        # Write frame to output video if requested
        if args.save_video and video_writer:
            video_writer.write(annotated_frame)

        # Display frame if requested
        if args.display:
            resized_frame = resize_frame_for_display(
                annotated_frame, 
                args.display_width, 
                args.display_height
            )
            cv2.imshow("Customer Tracking", resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Print final statistics
    print("\nProcessing complete!")
    print(f"Total customer tracked: {occupancy_stats['total_tracked']}")
    print(f"Final customer count: {occupancy_stats['current_count']}")
    print(f"Entered: {line_counter.in_count}, Exited: {line_counter.out_count}")
    
    # Print demographics summary if enabled
    if args.demographics and demographics_analyzer:
        demographics_analyzer.print_summary(line_counter.in_count)
    
    # Clean up resources
    if args.save_video and video_writer:
        video_writer.release()
        print(f"Output saved to: {args.output_video_path}")
        
    if args.display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 