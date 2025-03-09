import cv2
import numpy as np
import json
from datetime import datetime
import os  # Add this import at the top

###########################################
# Configuration Variables
###########################################

# Input/Output Settings
VIDEO_PATH = "data/customer-walking.mp4"
SCALE_FACTOR = 1.0  # Scale factor for resizing input frame (e.g., 0.5 for half size)
SAVE_JSON = False   # Whether to keep the JSON file after processing

# Grid Settings
INITIAL_GRID_SIZE = 16     # Initial size of grid cells in pixels
MIN_GRID_SIZE = 4         # Minimum grid size when using '-' key
MAX_GRID_SIZE = 64        # Maximum grid size when using '+' key
GRID_COLOR = (128, 128, 128)  # Grid line color (BGR)

# Zoom Settings
INITIAL_ZOOM = 1.0        # Initial zoom level
MIN_ZOOM = 0.1           # Minimum zoom level
MAX_ZOOM = 10.0          # Maximum zoom level
ZOOM_STEP = 1.2          # Zoom in/out multiplier (larger = faster zoom)

# Visualization Settings
POINT_SIZE = 5           # Size of points in pixels
LINE_THICKNESS = 2       # Thickness of lines connecting points
FONT_SIZE = 0.6         # Base font size for text
FONT_THICKNESS = 2      # Thickness of font outline
ZONE_OPACITY = 0.2      # Opacity of filled zones (0-1)

# Zone Colors (BGR format)
ACTIVE_POINT_COLOR = (0, 0, 255)    # Color of points in active zone
ACTIVE_LINE_COLOR = (255, 0, 0)     # Color of lines in active zone
ACTIVE_FILL_COLOR = (0, 255, 0)     # Color of fill in active zone
SAVED_ZONE_COLORS = [               # Colors for saved zones
    (0, 255, 0),    # Green
    (255, 165, 0),  # Orange
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Dark Blue
    (0, 128, 0)     # Dark Green
]

# Text Settings
TEXT_MARGIN = 10         # Margin for text from edges
TEXT_LINE_HEIGHT = 25    # Vertical space between lines of text
TEXT_START_Y = 30       # Starting Y position for instructions

###########################################
# Rest of the code
###########################################

def draw_grid(img, grid_size=INITIAL_GRID_SIZE):
    h, w = img.shape[:2]
    # Draw vertical lines
    for x in range(0, w, grid_size):
        cv2.line(img, (x, 0), (x, h), GRID_COLOR, 1)
    # Draw horizontal lines
    for y in range(0, h, grid_size):
        cv2.line(img, (0, y), (w, y), GRID_COLOR, 1)

def draw_instructions(img):
    instructions = [
        "Instructions:",
        "- Left click: Add point",
        "- Right click: Remove last point",
        "- Mouse wheel: Zoom in/out",
        "- Middle mouse + drag: Pan when zoomed",
        "- 'g': Toggle grid",
        "- 'r': Reset zoom",
        "- 's': Save current zone",
        "- 'n': Start new zone",
        "- 'c': Clear current zone",
        "- 'd': Delete last saved zone",
        "- 'e': Edit last saved zone",
        "- '+/-': Adjust grid size",
        "- 'q': Quit and save all"
    ]
    y = TEXT_START_Y
    for text in instructions:
        cv2.putText(img, text, (TEXT_MARGIN, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    FONT_SIZE, (255, 255, 255), FONT_THICKNESS)
        cv2.putText(img, text, (TEXT_MARGIN, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    FONT_SIZE, (0, 0, 0), 1)
        y += TEXT_LINE_HEIGHT

def get_zoomed_coordinates(x, y):
    # Convert window coordinates to original image coordinates
    original_x = int((x + pan_offset[0]) / zoom_scale)
    original_y = int((y + pan_offset[1]) / zoom_scale)
    return original_x, original_y

def update_frame():
    global frame_copy, zoom_scale, pan_offset
    
    # Create zoomed frame
    h, w = base_frame.shape[:2]
    M = np.float32([[zoom_scale, 0, -pan_offset[0]],
                    [0, zoom_scale, -pan_offset[1]]])
    frame_copy = cv2.warpAffine(base_frame, M, (w, h))
    
    # Draw grid if enabled
    if show_grid:
        draw_grid(frame_copy, grid_size)
    
    # Draw saved zones first
    for i, zone in enumerate(saved_zones):
        transformed_points = []
        color = SAVED_ZONE_COLORS[i % len(SAVED_ZONE_COLORS)]
        
        for pt in zone:
            zoomed_x = int(pt[0] * zoom_scale - pan_offset[0])
            zoomed_y = int(pt[1] * zoom_scale - pan_offset[1])
            transformed_points.append([zoomed_x, zoomed_y])
            cv2.circle(frame_copy, (zoomed_x, zoomed_y), POINT_SIZE, color, -1)
        
        if len(transformed_points) >= 2:
            for j in range(len(transformed_points)):
                cv2.line(frame_copy, 
                        tuple(transformed_points[j]),
                        tuple(transformed_points[(j + 1) % len(transformed_points)]),
                        color, LINE_THICKNESS)
        
        if len(transformed_points) >= 3:
            overlay = frame_copy.copy()
            cv2.fillPoly(overlay, [np.array(transformed_points)], color)
            cv2.addWeighted(overlay, ZONE_OPACITY, frame_copy, 1 - ZONE_OPACITY, 0, frame_copy)
            
            # Add zone number
            center = np.mean(transformed_points, axis=0).astype(int)
            cv2.putText(frame_copy, f"Zone {i+1}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE + 0.2, (255, 255, 255), FONT_THICKNESS)
            cv2.putText(frame_copy, f"Zone {i+1}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE + 0.2, (0, 0, 0), 1)
    
    # Draw current active zone
    if len(points) > 0:
        transformed_points = []
        for pt in points:
            zoomed_x = int(pt[0] * zoom_scale - pan_offset[0])
            zoomed_y = int(pt[1] * zoom_scale - pan_offset[1])
            transformed_points.append([zoomed_x, zoomed_y])
            cv2.circle(frame_copy, (zoomed_x, zoomed_y), POINT_SIZE, ACTIVE_POINT_COLOR, -1)
            
            # Show point numbers
            cv2.putText(frame_copy, str(len(transformed_points)), 
                       (zoomed_x + TEXT_MARGIN, zoomed_y + TEXT_MARGIN),
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE - 0.1, (255, 255, 255), FONT_THICKNESS)
            cv2.putText(frame_copy, str(len(transformed_points)), 
                       (zoomed_x + TEXT_MARGIN, zoomed_y + TEXT_MARGIN),
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE - 0.1, (0, 0, 0), 1)
        
        if len(transformed_points) >= 2:
            for i in range(len(transformed_points)):
                cv2.line(frame_copy, 
                        tuple(transformed_points[i]),
                        tuple(transformed_points[(i + 1) % len(transformed_points)]),
                        ACTIVE_LINE_COLOR, LINE_THICKNESS)
        
        if len(transformed_points) >= 3:
            overlay = frame_copy.copy()
            cv2.fillPoly(overlay, [np.array(transformed_points)], ACTIVE_FILL_COLOR)
            cv2.addWeighted(overlay, ZONE_OPACITY, frame_copy, 1 - ZONE_OPACITY, 0, frame_copy)
    
    # Show zone count and grid size
    status_text = [
        f"Zones: {len(saved_zones)} + {1 if points else 0} active",
        f"Grid size: {grid_size}",
        f"Zoom: {zoom_scale:.1f}x"
    ]
    
    y_offset = frame_copy.shape[0] - TEXT_MARGIN
    for text in reversed(status_text):
        cv2.putText(frame_copy, text, (TEXT_MARGIN, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), FONT_THICKNESS)
        cv2.putText(frame_copy, text, (TEXT_MARGIN, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 0), 1)
        y_offset -= TEXT_LINE_HEIGHT
    
    draw_instructions(frame_copy)

def save_current_zone():
    global points, saved_zones
    if len(points) >= 3:
        saved_zones.append(points.copy())
        points.clear()
        print(f"Zone {len(saved_zones)} saved")
        update_frame()
    else:
        print("Need at least 3 points to save a zone")

def edit_last_zone():
    global points, saved_zones
    if saved_zones:
        points = saved_zones.pop().copy()
        print("Editing last saved zone")
        update_frame()
    else:
        print("No saved zones to edit")

def delete_last_zone():
    global saved_zones
    if saved_zones:
        saved_zones.pop()
        print("Last zone deleted")
        update_frame()
    else:
        print("No saved zones to delete")

def select_points(event, x, y, flags, param):
    global points, frame_copy, show_coordinates, zoom_scale, pan_offset, is_panning, pan_start
    
    # Handle zooming with mouse wheel
    if event == cv2.EVENT_MOUSEWHEEL:
        old_x, old_y = get_zoomed_coordinates(x, y)
        
        if flags > 0:  # Zoom in
            zoom_scale = min(MAX_ZOOM, zoom_scale * ZOOM_STEP)
        else:  # Zoom out
            zoom_scale = max(MIN_ZOOM, zoom_scale / ZOOM_STEP)
        
        # Adjust pan offset to keep the mouse point fixed
        new_x = int(x / zoom_scale)
        new_y = int(y / zoom_scale)
        pan_offset[0] += int((new_x - old_x) * zoom_scale)
        pan_offset[1] += int((new_y - old_y) * zoom_scale)
        
        update_frame()
    
    # Handle panning with middle mouse button
    elif event == cv2.EVENT_MBUTTONDOWN:
        is_panning = True
        pan_start = (x, y)
    
    elif event == cv2.EVENT_MBUTTONUP:
        is_panning = False
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_panning:
            dx = x - pan_start[0]
            dy = y - pan_start[1]
            pan_offset[0] -= dx
            pan_offset[1] -= dy
            pan_start = (x, y)
            update_frame()
        
        # Show coordinates
        original_x, original_y = get_zoomed_coordinates(x, y)
        coord_text = f"({original_x}, {original_y})"
        frame_temp = frame_copy.copy()
        cv2.putText(frame_temp, coord_text, (x + TEXT_MARGIN, y - TEXT_MARGIN),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), FONT_THICKNESS)
        cv2.putText(frame_temp, coord_text, (x + TEXT_MARGIN, y - TEXT_MARGIN),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 0), 1)
        cv2.imshow("Frame", frame_temp)
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        original_x, original_y = get_zoomed_coordinates(x, y)
        points.append([original_x, original_y])
        print(f"Point added: [{original_x}, {original_y}]")
        update_frame()
    
    elif event == cv2.EVENT_RBUTTONDOWN and len(points) > 0:
        points.pop()
        print("Last point removed")
        update_frame()

# Main execution
if __name__ == "__main__":
    # Load the first frame of the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read video frame.")

    # Resize the frame to fit your screen
    frame_resized = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

    # Create copies of the frame
    base_frame = frame_resized.copy()
    frame_copy = base_frame.copy()

    # Initialize variables
    points = []
    saved_zones = []
    show_coordinates = None
    show_grid = True
    grid_size = INITIAL_GRID_SIZE
    zoom_scale = INITIAL_ZOOM
    pan_offset = [0, 0]
    is_panning = False
    pan_start = None

    # Display the frame and let the user click points
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", select_points)

    while True:
        cv2.imshow("Frame", frame_copy)
        key = cv2.waitKey(1) & 0xFF
        
        # Toggle grid with 'g' key
        if key == ord("g"):
            show_grid = not show_grid
            update_frame()
        
        # Reset zoom and pan with 'r' key
        elif key == ord("r"):
            zoom_scale = INITIAL_ZOOM
            pan_offset = [0, 0]
            update_frame()
        
        # Save current zone with 's' key
        elif key == ord("s"):
            save_current_zone()
        
        # Start new zone with 'n' key
        elif key == ord("n"):
            if len(points) >= 3:
                save_current_zone()
            points = []
            update_frame()
        
        # Clear current zone with 'c' key
        elif key == ord("c"):
            points = []
            update_frame()
        
        # Delete last saved zone with 'd' key
        elif key == ord("d"):
            delete_last_zone()
        
        # Edit last saved zone with 'e' key
        elif key == ord("e"):
            edit_last_zone()
        
        # Adjust grid size with +/- keys
        elif key == ord("+") or key == ord("="):
            grid_size = min(MAX_GRID_SIZE, grid_size * 2)
            update_frame()
        elif key == ord("-") or key == ord("_"):
            grid_size = max(MIN_GRID_SIZE, grid_size // 2)
            update_frame()
        
        # Exit the loop if 'q' is pressed
        elif key == ord("q"):
            if len(points) >= 3:
                save_current_zone()
            break

    # Clean up OpenCV windows
    cv2.destroyAllWindows()

    # Scale all zones back to the original resolution
    zones_original = [[[int(x / SCALE_FACTOR), int(y / SCALE_FACTOR)] for x, y in zone] for zone in saved_zones]

    # Save zones to a temporary JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"zones_{timestamp}.json"
    
    try:
        # Save to JSON file
        with open(output_file, "w") as f:
            json.dump({"zones": zones_original}, f, indent=2)
        
        # Print the zones
        print(f"\nProcessed {len(zones_original)} zones:")
        for i, zone in enumerate(zones_original):
            print(f"\nZone {i + 1}:", zone)
        
        # Remove the JSON file unless SAVE_JSON is True
        if not SAVE_JSON and os.path.exists(output_file):
            os.remove(output_file)
            print(f"\nTemporary file {output_file} has been cleaned up")
        elif SAVE_JSON:
            print(f"\nJSON file saved as: {output_file}")
    
    except Exception as e:
        print(f"Error processing zones: {str(e)}")
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Cleaned up temporary file {output_file} due to error")

