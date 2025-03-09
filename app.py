import streamlit as st
import cv2
import numpy as np
import supervision as sv
import tempfile
import os
from datetime import datetime
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
from collections import deque, defaultdict
import time

# Import from our modular implementation
from customer_tracker.core.detector import CustomerDetector
from customer_tracker.core.tracker import CustomerTracker
from customer_tracker.core.counter import OccupancyCounter
from customer_tracker.core.demographics import DemographicsAnalyzer
from customer_tracker.utils.video import resize_frame_for_display
from customer_tracker.visualization.annotator import ModernAnnotator, HeatmapGenerator, DemographicsAnnotator
from customer_tracker.config.settings import (
    DEFAULT_LINE_START,
    DEFAULT_LINE_END,
    MODEL_PATH,
    PERSON_CLASS_ID,
    HEATMAP_DECAY,
    TRACE_LENGTH_SECONDS,
    AGE_GROUPS
)

class FPSCounter:
    def __init__(self, avg_frames=30):
        self.fps_buffer = deque(maxlen=avg_frames)
        self.last_time = time.time()

    def update(self):
        current_time = time.time()
        self.fps_buffer.append(1 / (current_time - self.last_time))
        self.last_time = current_time

    def get_fps(self):
        return sum(self.fps_buffer) / len(self.fps_buffer) if self.fps_buffer else 0

# Set page configuration
st.set_page_config(
    page_title="Customer Detection, Tracking, and Demographics Analysis",
    page_icon="👥",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-bottom: 1rem;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5px 15px;
        border-radius: 8px;
        color: rgb(28, 131, 225);
        overflow-wrap: break-word;
    }

    div[data-testid="metric-container"] > div {
        color: rgb(28, 131, 225);
        font-family: "Source Sans Pro", sans-serif;
    }

    div[data-testid="stHorizontalBlock"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">Customer Detection, Tracking, and Demographics Analysis</div>', unsafe_allow_html=True)
st.markdown("""
This application uses YOLOv8, ByteTrack, and InsightFace to detect and track customer in video footage, providing real-time analytics
including count, demographics (age and gender), and movement patterns.
""")

# Input source selection
source_option = st.radio(
    "Select Input Source",
    ["Upload Video", "Webcam", "IP Camera"],
    horizontal=True
)

if source_option == "Upload Video":
    # Upload video
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    input_source = uploaded_file
elif source_option == "Webcam":
    # Camera selection
    available_cameras = []
    for i in range(3):  # Check first 3 camera indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()

    if not available_cameras:
        st.error("No webcams detected!")
    else:
        camera_index = st.selectbox(
            "Select Webcam",
            available_cameras,
            format_func=lambda x: f"Camera {x}"
        )
        input_source = camera_index
else:
    # IP Camera configuration
    st.markdown("### IP Camera Settings")

    # Camera presets
    camera_presets = {
        "Custom": {
            "rtsp": "/h264Preview_01_main",
            "http": "/stream",
            "default_port_rtsp": "554",
            "default_port_http": "80"
        },
        "Hikvision": {
            "rtsp": "/Streaming/Channels/101",
            "http": "/ISAPI/Streaming/channels/101/picture",
            "default_port_rtsp": "554",
            "default_port_http": "80"
        },
        "Dahua": {
            "rtsp": "/cam/realmonitor?channel=1&subtype=0",
            "http": "/cgi-bin/snapshot.cgi",
            "default_port_rtsp": "554",
            "default_port_http": "80"
        },
        "Axis": {
            "rtsp": "/axis-media/media.amp",
            "http": "/axis-cgi/jpg/image.cgi",
            "default_port_rtsp": "554",
            "default_port_http": "80"
        },
        "Reolink": {
            "rtsp": "/h264Preview_01_main",
            "http": "/cgi-bin/api.cgi?cmd=Snap&channel=0",
            "default_port_rtsp": "554",
            "default_port_http": "80"
        }
    }

    col1, col2 = st.columns(2)
    with col1:
        camera_brand = st.selectbox(
            "Camera Brand",
            list(camera_presets.keys()),
            help="Select your camera brand for automatic configuration"
        )

        protocol = st.selectbox(
            "Protocol",
            ["RTSP", "HTTP"],
            help="Select the streaming protocol"
        )

    with col2:
        # Advanced stream settings
        with st.expander("Advanced Stream Settings"):
            resolution = st.selectbox(
                "Resolution",
                ["Auto", "1920x1080", "1280x720", "640x480"],
                help="Select stream resolution"
            )

            quality = st.slider(
                "Stream Quality",
                min_value=1,
                max_value=100,
                value=75,
                help="Higher quality may impact performance"
            )

            enable_hardware_accel = st.checkbox(
                "Enable Hardware Acceleration",
                value=True,
                help="Use GPU for decoding (if available)"
            )

            reconnect_attempts = st.number_input(
                "Max Reconnection Attempts",
                min_value=1,
                max_value=100,
                value=10,
                help="Number of times to attempt reconnection"
            )

            connection_timeout = st.slider(
                "Connection Timeout (seconds)",
                min_value=1,
                max_value=30,
                value=5,
                help="Time to wait for connection"
            )

    # IP Camera connection details
    col1, col2 = st.columns(2)
    with col1:
        ip_address = st.text_input("IP Address", "192.168.1.100")
        username = st.text_input("Username (optional)")

        # Use preset stream path based on selected brand and protocol
        preset = camera_presets[camera_brand]
        default_path = preset["rtsp"] if protocol == "RTSP" else preset["http"]
        default_port = preset["default_port_rtsp"] if protocol == "RTSP" else preset["default_port_http"]

    with col2:
        port = st.text_input("Port", default_port)
        password = st.text_input("Password (optional)", type="password")

    # Stream path with preset
    stream_path = st.text_input(
        "Stream Path",
        default_path,
        help=f"Default path for {camera_brand} cameras"
    )

    # Construct the stream URL
    if username and password:
        auth = f"{username}:{password}@"
    else:
        auth = ""

    if protocol == "RTSP":
        stream_url = f"rtsp://{auth}{ip_address}:{port}{stream_path}"
    else:
        stream_url = f"http://{auth}{ip_address}:{port}{stream_path}"

    # Add stream settings to URL if supported
    if protocol == "RTSP" and resolution != "Auto":
        width, height = map(int, resolution.split("x"))
        stream_url += f"?width={width}&height={height}&quality={quality}"

    # Test connection button with detailed feedback
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test Connection", key="test_ip_camera"):
            try:
                with st.spinner("Testing connection..."):
                    test_cap = cv2.VideoCapture(stream_url)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret:
                            # Show a preview frame if connection was successful
                            preview_col = st.columns(3)[1]
                            with preview_col:
                                st.image(
                                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    caption="Preview Frame",
                                    use_column_width=True
                                )
                            
                            # Show stream info
                            st.success("Successfully connected to IP camera!")
                            st.info(f"""
                            Stream Information:
                            - Resolution: {int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
                            - FPS: {int(test_cap.get(cv2.CAP_PROP_FPS))}
                            - Codec: {int(test_cap.get(cv2.CAP_PROP_FOURCC))}
                            """)
                        else:
                            st.error("Connected to camera but couldn't read frame. Check stream settings.")
                    else:
                        st.error("Could not connect to IP camera. Please check your settings.")
                    test_cap.release()
            except Exception as e:
                st.error(f"Error connecting to camera: {str(e)}")

    input_source = stream_url

# Sidebar configuration
st.sidebar.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)

# Model selection
model_options = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLOv8l": "yolov8l.pt",
    "YOLOv8x": "yolov8x.pt",
}
selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()), index=0)
model_path = model_options[selected_model]

# Enable demographics analysis
enable_demographics = st.sidebar.checkbox("Enable Demographics Analysis", value=True)

# Visualization settings
st.sidebar.markdown("### Visualization Settings")
show_heatmap = st.sidebar.checkbox("Show Movement Heatmap", value=True)
show_traces = st.sidebar.checkbox("Show Movement Traces", value=True)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05
    )

    trace_length = st.slider(
        "Trace Length (seconds)",
        min_value=1,
        max_value=10,
        value=2
    )

    # Line settings
    st.markdown("### Counting Line Position")
    use_default_line = st.checkbox("Use Default Line", value=True)
    if not use_default_line:
        st.info("Custom line definition will be enabled when processing starts")

if input_source is not None:
    # Create placeholders for stats and video before processing
    stats_container = st.container()
    with stats_container:
        stats_cols = st.columns(3)

        # Create empty placeholders for each metric
        with stats_cols[0]:
            current_count_placeholder = st.empty()
        with stats_cols[1]:
            movement_placeholder = st.empty()
        with stats_cols[2]:
            demographics_placeholder = st.empty()

    # Create tab-based view to show video and demographics charts
    tabs = st.tabs(["Video Feed", "Demographics Dashboard"])

    with tabs[0]:
        video_placeholder = st.empty()

    with tabs[1]:
        # Demographics charts
        demo_charts_container = st.container()
        demo_col1, demo_col2 = demo_charts_container.columns(2)

        with demo_col1:
            gender_chart_placeholder = st.empty()

        with demo_col2:
            age_chart_placeholder = st.empty()

    # Initialize video source
    if source_option == "Upload Video":
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        cap = cv2.VideoCapture(video_path)
    else:
        # Handle both webcam and IP camera
        try:
            cap = cv2.VideoCapture(input_source)

            # Configure stream settings for IP camera
            if source_option == "IP Camera":
                # Set buffer size for IP camera to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Apply hardware acceleration if enabled
                if enable_hardware_accel:
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)

                # Set resolution if specified
                if resolution != "Auto":
                    width, height = map(int, resolution.split("x"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                # Add connection monitoring
                connection_monitor = {
                    "last_frame_time": time.time(),
                    "reconnect_count": 0,
                    "max_attempts": reconnect_attempts
                }
        except Exception as e:
            st.error(f"Error opening video source: {str(e)}")
            st.stop()

    if not cap.isOpened():
        st.error(f"Error: Could not open {source_option.lower()}")
        st.stop()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video info
    # Format for VideoInfo: width, height, fps, [total_frames]
    video_info = sv.VideoInfo(
        width=frame_width,
        height=frame_height,
        fps=fps
    )

    # Initialize components
    detector = CustomerDetector(model_path=model_path)
    byte_track = sv.ByteTrack(frame_rate=fps if fps > 0 else 30)
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
    if enable_demographics:
        demographics_analyzer = DemographicsAnalyzer()

    # Initialize annotators
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(frame_width, frame_height))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(frame_width, frame_height))

    modern_annotator = ModernAnnotator(
        thickness=thickness,
        text_scale=text_scale,
        text_thickness=thickness
    )

    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=fps * trace_length if fps > 0 else 30 * trace_length,
        position=sv.Position.CENTER,
        color=sv.Color.GREEN
    )

    line_counter_annotator = sv.LineZoneAnnotator(
        thickness=thickness,
        text_scale=text_scale,
        text_thickness=thickness
    )

    # Initialize heatmap if enabled
    heatmap_gen = None
    if show_heatmap:
        heatmap_gen = HeatmapGenerator(
            frame_resolution=(frame_width, frame_height),
            decay_factor=HEATMAP_DECAY
        )

    # Initialize tracking
    tracked_ids_for_demographics = set()
    start_time = datetime.now()

    # Add stop button for live feeds
    if source_option in ["Webcam", "IP Camera"]:
        stop_button_col = st.empty()
        stop_button = stop_button_col.button(
            f"Stop {source_option}",
            key="stop_stream_button"
        )

    # Process video/camera feed
    while True:
        if source_option in ["Webcam", "IP Camera"] and stop_button:
            break

        ret, frame = cap.read()
        if not ret:
            if source_option in ["Webcam", "IP Camera"]:
                # For live streams, try to reconnect
                time.sleep(1)  # Wait before reconnecting
                continue
            break  # For video files, break the loop

        # Detect and track customer
        detections = detector.detect(frame)
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
        if show_heatmap and heatmap_gen:
            heatmap_gen.update(points)

        # Analyze demographics if enabled
        demographics_info = None
        if enable_demographics and demographics_analyzer:
            demographics_info = demographics_analyzer.analyze_frame(frame, detections)
            demographics_analyzer.update_stats(new_entries, demographics_info, detections)

        # Create labels
        labels = []
        for i, (tracker_id, confidence) in enumerate(zip(detections.tracker_id, detections.confidence)):
            label = f"#{tracker_id}"

            # Add demographics info if available
            if enable_demographics and demographics_info and i < len(demographics_info) and demographics_info[i]:
                demo = demographics_info[i]
                label += f" {demo['gender']}, {demo['age']}y"

            labels.append(label)

        # Annotate frame
        annotated_frame = frame.copy()

        # Add visualizations based on enabled features
        if show_heatmap and heatmap_gen:
            heatmap, alpha = heatmap_gen.get_visualization()
            annotated_frame = cv2.addWeighted(annotated_frame, 1-alpha, heatmap, alpha, 0)

        # Draw line counter
        annotated_frame = line_counter_annotator.annotate(frame=annotated_frame, line_counter=line_counter)

        # Add traces
        if show_traces:
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

        # Add labels and halos
        annotated_frame = modern_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Update metrics
        occupancy_stats = occupancy_counter.get_stats()

        current_count_placeholder.metric(
            "Current Count 👥",
            occupancy_stats["current_count"],
            delta=f"{line_counter.in_count - line_counter.out_count} net change"
        )

        movement_placeholder.metric(
            "Movement 🚶",
            f"In: {line_counter.in_count}",
            delta=f"Out: {line_counter.out_count}"
        )

        # Update demographics metrics
        if enable_demographics and demographics_analyzer:
            demo_stats = demographics_analyzer.get_stats()
            total_analyzed = demo_stats["total_analyzed"]

            demographics_placeholder.metric(
                "Demographics 👤",
                f"Analyzed: {total_analyzed}",
                delta=f"{(total_analyzed/line_counter.in_count*100):.1f}% of entries" if line_counter.in_count > 0 else "No entries"
            )

            # Update gender chart
            with gender_chart_placeholder:
                if total_analyzed > 0:
                    gender_data = {"Gender": ["Male", "Female"],
                                  "Count": [demo_stats["gender_counts"]["Male"], demo_stats["gender_counts"]["Female"]]}
                    fig = px.pie(gender_data, values="Count", names="Gender",
                                title="Gender Distribution",
                                color_discrete_map={"Male": "#0072B2", "Female": "#D55E00"})
                    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True, key=f"gender_chart_{time.time()}")
                else:
                    st.info("No gender data available yet")

            # Update age chart
            with age_chart_placeholder:
                if total_analyzed > 0:
                    age_data = {"Age Group": list(demo_stats["age_groups"].keys()),
                               "Count": list(demo_stats["age_groups"].values())}
                    fig = px.bar(age_data, x="Age Group", y="Count",
                               title="Age Distribution",
                               color="Age Group",
                               color_discrete_sequence=px.colors.qualitative.Set3)
                    st.plotly_chart(fig, use_container_width=True, key=f"age_chart_{time.time()}")
                else:
                    st.info("No age data available yet")

        # Update video display with lower latency for live streams
        if source_option in ["Webcam", "IP Camera"]:
            # Reduce resolution for better performance
            display_frame = resize_frame_for_display(
                annotated_frame,
                max_width=1280,
                max_height=720
            )
        else:
            display_frame = annotated_frame

        video_placeholder.image(
            cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

        # Update stop button for live streams
        if source_option in ["Webcam", "IP Camera"]:
            stop_button = stop_button_col.button(
                f"Stop {source_option}",
                key=f"stop_stream_button_{time.time()}"
            )

        # In the video processing loop, add connection monitoring for IP camera
        if source_option == "IP Camera":
            if not ret:
                current_time = time.time()
                if current_time - connection_monitor["last_frame_time"] > connection_timeout:
                    if connection_monitor["reconnect_count"] < connection_monitor["max_attempts"]:
                        st.warning("Connection lost. Attempting to reconnect...")
                        cap.release()
                        time.sleep(1)
                        cap = cv2.VideoCapture(input_source)
                        connection_monitor["reconnect_count"] += 1
                    else:
                        st.error("Maximum reconnection attempts reached. Please check your connection.")
                        break
            else:
                connection_monitor["last_frame_time"] = time.time()
                connection_monitor["reconnect_count"] = 0

    # Cleanup
    cap.release()
    if source_option == "Upload Video":
        os.unlink(video_path)

    # Show final demographics summary if enabled
    if enable_demographics and demographics_analyzer and source_option == "Upload Video": 
        st.markdown("## Final Demographics Summary")
        col1, col2 = st.columns(2)

        with col1:
            gender_data = {"Gender": ["Male", "Female"],
                          "Count": [demographics_analyzer.gender_counts["Male"], demographics_analyzer.gender_counts["Female"]]}
            fig = px.pie(gender_data, values="Count", names="Gender",
                        title="Gender Distribution",
                        color_discrete_map={"Male": "#0072B2", "Female": "#D55E00"})
            st.plotly_chart(fig, use_container_width=True, key="final_gender_chart")

        with col2:
            age_data = {"Age Group": list(demographics_analyzer.age_groups.keys()),
                       "Count": list(demographics_analyzer.age_groups.values())}
            fig = px.bar(age_data, x="Age Group", y="Count",
                       title="Age Distribution",
                       color="Age Group",
                       color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True, key="final_age_chart")

        # Print text summary
        st.markdown("### Summary Statistics")
        summary_cols = st.columns(3)

        with summary_cols[0]:
            st.metric("Total Customer Entered", line_counter.in_count)

        with summary_cols[1]:
            st.metric("Total Customer Analyzed", demographics_analyzer.get_stats()["total_analyzed"])

        with summary_cols[2]:
            if line_counter.in_count > 0:
                percentage = (demographics_analyzer.get_stats()["total_analyzed"] / line_counter.in_count) * 100
                st.metric("Analysis Coverage", f"{percentage:.1f}%")
            else:
                st.metric("Analysis Coverage", "0%")

    # Show restart button for live streams
    if source_option in ["Webcam", "IP Camera"]:
        if st.button("Restart Stream", key="restart_stream_button"):
            st.experimental_rerun()

else:
    if source_option == "Upload Video":
        st.info("Please upload a video file to begin processing")
    elif source_option == "Webcam":
        st.info("Please select a webcam to begin processing")
    else:
        st.info("Please configure your IP camera settings to begin processing")
