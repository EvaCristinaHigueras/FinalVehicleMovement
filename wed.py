# Eva Cristina Higueras: Vehicle Movement Detection Verson 1.0
# Overview:
#This program processes video frames to detect lanes using edge detection and Hough Transform, 
#then overlays an arrow indicating upcoming turns. The main steps include preprocessing the video, 
#detecting lanes, merging and filtering lanes, drawing lanes and centerlines, and overlaying directional arrows.
#
#def overlay_arrow (frame, overlay, position_):
#    Convert frame from OpenCV format to PIL format
#    Paste the overlay image (arrow) onto the frame at the specified position
#    Convert the PIL image back to OpenCV format
#    Return the modified frame
#
#def preprocess_frame (frame_):
#    Convert frame to grayscale
#    Apply Gaussian blur to reduce noise
#    Apply Canny edge detection to detect strong edges
#    Return processed edge-detected frame
#
#def region_of_interest (image_):
#    Get the image height and width
#    Define polygon vertices for the area of interest
#    Create a mask and fill it with white in the defined polygon
#    Apply bitwise AND to keep only the region of interest
#    Return the masked image
#
#def detect_lanes (edges_):
#    Apply Hough Line Transform to detect lane lines in the image
#    Return the detected lane lines
#
#def draw_centerline (frame, left_lane, right_lane_):
#    If both left and right lanes exist:
#        Calculate the midpoint between left and right lanes
#        Draw a red line connecting the midpoints
#    Return the frame with the centerline
#
#def merge_and_filter_lanes (lines, width, height, prev_left_lane, prev_right_lane_):
#    Initialize empty lists for left and right lanes
#    Iterate through each detected line:
#        Compute the slope of the line
#        Categorize it as a left lane or right lane based on slope range
#    Find the closest lane to expected positions using closest_lane function
#    Ensure valid lane width by checking if detected lanes fall within constraints
#    Smooth out detected lanes using linear regression
#    Return the final left and right lane coordinates
#
#def draw_lanes (frame, lines_):
#    Iterate through each detected lane line
#    If a lane is detected, draw it in green
#    Return the updated frame
#
#Main Loop:
#Initialize video capture from file
#Loop while video is open:
#    Read the next frame
#    If the frame is not available, break
#    Get frame dimensions
#    Apply preprocessing (grayscale, blur, edge detection)
#    Define region of interest
#    Detect lane lines
#    Merge and filter lane lines
#    Update previous lane values for continuity
#    Draw lane lines and centerline
#    If the frame count matches a predefined turn:
#        Rotate the arrow overlay
#    Overlay the arrow onto the frame
#    Show the processed frame
#    Increment frame count
#    Stop processing if user presses 'q'
#Release the video and close all windows
#
##############################################################################################################

import cv2
import numpy as np
from PIL import Image

# Constants for lane color and thickness
LANE_COLOR = (0, 255, 0)  # Green
CENTERLINE_COLOR = (0, 0, 255)  # Red for centerline
THICKNESS = 5
LANE_WIDTH_ESTIMATE = 300  # Estimated lane width in pixels
MISSING_LANE_FRAMES_THRESHOLD = 10  
SMOOTHING_BUFFER = 5  
LEFT_LANE_SLOPE_RANGE = (-1.0, -0.7)  
RIGHT_LANE_SLOPE_RANGE = (0.5, 1.0)  
MIN_LANE_WIDTH = 290  
MAX_LANE_WIDTH = 350  

# Load the video
video_path = 'video.mp4'  
cap = cv2.VideoCapture(video_path)

# Load the arrow image 
overlay_image = Image.open("arrow.png").convert("RGBA")
overlay_image = overlay_image.resize((100, 100))  

turns = {
    0: 90, 50: 120, 100: 120, 150: 90, 790: 120, 890: 90, 
    1760: 120, 1800: 90, 2550: 60, 2650: 90
}

# Frame count tracking
frame_count = 0
current_angle = 90  # Default arrow angle (straight)

# Function to overlay arrow onto frame
def overlay_arrow(frame, overlay, position):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    frame_pil.paste(overlay, position, overlay)
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)

# Function to preprocess the frame
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Define region of interest
def region_of_interest(image):
    height, width = image.shape
    mask = np.zeros_like(image)
    polygon = np.array([[ 
        (int(width * 0.4), height), 
        (int(width * 0.6), height), 
        (int(width * 0.7), int(height * 0.7)), 
        (int(width * 0.3), int(height * 0.7))  
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

# Detect lanes using Hough Transform
def detect_lanes(edges):
    return cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=250)

# Function to calculate and draw the centerline
def draw_centerline(frame, left_lane, right_lane):
    if left_lane and right_lane:
        x1_left, y1, x2_left, y2 = left_lane
        x1_right, _, x2_right, _ = right_lane
        x1_center = (x1_left + x1_right) // 2
        x2_center = (x2_left + x2_right) // 2
        cv2.line(frame, (x1_center, y1), (x2_center, y2), CENTERLINE_COLOR, THICKNESS)
    return frame

# Merge and filter lanes
def merge_and_filter_lanes(lines, width, height, prev_left_lane, prev_right_lane):
    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            if LEFT_LANE_SLOPE_RANGE[0] <= slope <= LEFT_LANE_SLOPE_RANGE[1]:
                left_lines.append([x1, y1, x2, y2])
            elif RIGHT_LANE_SLOPE_RANGE[0] <= slope <= RIGHT_LANE_SLOPE_RANGE[1]:
                right_lines.append([x1, y1, x2, y2])

    def closest_lane(lines, opposite_x, prev_lane):
        if not lines:
            return prev_lane
        return min(lines, key=lambda line: abs(((line[0] + line[2]) / 2) - opposite_x))

    left_lane = closest_lane(left_lines, width * 0.65, prev_left_lane)
    right_lane = closest_lane(right_lines, width * 0.35, prev_right_lane)

    # Ensure valid lane width
    if left_lane and right_lane:
        lane_width = abs(left_lane[0] - right_lane[0])
        if lane_width < MIN_LANE_WIDTH or lane_width > MAX_LANE_WIDTH:
            right_lane = None  

    def average_lane(lines):
        if lines is None:
            return None
        x_points, y_points = [], []
        for line in [lines]:
            x_points.extend([line[0], line[2]])
            y_points.extend([line[1], line[3]])
        poly = np.polyfit(y_points, x_points, 1)
        y1, y2 = int(height * 0.7), height
        x1, x2 = int(np.polyval(poly, y1)), int(np.polyval(poly, y2))
        return [x1, y1, x2, y2]

    left_lane = average_lane(left_lane) if left_lane else prev_left_lane
    right_lane = average_lane(right_lane) if right_lane else prev_right_lane

    return left_lane, right_lane

# Draw lane lines on the frame
def draw_lanes(frame, lines):
    if lines:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(frame, (x1, y1), (x2, y2), LANE_COLOR, THICKNESS)
    return frame

# Process video
prev_left_lane, prev_right_lane = None, None
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    edges = preprocess_frame(frame)
    roi = region_of_interest(edges)
    lines = detect_lanes(roi)
    left_lane, right_lane = merge_and_filter_lanes(lines, width, height, prev_left_lane, prev_right_lane)

    prev_left_lane, prev_right_lane = left_lane, right_lane
    frame = draw_lanes(frame, [prev_left_lane, prev_right_lane])
    frame = draw_centerline(frame, prev_left_lane, prev_right_lane)

    # Update arrow angle if there's a change in turns dict
    if frame_count in turns:
        current_angle = turns[frame_count]

    # Rotate arrow based on hardcoded values
    overlay_rotated = overlay_image.rotate(current_angle, expand=True)

    # Overlay arrow on the frame
    position = ((frame_width - overlay_rotated.width) // 2, 10)
    frame_bgr = overlay_arrow(frame, overlay_rotated, position)

    # Show result
    cv2.imshow("Lane Detection with Turned Arrow", frame_bgr)
    
    # Increment frame count
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
