import cv2

from ultralytics import solutions

# Open the video file
cap = cv2.VideoCapture("media/TestVideo.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties: width, height, and frames per second (fps)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define points for a line or region of interest in the video frame
line_points = [(20, 400), (1080, 400)]  # Line coordinates

# Initialize the video writer to save the output video
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize the Object Counter with visualization options and other parameters
counter = solutions.ObjectCounter(
    show=True,  # Display the image during processing
    region=line_points,  # Region of interest points
    model="yolo11m.pt",  # Ultralytics YOLO11 model file
    line_width=2,  # Thickness of the lines and bounding boxes
)

# Process video frames in a loop
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Use the Object Counter to count objects in the frame and get the annotated image
    im0 = counter.count(im0)

    # Write the annotated frame to the output video
    video_writer.write(im0)

# Release the video capture and writer objects
cap.release()
video_writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()