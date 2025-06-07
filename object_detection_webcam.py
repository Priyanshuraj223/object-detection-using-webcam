import cv2
import numpy as np
import time
import os

# Create outputs directory if it doesn't exist
output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set your input source here:
# For webcam, use 0
# For video file, use "path_to_video.mp4"
# For image, use "path_to_image.jpg"
input_source = 0  # Change this to your video file path or image path if needed

# Adjustable confidence threshold
confidence_threshold = 0.6

# Class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

def detect_objects_in_frame(frame):
    (h, w) = frame.shape[:2]

    # Prepare input blob and perform forward pass
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            text = f"{label}: {confidence * 100:.2f}%"
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def run_on_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        output_frame = detect_objects_in_frame(frame)

        cv2.imshow("Object Detection - Webcam", output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = os.path.join(output_dir, f"saved_{int(time.time())}.jpg")
            cv2.imwrite(filename, output_frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()

def run_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break

        output_frame = detect_objects_in_frame(frame)

        cv2.imshow("Object Detection - Video", output_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = os.path.join(output_dir, f"saved_{int(time.time())}.jpg")
            cv2.imwrite(filename, output_frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()

def run_on_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    output_image = detect_objects_in_frame(image)

    cv2.imshow("Object Detection - Image", output_image)
    print("Press 's' to save the image, 'q' to quit.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = os.path.join(output_dir, f"saved_{int(time.time())}.jpg")
            cv2.imwrite(filename, output_image)
            print(f"Saved {filename}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if isinstance(input_source, int):
        run_on_webcam()
    elif input_source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        run_on_video(input_source)
    elif input_source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        run_on_image(input_source)
    else:
        print("Invalid input source! Change the 'input_source' variable to webcam (0), video file path, or image path.")
