import cv2
import os
import time

# Create output folder if it doesn't exist
output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = os.path.join(output_dir, f"test_{int(time.time())}.jpg")
        success = cv2.imwrite(filename, frame)
        print("Saving:", filename if success else "Failed to save")

cap.release()
cv2.destroyAllWindows()
