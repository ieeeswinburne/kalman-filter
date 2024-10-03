import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()

# Handle different indexing for getUnconnectedOutLayers
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Start webcam
cap = cv2.VideoCapture(0)

# Initialize Kalman Filter for tracking
kf = cv2.KalmanFilter(4, 2)  # 4 dynamic states (x, y, dx, dy), 2 measured (x, y)
kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)
# Process noise covariance matrix: represents uncertainty in the system's model (predicted position and velocity)
# Multiplied by a small value (1e-4) to assume very low process noise (system is stable)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4

# Measurement noise covariance matrix: represents uncertainty in the sensor measurements (detected position)
# Multiplied by 1e-1 to allow for moderate uncertainty in position measurements (sensor errors)
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

# Error covariance matrix after the update step: represents uncertainty in the initial state estimate (position and velocity)
# Initially, the error is set to an identity matrix (uncertainty is assumed to be 1 for each state variable)
kf.errorCovPost = np.eye(4, dtype=np.float32)

# Initial state estimate: sets the initial position (x, y) and velocity (dx, dy) to zero
# The object is assumed to be at the origin (0, 0) and stationary (zero velocity) at the start
kf.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32)


# Variables to control detection rate
frame_counter = 0
detection_interval = 10  # Run object detection every 10 frames
object_tracked = False  # Flag to start tracking
last_measurement = None  # Store the last measured position

# Cache for the last detected bounding box and label
last_box = None
last_label = None
last_detection_frame = -1  # Frame number when the last detection occurred

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Run YOLO detection every detection_interval frames
    if frame_counter % detection_interval == 0:
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

                # We are only interested in the "person" class
                if classes[class_ids[i]] == "person":
                    center_x = x + w // 2
                    center_y = y + h // 2
                    last_measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]], dtype=np.float32)

                    # If first detection, initialize Kalman filter
                    if not object_tracked:
                        kf.statePost = np.array([[np.float32(center_x)], [np.float32(center_y)], [0], [0]], dtype=np.float32)
                        object_tracked = True

                    # Cache the detection result (bounding box and label)
                    last_box = box
                    last_label = label
                    last_detection_frame = frame_counter

                    # Draw the detection box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break  # Only track the first detected person

    # Use Kalman Filter for prediction and correction
    if object_tracked:
        prediction = kf.predict()  # Predict the next position
        if last_measurement is not None and frame_counter % detection_interval == 0:
            kf.correct(last_measurement)  # Correct with the measurement every detection frame

        # Draw a single red dot representing the Kalman Filter's prediction
        predicted_x, predicted_y = int(prediction[0]), int(prediction[1])
        cv2.circle(frame, (predicted_x, predicted_y), 10, (0, 0, 255), -1)

    # Continue to draw the last detected bounding box and label if detection isn't running
    if last_box is not None and frame_counter - last_detection_frame < detection_interval:
        x, y, w, h = last_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, last_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("YOLO Object Detection with Kalman Filter", frame)

    # Increment frame counter
    frame_counter += 2

    # Exit on 'Esc'
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
