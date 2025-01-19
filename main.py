import cv2
import numpy as np
import time
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load YOLOv3 configuration and weights
net = cv2.dnn.readNet('./yolov3.weights', './yolov3.cfg')

# Load the COCO class labels
with open('./coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Initialize timers and detected objects
start_time = time.time()
clear_time = start_time
speech_time = start_time
detected_objects = set()  # Set to store unique detected objects

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to hold detection data
    class_ids = []
    confidences = []
    boxes = []

    # Process YOLO outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the frame
    new_detected_objects = set()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            new_detected_objects.add((label, confidence))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Store new detected objects every 3 seconds
    current_time = time.time()
    if current_time - start_time >= 3:
        if new_detected_objects - detected_objects:
            print("New detected objects:")
            for obj in new_detected_objects - detected_objects:
                print(f'{obj[0]}: {obj[1]:.2f}')
            detected_objects.update(new_detected_objects)
        start_time = current_time

    # Announce new detected objects every 10 seconds
    if current_time - speech_time >= 11:
        if new_detected_objects - detected_objects:
            for obj in new_detected_objects - detected_objects:
                engine.say(f"New object detected: {obj[0]}")
                engine.runAndWait()
        speech_time = current_time

    # Clear stored objects every 30 seconds
    if current_time - clear_time >= 30:
        detected_objects.clear()
        clear_time = current_time

    # Display the output frame
    cv2.imshow("YOLOv3 Object Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
