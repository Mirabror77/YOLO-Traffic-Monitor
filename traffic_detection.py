import cv2
import cvzone
import math
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_SOURCE = "traffic_intersection.mp4"  # Make sure this file is in the same folder!
MODEL_NAME = "yolov8n.pt"                  # Uses the 'Nano' model (Fastest)
CONFIDENCE_THRESHOLD = 0.3                 # Minimum confidence to detect

# Class IDs (COCO Dataset)
# 2=Car, 3=Motorbike, 5=Bus, 7=Truck
TARGET_CLASSES = [2, 3, 5, 7]

# Counting Line [x1, y1, x2, y2]
# Adjust these numbers if the line doesn't match your road!
LINE_COORDINATES = [0, 1000, 4000, 1000]

# Storage for Counts
total_counts = {2: [], 3: [], 5: [], 7: []}

# --- 1. LOAD MODEL ---
print("Loading YOLO Model...")
model = YOLO(MODEL_NAME)

# --- 2. START VIDEO ---
cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    success, img = cap.read()
    if not success:
        print("Video ended.")
        break

    # --- 3. TRACKING (The AI Brain) ---
    # persist=True keeps the ID attached to the car
    results = model.track(img, persist=True, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.id is not None:
                obj_id = int(box.id[0])
                cls = int(box.cls[0])
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Filter: Only Vehicles with High Confidence
                if cls in TARGET_CLASSES and conf > CONFIDENCE_THRESHOLD:
                    
                    # Coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cx, cy = x1 + w // 2, y1 + h // 2

                    # --- COLOR LOGIC (Your Request) ---
                    # Default Purple for Cars
                    color = (255, 0, 255) 
                    
                    # Orange for Trucks (7) and Buses (5)
                    if cls == 7 or cls == 5:
                        color = (0, 165, 255) 

                    # Draw Box & ID
                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=color)
                    cvzone.putTextRect(img, f'#{obj_id}', (max(0, x1), max(35, y1)), 
                                     scale=1, thickness=1, colorR=color, offset=3)

                    # Draw Center Dot
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

                    # --- COUNTING LOGIC ---
                    # Check if center point touches the line
                    if LINE_COORDINATES[0] < cx < LINE_COORDINATES[2] and \
                       LINE_COORDINATES[1] - 20 < cy < LINE_COORDINATES[1] + 20:
                        
                        # Count only if new
                        if total_counts[cls].count(obj_id) == 0:
                            total_counts[cls].append(obj_id)
                            # Flash Green
                            cv2.line(img, (LINE_COORDINATES[0], LINE_COORDINATES[1]), 
                                     (LINE_COORDINATES[2], LINE_COORDINATES[3]), (0, 255, 0), 5)

    # --- DRAW INTERFACE ---
    # Draw Counting Line
    cv2.line(img, (LINE_COORDINATES[0], LINE_COORDINATES[1]), 
             (LINE_COORDINATES[2], LINE_COORDINATES[3]), (0, 0, 255), 5)

    # Display Scores
    cars = len(total_counts[2])
    trucks = len(total_counts[7])
    buses = len(total_counts[5])
    total_veh = cars + trucks + buses + len(total_counts[3])

    cvzone.putTextRect(img, f'Total: {total_veh}', (50, 50), scale=2, thickness=2)
    cvzone.putTextRect(img, f'Cars: {cars}  Trucks: {trucks}', (50, 100), scale=1, thickness=1)

    cv2.imshow("Apex Research", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()