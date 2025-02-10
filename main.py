import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

# Use YOLOv8n model
model = YOLO('yolov8n.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('cr.mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

# Store previous frame's vehicle positions
prev_vehicles = []
crash_detected = False
crash_cooldown = 0
crash_frames = []  # Store frames where crashes are detected

while True:    
    ret,frame = cap.read()
    if not ret:
        break  # End video when it's finished

    frame = cv2.resize(frame,(1020,500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    # Current frame's vehicle positions
    current_vehicles = []
    current_crash = False

    for index,row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        d = int(row[5])
        
        if d < len(class_list):
            c = class_list[d].split('|')[-1]  # Handle format "1|person" -> "person"
        else:
            c = "unknown"

        # Only track vehicles
        if c in ['car', 'truck', 'bus', 'motorcycle']:
            current_vehicles.append([x1, y1, x2, y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)

    # Crash detection logic
    if len(prev_vehicles) > 0 and len(current_vehicles) > 0:
        for prev in prev_vehicles:
            for curr in current_vehicles:
                # Calculate IoU
                x_left = max(prev[0], curr[0])
                y_top = max(prev[1], curr[1])
                x_right = min(prev[2], curr[2])
                y_bottom = min(prev[3], curr[3])
                
                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    prev_area = (prev[2] - prev[0]) * (prev[3] - prev[1])
                    curr_area = (curr[2] - curr[0]) * (curr[3] - curr[1])
                    iou = intersection_area / float(prev_area + curr_area - intersection_area)
                    
                    # If vehicles overlap significantly
                    if iou > 0.3 and crash_cooldown == 0:
                        current_crash = True
                        crash_cooldown = 30
                        cv2.putText(frame, 'CRASH DETECTED!', (50,50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        # Save crash frame
                        crash_frames.append(frame.copy())
                        break

    if crash_cooldown > 0:
        crash_cooldown -= 1
        if current_crash:
            cv2.imshow("RGB", frame)
            cv2.waitKey(1)

    # Update previous vehicles
    prev_vehicles = current_vehicles.copy()
    
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()

# Show all detected crashes
print(f"Total crashes detected: {len(crash_frames)}")
for i, crash_frame in enumerate(crash_frames):
    cv2.imshow(f"Crash {i+1}", crash_frame)
    cv2.waitKey(0)  # Wait for key press before showing next crash

cv2.destroyAllWindows()




