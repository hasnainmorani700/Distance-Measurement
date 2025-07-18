import cv2
import math
from ultralytics import YOLO
#================================================================================
#Model                                                                         #=
model = YOLO("yolov8m.pt")                                                     #=
model.conf = 0.3                                                               #=
model.imgsz = 320                                                              #=
#===============================================================================
#video 
#===============================================================================
cap = cv2.VideoCapture("http://192.168.100.8:8080/video")                     #=
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                                           #=
if not cap.isOpened():                                                        #=
    print("Camera Error")                                                     #=
    exit()                                                                    #=
#===============================================================================
# Convert pix to cm 
#===============================================================================
px_per_cm = 40.0                                                              #=
while True:                                                                   #=
    ret, frame = cap.read()                                                   #=
    if not ret:                                                               #=
        print("Failed to grab frame")                                         #=
        break                                                                 #=
#===============================================================================
    # Run YOLOv8 inference
#===============================================================================
    results = model(frame, verbose=False)                                     #=
    centers = []                                                              #=
    for r in results:                                                         #=
        boxes = r.boxes                                                       #=
        if boxes is None:                                                     #=
            continue                                                          #=
        xywh = boxes.xywh.cpu().numpy()                                       #=
        confs = boxes.conf.cpu().numpy()                                      #=
        for box, conf in zip(xywh, confs):                                    #=
            if conf < model.conf:                                             #=
                continue                                                      #=
            x, y, w, h = box                                                  #=
            cx, cy = int(x), int(y)                                           #=
            centers.append((cx, cy))                                          #=
 #===============================================================================
            # Draw box and center point
 #===============================================================================
            x1, y1 = int(x - w/2), int(y - h/2)                                #=
            x2, y2 = int(x + w/2), int(y + h/2)                                #=
            x2, y2 = int(x + w/2), int(y + h/2)                                #=                               
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)           #=
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)                #=
 #===============================================================================
    # Measure distance between all pairs
 #===============================================================================
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            (x1, y1), (x2, y2) = centers[i], centers[j]
            d_px = math.hypot(x2 - x1, y2 - y1)
            d_cm = d_px / px_per_cm
 #===============================================================================
         # Draw line and distance text
 #===============================================================================
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            text = f"{d_cm:.1f}cm"
            cv2.putText(frame, text, (mx, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, text, (mx, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
 #===============================================================================
    # Show output
 #=============================================================================== 
    display = cv2.resize(frame, (800, 450))
    cv2.imshow("Distance Measurement", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#===============================================================================
# Cleanup
#=============================================================================== 
cap.release()
cv2.destroyAllWindows()
