# Distance-Measurement
### **Object Distance Measurement with YOLOv8**  
**Purpose**: Detect objects in real-time video and measure distances between them in centimeters.  

---

#### **1. Setup**  
```bash
pip install opencv-python ultralytics numpy
```  
- Replace camera URL with `0` for webcam:  
  ```python
  cap = cv2.VideoCapture(0)  # Webcam
  ```

#### **2. Key Settings**  
| Parameter      | Purpose                                  |  
|----------------|------------------------------------------|  
| `model.conf=0.3` | Confidence threshold (lower = more detections) |  
| `px_per_cm=40` | **Calibrate this!** Measure a known object to adjust |  

#### **3. Workflow**  
1. **Detection**: YOLOv8 finds objects → draws green boxes  
2. **Center Points**: White dots mark object centers  
3. **Distance**: Yellow lines + cm labels between objects  

#### **4. Run & Quit**  
```bash
python your_script.py
```  
- Press **`Q`** to exit  

---

#### **Troubleshooting**  
- **No video?** Verify camera source/connection  
- **Wrong distances?** Re-measure `px_per_cm`  
- **No detections?** Lower `model.conf` or use larger model (`yolov8l.pt`)  

> **Note**: Accuracy depends on camera angle and calibration. Best for objects on flat surfaces.
