import cv2
import numpy as np

# Try importing ultralytics. Handle missing package gracefully if possible
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

VEHICLE_CLASSES = {'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7}
VEHICLE_CLASS_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

def load_model(model_path: str = 'yolov8n.pt'):
    """Load YOLO model, handle download."""
    if YOLO is None:
        raise ImportError("ultralytics package is not installed. Please install it using 'pip install ultralytics'.")
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model from {model_path}: {str(e)}")

def detect_vehicles(model, image: np.ndarray, confidence: float = 0.25) -> list[dict]:
    """Run detection, filter to vehicle classes only."""
    results = model(image, verbose=False)[0]
    detections = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        
        if cls_id in VEHICLE_CLASS_NAMES and conf >= confidence:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            detections.append({
                'class_id': cls_id,
                'class_name': VEHICLE_CLASS_NAMES[cls_id],
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
                'center': [cx, cy]
            })
            
    return detections

def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes with labels and confidence on image copy."""
    img_copy = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_copy

def get_detection_summary(detections: list[dict]) -> dict:
    """Return counts by vehicle type."""
    summary = {name: 0 for name in VEHICLE_CLASSES.keys()}
    for det in detections:
        if det['class_name'] in summary:
            summary[det['class_name']] += 1
    return summary
