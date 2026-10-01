# Sequence Diagram

The sequence diagram below details the step-by-step interactions between system components during a typical image analysis request.

```mermaid
sequenceDiagram
    actor User
    participant Browser as Web Browser (UI)
    participant Flask as Flask Backend
    participant CV_Pre as Preprocessing Engine
    participant YOLO as YOLOv8 Detector
    participant Occupancy as Occupancy Classifier
    participant Viz as Visualization Module

    User->>Browser: Upload Image & Config
    Browser->>Flask: POST /analyze_image
    Flask->>CV_Pre: Forward raw image
    CV_Pre->>CV_Pre: Apply Blur, CLAHE, etc.
    CV_Pre-->>Flask: Preprocessed Image
    Flask->>YOLO: Pass preprocessed image
    YOLO->>YOLO: Run inference
    YOLO-->>Flask: Vehicle Bounding Boxes
    Flask->>Occupancy: Send BBoxes & Slot Polygons
    Occupancy->>Occupancy: Compute Polygon Overlaps
    Occupancy->>Occupancy: Apply Thresholds
    Occupancy-->>Flask: Slot Statuses (Occupied/Free)
    Flask->>Viz: Request annotation (Image, Statuses)
    Viz->>Viz: Draw polygons, text overlays
    Viz-->>Flask: Annotated Image file path
    Flask-->>Browser: JSON Stats + Annotated Image URL
    Browser-->>User: Display Results & Dashboard
```

## Explanation
1. **Initiation**: The user submits an image via the web form.
2. **Preprocessing**: The Flask backend delegates the image to the CV pipeline for noise reduction and enhancement.
3. **Detection**: YOLOv8 scans the enhanced image to find vehicles, returning coordinate data.
4. **Classification**: The Occupancy module calculates exactly how much of a detected vehicle intersects with a configured parking slot. If the intersection exceeds the user threshold, it's marked occupied.
5. **Visualization**: The Visualization module takes the raw image and the classification results to draw colored polygons (red for taken, green for empty) and text labels.
6. **Completion**: The final compiled data and image are sent back to the browser for user consumption.
