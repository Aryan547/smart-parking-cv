# Smart Parking Space Detection and Occupancy Analysis Using Computer Vision

## Overview
Brief description of the system - a computer vision application that analyzes parking lot images/videos to detect vehicles, determine parking space occupancy, and provide analytics. Built for VITyarthi Build Your Own Project submission.

## Problem Statement
Manual parking management is inefficient. This system uses CV and deep learning to automate parking occupancy detection.

## Objectives
- Implement image preprocessing pipeline using OpenCV
- Detect vehicles using YOLOv8 deep learning model
- Determine parking space occupancy through spatial analysis
- Provide real-time analytics and visualization
- Create a web-based interface for interaction

## Features
### Module 1: Image Preprocessing
- Image loading and validation
- Resize with aspect ratio preservation
- Grayscale conversion
- Gaussian blur
- CLAHE contrast enhancement
- Canny edge detection
- Morphological operations
- Configurable preprocessing pipeline

### Module 2: Vehicle Detection & Occupancy Analysis
- YOLOv8n object detection for vehicles (car, motorcycle, bus, truck)
- Polygon-based parking slot configuration
- Pixel-level vehicle-to-slot spatial matching using polygon overlap
- Configurable occupancy threshold
- Real-time slot status classification

### Module 3: Analytics & Visualization
- Occupancy statistics (total, occupied, available, percentage)
- Annotated output with color-coded slot overlays
- Vehicle type breakdown
- Video frame-by-frame occupancy tracking
- Interactive Chart.js visualizations

## Computer Vision Techniques Used
- **Image Preprocessing**: Grayscale conversion, Gaussian blur, CLAHE histogram equalization, Canny edge detection, morphological operations (dilation, erosion, opening, closing)
- **Object Detection**: YOLOv8 (You Only Look Once) deep learning model for real-time vehicle detection
- **Spatial Analysis**: Polygon-based parking slot definition, pixel-level overlap calculation using cv2.fillPoly and bitwise operations, IoU (Intersection over Union) computation
- **Video Processing**: Frame-by-frame analysis with configurable sampling interval

## CV Pipeline
```
Input Image/Video → Image Validation → Preprocessing → Slot Configuration → YOLO Detection → Spatial Matching → Occupancy Classification → Statistics → Annotated Output → Web Visualization
```

## Technologies
| Technology | Purpose |
|---|---|
| Python 3.9+ | Core language |
| Flask | Web framework |
| OpenCV | Image/video processing |
| NumPy | Numerical operations |
| Ultralytics YOLOv8 | Object detection |
| HTML5/CSS3/JS | Frontend |
| Chart.js | Data visualization |
| Pytest | Testing |

## Project Structure
```
smart-parking-vision/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── statement.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── vehicle_detector.py
│   ├── parking_slots.py
│   ├── occupancy.py
│   ├── analytics.py
│   ├── visualization.py
│   └── video_processor.py
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── image_analysis.html
│   ├── video_analysis.html
│   ├── slot_config.html
│   └── analytics.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── main.js
│       ├── image_analysis.js
│       ├── video_analysis.js
│       ├── slot_config.js
│       └── analytics.js
│
├── config/
│   └── parking_slots.json
│
├── input/
│   ├── README.md
│   ├── sample_parking.jpg
│   └── sample_vehicles.jpg
│
├── output/
│   └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_parking_slots.py
│   ├── test_occupancy.py
│   ├── test_analytics.py
│   ├── test_validation.py
│   └── test_app.py
│
└── docs/
    ├── architecture.md
    ├── workflow.md
    ├── use-case.md
    ├── sequence-diagram.md
    ├── component-diagram.md
    └── cv-pipeline.md
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Internet connection (for first-time YOLO model download)

### Step 1: Create Virtual Environment
```bash
python -m venv venv
```

### Step 2: Activate Virtual Environment
Windows:
```bash
venv\Scripts\activate
```
Linux/Mac:
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: YOLO Model Setup
The YOLOv8n model (6.2MB) is automatically downloaded on first run. No manual download needed. Requires internet connection for the initial download.

### Step 5: Run the Application
```bash
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

### CLI Options
```bash
python app.py --host 0.0.0.0 --port 8080 --debug --model yolov8n.pt
```

## Usage

### Image Analysis
1. Navigate to Image Analysis page
2. Upload a parking lot image (JPG/PNG)
3. Optionally select a parking slot configuration
4. Adjust confidence and occupancy thresholds
5. Click 'Analyze' to run the CV pipeline
6. View annotated results and statistics

### Video Analysis
1. Navigate to Video Analysis page
2. Upload a parking lot video (MP4/AVI)
3. Select slot configuration and set parameters
4. Click 'Process Video' to analyze
5. View occupancy over time chart and download annotated video

### Parking Slot Configuration
1. Navigate to Slot Configuration page
2. Upload a reference parking lot image
3. Click on the image to define polygon corners for each parking space
4. Press Enter or click near the starting point to close each polygon
5. Assign labels to each slot
6. Save the configuration as JSON

### Running Tests
```bash
pytest tests/ -v
```

## YOLO Model Selection Rationale
- **YOLOv8n** selected for being lightweight (6.2MB), fast inference, and sufficient accuracy for vehicle detection
- YOLO (You Only Look Once) provides single-pass detection, making it efficient for real-time analysis
- Pre-trained on COCO dataset which includes vehicle classes (car, motorcycle, bus, truck)
- No need for custom training as COCO vehicle classes match our requirements

## Occupancy Detection Methodology
- Parking slots defined as polygons (quadrilaterals) in JSON configuration
- Vehicle bounding boxes from YOLO compared against slot polygons
- Pixel-level overlap calculated using OpenCV polygon fill and bitwise operations
- Overlap ratio = intersection_pixels / slot_area_pixels
- If overlap ratio ≥ threshold (default 0.3), slot classified as occupied
- This method handles angled parking, irregular slot shapes, and partial occlusions

## Evaluation Methodology
- **No fabricated metrics**: Accuracy/precision/recall values are not claimed without actual labeled test data
- **Visual verification**: Results can be visually verified against input images
- **Unit & Integration tests**: 100+ pytest tests verify individual component correctness and pipeline integrations
- **Integration testing**: End-to-end pipeline tested with synthetic and real images
- To evaluate quantitatively: label a test set of parking images with ground-truth occupancy, run the pipeline, and compare predictions vs labels

## Limitations
1. Requires pre-configured parking slot polygons for accurate occupancy detection
2. Detection accuracy depends on image quality, angle, and lighting conditions
3. Heavily occluded vehicles may not be detected by YOLO
4. Night-time or poorly lit parking lots may reduce accuracy
5. Unusual vehicle types not in COCO dataset won't be detected
6. Very crowded scenes may cause overlapping detections
7. Video processing speed depends on hardware (GPU recommended for large videos)

## Future Enhancements
1. Automatic parking slot detection using line detection algorithms
2. GPU acceleration for faster video processing
3. Real-time RTSP camera feed support
4. Multi-floor parking lot support
5. License plate recognition integration
6. Historical occupancy trend analysis with database storage
7. Mobile-responsive progressive web app
8. Alert system for full parking lots

## Non-Functional Requirements
1. **Performance**: Image analysis completes within 5 seconds on modern hardware
2. **Scalability**: Supports parking lots with 100+ slots
3. **Usability**: Intuitive web interface requiring no CV knowledge
4. **Reliability**: Graceful error handling for invalid inputs
5. **Portability**: Runs on Windows, Linux, and macOS with Python 3.9+
