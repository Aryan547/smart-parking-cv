# Smart Parking Space Detection and Occupancy Analysis Using Computer Vision

## Project Overview

Smart Parking Space Detection and Occupancy Analysis is an automated computer vision application designed to evaluate vehicle occupancy in parking facilities from uploaded images and recorded video streams. The system couples classical digital image processing routines with deep learning-based object detection (Ultralytics YOLOv8) and geometric spatial analysis.

Rather than relying on hardware ground sensors or manual surveillance, this system processes visual inputs through an OpenCV image validation and preprocessing pipeline, detects vehicle instances, and projects their bounding geometry against user-defined polygon parking slot configurations. Spatial intersection algorithms determine whether each designated parking slot is occupied or available, displaying annotated overlays, tabular detection records, and occupancy metrics via an interactive web interface.

### Why this is a Computer Vision Project
While the application exposes a web interface for demonstration and testing, the core logic is centered on digital image processing and spatial computer vision:
- Multi-stage pixel processing using spatial filtering (Gaussian kernel), adaptive histogram equalization (CLAHE), and edge feature extraction (Canny).
- Deep learning inference using convolutional feature representations to localize object classes.
- Polygon rasterization (`cv2.fillPoly`) and bitwise logical masks (`cv2.bitwise_and`) to compute exact pixel-level intersection-over-slot ratios without relying on crude axis-aligned bounding box approximations.
- Temporal frame extraction and sequential visual analytics across video inputs.

---

## Problem Statement

Manual management of parking facilities is labor-intensive, inefficient, and prone to delays. Traditional approaches relying on physical sensors (ultrasonic, magnetic, or infrared embedded in individual stalls) require expensive installation, regular hardware maintenance, and complex wiring infrastructure. 

Computer vision offers a non-intrusive, scalable alternative by leveraging standard camera imagery. However, automated visual occupancy analysis presents specific engineering challenges:
- Variations in camera perspective, perspective distortion, and angled parking layouts.
- Ambient lighting fluctuations and shadows cast across empty or filled bays.
- Visual occlusions and vehicle boundaries extending beyond slot margins.

This project addresses these challenges by combining robust image preprocessing, bounding box localization via pre-trained vehicle feature detectors, and configurable polygon spatial overlap matching.

---

## Objectives

1. **OpenCV-based Image Preprocessing and Validation**: Implement a robust image pipeline to validate image integrity, normalize resolution while preserving aspect ratios, and generate intermediate visual representations (grayscale, Gaussian smoothing, CLAHE contrast enhancement, and Canny edge maps).
2. **YOLOv8-based Vehicle Detection**: Integrate a lightweight deep learning object detector (YOLOv8n) to detect, localize, and filter relevant vehicle classes from the COCO dataset with configurable detection confidence thresholds.
3. **Polygon-based Parking Slot Occupancy Detection**: Represent parking spaces using arbitrary quadrilateral polygon coordinates in JSON format, and evaluate occupancy status using pixel-level spatial intersection masks against detected vehicles.
4. **Visualization and Analytics via Web Interface**: Render annotated image and video outputs with color-coded slot overlays, provide interactive polygon slot configuration tools, and display structured statistical summaries through a Flask-powered web interface.

---

## Key Features

- **Image Upload and Validation**: Multi-format image ingestion (JPG, JPEG, PNG) with verification of array integrity, dimensionality, and channel properties.
- **OpenCV Preprocessing Pipeline**: Reusable functions for aspect-ratio-preserving resizing, grayscale transformation, Gaussian blur, CLAHE contrast adaptation, Canny edge detection, and morphological transformations.
- **YOLOv8 Vehicle Detection**: Single-stage deep learning object detection filtering specific transportation categories (`car`, `motorcycle`, `bus`, `truck`).
- **Vehicle-Class Filtering**: Strict filtering isolating vehicle categories from the 80 COCO classes, ignoring irrelevant objects (pedestrians, street furniture).
- **Polygon-based Parking Slot Geometry**: Slot representation using custom polygon coordinates, accommodating angled, parallel, and perspective-distorted parking layouts.
- **JSON-based Slot Storage**: Portable JSON schema for loading, validating, and saving multiple slot configurations per camera viewpoint.
- **Pixel-Level Spatial Overlap Calculation**: Accurate overlap ratio calculation using OpenCV mask generation (`cv2.fillPoly` and `cv2.bitwise_and`), comparing vehicle bounding boxes against target slot polygons.
- **Configurable Occupancy Classification**: Threshold-driven binary classification (`Occupied` vs `Available`) based on vehicle-to-slot spatial overlap ratios.
- **Annotated Visual Outputs**: Generated output images featuring semi-transparent color overlays (green for available, red for occupied), bounding boxes, slot IDs, and summary info panels.
- **Occupancy Statistics**: Computation of total slots, occupied count, available count, occupancy percentage, and vehicle frequency breakdown.
- **Video Analysis**: Frame-by-frame video processing at sampled frame intervals, aggregating occupancy timelines and exporting annotated video files.
- **Interactive Analytics**: Dashboard displaying occupancy metrics and vehicle category distributions using Chart.js charts.
- **Slot Configuration Editor**: Canvas-based interface allowing users to upload reference images, click polygon vertices, assign slot identifiers, and export JSON coordinate sets.
- **Robust Error Handling**: Structured API error feedback for empty files, unsupported extensions, unreadable buffers, missing coordinates, and malformed configurations.

---

## Computer Vision Pipeline

```
Input Image / Video
        ↓
Image Validation (Array structure, format, dimensions)
        ↓
OpenCV Preprocessing (Resize, Grayscale, Gaussian Blur, CLAHE, Canny Edges)
        ↓
YOLOv8 Vehicle Detection (Bounding boxes, class IDs, detection confidence)
        ↓
Parking Slot Polygon Configuration (JSON coordinate mapping)
        ↓
Spatial Overlap Analysis (Mask generation, cv2.bitwise_and, pixel intersection)
        ↓
Occupancy Classification (Threshold comparison → Occupied / Available)
        ↓
Visualization & Annotation (Alpha-blended polygons, bounding boxes, labels)
        ↓
Statistics & Analytics (Slot totals, occupancy rates, class distributions)
```

### Pipeline Stage Details

1. **Image Validation**: Checks that the input payload contains decodable image buffers of `uint8` data type with valid channel structures before initiating processing.
2. **Preprocessing**: Normalizes image resolution to a maximum bounding size (default 1280×720) while maintaining aspect ratio. Generates intermediate diagnostic representations including luminance reduction (grayscale), high-frequency noise suppression (Gaussian kernel $5 \times 5$), adaptive local contrast balancing (CLAHE clip limit 2.0), and gradient boundary extraction (Canny thresholds 50/150).
3. **Vehicle Detection**: Feeds the normalized image to the YOLOv8 convolutional backbone. Outputs bounding box coordinates $[x_1, y_1, x_2, y_2]$, class IDs, and detection confidence scores.
4. **Slot Configuration Mapping**: Loads quadrilateral or polygonal coordinates $[(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)]$ corresponding to the calibrated camera perspective.
5. **Spatial Overlap Matching**: Constructs binary raster masks for both the slot polygon and vehicle bounding rectangle on a common coordinate canvas. Calculates intersection pixel count using `cv2.bitwise_and` relative to the vehicle bounding area.
6. **Occupancy Classification**: If $\frac{\text{Area}(\text{Slot} \cap \text{Vehicle})}{\text{Area}(\text{Vehicle})} \ge \text{Threshold}$ (default 0.30), the slot is flagged as `Occupied`; otherwise, it remains `Available`.
7. **Annotation**: Alpha-blends semi-transparent color overlays ($35\%$ fill opacity) onto the image canvas using `cv2.addWeighted`, draws boundary polylines, marks vehicle bounding boxes, and prints status badges.
8. **Analytics**: Aggregates categorical counts, calculates overall facility utilization percentages, and formats data for frontend consumption.

---

## Technologies Used

| Technology | Category | Purpose in Project |
|---|---|---|
| **Python 3.9+** | Programming Language | Core execution runtime and pipeline logic |
| **OpenCV (`opencv-python`)** | Computer Vision Library | Spatial filtering, color conversions, morphological operations, polygon rasterization, and image annotation |
| **Ultralytics YOLOv8** | Deep Learning Framework | Object detection model loading and bounding box inference |
| **NumPy** | Numerical Computing | Matrix operations, binary mask manipulation, and coordinate transformations |
| **Flask** | Web Framework | Backend HTTP routing, multipart upload ingestion, and REST API services |
| **HTML5 / CSS3 / JavaScript** | Frontend UI | Responsive dark-themed user interface, canvas drawing, and DOM updates |
| **Chart.js** | Data Visualization | Client-side visual charts for occupancy timelines and vehicle distributions |
| **Pytest** | Testing Framework | Automated test suites for preprocessing, geometry, occupancy, and API routes |
| **JSON** | Data Format | Serialized storage of calibrated parking slot polygon coordinates |

---

## Project Architecture

The codebase follows a modular design separating vision routines, geometry operations, web routing, and presentation layers:

- **`app.py`**: Flask application entry point. Implements page routing, file upload endpoints, REST API routes (`/api/preprocess`, `/api/analyze-image`, `/api/process-video`, `/api/save-slots`, `/api/load-slots`), static serving, and CLI argument parsing.
- **`config.py`**: Centralized configuration parameters including upload folders, allowed extensions, model paths, default confidence values (0.25), and default overlap thresholds (0.30).
- **`src/preprocessing.py`**: Digital image processing routines using OpenCV: `load_image`, `validate_image`, `resize_image`, `to_grayscale`, `apply_gaussian_blur`, `enhance_contrast` (CLAHE), `canny_edge_detection`, `morphological_operation`, and `preprocess_pipeline`.
- **`src/vehicle_detector.py`**: Ultralytics YOLOv8 integration. Implements model loading, inference execution, bounding box filtering for vehicle classes, detection summary generation, and annotation rendering.
- **`src/parking_slots.py`**: Polygon configuration utilities. Handles JSON schema validation, polygon validation (point counts, non-negative integers), Shoelace formula area computation, bounding box conversions, and slot overlay rendering.
- **`src/occupancy.py`**: Spatial matching engine. Implements bounding box IoU calculation, OpenCV mask-based pixel overlap calculation (`calculate_overlap_ratio`), vehicle-in-slot threshold verification, and global occupancy classification.
- **`src/visualization.py`**: Rendering and visual annotation. Draws semi-transparent colored polygons, status badges, vehicle boxes, side-by-side comparison images, and statistics summary panels.
- **`src/analytics.py`**: Statistical calculation and data formatting for occupancy rates, vehicle category breakdowns, and frame-by-frame temporal tracking for video inputs.
- **`src/video_processor.py`**: Video stream processing. Handles OpenCV `VideoCapture` validation, video metadata extraction, sampled frame-interval processing, annotated video encoding (`mp4v` codec via `VideoWriter`), and temporal statistics aggregation.
- **`templates/`**: Jinja2 templates (`base.html`, `index.html`, `image_analysis.html`, `video_analysis.html`, `slot_config.html`, `analytics.html`).
- **`static/`**: Client-side assets including `css/style.css` and modular scripts (`main.js`, `image_analysis.js`, `video_analysis.js`, `slot_config.js`, `analytics.js`).
- **`config/`**: Directory containing calibrated parking slot JSON configurations (e.g., `parking_slots.json`).
- **`tests/`**: Automated Pytest test suite validating every module independently.

---

## YOLO Vehicle Detection

Object detection is handled by **YOLOv8n** (Nano), the lightweight variant of the Ultralytics YOLOv8 convolutional architecture (~6.2 MB). 

### Class Filtering
YOLOv8 is pre-trained on the COCO (Common Objects in Context) dataset containing 80 target categories. The system filters detections strictly to vehicle categories:
- **Class ID 2**: `car`
- **Class ID 3**: `motorcycle`
- **Class ID 5**: `bus`
- **Class ID 7**: `truck`

All non-vehicle detections (such as pedestrians, traffic lights, or animals) are discarded prior to spatial evaluation.

### Bounding Boxes and Detection Confidence
For each detected vehicle, the model outputs:
- Coordinate bounds $[x_1, y_1, x_2, y_2]$
- Centroid coordinates $[c_x, c_y]$
- Detected class identifier and label
- Detection confidence score (a value between $0.0$ and $1.0$)

> **Important Technical Distinction**: A detection confidence score (e.g., 0.88 or 88%) represents the model's probabilistic estimate that a localized bounding box corresponds to the specified class. It is **not** an evaluation metric for system-wide model accuracy, precision, or recall. Detections below the user-configurable confidence threshold (default 0.25) are pruned before spatial matching.

---

## Parking Slot Detection and Occupancy Logic

Rather than relying on basic rectangular approximations that fail when cameras view a lot from angled perspectives, parking spaces are modeled as **arbitrary convex or non-convex polygons**.

### Polygon Geometric Modeling
Each slot is defined in JSON by an ordered series of coordinate vertices:
```json
{
  "id": 1,
  "label": "A1",
  "points": [[80, 200], [200, 200], [200, 350], [80, 350]]
}
```

### Spatial Overlap Analysis via OpenCV Masking
Simple axis-aligned bounding box intersection (IoU) produces inaccurate results for angled parking spots. The system instead computes exact geometric overlap using OpenCV pixel masks:

1. A zero-initialized canvas matching the bounding coordinate extents is instantiated.
2. The parking slot polygon is drawn and filled with pixel value 255 using `cv2.fillPoly`.
3. A separate mask is populated with the vehicle bounding rectangle using `cv2.rectangle` with fill flag `-1`.
4. A bitwise logical AND operation (`cv2.bitwise_and`) extracts the intersecting pixels:
   $$\text{Overlap Pixels} = \text{countNonZero}(\text{Slot Mask} \land \text{Vehicle Mask})$$
5. The overlap ratio is calculated relative to the vehicle's bounding area:
   $$\text{Overlap Ratio} = \frac{\text{Overlap Pixels}}{\text{Vehicle Bounding Area}}$$

### Occupancy Threshold
A slot is classified as `Occupied` if the overlap ratio exceeds the user-configured occupancy threshold (default: `0.30` or 30%):
- If $\text{Overlap Ratio} \ge 0.30 \implies \textbf{Occupied}$
- If $\text{Overlap Ratio} < 0.30 \implies \textbf{Available}$

This threshold accounts for partial overlap caused by vehicles in adjacent aisles or minor perspective overhang without falsely declaring an adjacent stall occupied.

---

## Image Preprocessing

Before or alongside detection, images can be passed through a multi-stage digital image processing pipeline implemented in `src/preprocessing.py`. Each stage isolates distinct visual features:

1. **Original**: Ingested image decoded via OpenCV into BGR format and validated for channel/data integrity.
2. **Grayscale**: Color-space reduction using `cv2.COLOR_BGR2GRAY`. Strips color dependencies, simplifying intensity gradient analysis.
3. **Gaussian Blur**: Spatial convolution using an isotropic Gaussian kernel ($5 \times 5$, $\sigma = 0$) to attenuate high-frequency pixel noise and surface grain.
4. **Contrast Enhancement (CLAHE)**: Contrast Limited Adaptive Histogram Equalization applied with a clip limit of 2.0 and tile grid dimensions of $8 \times 8$. Prevents over-amplification of noise while equalizing local luminance variations caused by shadows or uneven sunlight.
5. **Canny Edge Detection**: Multi-stage gradient operator applying Sobel kernels, non-maximum suppression, and hysteresis thresholding ($T_{\text{low}} = 50$, $T_{\text{high}} = 150$) to extract structural contours, such as painted parking boundary lines.
6. **Morphological Operations**: Reusable implementations of dilation, erosion, opening, and closing using structured square kernels to close contour gaps or remove speckle noise where applicable.

The web interface includes an interactive **Preprocessing Preview** panel where users can trigger the pipeline and inspect all five output stages side-by-side via Base64 Data URIs.

---

## Web Interface

The system includes a responsive web interface served by Flask:

- **Home (`/`)**: Project overview, architectural flowchart, summary of computer vision modules, and direct navigation actions.
- **Image Analysis (`/image-analysis`)**: Upload parking images, select calibrated slot configurations, adjust detection confidence and occupancy thresholds via sliders, run OpenCV preprocessing inspection, and view detection overlays with detailed status tables.
- **Video Analysis (`/video-analysis`)**: Upload recorded parking lot video files, specify sampling frame intervals, monitor batch processing, review occupancy-over-time trend lines, and download annotated MP4 output videos.
- **Slot Configuration (`/slot-config`)**: Interactive HTML5 canvas editor allowing users to upload a parking lot background image, click vertices to outline parking polygons, assign slot labels, and export or load JSON coordinate definitions.
- **Analytics (`/analytics`)**: Dashboard rendering summary cards, vehicle class distribution doughnut charts, and temporal utilization charts populated from session analysis data.

The web application serves purely as the interaction and visualization layer for the underlying computer vision and geometric processing pipeline.

---

## Results and Example Output

When an image is processed through the analysis endpoint, the application produces the following structured output:

### Quantitative Metrics
- **Total Slots**: Total number of parking spaces defined in the active configuration (e.g., 12).
- **Occupied Slots**: Count of slots occupied by detected vehicles meeting the overlap threshold (e.g., 5).
- **Available Slots**: Count of unoccupied parking spaces (e.g., 7).
- **Occupancy Percentage**: Calculated facility utilization rate ($\frac{\text{Occupied}}{\text{Total}} \times 100$) (e.g., 41.7%).
- **Vehicles Detected**: Total count of vehicles localized by YOLO within the scene (e.g., 6).

### Tabular Detections
- **Detection Details**: Tabulated breakdown listing detected vehicle class (`car`, `truck`, etc.), bounding box coordinates $[x_1, y_1, x_2, y_2]$, and confidence score (e.g., 0.84).
- **Slot Status**: Tabulated list of each slot ID, user label (e.g., `A1`), assigned vehicle reference, and color-coded status indicator (`Occupied` / `Available`).

### Visual Overlays
- **Color-Coded Polygons**: Available slots shaded in semi-transparent green; occupied slots shaded in semi-transparent red.
- **Slot Badges**: Centroid labels indicating slot identifier and status.
- **Vehicle Boxes**: Distinct bounding rectangles delineating detected vehicles with confidence tags.
- **Downloadable Artifacts**: The annotated frame is exportable directly as a JPEG image or MP4 video.

---

## Testing

The project incorporates an automated Pytest test suite located in `tests/`. Synthetic images and deterministic geometric primitives generated via NumPy and OpenCV are used across unit tests to verify functionality without requiring external network downloads or model fetching during unit testing.

**104 automated tests passed successfully.**

### Test Coverage Breakdown
- **`tests/test_preprocessing.py` (29 tests)**: Verifies image validation, handling of empty/corrupt arrays, aspect-ratio preservation under scaling, grayscale conversion, Gaussian kernel constraints, CLAHE enhancements, Canny edge thresholds, morphological operators, and multi-step pipeline execution.
- **`tests/test_occupancy.py` (19 tests)**: Validates bounding box IoU logic, polygon-to-box overlap ratio computations using OpenCV masks, vehicle-in-slot threshold verification, complete occupancy classification edge cases, and utilization percentage calculations.
- **`tests/test_parking_slots.py` (18 tests)**: Tests polygon validation (coordinate bounds, minimum vertex constraints), Shoelace formula area calculations, bounding box derivation, JSON configuration serialization/deserialization, and polygon rendering functions.
- **`tests/test_analytics.py` (12 tests)**: Evaluates statistics calculations, vehicle category counters, frame-by-frame temporal timeline generation, and video metric aggregations (minimum, maximum, and average occupancy).
- **`tests/test_validation.py` (14 tests)**: Tests boundary conditions and defensive edge cases, including invalid data types, large image downsampling, zero-confidence inputs, zero-area polygons, non-existent video paths, and multiple vehicles overlapping single slots.
- **`tests/test_app.py` (12 tests)**: Integration testing for Flask routes, HTTP status codes, multipart file upload ingestion, JSON schema payloads, Base64 Data URI formatting for preprocessing stages, and error responses for corrupted inputs.

Run the test suite with:
```bash
pytest tests/ -v
```

---

## Error Handling

The application incorporates explicit error handling and validation routines across all endpoints to prevent unhandled exceptions or silent failures:

- **Missing Uploads**: Upload endpoints verify the presence of the `file` multipart key and reject empty requests with HTTP 400 and clear error messages.
- **Empty / Zero-Byte Files**: Files with zero bytes are caught before decoding and returned with explicit notifications.
- **Unsupported Extensions**: File names are validated against permitted extensions (`.jpg`, `.jpeg`, `.png` for images; `.mp4`, `.avi`, `.mov` for videos).
- **Corrupted / Unreadable Buffers**: Image decoding via `cv2.imdecode` is checked for `None` outputs or invalid array shapes. If decoding fails, the user is notified with an HTTP 400 error.
- **Non-Parking / Zero-Detection Images**: When an image contains no detectable vehicles or slot configurations, the interface issues informative warning badges rather than displaying broken elements.
- **Invalid Polygon Configurations**: Malformed slot JSON configurations (less than 3 vertices, negative coordinates, or missing fields) fail validation checks in `validate_slot_config` and revert safely.

---

## Project Structure

```
smart-parking-vision/
├── app.py                     # Flask web server, REST API endpoints, and CLI runner
├── config.py                  # Project-wide configuration parameters and paths
├── requirements.txt           # Minimal Python library dependencies
├── README.md                  # Comprehensive project documentation
├── statement.md               # Project scope and problem statement
├── .gitignore                 # Version control ignore rules
│
├── src/                       # Core Computer Vision and processing modules
│   ├── __init__.py            # Package initialization
│   ├── preprocessing.py       # OpenCV digital image processing pipeline
│   ├── vehicle_detector.py    # YOLOv8 vehicle detection and class filtering
│   ├── parking_slots.py       # Polygon slot geometry and JSON configuration
│   ├── occupancy.py           # Pixel-level spatial overlap and occupancy classifier
│   ├── analytics.py           # Statistical aggregation and temporal tracking
│   ├── visualization.py       # Visual overlays, alpha-blending, and annotations
│   └── video_processor.py     # Frame extraction and video processing pipeline
│
├── templates/                 # Frontend Jinja2 HTML templates
│   ├── base.html              # Base layout with navbar and footer
│   ├── index.html             # Landing page with pipeline overview
│   ├── image_analysis.html    # Image upload and detection interface
│   ├── video_analysis.html    # Video analysis and temporal charts
│   ├── slot_config.html       # Interactive polygon slot editor
│   └── analytics.html         # Occupancy analytics and distribution charts
│
├── static/                    # Static frontend assets
│   ├── css/
│   │   └── style.css          # Dark-theme stylesheet
│   └── js/
│       ├── main.js            # Shared UI utilities and dropzone logic
│       ├── image_analysis.js  # Image analysis and preprocessing handlers
│       ├── video_analysis.js  # Video processing handler and charts
│       ├── slot_config.js     # Canvas polygon drawing and slot editor
│       └── analytics.js       # Analytics dashboard visualization logic
│
├── config/                    # Calibrated slot configuration files
│   ├── parking_slots.json     # Default 12-slot parking configuration
│   └── vehicles_slots.json    # Alternative camera viewpoint configuration
│
├── input/                     # Input directory for sample media
│   ├── README.md              # Input guidelines and format documentation
│   ├── sample_parking.jpg     # Calibrated test parking lot image
│   └── sample_vehicles.jpg    # Verified sample vehicle image
│
├── output/                    # Destination directory for annotated outputs
│   └── .gitkeep               # Directory placeholder
│
├── tests/                     # Automated Pytest suite
│   ├── __init__.py            # Test package initialization
│   ├── test_preprocessing.py  # Unit tests for image preprocessing
│   ├── test_parking_slots.py  # Unit tests for polygon slot management
│   ├── test_occupancy.py      # Unit tests for spatial overlap logic
│   ├── test_analytics.py      # Unit tests for statistics and analytics
│   ├── test_validation.py     # Unit tests for input validation and edge cases
│   └── test_app.py            # Integration tests for Flask routes and APIs
│
└── docs/                      # Architectural specifications and diagrams
    ├── architecture.md        # System architecture specification
    ├── workflow.md            # Execution workflow documentation
    ├── use-case.md            # System use case models
    ├── sequence-diagram.md    # Component sequence diagrams
    ├── component-diagram.md   # System component diagrams
    └── cv-pipeline.md         # Detailed computer vision pipeline documentation
```

---

## How to Run

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 installed
- Git
- Internet access on first run (for automatic download of `yolov8n.pt` weights, ~6.2 MB)

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Aryan547/smart-parking-cv.git
   cd smart-parking-cv
   ```

2. **Create a Virtual Environment**:
   - On Windows:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - On Linux / macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

### Optional Command-Line Arguments
The application entry point `app.py` supports the following CLI options:
```bash
python app.py --host 127.0.0.1 --port 5000 --debug --model yolov8n.pt
```
- `--host`: Network interface to bind to (default: `127.0.0.1`).
- `--port`: Network port to listen on (default: `5000`).
- `--debug`: Enable Flask debug mode with auto-reloading (default: `False`).
- `--model`: Path or identifier for the YOLO model weights (default: `yolov8n.pt`).

### Running Automated Tests
To run all 104 unit and integration tests:
```bash
pytest tests/ -v
```

---

## Future Enhancements

The current implementation processes uploaded images and recorded video files. Planned future extensions include:
- **Automatic Parking Slot Detection**: Implementing Hough Line Transform and morphological line segment grouping to automatically detect parking stall markings without requiring manual polygon configuration.
- **Live CCTV / RTSP Stream Ingestion**: Direct network video stream ingestion (RTSP/HLS) allowing live surveillance feeds to be polled at set frame rates.
- **Hardware Acceleration**: Integrating TensorRT or CUDA GPU inference pipelines for processing high-framerate multi-camera video streams.
- **Historical Occupancy Tracking**: Integrating a persistent time-series database to track and query historical utilization patterns, peak parking hours, and turnover frequency.
- **Automated Alerts**: Email or webhook alerts notifying lot managers when facility capacity exceeds designated occupancy thresholds.
- **Automated License Plate Recognition (ALPR)**: Secondary OCR pipeline to associate vehicle license plates with specific occupied slot identifiers.
- **Multi-Floor Facility Mapping**: Hierarchical slot configuration schema supporting multi-level parking structures.

---

## Academic Project Context

This project was developed as a comprehensive Computer Vision submission demonstrating practical engineering principles:
- **Digital Image Processing**: Implementing foundational spatial transformations and feature extraction via OpenCV.
- **Deep Learning Application**: Applying pre-trained single-shot object detection architectures to practical domain problems.
- **Computational Geometry**: Designing spatial intersection algorithms using polygon rasterization masks to solve real-world perspective distortion challenges.
- **System Architecture**: Structuring clean separation between backend algorithmic logic, RESTful API services, and user interfaces.
- **Quality Assurance**: Validating software reliability through 104 automated unit and integration tests.

---

## Author

**Aryan Agrawal**  
B.Tech Computer Science and Engineering (Artificial Intelligence & Machine Learning)  
VIT Bhopal University
