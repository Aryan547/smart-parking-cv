# Computer Vision Pipeline

The CV Pipeline is the analytical core of the Smart Parking Space Detection system. It combines classic image processing techniques with modern deep learning.

## Pipeline Stages

### 1. Image Validation and Loading
- **Library Used**: OpenCV (`cv2.imread`, `cv2.VideoCapture`)
- **Action**: Load input media and verify integrity. Resize the image while maintaining aspect ratio to standardize processing and reduce computational load.

### 2. Preprocessing
Before passing images to detection algorithms, enhancing them can yield more reliable results, especially in poor lighting or noisy environments.
- **Grayscale Conversion (`cv2.cvtColor`)**: Simplifies the image for certain analytical tasks by removing color channels.
- **Gaussian Blur (`cv2.GaussianBlur`)**: Reduces high-frequency noise that might confuse edge detectors or bounding box models.
- **CLAHE (`cv2.createCLAHE`)**: Contrast Limited Adaptive Histogram Equalization. Used to improve visibility in overexposed or shadowed areas of the parking lot.
- **Canny Edge Detection & Morphology (`cv2.Canny`, `cv2.dilate`)**: While YOLO works on raw RGB frames, edge detection can optionally be used to dynamically detect parking lines in future enhancements.

### 3. Vehicle Detection
- **Library Used**: Ultralytics YOLOv8
- **Action**: The YOLOv8n (nano) model performs a forward pass on the input image. It returns bounding boxes `[x1, y1, x2, y2]` and confidence scores for detected classes (cars, trucks, buses, motorcycles).
- **Why YOLO**: It provides an excellent balance of speed and accuracy, operating in real-time on standard hardware.

### 4. Spatial Matching (Polygon Overlap)
This is the crux of the occupancy logic.
- **Action**: The system converts YOLO bounding boxes into polygon formats. It then compares these against the predefined parking slot polygons from the JSON config.
- **OpenCV Operations**:
  - `cv2.fillPoly`: Used to create binary mask representations of parking slots.
  - Bitwise operations (`cv2.bitwise_and`): Used to calculate the exact pixel intersection between the vehicle bounding box mask and the parking slot mask.
- **Why this method?**: Standard bounding box intersection (IoU) assumes rectangles. Parking slots are often angled parallelograms. Pixel-masking using `fillPoly` allows for accurate spatial overlap calculation regardless of the slot's geometry.

### 5. Occupancy Classification
- **Action**: An Overlap Ratio is calculated: `Overlap Area (pixels) / Slot Area (pixels)`.
- If `Ratio >= Configured Threshold (e.g., 0.3)`, the slot is marked as `Occupied`. Otherwise, it is `Available`.

### 6. Visualization & Annotation
- **Library Used**: OpenCV
- **Action**: 
  - `cv2.polylines`: Draws the parking slot borders on the output image.
  - `cv2.fillPoly` (with alpha blending): Applies a transparent red overlay for occupied slots and green for free ones.
  - `cv2.putText`: Adds slot IDs, occupancy status, and vehicle labels directly onto the image.
