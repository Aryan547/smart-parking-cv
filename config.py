import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload and output directories
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'input')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
CONFIG_FOLDER = os.path.join(BASE_DIR, 'config')

# Ensure directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, CONFIG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

# YOLO Configuration
YOLO_MODEL_PATH = 'yolov8n.pt'
DEFAULT_CONFIDENCE = 0.25

# Parking Configuration
DEFAULT_SLOT_CONFIG = os.path.join(CONFIG_FOLDER, 'parking_slots.json')
DEFAULT_OCCUPANCY_THRESHOLD = 0.3

# Image Processing
MAX_IMAGE_WIDTH = 1280
MAX_IMAGE_HEIGHT = 720
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

# Video Processing
VIDEO_FRAME_INTERVAL = 5  # Process every Nth frame
VIDEO_OUTPUT_CODEC = 'mp4v'

# Flask Configuration
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
FLASK_DEBUG = True
SECRET_KEY = 'smart-parking-vision-secret-key'
