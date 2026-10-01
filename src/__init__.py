"""
Smart Parking Space Detection System.
Provides modules for vehicle detection, occupancy classification, and analytics.
"""
from .preprocessing import load_image, preprocess_pipeline
from .vehicle_detector import load_model, detect_vehicles
from .parking_slots import load_slot_config, save_slot_config
from .occupancy import classify_occupancy
from .analytics import calculate_statistics
from .visualization import annotate_parking_image
from .video_processor import process_video
