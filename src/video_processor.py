import os
import cv2
import numpy as np

# Ensure required imports are available
from .vehicle_detector import detect_vehicles
from .occupancy import classify_occupancy
from .analytics import calculate_statistics, aggregate_video_statistics
from .visualization import annotate_parking_image, create_stats_overlay

def validate_video(file_path: str) -> bool:
    """Check video file exists and is readable by OpenCV."""
    if not os.path.exists(file_path):
        return False
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret

def get_video_info(file_path: str) -> dict:
    """Return fps, frame_count, width, height, duration."""
    if not validate_video(file_path):
        raise ValueError(f"Invalid or unreadable video file: {file_path}")
        
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    duration = frame_count / fps if fps > 0 else 0.0
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }

def process_video(video_path: str, slots: list[dict], detector_model, confidence: float = 0.25, threshold: float = 0.3, frame_interval: int = 5, output_path: str = None, progress_callback=None) -> dict:
    """Process video frame by frame."""
    if not validate_video(video_path):
        raise ValueError("Invalid video file.")
        
    cap = cv2.VideoCapture(video_path)
    info = get_video_info(video_path)
    total_frames = info['frame_count']
    fps = info['fps']
    width = info['width']
    height = info['height']
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps / frame_interval if frame_interval > 0 else fps, (width, height))
        
    frame_results = []
    frame_stats_list = []
    frames_processed = 0
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % frame_interval == 0:
            detections = detect_vehicles(detector_model, frame, confidence)
            occupancy = classify_occupancy(detections, slots, threshold)
            stats = calculate_statistics(occupancy, detections)
            
            annotated = annotate_parking_image(frame, slots, occupancy['slot_statuses'], detections)
            annotated = create_stats_overlay(annotated, stats)
            
            if writer:
                writer.write(annotated)
                
            stats['frame'] = current_frame
            frame_stats_list.append(stats)
            
            frame_results.append({
                'frame': current_frame,
                'detections': detections,
                'occupancy': occupancy
            })
            frames_processed += 1
            
            if progress_callback:
                progress_callback(current_frame, total_frames)
                
        current_frame += 1
        
    cap.release()
    if writer:
        writer.release()
        
    video_stats = aggregate_video_statistics(frame_stats_list)
    
    return {
        'frame_results': frame_results,
        'video_stats': video_stats,
        'output_path': output_path,
        'frames_processed': frames_processed
    }

def extract_frame(video_path: str, frame_number: int) -> np.ndarray:
    """Extract a specific frame."""
    if not validate_video(video_path):
        raise ValueError("Invalid video file.")
        
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError(f"Failed to extract frame {frame_number}")
        
    return frame
