import json
import os
import cv2
import numpy as np

def load_slot_config(config_path: str) -> list[dict]:
    """Load from JSON file, validate format."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        data = json.load(f)
    if not validate_slot_config(data):
        raise ValueError("Invalid slot configuration format.")
    return data.get('slots', [])

def save_slot_config(slots: list[dict], config_path: str) -> bool:
    """Save to JSON file."""
    config = {'slots': slots}
    if not validate_slot_config(config):
        return False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception:
        return False

def validate_slot_config(config: dict) -> bool:
    """Validate the config structure has 'slots' key with list of valid slots."""
    if not isinstance(config, dict) or 'slots' not in config:
        return False
    for slot in config['slots']:
        if not all(k in slot for k in ('id', 'points', 'label')):
            return False
        if not isinstance(slot['id'], int) or not isinstance(slot['label'], str):
            return False
        if not validate_polygon(slot['points']):
            return False
    return True

def validate_polygon(points: list[list[int]]) -> bool:
    """Check polygon has at least 3 points, all are [x,y] with non-negative ints."""
    if not isinstance(points, list) or len(points) < 3:
        return False
    for pt in points:
        if not isinstance(pt, list) or len(pt) != 2:
            return False
        if not isinstance(pt[0], int) or not isinstance(pt[1], int):
            return False
        if pt[0] < 0 or pt[1] < 0:
            return False
    return True

def create_default_config() -> dict:
    """Return empty config with 'slots': []"""
    return {'slots': []}

def polygon_to_bbox(points: list[list[int]]) -> tuple[int, int, int, int]:
    """Convert polygon points to bounding box (x_min, y_min, x_max, y_max)."""
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    return (min(xs), min(ys), max(xs), max(ys))

def calculate_polygon_area(points: list[list[int]]) -> float:
    """Calculate polygon area using shoelace formula."""
    x = [pt[0] for pt in points]
    y = [pt[1] for pt in points]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(area)

def draw_slots(image: np.ndarray, slots: list[dict], statuses: dict[int, str] = None) -> np.ndarray:
    """Draw slot polygons on image."""
    img_copy = image.copy()
    if statuses is None:
        statuses = {}
        
    for slot in slots:
        slot_id = slot['id']
        pts = np.array(slot['points'], np.int32).reshape((-1, 1, 2))
        
        status = statuses.get(slot_id, 'unknown')
        if status == 'available':
            color = (0, 255, 0)
        elif status == 'occupied':
            color = (0, 0, 255)
        else:
            color = (0, 255, 255) # Yellow for unknown
            
        cv2.polylines(img_copy, [pts], True, color, 2)
        
        # Draw label
        x_min, y_min, _, _ = polygon_to_bbox(slot['points'])
        cv2.putText(img_copy, str(slot_id), (x_min, max(y_min - 5, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
    return img_copy
