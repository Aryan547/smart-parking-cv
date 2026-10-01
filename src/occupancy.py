import cv2
import numpy as np

def calculate_bbox_iou(bbox1: tuple, bbox2: tuple) -> float:
    """IoU between two bounding boxes (x1,y1,x2,y2)."""
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    iou = intersection_area / float(area1 + area2 - intersection_area)
    return iou

def calculate_overlap_ratio(detection_bbox: list, slot_polygon: list[list[int]], threshold: float = 0.3) -> float:
    """Calculate how much of the detection bbox overlaps with the slot polygon using genuine OpenCV."""
    x1, y1, x2, y2 = map(int, detection_bbox)
    
    # Calculate bounding box area
    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area == 0:
        return 0.0
        
    # Determine canvas bounds to enclose both the bbox and the polygon
    max_x = max(x2, max(p[0] for p in slot_polygon))
    max_y = max(y2, max(p[1] for p in slot_polygon))
    
    canvas_shape = (max_y + 10, max_x + 10)
    
    poly_mask = np.zeros(canvas_shape, dtype=np.uint8)
    cv2.fillPoly(poly_mask, [np.array(slot_polygon, np.int32)], 255)
    
    bbox_mask = np.zeros(canvas_shape, dtype=np.uint8)
    cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
    
    # Calculate overlap using bitwise_and
    overlap = cv2.bitwise_and(poly_mask, bbox_mask)
    overlap_area = cv2.countNonZero(overlap)
    
    # Return ratio of overlap area to bbox area (how much of the vehicle is in the slot)
    return overlap_area / float(bbox_area)

def check_vehicle_in_slot(detection: dict, slot: dict, threshold: float = 0.3) -> bool:
    """Check if a detection is in a slot using overlap ratio."""
    overlap = calculate_overlap_ratio(detection['bbox'], slot['points'], threshold)
    return overlap >= threshold

def classify_occupancy(detections: list[dict], slots: list[dict], threshold: float = 0.3) -> dict:
    """For each slot, determine if occupied."""
    result = {
        'slot_statuses': {},
        'slot_vehicles': {},
        'occupied_count': 0,
        'available_count': 0,
        'total_slots': len(slots)
    }
    
    for slot in slots:
        slot_id = slot['id']
        result['slot_statuses'][slot_id] = 'available'
        result['slot_vehicles'][slot_id] = None
        
        for det in detections:
            if check_vehicle_in_slot(det, slot, threshold):
                result['slot_statuses'][slot_id] = 'occupied'
                result['slot_vehicles'][slot_id] = det
                break # Move to next slot once occupied
                
    # Tally up
    result['occupied_count'] = sum(1 for status in result['slot_statuses'].values() if status == 'occupied')
    result['available_count'] = result['total_slots'] - result['occupied_count']
    
    return result

def get_occupancy_percentage(occupied: int, total: int) -> float:
    """Calculate percentage."""
    if total == 0:
        return 0.0
    return (occupied / float(total)) * 100.0
