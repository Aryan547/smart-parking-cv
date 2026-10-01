import cv2
import numpy as np

def get_status_color(status: str) -> tuple:
    """Return BGR color for status."""
    if status == 'available':
        return (0, 255, 0) # Green
    elif status == 'occupied':
        return (0, 0, 255) # Red
    return (0, 255, 255) # Yellow

def annotate_parking_image(image: np.ndarray, slots: list[dict], slot_statuses: dict, detections: list[dict]) -> np.ndarray:
    """Complete annotation: transparent slots, bboxes, labels, status."""
    overlay = image.copy()
    output = image.copy()
    
    # Draw slots as semi-transparent polygons
    if slots:
        for slot in slots:
            slot_id = slot['id']
            pts = np.array(slot['points'], np.int32).reshape((-1, 1, 2))
            status = slot_statuses.get(slot_id, 'unknown')
            color = get_status_color(status)
            
            cv2.fillPoly(overlay, [pts], color)
            
        # Apply semi-transparency
        cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)
        
        # Draw borders and text
        for slot in slots:
            slot_id = slot['id']
            pts = np.array(slot['points'], np.int32).reshape((-1, 1, 2))
            status = slot_statuses.get(slot_id, 'unknown')
            color = get_status_color(status)
            label = slot.get('label', f"P{slot_id}")
            
            cv2.polylines(output, [pts], True, color, 2)
            
            xs = [p[0] for p in slot['points']]
            ys = [p[1] for p in slot['points']]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))
            
            display_text = f"{label}: {status.upper()}"
            (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(output, (cx - tw // 2 - 4, cy - th // 2 - 4), (cx + tw // 2 + 4, cy + th // 2 + 4), (20, 20, 20), -1)
            cv2.putText(output, display_text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
    # Draw detections (vehicles)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det.get('class_name', det.get('class', 'vehicle'))
        conf = det.get('confidence', 0.0)
        
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 140, 0), 2)
        tag = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(output, (x1, max(y1 - th - 6, 0)), (x1 + tw + 6, max(y1, th + 6)), (255, 140, 0), -1)
        cv2.putText(output, tag, (x1 + 3, max(y1 - 4, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        
    return output

def create_stats_overlay(image: np.ndarray, stats: dict) -> np.ndarray:
    """Draw semi-transparent statistics panel on top-right of image."""
    overlay = image.copy()
    output = image.copy()
    
    h, w = image.shape[:2]
    panel_w, panel_h = 250, 150
    x_offset, y_offset = w - panel_w - 20, 20
    
    cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
    
    occ = stats.get('occupied', 0)
    avail = stats.get('available', 0)
    pct = stats.get('occupancy_percentage', 0.0)
    
    texts = [
        f"Occupied: {occ}",
        f"Available: {avail}",
        f"Usage: {pct:.1f}%"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(output, text, (x_offset + 10, y_offset + 30 + (i * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
    return output

def create_comparison_image(original: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    """Side-by-side comparison."""
    h1, w1 = original.shape[:2]
    h2, w2 = annotated.shape[:2]
    
    target_h = max(h1, h2)
    
    w1_new = int(w1 * (target_h / h1)) if h1 > 0 else 0
    w2_new = int(w2 * (target_h / h2)) if h2 > 0 else 0
    
    orig_res = cv2.resize(original, (w1_new, target_h))
    ann_res = cv2.resize(annotated, (w2_new, target_h))
    
    return np.hstack((orig_res, ann_res))
