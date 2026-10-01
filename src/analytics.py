def calculate_statistics(occupancy_result: dict, detections: list[dict]) -> dict:
    """Comprehensive statistics including occupancy and vehicle types."""
    occupied = occupancy_result.get('occupied_count', 0)
    total_slots = occupancy_result.get('total_slots', 0)
    available = occupancy_result.get('available_count', 0)
    
    percentage = 0.0
    if total_slots > 0:
        percentage = (occupied / total_slots) * 100.0
        
    vehicles_by_type = {}
    for det in detections:
        cls_name = det.get('class_name', 'unknown')
        vehicles_by_type[cls_name] = vehicles_by_type.get(cls_name, 0) + 1
        
    return {
        'total_slots': total_slots,
        'occupied': occupied,
        'available': available,
        'occupancy_percentage': percentage,
        'vehicle_count': len(detections),
        'vehicles_by_type': vehicles_by_type
    }

def calculate_occupancy_over_time(frame_results: list[dict]) -> list[dict]:
    """Track occupancy across frames."""
    history = []
    for entry in frame_results:
        frame_idx = entry.get('frame', 0)
        occ_res = entry.get('occupancy', {})
        occ = occ_res.get('occupied_count', 0)
        avail = occ_res.get('available_count', 0)
        tot = occ_res.get('total_slots', 0)
        
        pct = (occ / tot * 100.0) if tot > 0 else 0.0
        
        history.append({
            'frame': frame_idx,
            'occupied': occ,
            'available': avail,
            'percentage': pct
        })
    return history

def aggregate_video_statistics(frame_stats: list[dict]) -> dict:
    """Aggregate statistics from video frames."""
    if not frame_stats:
        return {}
        
    total_frames = len(frame_stats)
    occupancies = [s['occupied'] for s in frame_stats]
    
    avg_occ = sum(occupancies) / total_frames if total_frames > 0 else 0
    max_occ = max(occupancies)
    min_occ = min(occupancies)
    
    peak_frame = next((s['frame'] for s in frame_stats if s['occupied'] == max_occ), 0)
    
    return {
        'avg_occupancy': avg_occ,
        'max_occupancy': max_occ,
        'min_occupancy': min_occ,
        'total_frames': total_frames,
        'peak_frame': peak_frame,
        'vehicles_detected_per_frame': avg_occ  # approximation based on occupancies
    }

def format_statistics(stats: dict) -> dict:
    """Format for display."""
    formatted = stats.copy()
    for k, v in formatted.items():
        if isinstance(v, float):
            formatted[k] = round(v, 2)
    return formatted
