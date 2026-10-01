import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import validate_image, to_grayscale, resize_image
from src.parking_slots import validate_slot_config, calculate_polygon_area
from src.occupancy import classify_occupancy, calculate_bbox_iou
from src.analytics import calculate_statistics
from src.video_processor import validate_video


def test_invalid_image_type_string():
    """Test passing string instead of image to validate_image."""
    assert validate_image("not_an_image") is False


def test_invalid_image_type_int():
    """Test passing integer instead of image to validate_image."""
    assert validate_image(12345) is False


def test_large_image_resize():
    """Test resizing a massive image."""
    large_img = np.zeros((5000, 5000, 3), dtype=np.uint8)
    resized = resize_image(large_img, max_width=800, max_height=600)
    assert resized.shape[1] <= 800
    assert resized.shape[0] <= 600


def test_grayscale_single_channel():
    """Test passing a single channel image to to_grayscale."""
    single_channel = np.zeros((100, 100), dtype=np.uint8)
    gray = to_grayscale(single_channel)
    assert len(gray.shape) == 2


def test_slot_config_with_extra_fields():
    """Test validating slot config with extraneous fields still passes if core structure valid."""
    config = {
        "slots": [{"id": 1, "points": [[0, 0], [10, 0], [10, 10], [0, 10]], "label": "A1"}],
        "extra_field": "should_be_ignored",
        "metadata": {"version": 1.0}
    }
    assert validate_slot_config(config) is True


def test_detection_with_zero_confidence():
    """Test handling detection with zero confidence in occupancy classification."""
    slots = [{"id": 1, "points": [[0, 0], [100, 0], [100, 100], [0, 100]], "label": "A1"}]
    detections = [{"bbox": [10, 10, 90, 90], "confidence": 0.0, "class_name": "car"}]
    result = classify_occupancy(detections, slots)
    assert result['slot_statuses'][1] == 'occupied'


def test_empty_video_path():
    """Test validating empty string for video path."""
    assert validate_video("") is False


def test_invalid_video_path():
    """Test validating non-existent video path."""
    assert validate_video("non_existent_video_path.mp4") is False


def test_occupancy_with_overlapping_detections():
    """Test multiple detections in same slot - slot should still be occupied."""
    slots = [{"id": 1, "points": [[0, 0], [100, 0], [100, 100], [0, 100]], "label": "A1"}]
    detections = [
        {"bbox": [10, 10, 40, 40], "class_name": "car", "confidence": 0.9},
        {"bbox": [60, 60, 90, 90], "class_name": "car", "confidence": 0.9}
    ]
    result = classify_occupancy(detections, slots)
    assert result['slot_statuses'][1] == 'occupied'
    assert result['occupied_count'] == 1


def test_statistics_with_large_numbers():
    """Test statistics with large number of slots and detections."""
    occupancy_result = {
        'occupied_count': 100,
        'available_count': 0,
        'total_slots': 100,
        'slot_statuses': {i: 'occupied' for i in range(100)}
    }
    detections = [{'class_name': 'car', 'confidence': 0.9, 'bbox': [0, 0, 10, 10]} for _ in range(100)]
    stats = calculate_statistics(occupancy_result, detections)
    assert stats['total_slots'] == 100
    assert stats['occupied'] == 100
    assert stats['occupancy_percentage'] == 100.0
    assert stats['vehicle_count'] == 100


def test_bbox_iou_zero_area():
    """Test IoU calculation with zero-area bounding box."""
    box1 = (0, 0, 0, 0)
    box2 = (10, 10, 20, 20)
    iou = calculate_bbox_iou(box1, box2)
    assert iou == 0.0


def test_polygon_area_zero():
    """Test area calculation for a degenerate polygon."""
    poly = [[0, 0], [0, 0], [0, 0]]
    area = calculate_polygon_area(poly)
    assert area == 0.0


def test_validate_image_with_list():
    """Test validate_image with list input."""
    assert validate_image([1, 2, 3]) is False


def test_classify_occupancy_high_threshold():
    """Test that high threshold makes detection not match."""
    slots = [{"id": 1, "points": [[0, 0], [100, 0], [100, 100], [0, 100]], "label": "A1"}]
    # Detection barely overlaps slot
    detections = [{"bbox": [90, 90, 150, 150], "class_name": "car", "confidence": 0.9}]
    result = classify_occupancy(detections, slots, threshold=0.9)
    assert result['slot_statuses'][1] == 'available'
