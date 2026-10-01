import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.occupancy import (
    calculate_bbox_iou, calculate_overlap_ratio, check_vehicle_in_slot,
    classify_occupancy, get_occupancy_percentage
)


@pytest.fixture
def slot_polygon():
    """Returns a simple square polygon."""
    return [[100, 100], [200, 100], [200, 200], [100, 200]]


@pytest.fixture
def slot_dict(slot_polygon):
    """Returns a slot dict."""
    return {"id": 1, "points": slot_polygon, "label": "A1"}


@pytest.fixture
def slot_list():
    """Returns a list of slots."""
    return [
        {"id": 1, "points": [[100, 100], [200, 100], [200, 200], [100, 200]], "label": "A1"},
        {"id": 2, "points": [[300, 100], [400, 100], [400, 200], [300, 200]], "label": "A2"}
    ]


def test_calculate_bbox_iou_perfect_overlap():
    """Test IoU with perfect overlap."""
    box1 = (0, 0, 100, 100)
    box2 = (0, 0, 100, 100)
    assert calculate_bbox_iou(box1, box2) == 1.0


def test_calculate_bbox_iou_no_overlap():
    """Test IoU with no overlap."""
    box1 = (0, 0, 100, 100)
    box2 = (200, 200, 300, 300)
    assert calculate_bbox_iou(box1, box2) == 0.0


def test_calculate_bbox_iou_partial_overlap():
    """Test IoU with partial overlap."""
    box1 = (0, 0, 100, 100)
    box2 = (50, 50, 150, 150)
    iou = calculate_bbox_iou(box1, box2)
    assert abs(iou - 1.0/7.0) < 0.01


def test_calculate_bbox_iou_contained():
    """Test IoU when one box is inside another."""
    box1 = (0, 0, 100, 100)
    box2 = (25, 25, 75, 75)
    iou = calculate_bbox_iou(box1, box2)
    assert abs(iou - 0.25) < 0.01


def test_calculate_overlap_ratio_full(slot_polygon):
    """Test overlap ratio when detection is fully inside slot."""
    detection_bbox = [120, 120, 180, 180]
    ratio = calculate_overlap_ratio(detection_bbox, slot_polygon)
    assert ratio > 0.9  # Should be very close to 1.0


def test_calculate_overlap_ratio_none(slot_polygon):
    """Test overlap ratio when detection is completely outside slot."""
    detection_bbox = [0, 0, 50, 50]
    ratio = calculate_overlap_ratio(detection_bbox, slot_polygon)
    assert ratio == 0.0


def test_calculate_overlap_ratio_partial(slot_polygon):
    """Test overlap ratio when detection is partially inside slot."""
    detection_bbox = [150, 150, 250, 250]
    ratio = calculate_overlap_ratio(detection_bbox, slot_polygon)
    assert ratio > 0.0 and ratio < 1.0


def test_check_vehicle_in_slot_true(slot_dict):
    """Test checking vehicle with high overlap."""
    detection = {"bbox": [110, 110, 190, 190], "class_name": "car", "confidence": 0.9}
    assert check_vehicle_in_slot(detection, slot_dict) is True


def test_check_vehicle_in_slot_false(slot_dict):
    """Test checking vehicle with no overlap."""
    detection = {"bbox": [0, 0, 50, 50], "class_name": "car", "confidence": 0.9}
    assert check_vehicle_in_slot(detection, slot_dict) is False


def test_check_vehicle_in_slot_threshold(slot_dict):
    """Test checking vehicle with different thresholds."""
    detection = {"bbox": [150, 150, 250, 250], "class_name": "car", "confidence": 0.9}
    # Partial overlap
    assert check_vehicle_in_slot(detection, slot_dict, threshold=0.1) is True
    assert check_vehicle_in_slot(detection, slot_dict, threshold=0.9) is False


def test_classify_occupancy_all_empty(slot_list):
    """Test classifying occupancy with no detections."""
    result = classify_occupancy([], slot_list)
    assert result['occupied_count'] == 0
    assert result['available_count'] == 2
    assert all(s == 'available' for s in result['slot_statuses'].values())


def test_classify_occupancy_all_occupied(slot_list):
    """Test classifying occupancy with all slots occupied."""
    detections = [
        {"bbox": [120, 120, 180, 180], "class_name": "car", "confidence": 0.9},
        {"bbox": [320, 120, 380, 180], "class_name": "car", "confidence": 0.9}
    ]
    result = classify_occupancy(detections, slot_list)
    assert result['occupied_count'] == 2
    assert result['available_count'] == 0
    assert all(s == 'occupied' for s in result['slot_statuses'].values())


def test_classify_occupancy_mixed(slot_list):
    """Test classifying occupancy with mixed availability."""
    detections = [
        {"bbox": [120, 120, 180, 180], "class_name": "car", "confidence": 0.9}
    ]
    result = classify_occupancy(detections, slot_list)
    assert result['slot_statuses'][1] == 'occupied'
    assert result['slot_statuses'][2] == 'available'
    assert result['occupied_count'] == 1
    assert result['available_count'] == 1


def test_classify_occupancy_no_slots():
    """Test classifying occupancy with empty slots."""
    detections = [{"bbox": [120, 120, 180, 180], "class_name": "car", "confidence": 0.9}]
    result = classify_occupancy(detections, [])
    assert result['total_slots'] == 0
    assert result['occupied_count'] == 0


def test_classify_occupancy_no_detections(slot_list):
    """Test classifying occupancy with slots but no detections."""
    result = classify_occupancy([], slot_list)
    assert result['occupied_count'] == 0
    assert result['available_count'] == 2


def test_get_occupancy_percentage_zero():
    """Test occupancy percentage for 0 occupied slots."""
    assert get_occupancy_percentage(0, 10) == 0.0


def test_get_occupancy_percentage_full():
    """Test occupancy percentage for all occupied slots."""
    assert get_occupancy_percentage(10, 10) == 100.0


def test_get_occupancy_percentage_half():
    """Test occupancy percentage for half occupied slots."""
    assert get_occupancy_percentage(5, 10) == 50.0


def test_get_occupancy_percentage_zero_total():
    """Test occupancy percentage handling division by zero."""
    assert get_occupancy_percentage(0, 0) == 0.0
