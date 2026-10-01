import pytest
import numpy as np
import cv2
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parking_slots import (
    validate_polygon, polygon_to_bbox, calculate_polygon_area,
    validate_slot_config, save_slot_config, load_slot_config,
    create_default_config, draw_slots
)


@pytest.fixture
def valid_polygon():
    """Returns a valid rectangle polygon."""
    return [[0, 0], [100, 0], [100, 100], [0, 100]]


@pytest.fixture
def valid_slot(valid_polygon):
    """Returns a valid slot dict."""
    return {"id": 1, "points": valid_polygon, "label": "A1"}


@pytest.fixture
def valid_config(valid_slot):
    """Returns a valid slot config dict."""
    return {"slots": [valid_slot]}


def test_validate_polygon_valid(valid_polygon):
    """Test valid polygon validation."""
    assert validate_polygon(valid_polygon) is True


def test_validate_polygon_too_few_points():
    """Test polygon with too few points."""
    assert validate_polygon([[0, 0], [100, 0]]) is False


def test_validate_polygon_empty():
    """Test empty polygon."""
    assert validate_polygon([]) is False


def test_validate_polygon_negative_coords():
    """Test polygon with negative coordinates."""
    assert validate_polygon([[-1, 0], [100, 0], [100, 100], [0, 100]]) is False


def test_validate_polygon_invalid_format():
    """Test polygon with invalid point formatting."""
    assert validate_polygon([[0], [100, 0, 5], [100, 100]]) is False


def test_polygon_to_bbox():
    """Test converting polygon to bounding box."""
    poly = [[50, 100], [200, 100], [200, 300], [50, 300]]
    assert polygon_to_bbox(poly) == (50, 100, 200, 300)


def test_polygon_to_bbox_irregular():
    """Test irregular polygon to bounding box."""
    poly = [[10, 20], [50, 5], [80, 60], [30, 90]]
    assert polygon_to_bbox(poly) == (10, 5, 80, 90)


def test_calculate_polygon_area_square(valid_polygon):
    """Test area calculation for a square."""
    area = calculate_polygon_area(valid_polygon)
    assert area == 10000


def test_calculate_polygon_area_triangle():
    """Test area calculation for a triangle."""
    poly = [[0, 0], [100, 0], [50, 100]]
    area = calculate_polygon_area(poly)
    assert area == 5000


def test_validate_slot_config_valid(valid_config):
    """Test validating correct slot config."""
    assert validate_slot_config(valid_config) is True


def test_validate_slot_config_missing_slots():
    """Test validating config missing slots key."""
    config = {"not_slots": []}
    assert validate_slot_config(config) is False


def test_validate_slot_config_invalid_slot():
    """Test validating config with invalid slot (missing points and label)."""
    config = {"slots": [{"id": 1}]}
    assert validate_slot_config(config) is False


def test_validate_slot_config_empty():
    """Test validating empty config dict."""
    assert validate_slot_config({}) is False


def test_save_and_load_config(valid_config):
    """Test saving and loading slot configuration."""
    slots = valid_config["slots"]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=".") as temp_file:
        temp_path = temp_file.name

    try:
        save_slot_config(slots, temp_path)
        loaded_slots = load_slot_config(temp_path)
        assert len(loaded_slots) == len(slots)
        assert loaded_slots[0]["id"] == slots[0]["id"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_config_nonexistent():
    """Test loading non-existent config file."""
    with pytest.raises(FileNotFoundError):
        load_slot_config("nonexistent_config_file.json")


def test_create_default_config():
    """Test creating default config."""
    config = create_default_config()
    assert isinstance(config, dict)
    assert "slots" in config
    assert config["slots"] == []


def test_draw_slots_no_crash(valid_slot):
    """Test drawing slots on an image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    slots = [valid_slot]
    out_img = draw_slots(img, slots)
    assert out_img is not None
    assert out_img.shape == img.shape


def test_draw_slots_with_statuses(valid_slot):
    """Test drawing slots with occupancy statuses."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    slots = [valid_slot]
    statuses = {1: "occupied"}
    out_img = draw_slots(img, slots, statuses)
    assert out_img is not None
    assert out_img.shape == img.shape
