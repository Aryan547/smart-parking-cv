import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics import (
    calculate_statistics, calculate_occupancy_over_time,
    aggregate_video_statistics, format_statistics
)


@pytest.fixture
def sample_occupancy_result():
    """Returns a sample occupancy result dict matching classify_occupancy output."""
    return {
        'slot_statuses': {1: 'occupied', 2: 'available', 3: 'occupied', 4: 'available'},
        'slot_vehicles': {1: {'class_name': 'car'}, 2: None, 3: {'class_name': 'truck'}, 4: None},
        'occupied_count': 2,
        'available_count': 2,
        'total_slots': 4
    }


@pytest.fixture
def sample_detections():
    """Returns sample detection dicts."""
    return [
        {'class_name': 'car', 'confidence': 0.9, 'bbox': [100, 100, 200, 200]},
        {'class_name': 'truck', 'confidence': 0.85, 'bbox': [300, 100, 400, 200]}
    ]


def test_calculate_statistics_basic(sample_occupancy_result, sample_detections):
    """Test basic statistics calculation."""
    stats = calculate_statistics(sample_occupancy_result, sample_detections)
    assert stats['total_slots'] == 4
    assert stats['occupied'] == 2
    assert stats['available'] == 2
    assert stats['occupancy_percentage'] == 50.0
    assert stats['vehicle_count'] == 2


def test_calculate_statistics_empty():
    """Test statistics calculation with empty inputs."""
    empty_result = {
        'slot_statuses': {},
        'slot_vehicles': {},
        'occupied_count': 0,
        'available_count': 0,
        'total_slots': 0
    }
    stats = calculate_statistics(empty_result, [])
    assert stats['total_slots'] == 0
    assert stats['occupied'] == 0
    assert stats['available'] == 0
    assert stats['occupancy_percentage'] == 0.0
    assert stats['vehicle_count'] == 0


def test_calculate_statistics_all_occupied():
    """Test statistics calculation when all slots occupied."""
    result = {
        'slot_statuses': {1: 'occupied', 2: 'occupied'},
        'occupied_count': 2,
        'available_count': 0,
        'total_slots': 2
    }
    detections = [
        {'class_name': 'car', 'confidence': 0.9, 'bbox': [0, 0, 10, 10]},
        {'class_name': 'car', 'confidence': 0.9, 'bbox': [20, 20, 30, 30]}
    ]
    stats = calculate_statistics(result, detections)
    assert stats['occupied'] == 2
    assert stats['occupancy_percentage'] == 100.0


def test_calculate_statistics_all_available():
    """Test statistics calculation when no slots occupied."""
    result = {
        'slot_statuses': {1: 'available', 2: 'available'},
        'occupied_count': 0,
        'available_count': 2,
        'total_slots': 2
    }
    stats = calculate_statistics(result, [])
    assert stats['occupied'] == 0
    assert stats['occupancy_percentage'] == 0.0


def test_calculate_statistics_vehicles_by_type(sample_occupancy_result, sample_detections):
    """Test that vehicle type breakdown is correct."""
    stats = calculate_statistics(sample_occupancy_result, sample_detections)
    assert stats['vehicles_by_type']['car'] == 1
    assert stats['vehicles_by_type']['truck'] == 1


def test_calculate_occupancy_over_time():
    """Test calculating occupancy timeline from frame results."""
    frame_results = [
        {'frame': 0, 'occupancy': {'occupied_count': 1, 'available_count': 1, 'total_slots': 2}},
        {'frame': 5, 'occupancy': {'occupied_count': 2, 'available_count': 0, 'total_slots': 2}},
        {'frame': 10, 'occupancy': {'occupied_count': 0, 'available_count': 2, 'total_slots': 2}}
    ]
    timeline = calculate_occupancy_over_time(frame_results)
    assert len(timeline) == 3
    assert timeline[0]['occupied'] == 1
    assert timeline[1]['occupied'] == 2
    assert timeline[2]['occupied'] == 0
    assert timeline[1]['percentage'] == 100.0


def test_calculate_occupancy_over_time_empty():
    """Test occupancy timeline with empty input."""
    timeline = calculate_occupancy_over_time([])
    assert timeline == []


def test_calculate_occupancy_over_time_single_frame():
    """Test occupancy timeline with single frame."""
    frame_results = [
        {'frame': 0, 'occupancy': {'occupied_count': 1, 'available_count': 0, 'total_slots': 1}}
    ]
    timeline = calculate_occupancy_over_time(frame_results)
    assert len(timeline) == 1
    assert timeline[0]['percentage'] == 100.0


def test_aggregate_video_statistics():
    """Test aggregating statistics across video frames."""
    frame_stats = [
        {'occupied': 1, 'frame': 0},
        {'occupied': 2, 'frame': 5},
        {'occupied': 0, 'frame': 10}
    ]
    agg = aggregate_video_statistics(frame_stats)
    assert agg['total_frames'] == 3
    assert agg['max_occupancy'] == 2
    assert agg['min_occupancy'] == 0
    assert abs(agg['avg_occupancy'] - 1.0) < 0.01


def test_aggregate_video_statistics_empty():
    """Test video statistics aggregation with empty input."""
    agg = aggregate_video_statistics([])
    assert agg == {}


def test_format_statistics():
    """Test statistics formatting rounds floats."""
    stats = {'occupancy_percentage': 33.333333, 'vehicle_count': 5}
    formatted = format_statistics(stats)
    assert formatted['occupancy_percentage'] == 33.33
    assert formatted['vehicle_count'] == 5


def test_format_statistics_edge_cases():
    """Test statistics formatting with edge values."""
    stats = {'occupancy_percentage': 0.0, 'avg_occupancy': 100.0}
    formatted = format_statistics(stats)
    assert formatted['occupancy_percentage'] == 0.0
    assert formatted['avg_occupancy'] == 100.0
