import pytest
import io
import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
import config as app_config


@pytest.fixture
def client():
    """Create test client for Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image_bytes():
    """Create a synthetic test image as JPEG bytes."""
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 120
    # Draw simple parking lot lines
    cv2.rectangle(img, (80, 200), (200, 350), (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return io.BytesIO(buffer.tobytes())


@pytest.fixture
def bus_image_bytes():
    """Create a realistic image with a vehicle or copy from ultralytics assets if available."""
    try:
        import ultralytics
        bus_path = os.path.join(os.path.dirname(ultralytics.__file__), 'assets', 'bus.jpg')
        if os.path.exists(bus_path):
            with open(bus_path, 'rb') as f:
                return io.BytesIO(f.read())
    except Exception:
        pass
    # Fallback to synthetic image
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 120
    _, buffer = cv2.imencode('.jpg', img)
    return io.BytesIO(buffer.tobytes())


def test_home_page(client):
    """Test home page loads."""
    res = client.get('/')
    assert res.status_code == 200
    assert b'ParkVision AI' in res.data


def test_image_analysis_page(client):
    """Test image analysis page loads."""
    res = client.get('/image-analysis')
    assert res.status_code == 200
    assert b'Image Analysis' in res.data


def test_health_check(client):
    """Test health check API."""
    res = client.get('/api/health')
    assert res.status_code == 200
    data = res.get_json()
    assert data['status'] == 'ok'


def test_analyze_image_valid(client, sample_image_bytes):
    """Test analyzing a valid image end-to-end."""
    data = {
        'file': (sample_image_bytes, 'test.jpg'),
        'config_name': 'parking_slots.json',
        'conf_threshold': '0.25',
        'occ_threshold': '0.3'
    }
    res = client.post('/api/analyze-image', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    resp = res.get_json()
    assert resp['success'] is True
    assert 'statistics' in resp
    assert 'images' in resp
    assert 'original' in resp['images']
    assert 'annotated' in resp['images']
    assert 'slot_status' in resp
    assert 'detections' in resp


def test_analyze_image_with_vehicles(client, bus_image_bytes):
    """Test analyzing an image containing vehicles."""
    data = {
        'file': (bus_image_bytes, 'bus.jpg'),
        'config_name': '',  # Test detect all vehicles without slots
        'conf_threshold': '0.25',
        'occ_threshold': '0.3'
    }
    res = client.post('/api/analyze-image', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    resp = res.get_json()
    assert resp['success'] is True
    assert resp['statistics']['total_vehicles'] >= 1
    assert len(resp['detections']) >= 1
    assert resp['detections'][0]['class'] in ['bus', 'car', 'truck', 'motorcycle']


def test_analyze_image_missing_file(client):
    """Test error when no file is submitted."""
    res = client.post('/api/analyze-image', data={}, content_type='multipart/form-data')
    assert res.status_code == 400
    resp = res.get_json()
    assert resp['success'] is False
    assert 'error' in resp


def test_analyze_image_empty_file(client):
    """Test error when empty file is submitted."""
    data = {
        'file': (io.BytesIO(b''), 'empty.jpg')
    }
    res = client.post('/api/analyze-image', data=data, content_type='multipart/form-data')
    assert res.status_code == 400
    resp = res.get_json()
    assert resp['success'] is False
    assert 'empty' in resp['error'].lower() or 'no image' in resp['error'].lower()


def test_analyze_image_invalid_extension(client):
    """Test error when non-image file is submitted."""
    data = {
        'file': (io.BytesIO(b'Not an image text'), 'file.txt')
    }
    res = client.post('/api/analyze-image', data=data, content_type='multipart/form-data')
    assert res.status_code == 400
    resp = res.get_json()
    assert resp['success'] is False
    assert 'unsupported' in resp['error'].lower() or 'invalid' in resp['error'].lower()


def test_analyze_image_corrupted_data(client):
    """Test error when corrupted image data is submitted."""
    data = {
        'file': (io.BytesIO(b'Corrupted bytes that cannot be decoded as image'), 'corrupted.jpg')
    }
    res = client.post('/api/analyze-image', data=data, content_type='multipart/form-data')
    assert res.status_code == 400
    resp = res.get_json()
    assert resp['success'] is False
    assert 'valid' in resp['error'].lower() or 'decode' in resp['error'].lower()


def test_analyze_non_parking_image(client):
    """Test that a non-parking image (blank white) returns meaningful warning notice."""
    blank = np.ones((500, 500, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode('.jpg', blank)
    data = {
        'file': (io.BytesIO(buf.tobytes()), 'blank.jpg'),
        'config_name': 'none'
    }
    res = client.post('/api/analyze-image', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    resp = res.get_json()
    assert resp['success'] is True
    assert resp['warning'] is not None
    assert 'No vehicles' in resp['warning']


def test_preprocess_file_upload(client, sample_image_bytes):
    """Test preprocessing via file upload returning standard data URI for all 5 stages."""
    data = {
        'file': (sample_image_bytes, 'sample.jpg')
    }
    res = client.post('/api/preprocess', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    resp = res.get_json()
    assert resp['success'] is True
    # Verify all 5 stages are present and correctly formatted
    for stage_key in ['original', 'gray', 'blurred', 'contrast', 'edges']:
        assert stage_key in resp, f"Stage {stage_key} missing from response"
        uri = resp[stage_key]
        assert isinstance(uri, str), f"Stage {stage_key} is not a string"
        assert uri.startswith('data:image/jpeg;base64,'), f"Stage {stage_key} missing standard data:image/jpeg;base64, prefix"
        # Ensure there is actual base64 payload after the prefix
        payload = uri.split('data:image/jpeg;base64,')[1]
        assert len(payload) > 100, f"Stage {stage_key} payload is too short or empty"



def test_save_and_load_slots_api(client):
    """Test saving and loading slot configuration via API."""
    slots_data = {
        'name': 'test_api_config',
        'slots': [
            {'id': 1, 'label': 'T1', 'points': [[50, 50], [150, 50], [150, 150], [50, 150]]}
        ]
    }
    save_res = client.post('/api/save-slots', json=slots_data)
    assert save_res.status_code == 200
    save_resp = save_res.get_json()
    assert save_resp['success'] is True
    assert save_resp['config_file'] == 'test_api_config.json'

    # Load it back
    load_res = client.get('/api/load-slots/test_api_config.json')
    assert load_res.status_code == 200
    load_resp = load_res.get_json()
    assert load_resp['success'] is True
    assert len(load_resp['slots']) == 1
    assert load_resp['slots'][0]['id'] == 1

    # Cleanup test config file
    test_path = os.path.join(app_config.CONFIG_FOLDER, 'test_api_config.json')
    if os.path.exists(test_path):
        os.remove(test_path)
