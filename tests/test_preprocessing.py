import pytest
import numpy as np
import cv2
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions assuming they are available in src.preprocessing
try:
    from src.preprocessing import (
        validate_image, validate_image_file, resize_image, to_grayscale,
        apply_gaussian_blur, enhance_contrast, canny_edge_detection,
        morphological_operation, preprocess_pipeline, load_image
    )
except ImportError:
    pass  # Allow tests to be written even if module is not fully implemented yet

@pytest.fixture
def sample_image():
    """Returns a valid 480x640x3 uint8 image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def sample_gray_image():
    """Returns a valid 480x640 uint8 grayscale image."""
    return np.zeros((480, 640), dtype=np.uint8)

def test_validate_image_valid(sample_image):
    """Test valid image validation."""
    assert validate_image(sample_image) is True

def test_validate_image_empty():
    """Test empty array image validation."""
    empty_img = np.array([])
    assert validate_image(empty_img) is False

def test_validate_image_none():
    """Test None image validation."""
    assert validate_image(None) is False

def test_validate_image_wrong_dtype():
    """Test float64 image validation."""
    wrong_dtype_img = np.zeros((100, 100, 3), dtype=np.float64)
    assert validate_image(wrong_dtype_img) is False

def test_validate_image_1d():
    """Test 1D array image validation."""
    img_1d = np.zeros((100,), dtype=np.uint8)
    assert validate_image(img_1d) is False

def test_validate_image_file_nonexistent():
    """Test non-existent image file validation."""
    assert validate_image_file("nonexistent_path.jpg") is False

def test_validate_image_file_valid(sample_image):
    """Test valid image file validation."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        cv2.imwrite(temp_file.name, sample_image)
        temp_path = temp_file.name
    
    try:
        assert validate_image_file(temp_path) is True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_validate_image_file_invalid_extension():
    """Test invalid extension image file validation."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(b"Not an image")
        temp_path = temp_file.name
        
    try:
        assert validate_image_file(temp_path) is False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_resize_image_larger():
    """Test resizing a larger image."""
    large_img = np.zeros((2000, 1000, 3), dtype=np.uint8)
    resized = resize_image(large_img, max_width=800, max_height=600)
    assert resized.shape[1] <= 800
    assert resized.shape[0] <= 600

def test_resize_image_smaller():
    """Test resizing a smaller image."""
    small_img = np.zeros((300, 200, 3), dtype=np.uint8)
    resized = resize_image(small_img, max_width=800, max_height=600)
    assert resized.shape == (300, 200, 3)

def test_resize_image_preserves_aspect_ratio():
    """Test that aspect ratio is preserved when resizing."""
    img = np.zeros((400, 800, 3), dtype=np.uint8) # 2:1 ratio (width:height)
    resized = resize_image(img, max_width=400, max_height=400)
    assert resized.shape[1] == 400
    assert resized.shape[0] == 200

def test_to_grayscale_bgr():
    """Test BGR to grayscale conversion."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = to_grayscale(img)
    assert len(gray.shape) == 2
    assert gray.shape == (100, 100)

def test_to_grayscale_already_gray(sample_gray_image):
    """Test converting already grayscale image."""
    gray = to_grayscale(sample_gray_image)
    assert gray.shape == sample_gray_image.shape

def test_gaussian_blur_odd_kernel(sample_image):
    """Test gaussian blur with odd kernel size."""
    blurred = apply_gaussian_blur(sample_image, kernel_size=5)
    assert blurred.shape == sample_image.shape

def test_gaussian_blur_even_kernel(sample_image):
    """Test gaussian blur with even kernel size."""
    with pytest.raises(ValueError):
        apply_gaussian_blur(sample_image, kernel_size=4)

def test_gaussian_blur_negative_kernel(sample_image):
    """Test gaussian blur with negative kernel size."""
    with pytest.raises(ValueError):
        apply_gaussian_blur(sample_image, kernel_size=-1)

def test_enhance_contrast(sample_gray_image):
    """Test CLAHE contrast enhancement."""
    sample_gray_image[10:50, 10:50] = 100
    enhanced = enhance_contrast(sample_gray_image)
    assert enhanced.shape == sample_gray_image.shape

def test_canny_edge_detection(sample_gray_image):
    """Test Canny edge detection."""
    edges = canny_edge_detection(sample_gray_image)
    assert len(edges.shape) == 2
    assert edges.shape == sample_gray_image.shape

def test_canny_thresholds(sample_gray_image):
    """Test Canny edge detection with thresholds."""
    edges = canny_edge_detection(sample_gray_image, low_threshold=50, high_threshold=150)
    assert edges is not None

def test_morphological_dilate(sample_gray_image):
    """Test dilate morphological operation."""
    res = morphological_operation(sample_gray_image, operation='dilate')
    assert res.shape == sample_gray_image.shape

def test_morphological_erode(sample_gray_image):
    """Test erode morphological operation."""
    res = morphological_operation(sample_gray_image, operation='erode')
    assert res.shape == sample_gray_image.shape

def test_morphological_open(sample_gray_image):
    """Test open morphological operation."""
    res = morphological_operation(sample_gray_image, operation='open')
    assert res.shape == sample_gray_image.shape

def test_morphological_close(sample_gray_image):
    """Test close morphological operation."""
    res = morphological_operation(sample_gray_image, operation='close')
    assert res.shape == sample_gray_image.shape

def test_morphological_invalid_op(sample_gray_image):
    """Test invalid morphological operation."""
    with pytest.raises(ValueError):
        morphological_operation(sample_gray_image, operation='invalid_op')

def test_preprocess_pipeline_default(sample_image):
    """Test default preprocessing pipeline."""
    res = preprocess_pipeline(sample_image)
    assert isinstance(res, dict)
    assert 'original' in res

def test_preprocess_pipeline_custom(sample_image):
    """Test custom preprocessing pipeline."""
    res = preprocess_pipeline(sample_image, steps=['original', 'grayscale', 'blurred'])
    assert 'original' in res
    assert 'grayscale' in res
    assert 'blurred' in res

def test_preprocess_pipeline_empty_steps(sample_image):
    """Test preprocessing pipeline with empty steps."""
    res = preprocess_pipeline(sample_image, steps=[])
    assert isinstance(res, dict)
    assert len(res) == 0

def test_load_image_valid(sample_image):
    """Test valid image loading."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        cv2.imwrite(temp_file.name, sample_image)
        temp_path = temp_file.name
        
    try:
        img = load_image(temp_path)
        assert isinstance(img, np.ndarray)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_load_image_invalid_path():
    """Test loading invalid image path."""
    with pytest.raises(ValueError):
        load_image("nonexistent_image_path.jpg")
