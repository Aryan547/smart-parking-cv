import os
import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    """Load image from file path, raise ValueError if invalid."""
    if not os.path.exists(path):
        raise ValueError(f"Image path does not exist: {path}")
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image at: {path}")
    return image

def validate_image(image: np.ndarray) -> bool:
    """Check if array is valid image (non-empty, proper dimensions, proper dtype)."""
    if not isinstance(image, np.ndarray):
        return False
    if image.size == 0 or len(image.shape) < 2:
        return False
    if image.dtype != np.uint8:
        return False
    return True

def validate_image_file(file_path: str) -> bool:
    """Check if file exists, has valid extension, is readable as image."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in valid_extensions:
        return False
    if not os.path.exists(file_path):
        return False
    # Try reading header (or just imread since we want to know if it's readable)
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return img is not None

def resize_image(image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """Resize preserving aspect ratio."""
    if not validate_image(image):
        raise ValueError("Invalid image provided.")
    h, w = image.shape[:2]
    scale_w = max_width / w if w > max_width else 1.0
    scale_h = max_height / h if h > max_height else 1.0
    scale = min(scale_w, scale_h)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image.copy()

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale."""
    if not validate_image(image):
        raise ValueError("Invalid image.")
    if len(image.shape) == 2:
        return image.copy() # Already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur, validate kernel_size is odd and positive."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be positive and odd.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if len(image.shape) == 3:
        # If original was BGR, returning grayscale enhanced for consistency or we can apply it to lightness channel.
        # Let's convert LAB and apply to L channel for BGR.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_enhanced = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

def canny_edge_detection(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """Canny edges."""
    gray = to_grayscale(image)
    return cv2.Canny(gray, low_threshold, high_threshold)

def morphological_operation(image: np.ndarray, operation: str = 'dilate', kernel_size: int = 5) -> np.ndarray:
    """Support 'dilate', 'erode', 'open', 'close'."""
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    ops = {
        'dilate': cv2.dilate,
        'erode': cv2.erode,
    }
    morph_ops = {
        'open': cv2.MORPH_OPEN,
        'close': cv2.MORPH_CLOSE
    }
    
    if operation in ops:
        return ops[operation](image, kernel, iterations=1)
    elif operation in morph_ops:
        return cv2.morphologyEx(image, morph_ops[operation], kernel)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

def preprocess_pipeline(image: np.ndarray, steps: list[str] = None) -> dict[str, np.ndarray]:
    """Run a pipeline of named steps, return dict mapping step name to result image."""
    if steps is None:
        steps = ['original', 'resized', 'grayscale', 'blurred', 'contrast_enhanced', 'edges']
    
    results = {}
    current = image.copy()
    
    for step in steps:
        if step == 'original':
            results[step] = current.copy()
        elif step == 'resized':
            current = resize_image(current)
            results[step] = current.copy()
        elif step == 'grayscale':
            current = to_grayscale(current)
            results[step] = current.copy()
        elif step == 'blurred':
            current = apply_gaussian_blur(current)
            results[step] = current.copy()
        elif step == 'contrast_enhanced':
            current = enhance_contrast(current)
            results[step] = current.copy()
        elif step == 'edges':
            current = canny_edge_detection(current)
            results[step] = current.copy()
        else:
            raise ValueError(f"Unknown pipeline step: {step}")
            
    return results
