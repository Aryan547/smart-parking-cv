import os
import uuid
import json
import base64
import argparse
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Import configuration
import config as app_config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(app_config)
app.config['MAX_CONTENT_LENGTH'] = app_config.MAX_UPLOAD_SIZE
app.secret_key = app_config.SECRET_KEY

# Global variable for YOLO model
MODEL = None

try:
    from src.preprocessing import (
        load_image, validate_image, validate_image_file, preprocess_pipeline,
        resize_image, to_grayscale, apply_gaussian_blur, enhance_contrast, canny_edge_detection
    )
    from src.vehicle_detector import load_model, detect_vehicles, draw_detections, get_detection_summary
    from src.parking_slots import load_slot_config, save_slot_config, validate_slot_config, draw_slots
    from src.occupancy import classify_occupancy
    from src.analytics import calculate_statistics, calculate_occupancy_over_time
    from src.visualization import annotate_parking_image, create_stats_overlay
    from src.video_processor import validate_video, process_video, get_video_info
except ImportError:
    pass

def allowed_file(filename: str, extensions: set) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def get_model(model_path: str = app_config.YOLO_MODEL_PATH):
    """Lazy load YOLO model."""
    global MODEL
    if MODEL is None:
        try:
            MODEL = load_model(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
    return MODEL

def mat_to_data_uri(cv_img: np.ndarray, ext: str = '.jpg') -> str:
    """Convert an OpenCV image matrix to a standard base64 data URI."""
    if cv_img is None or not isinstance(cv_img, np.ndarray) or cv_img.size == 0:
        return ""
    success, buf = cv2.imencode(ext, cv_img)
    if not success:
        return ""
    b64_str = base64.b64encode(buf).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

# --- Page Routes ---

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/image-analysis')
def image_analysis():
    """Image analysis page."""
    return render_template('image_analysis.html')

@app.route('/video-analysis')
def video_analysis():
    """Video analysis page."""
    return render_template('video_analysis.html')

@app.route('/slot-config')
def slot_config():
    """Slot configuration page."""
    return render_template('slot_config.html')

@app.route('/analytics')
def analytics():
    """Analytics page."""
    return render_template('analytics.html')

# --- API Routes ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None
    })

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Upload image file."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, app_config.ALLOWED_IMAGE_EXTENSIONS):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'filename': filename,
            'image_url': f"/uploads/{filename}"
        })
    return jsonify({'success': False, 'error': 'Invalid file type'}), 400

@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    """Run preprocessing pipeline on image file or path."""
    try:
        # 1. Handle multipart/form-data upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            file_bytes = file.read()
            if not file_bytes:
                return jsonify({'success': False, 'error': 'Uploaded file is empty.'}), 400
            
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None or not validate_image(img):
                return jsonify({'success': False, 'error': 'Could not decode image for preprocessing. Please upload a valid JPG or PNG.'}), 400
            
            resized_img = resize_image(img, app_config.MAX_IMAGE_WIDTH, app_config.MAX_IMAGE_HEIGHT)
            results = preprocess_pipeline(resized_img)

            # Convert all pipeline outputs to complete Data URIs
            orig_uri = mat_to_data_uri(results.get('original', resized_img))
            gray_uri = mat_to_data_uri(results.get('grayscale', to_grayscale(resized_img)))
            blur_uri = mat_to_data_uri(results.get('blurred', apply_gaussian_blur(to_grayscale(resized_img))))
            contrast_uri = mat_to_data_uri(results.get('contrast_enhanced', enhance_contrast(resized_img)))
            edges_uri = mat_to_data_uri(results.get('edges', canny_edge_detection(resized_img)))

            return jsonify({
                'success': True,
                'original': orig_uri,
                'gray': gray_uri,
                'grayscale': gray_uri,
                'blurred': blur_uri,
                'blur': blur_uri,
                'contrast': contrast_uri,
                'contrast_enhanced': contrast_uri,
                'edges': edges_uri
            })

        # 2. Handle JSON payload
        data = request.json or {}
        filename = data.get('filename')
        steps = data.get('steps', None)
        if not filename:
            return jsonify({'success': False, 'error': 'Filename required'}), 400
        
        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
        img = load_image(filepath)
        results = preprocess_pipeline(img, steps)
        
        response_results = {}
        for step_name, res_img in results.items():
            out_filename = f"prep_{step_name}_{filename}"
            out_path = os.path.join(app_config.OUTPUT_FOLDER, out_filename)
            cv2.imwrite(out_path, res_img)
            response_results[step_name] = f"/outputs/{out_filename}"
            
        return jsonify({
            'success': True,
            'results': response_results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Run full CV pipeline on uploaded image for vehicle detection and parking occupancy."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded. Please select an image.'}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected.'}), 400
            
        if not allowed_file(file.filename, app_config.ALLOWED_IMAGE_EXTENSIONS):
            allowed_exts = ", ".join(app_config.ALLOWED_IMAGE_EXTENSIONS).upper()
            return jsonify({'success': False, 'error': f'Unsupported file format. Please upload {allowed_exts}.'}), 400

        file_bytes = file.read()
        if not file_bytes or len(file_bytes) == 0:
            return jsonify({'success': False, 'error': 'The uploaded image file is empty.'}), 400

        # Decode image using OpenCV
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None or not validate_image(img):
            return jsonify({'success': False, 'error': 'The uploaded file is not a valid or readable image. Please upload a valid JPG or PNG image.'}), 400

        # Resize preserving aspect ratio if needed
        img = resize_image(img, app_config.MAX_IMAGE_WIDTH, app_config.MAX_IMAGE_HEIGHT)

        # Parse form parameters
        config_name = request.form.get('config_name', '').strip()
        try:
            confidence = float(request.form.get('conf_threshold', app_config.DEFAULT_CONFIDENCE))
        except (ValueError, TypeError):
            confidence = app_config.DEFAULT_CONFIDENCE

        try:
            threshold = float(request.form.get('occ_threshold', app_config.DEFAULT_OCCUPANCY_THRESHOLD))
        except (ValueError, TypeError):
            threshold = app_config.DEFAULT_OCCUPANCY_THRESHOLD

        # Load slot configuration if requested
        slots = []
        if config_name and config_name.lower() not in ['none', 'null', '']:
            config_path = os.path.join(app_config.CONFIG_FOLDER, config_name)
            if not os.path.exists(config_path):
                config_path = app_config.DEFAULT_SLOT_CONFIG
            if os.path.exists(config_path):
                try:
                    slots = load_slot_config(config_path)
                except Exception as e:
                    print(f"Warning: Failed to load slot config {config_name}: {e}")
                    slots = []
        elif not config_name and os.path.exists(app_config.DEFAULT_SLOT_CONFIG):
            try:
                slots = load_slot_config(app_config.DEFAULT_SLOT_CONFIG)
            except Exception:
                slots = []

        # YOLO Vehicle Detection
        model = get_model()
        if model is None:
            return jsonify({'success': False, 'error': 'Failed to load YOLO detection model. Please check ultralytics installation.'}), 500

        detections = detect_vehicles(model, img, confidence=confidence)

        # Spatial Matching & Occupancy Classification
        if slots and len(slots) > 0:
            occupancy_result = classify_occupancy(detections, slots, threshold=threshold)
            slot_statuses = occupancy_result['slot_statuses']
            annotated_img = annotate_parking_image(img.copy(), slots, slot_statuses, detections)
            stats = calculate_statistics(occupancy_result, detections)
            annotated_img = create_stats_overlay(annotated_img, stats)

            slot_status_list = [
                {
                    'id': s['id'],
                    'label': s.get('label', f"P{s['id']}"),
                    'status': 'Occupied' if slot_statuses.get(s['id']) == 'occupied' else 'Available'
                }
                for s in slots
            ]
        else:
            # No slots configured: draw vehicle detections only
            annotated_img = draw_detections(img.copy(), detections)
            slot_status_list = []
            det_summary = get_detection_summary(detections)
            stats = {
                'total_slots': 0,
                'occupied': 0,
                'available': 0,
                'occupancy_percentage': 0.0,
                'vehicle_count': len(detections),
                'vehicles_by_type': det_summary
            }

        # Meaningful feedback for non-parking / zero detection scenes
        warning = None
        if len(detections) == 0 and len(slots) == 0:
            warning = "Notice: No vehicles or parking slots detected. The uploaded image might not be a parking lot."
        elif len(detections) == 0 and len(slots) > 0:
            warning = "Notice: No vehicles detected in the parking spaces. All configured slots are marked Available."

        # Format detections for frontend
        formatted_detections = [
            {
                'class': det['class_name'],
                'confidence': round(det['confidence'], 3),
                'bbox': det['bbox'],
                'center': det.get('center', [0, 0])
            }
            for det in detections
        ]

        total_slots = stats.get('total_slots', 0)
        occupied_slots = stats.get('occupied', 0)
        available_slots = stats.get('available', 0)
        occ_pct = stats.get('occupancy_percentage', 0.0)
        occ_rate = round(occ_pct / 100.0, 4) if total_slots > 0 else 0.0

        statistics = {
            'total_slots': total_slots,
            'occupied_slots': occupied_slots,
            'available_slots': available_slots,
            'occupancy_rate': occ_rate,
            'occupancy_percentage': round(occ_pct, 2),
            'total_vehicles': stats.get('vehicle_count', len(detections)),
            'vehicles_by_type': stats.get('vehicles_by_type', {})
        }

        # Save files to disk
        uid = str(uuid.uuid4())
        orig_filename = f"{uid}.jpg"
        annot_filename = f"det_{uid}.jpg"
        cv2.imwrite(os.path.join(app_config.UPLOAD_FOLDER, orig_filename), img)
        cv2.imwrite(os.path.join(app_config.OUTPUT_FOLDER, annot_filename), annotated_img)

        # Base64 data URIs for immediate rendering
        orig_data_uri = mat_to_data_uri(img)
        annot_data_uri = mat_to_data_uri(annotated_img)

        return jsonify({
            'success': True,
            'statistics': statistics,
            'images': {
                'original': orig_data_uri,
                'annotated': annot_data_uri,
                'original_url': f"/uploads/{orig_filename}",
                'annotated_url': f"/outputs/{annot_filename}"
            },
            'slot_status': slot_status_list,
            'detections': formatted_detections,
            'warning': warning
        })
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return jsonify({'success': False, 'error': f'Image processing error: {str(e)}'}), 500

@app.route('/api/detect', methods=['POST'])
def detect():
    """Run detection and occupancy classification on an image (JSON API)."""
    try:
        data = request.json or {}
        filename = data.get('filename')
        config_file = data.get('config_file', 'parking_slots.json')
        confidence = float(data.get('confidence', app_config.DEFAULT_CONFIDENCE))
        threshold = float(data.get('threshold', app_config.DEFAULT_OCCUPANCY_THRESHOLD))
        
        if not filename:
            return jsonify({'success': False, 'error': 'Filename required'}), 400
            
        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
        config_path = os.path.join(app_config.CONFIG_FOLDER, config_file)
        if not os.path.exists(config_path):
            config_path = app_config.DEFAULT_SLOT_CONFIG
            
        img = load_image(filepath)
        slots = load_slot_config(config_path) if os.path.exists(config_path) else []
        model = get_model()
        if not model:
            return jsonify({'success': False, 'error': 'Failed to load model'}), 500
            
        detections = detect_vehicles(model, img, confidence=confidence)
        occupancy_result = classify_occupancy(detections, slots, threshold)
        slot_statuses = occupancy_result['slot_statuses']
        annotated_img = annotate_parking_image(img.copy(), slots, slot_statuses, detections)
        stats = calculate_statistics(occupancy_result, detections)
        
        out_filename = f"det_{filename}"
        out_path = os.path.join(app_config.OUTPUT_FOLDER, out_filename)
        cv2.imwrite(out_path, annotated_img)
        
        return jsonify({
            'success': True,
            'annotated_url': f"/outputs/{out_filename}",
            'stats': stats,
            'detections': detections,
            'detection_summary': get_detection_summary(detections),
            'slot_statuses': {str(k): v for k, v in slot_statuses.items()}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """Upload video file."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    if file and allowed_file(file.filename, app_config.ALLOWED_VIDEO_EXTENSIONS):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'filename': filename
        })
    return jsonify({'success': False, 'error': 'Invalid file type'}), 400

@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """Process uploaded video file for parking occupancy (multipart/form-data)."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided.'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected video file.'}), 400
        if not allowed_file(file.filename, app_config.ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({'success': False, 'error': 'Invalid video format. Allowed: MP4, AVI, MOV.'}), 400

        ext = file.filename.rsplit('.', 1)[1].lower()
        vid_id = str(uuid.uuid4())
        filename = f"{vid_id}.{ext}"
        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)
        file.save(filepath)

        config_name = request.form.get('config_name', 'parking_slots.json').strip()
        confidence = float(request.form.get('conf_threshold', app_config.DEFAULT_CONFIDENCE))
        threshold = float(request.form.get('occ_threshold', app_config.DEFAULT_OCCUPANCY_THRESHOLD))
        frame_interval = int(request.form.get('frame_interval', app_config.VIDEO_FRAME_INTERVAL))

        config_path = os.path.join(app_config.CONFIG_FOLDER, config_name)
        if not os.path.exists(config_path):
            config_path = app_config.DEFAULT_SLOT_CONFIG
        slots = load_slot_config(config_path) if os.path.exists(config_path) else []

        model = get_model()
        if not model:
            return jsonify({'success': False, 'error': 'Failed to load model.'}), 500

        out_filename = f"out_{vid_id}.mp4"
        out_path = os.path.join(app_config.OUTPUT_FOLDER, out_filename)

        result = process_video(
            filepath, slots, model,
            confidence=confidence, threshold=threshold,
            frame_interval=frame_interval, output_path=out_path
        )

        frame_results = result.get('frame_results', [])
        frame_data = []
        for fr in frame_results:
            occ = fr.get('occupancy', {})
            tot = occ.get('total_slots', len(slots))
            occupied = occ.get('occupied_count', 0)
            available = occ.get('available_count', tot - occupied)
            rate = (occupied / tot) if tot > 0 else 0.0

            frame_data.append({
                'frame': fr['frame'],
                'total_slots': tot,
                'occupied': occupied,
                'available': available,
                'occupancy_rate': round(rate, 4),
                'total_vehicles': len(fr.get('detections', []))
            })

        return jsonify({
            'success': True,
            'output_video_url': f"/outputs/{out_filename}",
            'frame_data': frame_data,
            'stats': result.get('video_stats', {}),
            'frames_processed': result.get('frames_processed', 0)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process-video', methods=['POST'])
def process_video_route():
    """Process video for parking occupancy (JSON API)."""
    try:
        data = request.json or {}
        filename = data.get('filename')
        config_file = data.get('config_file', 'parking_slots.json')
        confidence = float(data.get('confidence', app_config.DEFAULT_CONFIDENCE))
        threshold = float(data.get('threshold', app_config.DEFAULT_OCCUPANCY_THRESHOLD))
        frame_interval = int(data.get('frame_interval', app_config.VIDEO_FRAME_INTERVAL))
        
        if not filename:
            return jsonify({'success': False, 'error': 'Filename required'}), 400
            
        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
        config_path = os.path.join(app_config.CONFIG_FOLDER, config_file)
        if not os.path.exists(config_path):
            config_path = app_config.DEFAULT_SLOT_CONFIG
            
        slots = load_slot_config(config_path) if os.path.exists(config_path) else []
        model = get_model()
        if not model:
            return jsonify({'success': False, 'error': 'Failed to load model'}), 500
            
        out_filename = f"out_{filename}"
        out_path = os.path.join(app_config.OUTPUT_FOLDER, out_filename)
        
        result = process_video(
            filepath, slots, model,
            confidence=confidence, threshold=threshold,
            frame_interval=frame_interval, output_path=out_path
        )
        
        occupancy_history = result.get('frame_results', [])
        video_stats = result.get('video_stats', {})
        occupancy_timeline = calculate_occupancy_over_time(occupancy_history)
        
        return jsonify({
            'success': True,
            'output_url': f"/outputs/{out_filename}",
            'stats': video_stats,
            'occupancy_over_time': occupancy_timeline,
            'frames_processed': result.get('frames_processed', 0)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save-slots', methods=['POST'])
def save_slots():
    """Save slot configuration."""
    try:
        data = request.json or {}
        slots = data.get('slots')
        name = data.get('name') or data.get('filename') or 'parking_slots.json'
        if not name.endswith('.json'):
            name = f"{name}.json"
            
        if not slots:
            return jsonify({'success': False, 'error': 'Slots data required'}), 400
            
        filepath = os.path.join(app_config.CONFIG_FOLDER, name)
        save_slot_config(slots, filepath)
        return jsonify({
            'success': True,
            'config_file': name
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/load-slots', methods=['GET'])
def load_default_slots():
    """Load default slot configuration."""
    try:
        slots = load_slot_config(app_config.DEFAULT_SLOT_CONFIG)
        return jsonify({
            'success': True,
            'slots': slots
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load-slots/<config_file>', methods=['GET'])
def load_slots(config_file):
    """Load specific slot configuration."""
    try:
        filepath = os.path.join(app_config.CONFIG_FOLDER, config_file)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Config not found'}), 404
        slots = load_slot_config(filepath)
        return jsonify({
            'success': True,
            'slots': slots
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/list-configs', methods=['GET'])
def list_configs():
    """List available slot configurations."""
    try:
        configs = [f for f in os.listdir(app_config.CONFIG_FOLDER) if f.endswith('.json')]
        return jsonify({
            'success': True,
            'configs': configs
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/list-images', methods=['GET'])
def list_images():
    """List uploaded images."""
    try:
        images = [f for f in os.listdir(app_config.UPLOAD_FOLDER) if allowed_file(f, app_config.ALLOWED_IMAGE_EXTENSIONS)]
        return jsonify({
            'success': True,
            'images': images
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# --- Static File Serving ---

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """Serve files from input/ folder."""
    return send_from_directory(app_config.UPLOAD_FOLDER, filename)

@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    """Serve files from output/ folder."""
    return send_from_directory(app_config.OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smart Parking Space Detection System')
    parser.add_argument('--host', default=app_config.FLASK_HOST, help='Host to bind to')
    parser.add_argument('--port', type=int, default=app_config.FLASK_PORT, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model', default=app_config.YOLO_MODEL_PATH, help='YOLO model path')
    args = parser.parse_args()
    
    app_config.FLASK_HOST = args.host
    app_config.FLASK_PORT = args.port
    app_config.FLASK_DEBUG = args.debug
    app_config.YOLO_MODEL_PATH = args.model
    
    app.run(host=args.host, port=args.port, debug=args.debug)
