from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
import json
from insightface.app import FaceAnalysis
import threading
import time
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Global variables
model = None
camera = None
camera_lock = threading.Lock()


def initialize_model():
    """Initialize the InsightFace model"""
    global model
    try:
        model = FaceAnalysis(name="buffalo_l")
        model.prepare(ctx_id=-1)  # Use CPU for deployment compatibility
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        model = None


def get_camera():
    """Get camera instance with thread safety"""
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
        return camera


def release_camera():
    """Release camera resources"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None


def process_frame_for_age(frame):
    """Process a single frame for age prediction"""
    if model is None:
        return frame, []

    try:
        # Analyze faces
        faces = model.get(frame)
        results = []

        for face in faces:
            age = int(face.age)
            box = face.bbox.astype(int)

            # Draw rectangle and age text
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            results.append({
                'age': age,
                'bbox': box.tolist()
            })

        return frame, results
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/capture_mode')
def capture_mode():
    return render_template('capture_mode.html')


@app.route('/realtime_mode')
def realtime_mode():
    return render_template('realtime_mode.html')


@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Capture and process a single image"""
    try:
        camera = get_camera()
        ret, frame = camera.read()

        if not ret:
            return jsonify({'error': 'Failed to capture image'}), 500

        # Process frame for age prediction
        processed_frame, results = process_frame_for_age(frame.copy())

        # Convert to base64 for web display
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': img_base64,
            'faces': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Process uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Read and decode image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Process frame for age prediction
        processed_frame, results = process_frame_for_age(frame.copy())

        # Convert to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': img_base64,
            'faces': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_frames():
    """Generate frames for video streaming"""
    camera = get_camera()

    while True:
        try:
            ret, frame = camera.read()
            if not ret:
                break

            # Process frame for age prediction
            processed_frame, _ = process_frame_for_age(frame.copy())

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Small delay to prevent overwhelming the CPU
            time.sleep(0.03)  # ~30 FPS

        except Exception as e:
            print(f"Error in frame generation: {e}")
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera')
def start_camera():
    """Initialize camera for streaming"""
    try:
        get_camera()
        return jsonify({'status': 'Camera started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stop_camera')
def stop_camera():
    """Stop camera streaming"""
    try:
        release_camera()
        return jsonify({'status': 'Camera stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()

    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)