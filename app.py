from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import base64
import json
import pickle
import os
from insightface.app import FaceAnalysis
import threading
import time
from io import BytesIO
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Global variables
model = None
camera = None
camera_lock = threading.Lock()
learned_faces = {}  # Store learned faces: {person_id: {'embedding': array, 'age': int, 'name': str, 'count': int}}
face_id_counter = 0
SIMILARITY_THRESHOLD = 0.6  # Threshold for face recognition
FACES_DB_FILE = 'learned_faces.pkl'
frame_skip_counter = 0  # For performance optimization


def initialize_model():
    """Initialize the InsightFace model"""
    global model
    try:
        model = FaceAnalysis(name="buffalo_l")
        model.prepare(ctx_id=-1)  # Use CPU for deployment compatibility
        print("Model initialized successfully")
        load_learned_faces()
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        model = None
        return False


def save_learned_faces():
    """Save learned faces to disk"""
    try:
        with open(FACES_DB_FILE, 'wb') as f:
            pickle.dump(learned_faces, f)
        print(f"Saved {len(learned_faces)} learned faces to disk")
    except Exception as e:
        print(f"Error saving learned faces: {e}")


def load_learned_faces():
    """Load learned faces from disk"""
    global learned_faces, face_id_counter
    try:
        if os.path.exists(FACES_DB_FILE):
            with open(FACES_DB_FILE, 'rb') as f:
                learned_faces = pickle.load(f)
            face_id_counter = max(learned_faces.keys()) + 1 if learned_faces else 0
            print(f"Loaded {len(learned_faces)} learned faces from disk")
        else:
            learned_faces = {}
            face_id_counter = 0
            print("No previous learned faces found")
    except Exception as e:
        print(f"Error loading learned faces: {e}")
        learned_faces = {}
        face_id_counter = 0


def find_matching_face(face_embedding):
    """Find if this face matches any learned face"""
    if not learned_faces:
        return None, 0

    best_match_id = None
    best_similarity = 0

    for person_id, person_data in learned_faces.items():
        stored_embedding = person_data['embedding']

        # Calculate cosine similarity
        similarity = cosine_similarity(
            face_embedding.reshape(1, -1),
            stored_embedding.reshape(1, -1)
        )[0][0]

        if similarity > SIMILARITY_THRESHOLD and similarity > best_similarity:
            best_similarity = similarity
            best_match_id = person_id

    return best_match_id, best_similarity


def learn_new_face(face_embedding, age):
    """Learn a new face and assign it an ID"""
    global face_id_counter

    person_id = face_id_counter
    person_name = f"Person_{person_id}"

    learned_faces[person_id] = {
        'embedding': face_embedding,
        'age': age,
        'name': person_name,
        'count': 1,
        'last_seen': time.time()
    }

    face_id_counter += 1
    save_learned_faces()

    return person_id, person_name


def update_learned_face(person_id, face_embedding, age):
    """Update an existing learned face (running average of embeddings and age)"""
    if person_id in learned_faces:
        person_data = learned_faces[person_id]

        # Update count
        person_data['count'] += 1

        # Running average of embeddings (for better stability)
        alpha = 0.1  # Learning rate
        person_data['embedding'] = (1 - alpha) * person_data['embedding'] + alpha * face_embedding

        # Running average of age
        person_data['age'] = int((person_data['age'] + age) / 2)

        # Update last seen
        person_data['last_seen'] = time.time()

        # Save every 10 recognitions to avoid too frequent disk writes
        if person_data['count'] % 10 == 0:
            save_learned_faces()


def get_camera():
    """Get camera instance with thread safety"""
    global camera
    with camera_lock:
        if camera is None:
            try:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    print("Trying alternative camera indices...")
                    camera.release()
                    # Try different camera indices
                    for i in range(1, 4):
                        camera = cv2.VideoCapture(i)
                        if camera.isOpened():
                            print(f"Camera opened on index {i}")
                            break

                if camera.isOpened():
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for real-time
                    print("Camera configured successfully")
                else:
                    print("Failed to open camera")
                    camera = None
            except Exception as e:
                print(f"Error opening camera: {e}")
                camera = None
        return camera


def release_camera():
    """Release camera resources"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            print("Camera released")


def process_frame_for_age_and_recognition(frame, skip_processing=False):
    """Process a single frame for age prediction and face recognition"""
    if model is None:
        # Draw error message on frame
        cv2.putText(frame, "Model not initialized", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, []

    try:
        # Skip face detection every few frames for performance (but still return the frame)
        if skip_processing:
            return frame, []

        # Analyze faces
        faces = model.get(frame)
        results = []

        if len(faces) == 0:
            # Draw "No faces detected" message
            cv2.putText(frame, "No faces detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for i, face in enumerate(faces):
            age = int(face.age)
            box = face.bbox.astype(int)
            face_embedding = face.embedding

            # Check if this face matches any learned face
            match_id, similarity = find_matching_face(face_embedding)

            if match_id is not None:
                # Found a match - use stored information
                person_data = learned_faces[match_id]
                stored_age = person_data['age']
                person_name = person_data['name']

                # Update the learned face
                update_learned_face(match_id, face_embedding, age)

                # Use stored age for stability
                display_age = stored_age
                status = "RECOGNIZED"
                color = (0, 255, 0)  # Green for recognized

                label = f"{person_name}: Age {display_age}"
                confidence_label = f"Confidence: {similarity:.2f}"

            else:
                # New face - learn it
                person_id, person_name = learn_new_face(face_embedding, age)
                display_age = age
                status = "LEARNING"
                color = (0, 165, 255)  # Orange for learning

                label = f"{person_name}: Age {display_age}"
                confidence_label = "LEARNING NEW FACE"

            # Ensure box coordinates are within frame bounds
            h, w = frame.shape[:2]
            box[0] = max(0, min(box[0], w - 1))
            box[1] = max(0, min(box[1], h - 1))
            box[2] = max(0, min(box[2], w - 1))
            box[3] = max(0, min(box[3], h - 1))

            # Draw rectangle and labels
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Main age label
            cv2.putText(frame, label, (box[0], max(box[1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Status/confidence indicator
            cv2.putText(frame, confidence_label, (box[0], min(box[3] + 20, h - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            results.append({
                'person_id': match_id if match_id is not None else person_id,
                'name': person_name,
                'age': display_age,
                'bbox': box.tolist(),
                'status': status,
                'similarity': similarity if match_id is not None else 0.0
            })

        return frame, results
    except Exception as e:
        print(f"Error processing frame: {e}")
        # Draw error message on frame
        cv2.putText(frame, f"Processing error: {str(e)[:50]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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


@app.route('/learned_faces')
def learned_faces_page():
    return render_template('learned_faces.html')


@app.route('/api/learned_faces')
def get_learned_faces():
    """Get all learned faces data"""
    faces_data = []
    for person_id, data in learned_faces.items():
        faces_data.append({
            'id': person_id,
            'name': data['name'],
            'age': data['age'],
            'count': data['count'],
            'last_seen': data['last_seen']
        })
    return jsonify({'faces': faces_data, 'total': len(faces_data)})


@app.route('/api/reset_learned_faces', methods=['POST'])
def reset_learned_faces():
    """Reset all learned faces"""
    global learned_faces, face_id_counter
    learned_faces = {}
    face_id_counter = 0

    # Remove the file
    if os.path.exists(FACES_DB_FILE):
        os.remove(FACES_DB_FILE)

    return jsonify({'status': 'success', 'message': 'All learned faces have been reset'})


@app.route('/api/rename_person', methods=['POST'])
def rename_person():
    """Rename a learned person"""
    data = request.json
    person_id = data.get('person_id')
    new_name = data.get('new_name', '').strip()

    if person_id in learned_faces and new_name:
        learned_faces[person_id]['name'] = new_name
        save_learned_faces()
        return jsonify({'status': 'success', 'message': f'Person renamed to {new_name}'})

    return jsonify({'status': 'error', 'message': 'Invalid person ID or name'}), 400


@app.route('/api/model_status')
def model_status():
    """Check if model is properly initialized"""
    return jsonify({
        'model_initialized': model is not None,
        'learned_faces_count': len(learned_faces)
    })


@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Capture and process a single image"""
    try:
        camera = get_camera()
        if camera is None:
            return jsonify({'error': 'Failed to access camera'}), 500

        ret, frame = camera.read()

        if not ret:
            return jsonify({'error': 'Failed to capture image'}), 500

        # Process frame for age prediction and recognition
        processed_frame, results = process_frame_for_age_and_recognition(frame.copy())

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

        # Process frame for age prediction and recognition
        processed_frame, results = process_frame_for_age_and_recognition(frame.copy())

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
    """Generate frames for video streaming with face recognition"""
    global frame_skip_counter
    camera = get_camera()

    if camera is None:
        print("Camera not available for streaming")
        return

    print("Starting video stream generation...")

    while True:
        try:
            ret, frame = camera.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            # Skip face processing every few frames for better performance
            frame_skip_counter += 1
            skip_processing = (frame_skip_counter % 3 != 0)  # Process every 3rd frame

            # Process frame for age prediction and recognition
            processed_frame, _ = process_frame_for_age_and_recognition(frame.copy(), skip_processing)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Small delay to prevent overwhelming the CPU
            time.sleep(0.05)  # ~20 FPS for better stability

        except Exception as e:
            print(f"Error in frame generation: {e}")
            break

    print("Video stream generation ended")


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera')
def start_camera():
    """Initialize camera for streaming"""
    try:
        camera = get_camera()
        if camera is None:
            return jsonify({'error': 'Failed to access camera. Please check camera permissions and connection.'}), 500
        return jsonify({'status': 'Camera started successfully'})
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
    print("Initializing model...")
    if initialize_model():
        print("Model initialized successfully. Starting Flask app...")
    else:
        print("Failed to initialize model. App may not work properly.")

    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)