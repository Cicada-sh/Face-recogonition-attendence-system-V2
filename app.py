from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import sqlite3
import os
import cv2
import numpy as np
import base64
import json
from datetime import datetime, timedelta
import hashlib
import secrets
import threading
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch.nn.functional as F

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

# Database setup
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Create students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            face_encodings TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')
    
    # Create teachers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    # Insert default admin teacher
    admin_password = hashlib.sha256('admin'.encode()).hexdigest()
    cursor.execute('''
        INSERT OR IGNORE INTO teachers (username, password_hash) 
        VALUES (?, ?)
    ''', ('admin', admin_password))
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Global variables for face recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mtcnn = MTCNN(keep_all=True, device='cpu')  # Face detection
facenet = InceptionResnetV1(pretrained='vggface2').eval()  # Face recognition
known_face_encodings = []
known_face_names = []
model_ready = False

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def extract_face_encoding(face_image):
    """Extract face encoding using FaceNet deep learning model"""
    try:
        # Convert OpenCV image to PIL Image
        if len(face_image.shape) == 2:
            # Grayscale to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        else:
            face_rgb = face_image
        
        # Convert BGR to RGB for PIL
        face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        
        # Resize to 160x160 (FaceNet input size)
        pil_image = pil_image.resize((160, 160))
        
        # Convert to tensor and normalize
        face_tensor = torch.tensor(np.array(pil_image)).permute(2, 0, 1).float()
        face_tensor = (face_tensor - 127.5) / 128.0  # Normalize to [-1, 1]
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        # Extract face embedding using FaceNet
        with torch.no_grad():
            face_embedding = facenet(face_tensor)
            face_embedding = F.normalize(face_embedding, p=2, dim=1)  # L2 normalize
        
        return face_embedding.squeeze().numpy()
    except Exception as e:
        print(f"Error extracting face encoding: {e}")
        return None

def load_face_encodings():
    global known_face_encodings, known_face_names, model_ready
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT name, student_id, face_encodings FROM students')
    students = cursor.fetchall()
    
    if len(students) == 0:
        conn.close()
        return
    
    known_face_encodings = []
    known_face_names = []
    
    for name, student_id, encodings_json in students:
        encodings = json.loads(encodings_json)
        for encoding in encodings:
            face_array = np.array(encoding, dtype=np.uint8)
            
            # Extract FaceNet features
            face_encoding = extract_face_encoding(face_array)
            if face_encoding is not None:
                known_face_encodings.append(face_encoding)
                known_face_names.append(f"{name} ({student_id})")
    
    if len(known_face_encodings) > 0:
        # Save the trained model
        with open('face_model.pkl', 'wb') as f:
            pickle.dump({
                'encodings': known_face_encodings,
                'names': known_face_names
            }, f)
        
        model_ready = True
        print(f"Loaded {len(known_face_names)} face encodings")
    else:
        model_ready = False
        print("No valid face encodings found")
    
    conn.close()

def load_saved_model():
    """Load saved face recognition model if it exists"""
    global known_face_encodings, known_face_names, model_ready
    if os.path.exists('face_model.pkl'):
        try:
            with open('face_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                # Check if it's the new format
                if 'encodings' in model_data and 'names' in model_data:
                    known_face_encodings = model_data['encodings']
                    known_face_names = model_data['names']
                    model_ready = True
                    print(f"Loaded face recognition model with {len(known_face_names)} students")
                else:
                    # Old format, retrain the model
                    print("Old model format detected. Retraining with new system...")
                    load_face_encodings()
        except Exception as e:
            print(f"Error loading saved model: {e}")
            load_face_encodings()
    else:
        load_face_encodings()

# Load face encodings on startup (async to avoid blocking)
def load_model_async():
    """Load the face recognition model in a separate thread"""
    try:
        load_saved_model()
        print("Face recognition model loaded successfully")
    except Exception as e:
        print(f"Error loading face recognition model: {e}")

# Start model loading in background thread
model_loading_thread = threading.Thread(target=load_model_async, daemon=True)
model_loading_thread.start()

@app.route('/')
def index():
    return render_template('attendance.html')

@app.route('/teacher-login')
def teacher_login():
    return render_template('teacher_login.html')

@app.route('/teacher-dashboard')
def teacher_dashboard():
    if 'teacher_logged_in' not in session:
        return redirect(url_for('teacher_login'))
    return render_template('teacher_dashboard.html')

@app.route('/setup')
def setup():
    return render_template('setup.html')

@app.route('/debug')
def debug():
    return render_template('debug.html')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'})
    
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute('SELECT id FROM teachers WHERE username = ? AND password_hash = ?', 
                   (username, password_hash))
    
    if cursor.fetchone():
        session['teacher_logged_in'] = True
        session['username'] = username
        conn.close()
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        conn.close()
        return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('teacher_logged_in', None)
    session.pop('username', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/process-attendance', methods=['POST'])
def process_attendance():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data received'})
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with better parameters
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1,   # Less sensitive scaling
            minNeighbors=8,    # Require more neighbors
            minSize=(50, 50),  # Larger minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        attendance_marked = []
        
        # Only process the largest face to avoid confusion
        if len(faces) > 0:
            # Find the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            (x, y, w, h) = largest_face
            # Extract face region
            face_roi = gray_frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))  # Resize for consistency
            
            if model_ready and len(known_face_encodings) > 0:
                try:
                    # Extract LBP features from the detected face
                    face_encoding = extract_face_encoding(face_roi)
                    
                    if face_encoding is not None:
                        # Calculate cosine similarities to all known faces
                        similarities = []
                        for known_encoding in known_face_encodings:
                            # Cosine similarity (higher = more similar)
                            similarity = np.dot(face_encoding, known_encoding) / (
                                np.linalg.norm(face_encoding) * np.linalg.norm(known_encoding)
                            )
                            similarities.append(similarity)
                        
                        similarities = np.array(similarities)
                        
                        # Find the best match
                        best_match_index = np.argmax(similarities)
                        best_similarity = similarities[best_match_index]
                        
                        # Convert similarity to confidence percentage
                        # FaceNet similarity: 1.0 = perfect match, 0.0 = no similarity
                        # Typical good matches: 0.6-1.0, poor matches: 0.0-0.6
                        if best_similarity >= 0.8:  # Excellent match
                            confidence_percentage = min(100, 80 + (best_similarity - 0.8) * 100)
                        elif best_similarity >= 0.7:  # Good match
                            confidence_percentage = 60 + (best_similarity - 0.7) * 200
                        elif best_similarity >= 0.6:  # Moderate match
                            confidence_percentage = 40 + (best_similarity - 0.6) * 200
                        else:  # Poor match
                            confidence_percentage = best_similarity * 66.67
                        
                        # Only accept if confidence is high enough
                        if confidence_percentage > 75:  # High threshold for reliability
                            name = known_face_names[best_match_index]
                            student_id = name.split('(')[1].split(')')[0]
                            
                            # Check if already marked today
                            conn = sqlite3.connect('attendance.db')
                            cursor = conn.cursor()
                            
                            today = datetime.now().date()
                            now = datetime.now()
                            one_hour_ago = now - timedelta(hours=1)
                            
                            cursor.execute('''
                                SELECT id, time FROM attendance 
                                WHERE student_id = ? AND date = ? AND time >= ?
                                ORDER BY time DESC LIMIT 1
                            ''', (student_id, today, one_hour_ago.strftime('%H:%M:%S')))
                            
                            recent_attendance = cursor.fetchone()
                            
                            if not recent_attendance:
                                # Mark attendance
                                cursor.execute('''
                                    INSERT INTO attendance (student_id, date, time, confidence)
                                    VALUES (?, ?, ?, ?)
                                ''', (student_id, today, now.strftime('%H:%M:%S'), confidence_percentage / 100))
                                conn.commit()
                                
                                attendance_marked.append({
                                    'name': name.split('(')[0].strip(),
                                    'student_id': student_id,
                                    'confidence': round(confidence_percentage, 2),
                                    'time': now.strftime('%H:%M:%S'),
                                    'status': 'marked'
                                })
                            else:
                                # Already marked in last hour
                                last_time = recent_attendance[1]
                                attendance_marked.append({
                                    'name': name.split('(')[0].strip(),
                                    'student_id': student_id,
                                    'confidence': round(confidence_percentage, 2),
                                    'time': last_time,
                                    'status': 'already_marked'
                                })
                            
                            conn.close()
                except Exception as e:
                    print(f"Error in face recognition: {e}")
                    pass
        
        return jsonify({
            'success': True,
            'attendance_marked': attendance_marked,
            'faces_detected': len(faces)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/train-face', methods=['POST'])
def train_face():
    try:
        data = request.get_json()
        name = data.get('name')
        student_id = data.get('student_id')
        images = data.get('images')  # Array of base64 images
        
        if not name or not student_id or not images:
            return jsonify({'success': False, 'message': 'Missing required data'})
        
        face_encodings = []
        
        # Use only the first (and best) image
        if len(images) > 0:
            image_data = images[0]  # Take only the first image
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with better parameters
            faces = face_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=1.1,   # Less sensitive scaling to avoid false positives
                minNeighbors=8,    # Require more neighbors for better detection
                minSize=(50, 50),  # Larger minimum face size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                # Use the largest face found
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                (x, y, w, h) = largest_face
                face_roi = gray_frame[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))  # Resize for consistency
                
                # Apply histogram equalization for better contrast
                face_roi = cv2.equalizeHist(face_roi)
                
                # Store the face image for later encoding
                face_encodings.append(face_roi.tolist())
        
        if len(face_encodings) == 0:
            return jsonify({'success': False, 'message': 'No faces detected in any of the images'})
        
        # Save to database
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Check if student already exists
        cursor.execute('SELECT id FROM students WHERE student_id = ?', (student_id,))
        existing_student = cursor.fetchone()
        
        if existing_student:
            # Update existing student - replace all face encodings
            cursor.execute('''
                UPDATE students 
                SET name = ?, face_encodings = ?
                WHERE student_id = ?
            ''', (name, json.dumps(face_encodings), student_id))
            print(f"Updated existing student: {name} ({student_id})")
        else:
            # Insert new student
            cursor.execute('''
                INSERT INTO students (name, student_id, face_encodings)
                VALUES (?, ?, ?)
            ''', (name, student_id, json.dumps(face_encodings)))
            print(f"Added new student: {name} ({student_id})")
        
        conn.commit()
        conn.close()
        
        # Reload face encodings
        load_face_encodings()
        
        return jsonify({
            'success': True,
            'message': f'Successfully trained {len(face_encodings)} face encodings for {name}',
            'encodings_count': len(face_encodings)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/attendance-stats')
def attendance_stats():
    if 'teacher_logged_in' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Get total students
    cursor.execute('SELECT COUNT(*) FROM students')
    total_students = cursor.fetchone()[0]
    
    # Get today's attendance
    today = datetime.now().date()
    cursor.execute('''
        SELECT COUNT(DISTINCT student_id) FROM attendance 
        WHERE date = ?
    ''', (today,))
    today_present = cursor.fetchone()[0]
    
    # Get attendance by date for the last 7 days
    cursor.execute('''
        SELECT date, COUNT(DISTINCT student_id) as present_count
        FROM attendance 
        WHERE date >= date('now', '-7 days')
        GROUP BY date
        ORDER BY date
    ''')
    weekly_attendance = cursor.fetchall()
    
    # Get student-wise attendance
    cursor.execute('''
        SELECT s.name, s.student_id, 
               COUNT(a.id) as total_days,
               COUNT(CASE WHEN a.date = ? THEN 1 END) as today_present
        FROM students s
        LEFT JOIN attendance a ON s.student_id = a.student_id
        GROUP BY s.id, s.name, s.student_id
        ORDER BY s.name
    ''', (today,))
    student_stats = cursor.fetchall()
    
    conn.close()
    
    return jsonify({
        'success': True,
        'total_students': total_students,
        'today_present': today_present,
        'attendance_rate': round((today_present / total_students * 100) if total_students > 0 else 0, 2),
        'weekly_attendance': [{'date': str(row[0]), 'present': row[1]} for row in weekly_attendance],
        'student_stats': [{'name': row[0], 'student_id': row[1], 'total_days': row[2], 'today_present': row[3]} for row in student_stats]
    })

@app.route('/api/model-status')
def get_model_status():
    """Check if the face recognition model is ready"""
    return jsonify({
        'model_ready': model_ready,
        'student_count': len(known_face_names),
        'students': known_face_names
    })

@app.route('/api/students')
def get_students():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT name, student_id, created_at FROM students ORDER BY name')
    students = cursor.fetchall()
    
    conn.close()
    
    return jsonify({
        'success': True,
        'students': [{'name': row[0], 'student_id': row[1], 'created_at': row[2]} for row in students]
    })

@app.route('/api/cleanup-duplicates', methods=['POST'])
def cleanup_duplicates():
    """Remove duplicate student entries"""
    if 'teacher_logged_in' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'})
    
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Find and remove duplicates, keeping the latest entry
        cursor.execute('''
            DELETE FROM students 
            WHERE id NOT IN (
                SELECT MAX(id) 
                FROM students 
                GROUP BY student_id
            )
        ''')
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        # Reload face encodings after cleanup
        load_face_encodings()
        
        return jsonify({
            'success': True,
            'message': f'Removed {deleted_count} duplicate entries',
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/clear-all-data', methods=['POST'])
def clear_all_data():
    """Clear all training data and start fresh"""
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Clear all students
        cursor.execute('DELETE FROM students')
        deleted_students = cursor.rowcount
        
        # Clear all attendance records
        cursor.execute('DELETE FROM attendance')
        deleted_attendance = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        # Delete model file
        if os.path.exists('face_model.pkl'):
            os.remove('face_model.pkl')
        
        # Reset global variables
        global known_face_encodings, known_face_names, model_ready
        known_face_encodings = []
        known_face_names = []
        model_ready = False
        
        return jsonify({
            'success': True,
            'message': f'Cleared {deleted_students} students and {deleted_attendance} attendance records',
            'deleted_students': deleted_students,
            'deleted_attendance': deleted_attendance
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete-student/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    """Delete a specific student"""
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Delete student
        cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
        deleted_student = cursor.rowcount
        
        # Delete attendance records for this student
        cursor.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))
        deleted_attendance = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        # Reload face encodings to retrain model
        load_face_encodings()
        
        return jsonify({
            'success': True,
            'message': f'Deleted student {student_id} and {deleted_attendance} attendance records',
            'deleted_student': deleted_student,
            'deleted_attendance': deleted_attendance
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/debug-attendance', methods=['POST'])
def debug_attendance():
    """Debug endpoint to see what's happening with face recognition"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data received'})
        
        debug_info = {
            'faces_detected': 0,
            'face_recognizer_status': model_ready,
            'student_count': len(known_face_names),
            'student_names': known_face_names,
            'knn_neighbors': len(known_face_encodings),
            'recognition_attempts': [],
            'errors': []
        }
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with better parameters
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1,   # Less sensitive scaling
            minNeighbors=8,    # Require more neighbors
            minSize=(50, 50),  # Larger minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        debug_info['faces_detected'] = len(faces)
        
        # Only process the largest face to avoid confusion
        if len(faces) > 0:
            # Find the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            (x, y, w, h) = largest_face
            idx = 0  # Only one face to process
            face_debug = {
                'face_index': idx,
                'face_size': f"{w}x{h}",
                'face_position': f"({x},{y})",
                'recognition_attempted': False,
                'features_extracted': False,
                'prediction_made': False,
                'confidence': 0,
                'predicted_student': None,
                'error': None
            }
            
            try:
                # Extract face region
                face_roi = gray_frame[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                
                if model_ready and len(known_face_encodings) > 0:
                    face_debug['recognition_attempted'] = True
                    
                    # Extract LBP features from the detected face
                    face_encoding = extract_face_encoding(face_roi)
                    face_debug['features_extracted'] = face_encoding is not None
                    
                    if face_encoding is not None:
                        face_debug['feature_count'] = len(face_encoding)
                        face_debug['feature_range'] = f"{float(np.min(face_encoding)):.3f} to {float(np.max(face_encoding)):.3f}"
                        face_debug['feature_mean'] = f"{float(np.mean(face_encoding)):.3f}"
                        
                        # Calculate cosine similarities to all known faces
                        similarities = []
                        for known_encoding in known_face_encodings:
                            similarity = np.dot(face_encoding, known_encoding) / (
                                np.linalg.norm(face_encoding) * np.linalg.norm(known_encoding)
                            )
                            similarities.append(float(similarity))  # Convert to Python float
                        
                        face_debug['prediction_made'] = True
                        face_debug['all_similarities'] = [round(float(s), 3) for s in similarities]
                        
                        # Find the best match
                        best_match_index = int(np.argmax(similarities))
                        best_similarity = float(similarities[best_match_index])
                        face_debug['best_match_index'] = best_match_index
                        face_debug['best_similarity'] = round(best_similarity, 3)
                        
                        # Convert similarity to confidence percentage
                        if best_similarity >= 0.8:  # Excellent match
                            confidence_percentage = min(100, 80 + (best_similarity - 0.8) * 100)
                        elif best_similarity >= 0.7:  # Good match
                            confidence_percentage = 60 + (best_similarity - 0.7) * 200
                        elif best_similarity >= 0.6:  # Moderate match
                            confidence_percentage = 40 + (best_similarity - 0.6) * 200
                        else:  # Poor match
                            confidence_percentage = best_similarity * 66.67
                        
                        face_debug['confidence'] = round(float(confidence_percentage), 2)
                        face_debug['confidence_formula'] = f"FaceNet cosine similarity: {best_similarity:.3f}"
                        
                        # Only accept if confidence is high enough
                        if confidence_percentage > 75:
                            face_debug['predicted_student'] = known_face_names[best_match_index]
                        else:
                            face_debug['error'] = f"Confidence too low: {confidence_percentage:.2f}% (threshold: 75%)"
                    else:
                        face_debug['error'] = "Could not extract FaceNet features"
                else:
                    if not model_ready:
                        face_debug['error'] = "Face recognition model not ready"
                    if len(known_face_encodings) == 0:
                        face_debug['error'] = "No students registered"
                        
            except Exception as e:
                face_debug['error'] = str(e)
                debug_info['errors'].append(f"Face {idx}: {str(e)}")
            
            debug_info['recognition_attempts'].append(face_debug)
        
        # Convert all NumPy types to Python types for JSON serialization
        debug_info = convert_numpy_types(debug_info)
        
        return jsonify({
            'success': True,
            'debug_info': debug_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
