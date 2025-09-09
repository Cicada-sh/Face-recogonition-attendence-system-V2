# 🎯 Face Recognition Attendance System

A high-accuracy face recognition attendance system built with OpenCV, Flask, and modern web technologies. This system provides real-time attendance tracking with multiple face encodings for improved accuracy.

## ✨ Features

- **High Accuracy Face Recognition**: Uses OpenCV LBPH (Local Binary Patterns Histograms) for reliable face recognition
- **Real-time Attendance Tracking**: Live camera feed with instant attendance marking
- **Teacher Dashboard**: Comprehensive analytics and student management
- **Secure Authentication**: Teacher login with encrypted passwords
- **Database Storage**: SQLite database for persistent data storage
- **Responsive Design**: Modern, mobile-friendly web interface
- **Multiple Face Training**: Capture 3-8 photos per student for optimal accuracy
- **Easy Installation**: No complex dependencies like dlib or CMake required

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Webcam/Camera access
- Windows/macOS/Linux

### Installation

#### Option 1: Quick Install (Recommended)
1. **Run the installation script**
   ```bash
   # On Windows
   install.bat
   
   # On Linux/Mac
   chmod +x install.sh && ./install.sh
   ```

#### Option 2: Manual Install
1. **Clone or download the project**
   ```bash
   cd face_recognition_attendance_system
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python==4.8.1.78
   pip install numpy==1.24.3
   pip install flask==2.3.3
   pip install flask-cors==4.0.0
   pip install pillow==10.0.1
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the system**
   - Open your browser and go to `http://localhost:5000`
   - The system will automatically create the database and default admin account

## 📱 Usage Guide

### 1. Setup (First Time)

1. Navigate to the **Setup** page
2. Enter student information (Name and Student ID)
3. Start the camera and capture 5-8 photos from different angles
4. Click "Train Face Recognition" to save the student's face data
5. Repeat for all students

### 2. Attendance Tracking

1. Go to the main **Attendance** page
2. Click "Start Camera" to begin face detection
3. Students will be automatically marked present when detected
4. View real-time attendance logs and statistics

### 3. Teacher Dashboard

1. Click "Teacher Login" and use credentials:
   - **Username**: `admin`
   - **Password**: `admin`
2. View comprehensive attendance analytics
3. Monitor student attendance rates
4. Access weekly attendance trends

## 🎯 Key Features for Accuracy

### Multiple Face Encodings
- Each student is trained with 3-8 different face images
- Captures various angles and expressions
- Reduces false negatives and improves recognition

### High Confidence Threshold
- Uses 60% confidence threshold for attendance marking
- Prevents false positives from similar-looking faces
- Displays confidence percentage for each detection

### Advanced Face Detection
- Uses OpenCV's Haar cascades for face detection
- Face recognition library for encoding comparison
- Real-time processing with optimized performance

## 🗄️ Database Schema

### Students Table
- `id`: Primary key
- `name`: Student's full name
- `student_id`: Unique student identifier
- `face_encodings`: JSON array of face encodings
- `created_at`: Registration timestamp

### Attendance Table
- `id`: Primary key
- `student_id`: Foreign key to students
- `date`: Attendance date
- `time`: Attendance time
- `confidence`: Recognition confidence score

### Teachers Table
- `id`: Primary key
- `username`: Teacher username
- `password_hash`: Encrypted password

## 🔧 API Endpoints

### Authentication
- `POST /api/login` - Teacher login
- `POST /api/logout` - Teacher logout

### Face Recognition
- `POST /api/process-attendance` - Process camera frame for attendance
- `POST /api/train-face` - Train face recognition for new student

### Data Retrieval
- `GET /api/attendance-stats` - Get attendance statistics
- `GET /api/students` - Get list of registered students

## 🎨 Web Pages

1. **Attendance Page** (`/`) - Real-time face detection and attendance marking
2. **Teacher Login** (`/teacher-login`) - Secure authentication for teachers
3. **Teacher Dashboard** (`/teacher-dashboard`) - Analytics and student management
4. **Setup Page** (`/setup`) - Face training and student registration

## 🔒 Security Features

- Password hashing using SHA-256
- Session-based authentication
- Input validation and sanitization
- Secure API endpoints with authentication checks

## 📊 Performance Optimization

- Efficient face encoding storage
- Optimized camera frame processing
- Database indexing for fast queries
- Responsive web design for all devices

## 🛠️ Troubleshooting

### Common Issues

1. **Camera not working**
   - Check browser permissions for camera access
   - Ensure no other applications are using the camera
   - Try refreshing the page

2. **Low recognition accuracy**
   - Capture more photos during training (5-8 recommended)
   - Ensure good lighting conditions
   - Make sure faces are clearly visible and centered

3. **Database errors**
   - Delete `attendance.db` file and restart the application
   - Check file permissions in the project directory

### System Requirements

- **Minimum**: 4GB RAM, 2GB free disk space
- **Recommended**: 8GB RAM, 5GB free disk space
- **Camera**: 720p or higher resolution webcam
- **Browser**: Chrome, Firefox, Safari, or Edge (latest versions)

## 🔄 Updates and Maintenance

### Adding New Students
1. Go to Setup page
2. Enter student information
3. Capture multiple photos
4. Train the system

### Changing Admin Password
1. Access the database directly
2. Update the password hash in the teachers table
3. Use SHA-256 encryption for the new password

### Backup Data
- Copy the `attendance.db` file to backup location
- Database contains all student data and attendance records

## 📈 Future Enhancements

- Export attendance reports to Excel/PDF
- Email notifications for attendance
- Mobile app integration
- Advanced analytics and reporting
- Multi-classroom support
- Integration with school management systems

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify system requirements
3. Ensure all dependencies are installed correctly
4. Check browser console for JavaScript errors

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ for accurate and efficient attendance tracking**
