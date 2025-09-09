#!/bin/bash

echo "Installing Face Recognition Attendance System..."
echo

echo "Installing Python dependencies..."
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install flask==2.3.3
pip install flask-cors==4.0.0
pip install pillow==10.0.1
pip install scikit-learn==1.3.0
pip install scipy==1.11.1

echo
echo "Installation complete!"
echo
echo "To run the system:"
echo "1. Run: python app.py"
echo "2. Open browser to: http://localhost:5000"
echo
echo "Default teacher login:"
echo "Username: admin"
echo "Password: admin"
echo
echo "Note: This version uses OpenCV-based face recognition for better compatibility."
echo
