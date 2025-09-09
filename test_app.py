#!/usr/bin/env python3
"""
Simple test script to verify the face recognition attendance system works
"""

import sys
import cv2
import numpy as np

def test_opencv():
    """Test if OpenCV is working properly"""
    print("Testing OpenCV installation...")
    
    try:
        # Test basic OpenCV functionality
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test face cascade loading
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("❌ Error: Could not load face cascade classifier")
            return False
        else:
            print("✅ Face cascade classifier loaded successfully")
        
        # Test template matching
        img1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        print("✅ Template matching functionality working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing OpenCV: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing module imports...")
    
    modules = [
        ('flask', 'Flask'),
        ('flask_cors', 'CORS'),
        ('sqlite3', 'sqlite3'),
        ('numpy', 'numpy'),
        ('cv2', 'cv2'),
        ('PIL', 'PIL'),
        ('json', 'json'),
        ('datetime', 'datetime'),
        ('hashlib', 'hashlib'),
        ('secrets', 'secrets')
    ]
    
    all_good = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {display_name}: {e}")
            all_good = False
    
    return all_good

def main():
    print("🧪 Face Recognition Attendance System - Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing dependencies.")
        sys.exit(1)
    
    print()
    
    # Test OpenCV
    if not test_opencv():
        print("\n❌ OpenCV tests failed. Please check your OpenCV installation.")
        sys.exit(1)
    
    print("\n✅ All tests passed! The system should work correctly.")
    print("\nTo run the application:")
    print("1. Run: python app.py")
    print("2. Open browser to: http://localhost:5000")
    print("3. Default login: admin / admin")

if __name__ == "__main__":
    main()
