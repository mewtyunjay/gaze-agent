import cv2
import numpy as np
from gaze_agent.config.settings import CAMERA_WIDTH, CAMERA_HEIGHT


class Camera:
    def __init__(self, camera_id=0, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=30):
        """
        Initialize camera with specified parameters
        
        Args:
            camera_id: Camera device ID (default: 0, primary camera)
            width: Frame width (default: from settings)
            height: Frame height (default: from settings)
            fps: Frames per second (default: 30)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera with ID {self.camera_id}")
                
        except Exception as e:
            print(f"Error initializing camera: {e}")
            if self.cap is not None:
                self.cap.release()
            raise
        
    def get_frame(self):
        """
        Capture and return a frame from the camera
        
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the captured image
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        try:
            success, frame = self.cap.read()
            if not success:
                return False, None
            return True, frame
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return False, None
        
    def is_opened(self):
        """Check if camera is opened and available
        
        Returns:
            bool: True if camera is available, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
        
    def release(self):
        """Release camera resources"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None