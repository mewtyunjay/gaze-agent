import cv2
import numpy as np
import dlib
from imutils import face_utils
import os

class GazeDetector:
    def __init__(self):
        # Initialize face and eye detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Get the path to the shape predictor file
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                "Please download the shape predictor file from:\n"
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                "Extract it and place it in the same directory as this script."
            )
        
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Define eye indices for dlib's 68-point model
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        # Constants for eye aspect ratio
        self.EAR_THRESHOLD = 0.2
        self.CONSEC_FRAMES = 2
        self.counter = 0

    def get_eye_aspect_ratio(self, eye):
        # Compute euclidean distances between eye landmarks
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        
        # Calculate eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def get_gaze_direction(self, eye):
        # Get the eye region
        eye_region = np.array([(p.x, p.y) for p in eye])
        
        # Get the eye center
        eye_center = np.mean(eye_region, axis=0).astype(int)
        
        # Get the leftmost and rightmost points
        left_point = tuple(eye_region[0])
        right_point = tuple(eye_region[3])
        
        # Calculate relative pupil position
        relative_pos = (eye_center[0] - left_point[0]) / (right_point[0] - left_point[0])
        
        # Determine gaze direction
        if relative_pos < 0.4:
            return "left"
        elif relative_pos > 0.6:
            return "right"
        return "center"

    def analyze_gaze(self, shape, face_width):
        # Convert shape to numpy array
        shape = face_utils.shape_to_np(shape)
        
        # Get both eyes
        leftEye = shape[self.lStart:self.lEnd]
        rightEye = shape[self.rStart:self.rEnd]
        
        # Calculate eye aspect ratios
        leftEAR = self.get_eye_aspect_ratio(leftEye)
        rightEAR = self.get_eye_aspect_ratio(rightEye)
        
        # Average the eye aspect ratio
        ear = (leftEAR + rightEAR) / 2.0
        
        # Check if eyes are open enough
        if ear < self.EAR_THRESHOLD:
            self.counter += 1
            if self.counter >= self.CONSEC_FRAMES:
                return "Eyes Closed"
        else:
            self.counter = 0
        
        # Get eye centers
        left_eye_center = np.mean(leftEye, axis=0)
        right_eye_center = np.mean(rightEye, axis=0)
        
        # Calculate relative position of eye centers
        left_eye_x = left_eye_center[0]
        right_eye_x = right_eye_center[0]
        
        # Get frame width from first point's position
        frame_width = shape[0][0] * 3  # Approximate frame width
        
        # Stricter center check and minimum face size requirement
        left_centered = 0.4 < (left_eye_x / frame_width) < 0.6
        right_centered = 0.4 < (right_eye_x / frame_width) < 0.6
        
        # Minimum face width requirement (adjust this value as needed)
        MIN_FACE_WIDTH = 200  # pixels
        
        if left_centered and right_centered and face_width >= MIN_FACE_WIDTH:
            return "Looking at Camera"
        elif face_width < MIN_FACE_WIDTH:
            return "Move Closer"
        return "Looking Away"

    def run(self):
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        # Set window properties
        cv2.namedWindow('Gaze Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gaze Detection', 800, 600)
        
        while True:
            success, image = cap.read()
            if not success:
                print("\nFailed to grab frame")
                break

            # Flip and convert to grayscale
            image = cv2.flip(image, 1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray, 0)
            
            if len(faces) > 0:
                # Get facial landmarks
                face = faces[0]
                shape = self.predictor(gray, face)
                
                # Get face width
                (x, y, face_width, h) = face_utils.rect_to_bb(face)
                
                # Analyze gaze with face width
                gaze_status = self.analyze_gaze(shape, face_width)
                
                # Convert shape to numpy array for drawing
                shape = face_utils.shape_to_np(shape)
                
                # Draw eye regions
                for (x, y) in shape[self.lStart:self.lEnd]:
                    cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
                for (x, y) in shape[self.rStart:self.rEnd]:
                    cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
                
                # Display status on image
                color = (0, 255, 0) if gaze_status == "Looking at Camera" else (0, 0, 255)
                cv2.putText(image, f"Status: {gaze_status}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw face rectangle
                (x, y, w, h) = face_utils.rect_to_bb(face)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
            else:
                cv2.putText(image, "No Face Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show the image
            cv2.imshow('Gaze Detection', image)
            
            # Break loop with 'q' - using waitKey(1) & 0xFF for compatibility
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Additional waitKey to ensure windows are closed

if __name__ == "__main__":
    print("Starting gaze detection... Press 'q' to quit")
    try:
        detector = GazeDetector()
        detector.run()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")