import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class FaceDetectionResult:
    landmarks: np.ndarray  # Full set of facial landmarks
    face_bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    left_eye_landmarks: np.ndarray  # Left eye landmarks
    right_eye_landmarks: np.ndarray  # Right eye landmarks
    left_eye_ratio: float  # Left eye aspect ratio
    right_eye_ratio: float  # Right eye aspect ratio
    is_blinking: bool  # True if eyes are closed/blinking
    head_pose: Tuple[float, float, float] = None  # Head pose (roll, pitch, yaw) in radians


class FaceDetector:
    # MediaPipe Face Mesh landmarks indices
    # Left eye indices
    LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    # Right eye indices
    RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    # Blink detection threshold
    BLINK_THRESHOLD = 0.2
    
    # Head pose estimation landmarks
    # Nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
    FACE_MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye corner
        (225.0, 170.0, -135.0),    # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)    # Right mouth corner
    ]) / 1000.0  # Convert to meters
    
    # Corresponding MediaPipe Face Mesh indices
    FACE_POSE_INDICES = [1, 199, 33, 263, 61, 291]
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize face mesh detector with MediaPipe
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect(self, frame):
        """
        Process frame and detect facial landmarks
        
        Args:
            frame: Input frame from camera
            
        Returns:
            FaceDetectionResult or None if no face detected
        """
        if frame is None:
            return None
            
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Process the image
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract all landmarks
        landmarks = np.array([
            [int(landmark.x * w), int(landmark.y * h)]
            for landmark in face_landmarks.landmark
        ])
        
        # Get face bounding box
        x_min = np.min(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        x_max = np.max(landmarks[:, 0])
        y_max = np.max(landmarks[:, 1])
        face_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        
        # Extract eye landmarks
        left_eye_landmarks = self.get_eye_landmarks(landmarks, self.LEFT_EYE_INDICES)
        right_eye_landmarks = self.get_eye_landmarks(landmarks, self.RIGHT_EYE_INDICES)
        
        # Calculate eye aspect ratios
        left_eye_ratio = self.calculate_eye_ratio(left_eye_landmarks)
        right_eye_ratio = self.calculate_eye_ratio(right_eye_landmarks)
        
        # Check for blink
        avg_ratio = (left_eye_ratio + right_eye_ratio) / 2
        is_blinking = avg_ratio < self.BLINK_THRESHOLD
        
        # Estimate head pose
        head_pose = self.estimate_head_pose(frame, landmarks)
        
        return FaceDetectionResult(
            landmarks=landmarks,
            face_bbox=face_bbox,
            left_eye_landmarks=left_eye_landmarks,
            right_eye_landmarks=right_eye_landmarks,
            left_eye_ratio=left_eye_ratio,
            right_eye_ratio=right_eye_ratio,
            is_blinking=is_blinking,
            head_pose=head_pose
        )
        
    def get_eye_landmarks(self, landmarks, eye_indices):
        """
        Extract eye landmarks from face landmarks
        
        Args:
            landmarks: Full set of facial landmarks
            eye_indices: Indices of eye landmarks
            
        Returns:
            np.ndarray: Eye landmarks
        """
        return np.array([landmarks[idx] for idx in eye_indices])
        
    def calculate_eye_ratio(self, eye_landmarks):
        """
        Calculate the eye aspect ratio (EAR)
        EAR = (vertical distances) / (horizontal distances)
        
        Args:
            eye_landmarks: Eye landmarks from MediaPipe Face Mesh
            
        Returns:
            float: Eye aspect ratio
        """
        # For MediaPipe eye landmarks, we use a different approach than traditional EAR
        # The eye landmarks form a contour around the eye
        
        # Get the width of the eye (horizontal distance)
        eye_width = np.linalg.norm(eye_landmarks[0] - eye_landmarks[8])
        
        # Get the height of the eye (vertical distance)
        # We take the average of several points for better accuracy
        v1 = np.linalg.norm(eye_landmarks[12] - eye_landmarks[4])
        v2 = np.linalg.norm(eye_landmarks[13] - eye_landmarks[3])
        v3 = np.linalg.norm(eye_landmarks[14] - eye_landmarks[2])
        eye_height = (v1 + v2 + v3) / 3
        
        # Calculate ratio
        ratio = eye_height / eye_width if eye_width > 0 else 0
        return ratio
        
    def estimate_head_pose(self, frame, landmarks):
        """
        Estimate head pose (roll, pitch, yaw) using PnP algorithm
        
        Args:
            frame: Input frame from camera
            landmarks: Full set of facial landmarks
            
        Returns:
            tuple: (roll, pitch, yaw) in radians or None if estimation fails
        """
        try:
            h, w, _ = frame.shape
            
            # Camera matrix
            focal_length = w
            camera_center = (w / 2, h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, camera_center[0]],
                 [0, focal_length, camera_center[1]],
                 [0, 0, 1]], dtype=np.float32
            )
            
            # Distortion coefficients
            dist_coeffs = np.zeros((4, 1))
            
            # Get specific face landmarks for pose estimation
            face_landmarks_2d = np.array([
                landmarks[self.FACE_POSE_INDICES[i]] for i in range(len(self.FACE_POSE_INDICES))
            ], dtype=np.float32)
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.FACE_MODEL_POINTS, face_landmarks_2d, camera_matrix, dist_coeffs
            )
            
            if not success:
                return None
                
            # Convert rotation vector to rotation matrix and then to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pose_matrix = np.hstack((rotation_matrix, translation_vector))
            
            # Calculate Euler angles
            euler_angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
            
            # Return angles in radians (roll, pitch, yaw)
            return euler_angles
            
        except Exception as e:
            print(f"Head pose estimation error: {e}")
            return None
            
    def rotation_matrix_to_euler_angles(self, R):
        """
        Convert rotation matrix to Euler angles (roll, pitch, yaw)
        
        Args:
            R: Rotation matrix
            
        Returns:
            tuple: (roll, pitch, yaw) in radians
        """
        # Check if the rotation matrix is valid
        # The rotation matrix is valid if det(R) = 1
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
            
        return (roll, pitch, yaw)