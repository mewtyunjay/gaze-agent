import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class GazeResult:
    left_pupil: Tuple[int, int]  # (x, y) coordinates of left pupil
    right_pupil: Tuple[int, int]  # (x, y) coordinates of right pupil
    gaze_vector: Tuple[float, float]  # Normalized gaze direction vector
    gaze_point: Optional[Tuple[int, int]] = None  # Estimated gaze point on screen
    head_adjusted: bool = False  # Whether the gaze vector was adjusted for head pose


class GazeDetector:
    # Gaze estimation parameters
    SMOOTH_FACTOR = 0.6  # Reduced to make system more responsive to changes
    HEAD_POSE_WEIGHT = 0.9  # Weight for head pose influence on gaze direction (increased)
    
    # Maximum angles (in radians) for looking at camera check
    MAX_YAW_ANGLE = 0.25  # ~15 degrees - max head rotation for looking at camera
    MAX_PITCH_ANGLE = 0.25  # ~15 degrees - max head tilt for looking at camera
    
    def __init__(self):
        """Initialize gaze detector"""
        self.last_gaze_vector = None
        self.last_left_pupil = None
        self.last_right_pupil = None
    
    def detect(self, frame, face_results):
        """
        Detect gaze direction based on face detection results
        
        Args:
            frame: Input frame from camera
            face_results: FaceDetectionResult from FaceDetector
            
        Returns:
            GazeResult or None if gaze cannot be estimated
        """
        if frame is None or face_results is None:
            return None
            
        if face_results.is_blinking:
            # Don't update gaze during blinks
            if self.last_gaze_vector is not None:
                return GazeResult(
                    left_pupil=self.last_left_pupil,
                    right_pupil=self.last_right_pupil,
                    gaze_vector=self.last_gaze_vector
                )
            return None
        
        # If head pose is available, check if head is facing camera
        if face_results.head_pose is not None:
            roll, pitch, yaw = face_results.head_pose
            # If head is turned too far to the side, don't detect gaze
            if abs(yaw) > self.MAX_YAW_ANGLE or abs(pitch) > self.MAX_PITCH_ANGLE:
                return GazeResult(
                    left_pupil=(0, 0),  # Placeholder
                    right_pupil=(0, 0),  # Placeholder
                    gaze_vector=(0, 0),  # Neutral gaze
                    head_adjusted=True
                )
        
        # Find pupil positions (center of eye contours)
        left_pupil = self._find_pupil(frame, face_results.left_eye_landmarks)
        right_pupil = self._find_pupil(frame, face_results.right_eye_landmarks)
        
        if left_pupil is None or right_pupil is None:
            return None
            
        # Calculate basic gaze vector based on pupil positions relative to eye corners
        gaze_vector = self._calculate_gaze_vector(
            face_results.left_eye_landmarks,
            face_results.right_eye_landmarks,
            left_pupil,
            right_pupil
        )
        
        # Adjust gaze vector based on head pose if available
        head_adjusted = False
        if face_results.head_pose is not None:
            gaze_vector = self._adjust_gaze_for_head_pose(gaze_vector, face_results.head_pose)
            head_adjusted = True
        
        # Apply smoothing if we have previous gaze data
        if self.last_gaze_vector is not None:
            gaze_vector = (
                gaze_vector[0] * (1 - self.SMOOTH_FACTOR) + self.last_gaze_vector[0] * self.SMOOTH_FACTOR,
                gaze_vector[1] * (1 - self.SMOOTH_FACTOR) + self.last_gaze_vector[1] * self.SMOOTH_FACTOR
            )
        
        # Update last known values
        self.last_gaze_vector = gaze_vector
        self.last_left_pupil = left_pupil
        self.last_right_pupil = right_pupil
        
        return GazeResult(
            left_pupil=left_pupil,
            right_pupil=right_pupil,
            gaze_vector=gaze_vector,
            head_adjusted=head_adjusted
        )
    
    def _find_pupil(self, frame, eye_landmarks):
        """
        Find pupil center using image processing techniques
        
        Args:
            frame: Input frame
            eye_landmarks: Array of eye contour landmarks
            
        Returns:
            (x, y) coordinates of pupil center or None
        """
        # Create a mask of the eye region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        eye_region = np.array(eye_landmarks, dtype=np.int32)
        cv2.fillPoly(mask, [eye_region], 255)
        
        # Extract eye region
        eye = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert to grayscale
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate pupil (dark region)
        _, thresholded = cv2.threshold(gray_eye, 40, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours found, estimate pupil as center of eye contour
            return (
                int(np.mean(eye_landmarks[:, 0])),
                int(np.mean(eye_landmarks[:, 1]))
            )
        
        # Find the largest contour (likely the pupil)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            return None
            
        # Calculate center of mass
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy)
    
    def _calculate_gaze_vector(self, left_eye, right_eye, left_pupil, right_pupil):
        """
        Calculate normalized gaze vector based on pupil positions
        
        Args:
            left_eye: Left eye landmarks
            right_eye: Right eye landmarks
            left_pupil: Left pupil center coordinates
            right_pupil: Right pupil center coordinates
            
        Returns:
            (x, y) normalized gaze vector
        """
        # Calculate eye centers
        left_eye_center = (
            int(np.mean(left_eye[:, 0])),
            int(np.mean(left_eye[:, 1]))
        )
        right_eye_center = (
            int(np.mean(right_eye[:, 0])),
            int(np.mean(right_eye[:, 1]))
        )
        
        # Calculate displacement of pupils from eye centers
        left_dx = left_pupil[0] - left_eye_center[0]
        left_dy = left_pupil[1] - left_eye_center[1]
        right_dx = right_pupil[0] - right_eye_center[0]
        right_dy = right_pupil[1] - right_eye_center[1]
        
        # Average displacement
        dx = (left_dx + right_dx) / 2
        dy = (left_dy + right_dy) / 2
        
        # Normalize by eye size
        left_eye_width = np.max(left_eye[:, 0]) - np.min(left_eye[:, 0])
        right_eye_width = np.max(right_eye[:, 0]) - np.min(right_eye[:, 0])
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        left_eye_height = np.max(left_eye[:, 1]) - np.min(left_eye[:, 1])
        right_eye_height = np.max(right_eye[:, 1]) - np.min(right_eye[:, 1])
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        # Final normalized vector
        gaze_x = dx / (avg_eye_width / 2) if avg_eye_width > 0 else 0
        gaze_y = dy / (avg_eye_height / 2) if avg_eye_height > 0 else 0
        
        # Clip values to range [-1, 1]
        gaze_x = max(-1, min(1, gaze_x))
        gaze_y = max(-1, min(1, gaze_y))
        
        return (gaze_x, gaze_y)
        
    def _adjust_gaze_for_head_pose(self, gaze_vector, head_pose):
        """
        Adjust gaze vector based on head pose (roll, pitch, yaw)
        
        Args:
            gaze_vector: Original gaze vector (x, y)
            head_pose: Head pose as (roll, pitch, yaw) in radians
            
        Returns:
            Adjusted gaze vector (x, y)
        """
        roll, pitch, yaw = head_pose
        
        # Convert the 2D gaze vector to a 3D vector (assuming z=-1 for gaze direction)
        gaze_3d = np.array([gaze_vector[0], gaze_vector[1], -1.0])
        gaze_3d = gaze_3d / np.linalg.norm(gaze_3d)  # Normalize
        
        # Create rotation matrices for pitch and yaw (we ignore roll as it's less significant for gaze)
        # Pitch rotation (around x-axis)
        pitch_rot = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        
        # Yaw rotation (around y-axis)
        yaw_rot = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        
        # Roll rotation (around z-axis)
        roll_rot = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations
        rotation_matrix = roll_rot @ pitch_rot @ yaw_rot
        
        # Compute head direction vector (where the head is facing)
        head_direction = np.array([0, 0, -1.0])  # Facing forward
        head_direction = rotation_matrix @ head_direction
        
        # Calculate angle between head direction and camera direction
        # Camera direction is [0, 0, -1] (negative z-axis)
        camera_direction = np.array([0, 0, -1.0])
        head_camera_angle = np.arccos(np.clip(np.dot(head_direction, camera_direction) / 
                                     (np.linalg.norm(head_direction) * np.linalg.norm(camera_direction)), -1.0, 1.0))
        
        # If head is not facing camera (angle > threshold), force very high head weight
        if head_camera_angle > 0.2:  # ~11 degrees (stricter than before)
            head_weight = 0.95  # Almost entirely use head pose
        else:
            head_weight = self.HEAD_POSE_WEIGHT  # Use configured weight
        
        # Blend between the eye-based gaze vector and the head-based direction
        adjusted_gaze_3d = (
            gaze_3d * (1 - head_weight) + 
            head_direction * head_weight
        )
        adjusted_gaze_3d = adjusted_gaze_3d / np.linalg.norm(adjusted_gaze_3d)  # Normalize
        
        # Check if X component (horizontal gaze) is too large
        # If person is looking significantly to the side, adjust to look more forward
        if abs(adjusted_gaze_3d[0]) > 0.25:  # If horizontal component is significant
            side_factor = 0.5  # Reduce horizontal component by this factor
            adjusted_gaze_3d[0] *= side_factor  # Reduce side-to-side gaze
            # Renormalize after adjustment
            adjusted_gaze_3d = adjusted_gaze_3d / np.linalg.norm(adjusted_gaze_3d)
        
        # Project back to 2D space for display purposes
        # We assume a simple perspective projection
        adjusted_gaze_2d = (
            adjusted_gaze_3d[0] / -adjusted_gaze_3d[2],
            adjusted_gaze_3d[1] / -adjusted_gaze_3d[2]
        )
        
        # Clip values to range [-1, 1]
        adjusted_gaze_x = max(-1, min(1, adjusted_gaze_2d[0]))
        adjusted_gaze_y = max(-1, min(1, adjusted_gaze_2d[1]))
        
        return (adjusted_gaze_x, adjusted_gaze_y)
