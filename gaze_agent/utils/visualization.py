import cv2
import numpy as np
from typing import Tuple, Optional, List


class Visualizer:
    # Visualization settings
    FACE_BOX_COLOR = (0, 255, 0)  # Green
    EYE_COLOR = (255, 0, 0)  # Blue
    PUPIL_COLOR = (0, 0, 255)  # Red
    GAZE_COLOR = (255, 255, 0)  # Yellow
    LEFT_LASER_COLOR = (0, 0, 255)  # Red for left eye laser
    RIGHT_LASER_COLOR = (255, 0, 0)  # Blue for right eye laser
    TEXT_COLOR = (255, 255, 255)  # White
    LANDMARK_COLOR = (0, 255, 255)  # Cyan
    STATUS_GREEN = (0, 255, 0)  # Green for looking at screen
    STATUS_RED = (0, 0, 255)  # Red for looking away
    HEAD_POSE_X_COLOR = (0, 0, 255)  # Red for X-axis
    HEAD_POSE_Y_COLOR = (0, 255, 0)  # Green for Y-axis
    HEAD_POSE_Z_COLOR = (255, 0, 0)  # Blue for Z-axis
    
    # Default thickness values
    FACE_BOX_THICKNESS = 2
    EYE_CONTOUR_THICKNESS = 1
    PUPIL_RADIUS = 3
    GAZE_LINE_THICKNESS = 2
    LASER_THICKNESS = 3
    LANDMARK_RADIUS = 1
    HEAD_POSE_LINE_THICKNESS = 2
    
    # Screen region parameters (relative to frame size)
    SCREEN_REGION_WIDTH_RATIO = 0.5  # Width of the region considered as "looking at screen"
    SCREEN_REGION_HEIGHT_RATIO = 0.4  # Height of the region considered as "looking at screen"
    
    def __init__(self, show_landmarks=True, show_gaze=True, show_head_pose=True):
        """
        Initialize visualizer
        
        Args:
            show_landmarks: Whether to display facial landmarks
            show_gaze: Whether to display gaze vector
            show_head_pose: Whether to display head pose axes
        """
        self.show_landmarks = show_landmarks
        self.show_gaze = show_gaze
        self.show_head_pose = show_head_pose
    
    def draw_results(self, frame, face_results, gaze_results):
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame from camera
            face_results: FaceDetectionResult from FaceDetector or None
            gaze_results: GazeResult from GazeDetector or None
            
        Returns:
            Frame with visualizations
        """
        if frame is None:
            return frame
            
        # Create a copy of the frame to draw on
        vis_frame = frame.copy()
        
        # Define screen region for visualization
        h, w = frame.shape[:2]
        screen_region = self._get_screen_region(w, h)
        
        # Draw screen region
        self._draw_screen_region(vis_frame, screen_region)
        
        # Draw face detection results
        if face_results is not None:
            vis_frame = self._draw_face(vis_frame, face_results)
            vis_frame = self._draw_eyes(vis_frame, face_results)
            
            if self.show_landmarks:
                vis_frame = self._draw_facial_landmarks(vis_frame, face_results.landmarks)
            
            # Display blink status
            if face_results.is_blinking:
                self._draw_text(vis_frame, "BLINK DETECTED", (20, 70))
            
            # Display eye aspect ratios
            ear_text = f"EAR: L={face_results.left_eye_ratio:.2f}, R={face_results.right_eye_ratio:.2f}"
            self._draw_text(vis_frame, ear_text, (20, 40))
            
            # Draw head pose visualization if available
            if self.show_head_pose and face_results.head_pose is not None:
                vis_frame = self._draw_head_pose(vis_frame, face_results)
                
                # Display head pose angles
                roll, pitch, yaw = face_results.head_pose
                head_pose_text = f"Head: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°"
                self._draw_text(vis_frame, head_pose_text, (20, 100))
        
        # Draw gaze detection results
        is_looking_at_screen = False
        if gaze_results is not None and face_results is not None:
            if self.show_gaze:
                vis_frame = self._draw_eye_lasers(vis_frame, gaze_results, face_results)
            
            # Determine if looking at screen by projecting the gaze rays onto the screen
            is_looking_at_screen = self._is_looking_at_screen(
                vis_frame.shape[1], 
                vis_frame.shape[0],
                gaze_results, 
                screen_region
            )
            
            # Display if gaze is head-adjusted
            if gaze_results.head_adjusted:
                self._draw_text(vis_frame, "Head-adjusted gaze", (20, 130))
            
        # Draw status indicator
        vis_frame = self.draw_status_indicator(vis_frame, is_looking_at_screen)
        
        return vis_frame
    
    def _draw_face(self, frame, face_results):
        """Draw face bounding box"""
        x, y, w, h = face_results.face_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.FACE_BOX_COLOR, self.FACE_BOX_THICKNESS)
        return frame
    
    def _draw_eyes(self, frame, face_results):
        """Draw eye contours"""
        # Draw left eye contour
        cv2.polylines(
            frame, 
            [face_results.left_eye_landmarks.astype(np.int32)], 
            True, 
            self.EYE_COLOR, 
            self.EYE_CONTOUR_THICKNESS
        )
        
        # Draw right eye contour
        cv2.polylines(
            frame, 
            [face_results.right_eye_landmarks.astype(np.int32)], 
            True, 
            self.EYE_COLOR, 
            self.EYE_CONTOUR_THICKNESS
        )
        
        return frame
    
    def _draw_head_pose(self, frame, face_results):
        """
        Draw head pose axes visualization
        
        Args:
            frame: Input frame
            face_results: FaceDetectionResult containing head pose
            
        Returns:
            Frame with head pose visualization
        """
        if face_results.head_pose is None:
            return frame
            
        h, w = frame.shape[:2]
        
        # Use nose tip as origin for the pose axes
        nose_tip = None
        if 1 in face_results.landmarks:  # Assuming index 1 is nose tip in MediaPipe
            nose_tip = tuple(face_results.landmarks[1].astype(int))
        else:
            # If nose tip not available, use center of face
            x, y, width, height = face_results.face_bbox
            nose_tip = (x + width // 2, y + height // 2)
        
        # Camera matrix (estimated)
        focal_length = w
        camera_matrix = np.array(
            [[focal_length, 0, w / 2],
             [0, focal_length, h / 2],
             [0, 0, 1]], dtype=np.float32
        )
        
        # Get rotation from head pose
        roll, pitch, yaw = face_results.head_pose
        
        # Create rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations
        R = Rz @ Ry @ Rx
        
        # Define the axes in 3D space (length relative to face size)
        face_size = max(face_results.face_bbox[2], face_results.face_bbox[3])
        axis_length = face_size / 2
        
        # 3D axes
        axes = np.array([
            [axis_length, 0, 0],  # X-axis
            [0, axis_length, 0],  # Y-axis
            [0, 0, -axis_length]   # Z-axis (negative to point toward camera)
        ])
        
        # Project axes to image plane
        axes_2d = []
        for axis in axes:
            # Apply rotation
            rotated_axis = R @ axis
            
            # Calculate 2D projection (simple perspective)
            x = rotated_axis[0] + nose_tip[0]
            y = rotated_axis[1] + nose_tip[1]
            
            axes_2d.append((int(x), int(y)))
        
        # Draw the axes
        cv2.line(frame, nose_tip, axes_2d[0], self.HEAD_POSE_X_COLOR, self.HEAD_POSE_LINE_THICKNESS)  # X-axis
        cv2.line(frame, nose_tip, axes_2d[1], self.HEAD_POSE_Y_COLOR, self.HEAD_POSE_LINE_THICKNESS)  # Y-axis
        cv2.line(frame, nose_tip, axes_2d[2], self.HEAD_POSE_Z_COLOR, self.HEAD_POSE_LINE_THICKNESS)  # Z-axis
        
        return frame
    
    def _draw_facial_landmarks(self, frame, landmarks):
        """
        Draw facial landmarks on the frame
        
        Args:
            frame: Input frame
            landmarks: Array of facial landmark coordinates
            
        Returns:
            Frame with landmarks drawn
        """
        # Draw each landmark point
        for point in landmarks:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), self.LANDMARK_RADIUS, self.LANDMARK_COLOR, -1)
            
        return frame
    
    def _draw_eye_lasers(self, frame, gaze_results, face_results):
        """
        Draw laser beams from each eye following the gaze direction
        
        Args:
            frame: Input frame
            gaze_results: GazeResult object
            face_results: FaceDetectionResult object
            
        Returns:
            Frame with laser beams drawn
        """
        h, w = frame.shape[:2]
        
        # Get pupils
        left_pupil = gaze_results.left_pupil
        right_pupil = gaze_results.right_pupil
        
        # Calculate individual gaze directions from each eye
        # (Simplified: using the shared gaze vector for both eyes)
        gaze_x, gaze_y = gaze_results.gaze_vector
        
        # Scale the laser length based on frame size
        laser_length = max(w, h)
        
        # Draw left eye laser (red)
        left_endpoint = (
            int(left_pupil[0] + gaze_x * laser_length),
            int(left_pupil[1] + gaze_y * laser_length)
        )
        cv2.line(frame, left_pupil, left_endpoint, self.LEFT_LASER_COLOR, self.LASER_THICKNESS)
        
        # Draw right eye laser (blue)
        right_endpoint = (
            int(right_pupil[0] + gaze_x * laser_length),
            int(right_pupil[1] + gaze_y * laser_length)
        )
        cv2.line(frame, right_pupil, right_endpoint, self.RIGHT_LASER_COLOR, self.LASER_THICKNESS)
        
        # For visualization of the gaze point, calculate intersection with bottom/top/left/right of frame
        left_intersection = self._get_line_frame_intersection(left_pupil, left_endpoint, w, h)
        right_intersection = self._get_line_frame_intersection(right_pupil, right_endpoint, w, h)
        
        # Draw intersection points (where the lasers hit the edge of the frame)
        if left_intersection:
            cv2.circle(frame, left_intersection, 5, self.LEFT_LASER_COLOR, -1)
        
        if right_intersection:
            cv2.circle(frame, right_intersection, 5, self.RIGHT_LASER_COLOR, -1)
            
        return frame
    
    def _get_line_frame_intersection(self, start_point, end_point, frame_width, frame_height):
        """
        Find where a line (laser) intersects with the frame boundary
        
        Args:
            start_point: Starting point of the line (pupil)
            end_point: End point of the line
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Intersection point or None if no intersection
        """
        # Convert to floats for line equation
        x1, y1 = float(start_point[0]), float(start_point[1])
        x2, y2 = float(end_point[0]), float(end_point[1])
        
        # Check if points are the same
        if x1 == x2 and y1 == y2:
            return None
        
        # Line parameters
        if x2 - x1 == 0:  # Vertical line
            m = float('inf')
            b = x1
        else:
            m = (y2 - y1) / (x2 - x1)  # Slope
            b = y1 - m * x1  # Y-intercept
        
        # Intersection points with frame boundaries
        intersections = []
        
        # Top edge (y = 0)
        if m != 0:
            x_top = (0 - b) / m if m != float('inf') else x1
            if 0 <= x_top <= frame_width and self._is_between(y1, y2, 0):
                intersections.append((int(x_top), 0))
        
        # Bottom edge (y = frame_height)
        if m != 0:
            x_bottom = (frame_height - b) / m if m != float('inf') else x1
            if 0 <= x_bottom <= frame_width and self._is_between(y1, y2, frame_height):
                intersections.append((int(x_bottom), frame_height))
        
        # Left edge (x = 0)
        if m != float('inf'):
            y_left = b if m != 0 else y1
            if 0 <= y_left <= frame_height and self._is_between(x1, x2, 0):
                intersections.append((0, int(y_left)))
        
        # Right edge (x = frame_width)
        if m != float('inf'):
            y_right = m * frame_width + b if m != 0 else y1
            if 0 <= y_right <= frame_height and self._is_between(x1, x2, frame_width):
                intersections.append((frame_width, int(y_right)))
        
        # Find the closest intersection point in the direction of the ray
        if not intersections:
            return None
            
        # Calculate the vector from start to end
        direction_x, direction_y = x2 - x1, y2 - y1
        
        # Find the intersection point that is in the direction of the ray
        valid_intersections = []
        for point in intersections:
            vector_x, vector_y = point[0] - x1, point[1] - y1
            # Check if the vectors point in the same direction
            if (vector_x * direction_x >= 0) and (vector_y * direction_y >= 0):
                valid_intersections.append(point)
        
        if not valid_intersections:
            return None
            
        # Return the first valid intersection
        return valid_intersections[0]
    
    def _is_between(self, a, b, c):
        """Check if c is between a and b (or equal to either)"""
        return (a <= c <= b) or (b <= c <= a)
    
    def _get_screen_region(self, frame_width, frame_height):
        """
        Define a small region in the center of the frame that represents the camera area
        
        Args:
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Tuple of (x, y, width, height) representing the camera region
        """
        # Much smaller region focused on camera location
        region_width = int(frame_width * 0.1)  # 10% of frame width (reduced from 15%)
        region_height = int(frame_height * 0.1)  # 10% of frame height (reduced from 15%)
        
        x = (frame_width - region_width) // 2
        y = (frame_height - region_height) // 2
        
        return (x, y, region_width, region_height)
    
    def _draw_screen_region(self, frame, screen_region):
        """
        Draw a rectangle representing the camera region
        
        Args:
            frame: Input frame
            screen_region: Tuple of (x, y, width, height)
            
        Returns:
            Frame with camera region drawn
        """
        x, y, w, h = screen_region
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),  # Blue color
            1  # Thin line
        )
        return frame
    
    def _is_looking_at_screen(self, frame_width, frame_height, gaze_results, screen_region):
        """
        Determine if either eye's laser intersects with the camera region
        
        Args:
            frame_width: Width of the frame
            frame_height: Height of the frame
            gaze_results: GazeResult object
            screen_region: Tuple of (x, y, width, height)
            
        Returns:
            Boolean indicating if looking at camera
        """
        # Extract screen region coordinates
        screen_x, screen_y, screen_w, screen_h = screen_region
        
        # Get pupils and gaze vector
        left_pupil = gaze_results.left_pupil
        right_pupil = gaze_results.right_pupil
        gaze_x, gaze_y = gaze_results.gaze_vector
        
        # Calculate the angle of the gaze vector (in degrees)
        gaze_angle = np.degrees(np.arctan2(gaze_y, gaze_x))
        
        # Reject gazes with significant horizontal component (looking sideways)
        # Allow only gazes within ±20 degrees from vertical (±90° is vertical)
        if abs(abs(gaze_angle) - 90) > 20:
            return False
            
        # Calculate laser endpoints (far beyond the frame)
        laser_length = max(frame_width, frame_height) * 2
        
        left_endpoint = (
            int(left_pupil[0] + gaze_x * laser_length),
            int(left_pupil[1] + gaze_y * laser_length)
        )
        
        right_endpoint = (
            int(right_pupil[0] + gaze_x * laser_length),
            int(right_pupil[1] + gaze_y * laser_length)
        )
        
        # Check if left eye laser intersects with screen region
        left_intersection = self._line_rect_intersection(
            left_pupil, left_endpoint, 
            (screen_x, screen_y), 
            (screen_x + screen_w, screen_y + screen_h)
        )
        
        # Check if right eye laser intersects with screen region
        right_intersection = self._line_rect_intersection(
            right_pupil, right_endpoint, 
            (screen_x, screen_y), 
            (screen_x + screen_w, screen_y + screen_h)
        )
        
        # Looking at camera only if BOTH lasers intersect with the camera region (stricter)
        return left_intersection and right_intersection
    
    def _line_rect_intersection(self, line_start, line_end, rect_top_left, rect_bottom_right):
        """
        Check if a line intersects with a rectangle
        
        Args:
            line_start: Starting point of the line (x, y)
            line_end: End point of the line (x, y)
            rect_top_left: Top-left point of the rectangle (x, y)
            rect_bottom_right: Bottom-right point of the rectangle (x, y)
            
        Returns:
            Boolean indicating if the line intersects with the rectangle
        """
        # Rectangle edges
        x1, y1 = rect_top_left
        x2, y2 = rect_bottom_right
        
        # Check if the line intersects with any of the rectangle's edges
        edges = [
            [(x1, y1), (x2, y1)],  # Top edge
            [(x1, y2), (x2, y2)],  # Bottom edge
            [(x1, y1), (x1, y2)],  # Left edge
            [(x2, y1), (x2, y2)]   # Right edge
        ]
        
        for edge_start, edge_end in edges:
            if self._line_segments_intersect(line_start, line_end, edge_start, edge_end):
                return True
        
        # Also check if the starting point is inside the rectangle
        if (x1 <= line_start[0] <= x2 and y1 <= line_start[1] <= y2):
            return True
            
        return False
    
    def _line_segments_intersect(self, p1, p2, p3, p4):
        """
        Check if two line segments intersect
        
        Args:
            p1, p2: Points defining the first line segment
            p3, p4: Points defining the second line segment
            
        Returns:
            Boolean indicating if the line segments intersect
        """
        def orientation(p, q, r):
            """Calculate orientation of triplet (p, q, r)"""
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise
        
        def on_segment(p, q, r):
            """Check if point q lies on line segment 'pr'"""
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        # Calculate orientations
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
            
        # Special cases
        if o1 == 0 and on_segment(p1, p3, p2): return True
        if o2 == 0 and on_segment(p1, p4, p2): return True
        if o3 == 0 and on_segment(p3, p1, p4): return True
        if o4 == 0 and on_segment(p3, p2, p4): return True
        
        return False
    
    def draw_status_indicator(self, frame, is_looking_at_screen):
        """
        Draw a colored indicator showing whether the user is looking at the camera
        
        Args:
            frame: Input frame
            is_looking_at_screen: Boolean indicating if the user is looking at the camera
            
        Returns:
            Frame with status indicator
        """
        h, w = frame.shape[:2]
        indicator_size = 30
        indicator_margin = 20
        indicator_pos = (w - indicator_size - indicator_margin, indicator_margin)
        
        # Draw circle indicator
        color = self.STATUS_GREEN if is_looking_at_screen else self.STATUS_RED
        cv2.circle(
            frame,
            (indicator_pos[0] + indicator_size // 2, indicator_pos[1] + indicator_size // 2),
            indicator_size // 2,
            color,
            -1
        )
        
        # Add text label
        status_text = "Looking at camera" if is_looking_at_screen else "Not looking at camera"
        self._draw_text(
            frame,
            status_text,
            (indicator_pos[0] - 120, indicator_pos[1] + indicator_size // 2 + 5)
        )
        
        return frame
    
    def _draw_text(self, frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7):
        """Helper method to draw text with background"""
        # Get text size
        text_size = cv2.getTextSize(text, font, scale, 1)[0]
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (position[0] - 5, position[1] - text_size[1] - 5),
            (position[0] + text_size[0] + 5, position[1] + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            position,
            font,
            scale,
            self.TEXT_COLOR,
            1,
            cv2.LINE_AA
        )
        
        return frame
