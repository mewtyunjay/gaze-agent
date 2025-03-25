from gaze_agent.detection.camera import Camera
from gaze_agent.detection.face import FaceDetector
from gaze_agent.detection.gaze import GazeDetector
from gaze_agent.utils.visualization import Visualizer
import cv2
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Gaze Detection System")
    parser.add_argument("--no-landmarks", action="store_true", help="Disable facial landmark visualization")
    parser.add_argument("--no-gaze", action="store_true", help="Disable gaze vector visualization")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize components
    camera = Camera(camera_id=args.camera)
    face_detector = FaceDetector()
    gaze_detector = GazeDetector()
    visualizer = Visualizer(
        show_landmarks=not args.no_landmarks,
        show_gaze=not args.no_gaze
    )
    
    print("Starting gaze detection...")
    print("Press 'q' to quit")
    
    # Main loop
    while True:
        success, frame = camera.get_frame()
        if not success or frame is None:
            continue
            
        # Detect face and eyes
        face_results = face_detector.detect(frame)
        
        # Detect gaze
        gaze_results = None
        if face_results:
            gaze_results = gaze_detector.detect(frame, face_results)
            
        # Visualize results
        output_frame = visualizer.draw_results(frame, face_results, gaze_results)
        
        # Display output
        cv2.imshow("Gaze Detection", output_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()