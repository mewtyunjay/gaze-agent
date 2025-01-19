# Gaze Detection Application

A real-time gaze detection application that tracks whether a user is looking at the camera, looking away, or has their eyes closed. The application uses computer vision to detect faces and analyze eye positions.

## Features

- Real-time face detection
- Eye gaze tracking
- Proximity detection (ensures user is close enough to camera)
- Visual feedback with face and eye landmarks
- Status display (Looking at Camera, Looking Away, Eyes Closed, Move Closer)

## Requirements

- Python 3.6+
- Webcam
- Required Python packages (install via `pip install -r requirements.txt`):
  - opencv-python
  - numpy
  - dlib
  - imutils

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the shape predictor file:
   - Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract the `.dat` file
   - Place `shape_predictor_68_face_landmarks.dat` in the project root directory

## Usage

1. Run the application:
```bash
python app.py
```

2. Position yourself in front of the webcam:
   - Ensure your face is clearly visible
   - Move closer if prompted
   - Center your face in the frame

3. The application will display:
   - Green box around your detected face
   - Yellow dots marking eye landmarks
   - Current status (Looking at Camera, Looking Away, etc.)

4. Press 'q' to quit the application

## Status Indicators

- **Looking at Camera**: Face is centered and at proper distance
- **Looking Away**: Face detected but not centered
- **Eyes Closed**: Eyes are detected as closed
- **Move Closer**: Face is too far from the camera
- **No Face Detected**: No face found in frame

## Troubleshooting

1. If the shape predictor file is missing, you'll see instructions to download it
2. Ensure proper lighting for better face detection
3. Check that your webcam is properly connected and accessible
4. If using a virtual environment, ensure it's activated before running

## License

[Your chosen license] 