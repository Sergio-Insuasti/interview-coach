import cv2
import numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

def decode_frame(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

def preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalised = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalised, (3, 3), 0)
    return blurred

def detect_and_crop_face(frame: np.ndarray) -> np.ndarray | None:
    with mp_face.FaceDetection(min_detection_confidence=0.6) as detector:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)
        if not result.detections:
            return None
        det = result.detections[0]
        box = det.location_data.relative_bounding_box
        h, w = frame.shape[:2]
        x = max(0, int(box.xmin * w))
        y = max(0, int(box.ymin * h))
        box_width = int(box.width * w)
        box_height = int(box.height * h)
        return frame[y:y + box_height, x: x + box_width]
    
def extract_hog_features(face_crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_crop, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize = (64, 64),
        _blockSize = (16, 16),
        _blockStride = (8, 8),
        _cellSize = (8, 8),
        _nbins = 9        
    )
    return hog.compute(gray).flatten()