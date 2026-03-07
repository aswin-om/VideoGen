import os
import numpy as np
import cv2

_yolo_model = None
_TRACK_CLASSES = [0, 1, 2, 3]  # person, bicycle, car, motorcycle


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
        _yolo_model = YOLO(model_path)
    return _yolo_model


def prescan_person_positions(video_files):
    """
    Pre-scan ALL frames of each video with YOLO detection for
    people, bicycles, cars, and motorcycles.
    For people: refines to eye-center using face/eye cascades.
    For vehicles: centers on bounding box center.
    Returns: {video_path: {frame_idx: (cx, cy)}} in pixel coords.
    """
    model = _get_yolo_model()
    cascade_dir = os.path.join(os.path.dirname(cv2.__file__), "data")
    face_cascade = cv2.CascadeClassifier(
        os.path.join(cascade_dir, "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(
        os.path.join(cascade_dir, "haarcascade_eye.xml"))

    person_cache = {}

    for v_path in video_files:
        detections = {}
        cap = cv2.VideoCapture(v_path)
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, classes=_TRACK_CLASSES, conf=0.35,
                                    verbose=False, imgsz=320)

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                areas = (boxes.xywh[:, 2] * boxes.xywh[:, 3]).cpu().numpy()
                best = int(np.argmax(areas))
                cls_id = int(boxes.cls[best].cpu().item())
                x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int).tolist()

                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if cls_id == 0:
                    person_roi = frame[y1:y2, x1:x2]
                    focus = _detect_eyes_in_roi(person_roi, face_cascade, eye_cascade)
                    if focus is not None:
                        detections[idx] = (x1 + focus[0], y1 + focus[1])
                    else:
                        detections[idx] = ((x1 + x2) // 2, y1 + (y2 - y1) // 4)
                else:
                    detections[idx] = ((x1 + x2) // 2, (y1 + y2) // 2)

            idx += 1

        cap.release()
        person_cache[v_path] = detections

    return person_cache


def _detect_eyes_in_roi(person_roi, face_cascade, eye_cascade):
    """Detect eyes within a person ROI. Returns (cx, cy) relative to ROI, or None."""
    if person_roi.size == 0:
        return None

    gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape[:2]

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4,
                                           minSize=(rw // 8, rh // 8))
    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_roi_gray = gray[fy:fy + fh, fx:fx + fw]

    eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(fw // 6, fh // 6))

    if len(eyes) >= 2:
        eyes_sorted = sorted(eyes, key=lambda e: e[0])[:2]
        ex1, ey1, ew1, eh1 = eyes_sorted[0]
        ex2, ey2, ew2, eh2 = eyes_sorted[1]
        mid_x = fx + (ex1 + ew1 // 2 + ex2 + ew2 // 2) // 2
        mid_y = fy + (ey1 + eh1 // 2 + ey2 + eh2 // 2) // 2
        return (mid_x, mid_y)

    return (fx + fw // 2, fy + fh // 3)


def analyze_video_motion(video_files, max_samples=25):
    """
    Measure motion intensity per video using fast frame differencing.
    Optimized for CPU/M2 by avoiding Farneback optical flow.
    """
    motion_scores = {}
    for v_path in video_files:
        try:
            cap = cv2.VideoCapture(v_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 2:
                motion_scores[v_path] = 0.0
                cap.release()
                continue

            # Sample fewer frames for faster analysis
            sample_indices = np.linspace(0, total_frames - 2, 
                                         min(max_samples, total_frames - 1), dtype=int)
            motion_values = []
            
            # Pre-read first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_indices[0]))
            ret, prev_frame = cap.read()
            if not ret:
                motion_scores[v_path] = 0.5
                cap.release()
                continue
            
            # Downscale and gray for speed
            prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (160, 120)), cv2.COLOR_BGR2GRAY)

            for idx in sample_indices[1:]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
                # Use absolute difference - much faster than optical flow
                diff = cv2.absdiff(prev_gray, gray)
                motion_values.append(float(np.mean(diff)))
                prev_gray = gray

            cap.release()
            if motion_values:
                # Normalize score to 0.0 - 1.0 range
                avg_diff = np.mean(motion_values)
                motion_scores[v_path] = round(min(1.0, avg_diff / 20.0), 2)
            else:
                motion_scores[v_path] = 0.0
        except Exception as e:
            print(f"Video motion analysis error for {v_path}: {e}")
            motion_scores[v_path] = 0.5

    return motion_scores
