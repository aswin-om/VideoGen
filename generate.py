import os
import random
import numpy as np
import glob
import cv2
import librosa
from ultralytics import YOLO
from moviepy import VideoFileClip, AudioFileClip

# These will be computed dynamically
fps = 24
height = 1080
width = 1080
duration_seconds = 30

song_path = "bunt-trippin.mp3"
start_time_str = "0:50"
end_time_str = "1:20"

def parse_time(time_str):
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(time_str)

start_time = parse_time(start_time_str)
end_time = parse_time(end_time_str)
duration_seconds = end_time - start_time

output_dir = "output"
frames_dir = "all_frames"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize YOLO model (will automatically download yolov8n.pt if not present)
model = YOLO('yolov8n.pt')

def get_focal_point(img):
    """
    Uses YOLO to detect the main subject (bike/vehicle/person) 
    and returns its center coordinates (x, y).
    If nothing is found, returns the center of the image.
    """
    # Disable verbose output to keep console clean
    results = model(img, verbose=False)
    boxes = results[0].boxes
    
    img_h, img_w = img.shape[:2]
    focal_x, focal_y = img_w // 2, img_h // 2
    
    max_score = 0
    best_box = None
    
    for box in boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        
        # Prioritize bicycles (1) and motorcycles (3)
        score = area
        if cls in [1, 3]:  
            score *= 10
        elif cls in [0, 2, 5, 7]: # person, car, bus, truck
            score *= 2
            
        if score > max_score:
            max_score = score
            best_box = (x1, y1, x2, y2)
            
    if best_box:
        x1, y1, x2, y2 = best_box
        focal_x = int((x1 + x2) / 2)
        focal_y = int((y1 + y2) / 2)
        
    return focal_x, focal_y

def calculate_audio_features(audio_path, start_t, end_t):
    """
    Load the specific segment of audio, calculate its tempo, 
    and derive an appropriate frames per second.
    """
    print(f"Loading audio '{audio_path}' from {start_t}s to {end_t}s...")
    # Load audio segment
    y, sr = librosa.load(audio_path, offset=start_t, duration=(end_t - start_t))
    
    # Calculate tempo (beats per minute)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo[0]) if isinstance(tempo, (list, tuple, np.ndarray)) else float(tempo)
    print(f"Detected tempo: {tempo:.2f} BPM")
    
    # We prefer a tempo above 120 so the video is fast-paced
    # If the detected tempo is slow, we can double it to get the "double time" feel
    target_tempo = tempo
    while target_tempo < 120:
        target_tempo *= 2
        
    print(f"Adjusted target tempo: {target_tempo:.2f} BPM")
    
    # Calculate FPS based on the tempo so frames change on the beat
    # For example, if tempo is 120 BPM, that's 2 beats per second.
    # We might want frames to change every half beat, or on the beat.
    # Let's say we want a frame duration to match 1/4 of a beat for fast cuts.
    # bps = target_tempo / 60.0
    # fps = bps * 4 # e.g., 2 bps * 4 = 8 fps (too slow for smooth video?)
    # Wait, the user wants the video FPS to match the tempo. 
    # Usually we want the video output to be a standard frame rate (like 24 or 30), 
    # and we hold frames for exactly X seconds...
    # BUT if the user strictly wants the FPS calculated from tempo:
    
    beats_per_second = target_tempo / 60.0
    
    # Let's have the framerate be a multiple of the beats so they sync naturally.
    # Say we want exactly 4 frames per beat:
    calculated_fps = int(beats_per_second * 4) 
    
    # Ensure a reasonable minimum/maximum bounds for video FPS
    calculated_fps = max(12, min(calculated_fps, 60))
    
    print(f"Calculated FPS based on tempo: {calculated_fps}")
    return calculated_fps

def generate_random_video():
    global fps
    
    if os.path.exists(song_path):
        fps = calculate_audio_features(song_path, start_time, end_time)
    else:
        print(f"Warning: Audio file '{song_path}' not found. Using default FPS: {fps}")
        
    print(f"Scanning for frames in '{frames_dir}'...")
    # Support common image formats
    all_frames = glob.glob(os.path.join(frames_dir, "*.jpg"))
    all_frames.extend(glob.glob(os.path.join(frames_dir, "*.png")))
    all_frames.extend(glob.glob(os.path.join(frames_dir, "*.jpeg")))
    
    if not all_frames:
        print(f"Error: No frames found in '{frames_dir}' folder.")
        return

    num_frames_needed = duration_seconds * fps
    print(f"Total available frames: {len(all_frames)}")
    print(f"Target video duration: {duration_seconds}s at {fps} fps ({num_frames_needed} frames)")

    # Select random frames (with replacement if we need more frames than available)
    if len(all_frames) >= num_frames_needed:
        selected_frames = random.sample(all_frames, num_frames_needed)
    else:
        selected_frames = random.choices(all_frames, k=num_frames_needed)

    output_path = os.path.join(output_dir, "random_video.mp4")
    
    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Starting video generation at: {output_path}")

    for idx, frame_path in enumerate(selected_frames, start=1):
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
            
        # Get the focal point using YOLO
        focal_x, focal_y = get_focal_point(img)
            
        # Center crop frame to match target aspect ratio based on focal point
        img_h, img_w = img.shape[:2]
        target_aspect = width / height
        img_aspect = img_w / img_h
        
        if img_aspect > target_aspect:
            # Image is wider than target aspect
            new_w = int(img_h * target_aspect)
            x_offset = int(focal_x - (new_w / 2))
            # Clamp offset to make sure crop box stays inside image boundaries
            x_offset = max(0, min(x_offset, img_w - new_w))
            cropped_img = img[:, x_offset:x_offset+new_w]
        elif img_aspect < target_aspect:
            # Image is taller than target aspect
            new_h = int(img_w / target_aspect)
            y_offset = int(focal_y - (new_h / 2))
            # Clamp offset to make sure crop box stays inside image boundaries
            y_offset = max(0, min(y_offset, img_h - new_h))
            cropped_img = img[y_offset:y_offset+new_h, :]
        else:
            cropped_img = img
            
        # Resize frame to match target dimensions
        resized_img = cv2.resize(cropped_img, (width, height))
        out.write(resized_img)
        
        if idx % 50 == 0 or idx == num_frames_needed:
            print(f"Processed {idx}/{num_frames_needed} frames")

    out.release()
    print("Done! Video successfully created.")
    
    # Now merge the audio
    if os.path.exists(song_path):
        print(f"Merging audio from {start_time_str} to {end_time_str}...")
        final_output_path = os.path.join(output_dir, "random_video_with_audio.mp4")
        
        try:
            video_clip = VideoFileClip(output_path)
            # In moviepy v2+, subclip is replaced with subclipped
            audio_clip = AudioFileClip(song_path).subclipped(start_time, end_time)
            
            # Ensure audio matches video duration
            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclipped(0, video_clip.duration)
                
            final_clip = video_clip.with_audio(audio_clip)
            final_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
            
            print(f"Final video with audio saved to: {final_output_path}")
            
            # Close clips to free memory
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
        except Exception as e:
            print(f"Error merging audio: {e}")

if __name__ == "__main__":
    generate_random_video()