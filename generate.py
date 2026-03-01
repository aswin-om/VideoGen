import os
import random
import numpy as np
import cv2
import librosa
from moviepy import VideoFileClip, AudioFileClip

def parse_time(time_str):
    if isinstance(time_str, (int, float)):
        return float(time_str)
    parts = str(time_str).split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    try:
        return float(time_str)
    except ValueError:
        return 0

def detect_scenes(video_path, threshold=25.0):
    """
    Detects scene changes by comparing frame intensity differences.
    """
    cap = cv2.VideoCapture(video_path)
    scene_indices = [0]
    prev_frame = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only check every 3rd frame for speed
        if frame_count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                mean_diff = np.mean(diff)
                if mean_diff > threshold:
                    scene_indices.append(frame_count)
            prev_frame = gray
        
        frame_count += 1
    
    cap.release()
    return scene_indices

def get_center_crop(img, target_w, target_h):
    img_h, img_w = img.shape[:2]
    target_aspect = target_w / target_h
    img_aspect = img_w / img_h
    
    if img_aspect > target_aspect:
        new_w = int(img_h * target_aspect)
        x_offset = (img_w - new_w) // 2
        cropped_img = img[:, x_offset:x_offset+new_w]
    elif img_aspect < target_aspect:
        new_h = int(img_w / target_aspect)
        y_offset = (img_h - new_h) // 2
        cropped_img = img[y_offset:y_offset+new_h, :]
    else:
        cropped_img = img
        
    return cv2.resize(cropped_img, (target_w, target_h))

def generate_beat_sync_video(video_files, audio_file, start_time_str, clip_duration, output_dir="output"):
    start_time = parse_time(start_time_str)
    duration_seconds = float(clip_duration)
    width, height = 1080, 1080
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Audio Analysis (BPM & Beats)
    print(f"Analyzing audio: {audio_file}")
    # librosa.load arguments: path, offset, duration
    y, sr = librosa.load(audio_file, offset=start_time, duration=duration_seconds)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    if len(beat_times) == 0:
        tempo = float(tempo[0]) if isinstance(tempo, (list, tuple, np.ndarray)) else float(tempo)
        beat_times = np.arange(0, duration_seconds, 60.0 / max(tempo, 60.0))
    
    print(f"Detected BPM: {tempo}. Total beats: {len(beat_times)}")
    
    # 2. Extract Scenes
    all_scenes = []
    for v_path in video_files:
        print(f"Processing scenes in {v_path}...")
        scenes = detect_scenes(v_path)
        cap = cv2.VideoCapture(v_path)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(len(scenes)):
            s = scenes[i]
            e = scenes[i+1] if i+1 < len(scenes) else total_f
            if e - s > 15: # Filter out very short clips
                all_scenes.append({'path': v_path, 'start': s, 'end': e})
        cap.release()

    if not all_scenes:
        print("No scenes found, using full videos as scenes.")
        for v_path in video_files:
            cap = cv2.VideoCapture(v_path)
            all_scenes.append({'path': v_path, 'start': 0, 'end': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})
            cap.release()

    # 3. Build Video with Motion Effect
    fps = 30
    raw_video_path = os.path.join(output_dir, "raw_generated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_video_path, fourcc, fps, (width, height))

    # We map beats to scene changes
    # Ensure beats are relative to the start of the audio segment (already handled by librosa load offset)
    beat_checkpoints = sorted(list(set([0.0] + list(beat_times) + [duration_seconds])))
    
    total_frames = 0
    for i in range(len(beat_checkpoints) - 1):
        interval_start = beat_checkpoints[i]
        interval_end = beat_checkpoints[i+1]
        num_frames = int((interval_end - interval_start) * fps)
        
        if num_frames <= 0: continue
        
        # New scene for every beat
        scene = random.choice(all_scenes)
        cap = cv2.VideoCapture(scene['path'])
        
        # Pick a random starting point in the scene that has enough buffer
        # Amplitude of motion is about 5-10 frames
        amplitude = 6
        scene_start = scene['start']
        scene_end = scene['end']
        
        # Start somewhere safe in the middle/start of the clip
        base_frame = random.randint(scene_start + amplitude, max(scene_start + amplitude + 1, scene_end - amplitude - 1))
        
        for f_idx in range(num_frames):
            # Back and forth: 0 -> 1 -> 2 -> 1 -> 0 ...
            # Using a triangle wave for ping-pong
            # period = 10 frames
            period = 8
            offset = amplitude - abs((f_idx % (2 * period)) - period)
            
            target_f = base_frame + offset
            target_f = max(scene_start, min(target_f, scene_end - 1))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            ret, frame = cap.read()
            if ret:
                out.write(get_center_crop(frame, width, height))
                total_frames += 1
            else:
                break
        cap.release()

    out.release()

    # 4. Final Merge
    final_output_path = os.path.join(output_dir, "sync_motion_final.mp4")
    try:
        video_clip = VideoFileClip(raw_video_path)
        audio_clip = AudioFileClip(audio_file).subclipped(start_time, start_time + (total_frames / fps))
        
        final_clip = video_clip.with_audio(audio_clip)
        final_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac", fps=fps, logger=None)
        
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        return final_output_path
    except Exception as e:
        print(f"Error: {e}")
        return raw_video_path
