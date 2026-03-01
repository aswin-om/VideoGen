import os
import random
import numpy as np
import cv2
import librosa
import torch
import torchvision.transforms as transforms
import datetime
from PIL import Image
from moviepy import VideoFileClip, AudioFileClip

# 1. Device Setup for MacBook M4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using Device: {device}")

# 2. Semantic Model: DINOv2 (Small - ViT-S/14)
print("Loading DINOv2 for Car Orbit Tracking...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_semantic_features(frame):
    with torch.no_grad():
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = preprocess(img).unsqueeze(0).to(device)
        features = model(img_t)
        return features.squeeze().cpu().numpy()

def parse_time(time_str):
    if isinstance(time_str, (int, float)): return float(time_str)
    parts = str(time_str).split(':')
    if len(parts) == 2: return int(parts[0]) * 60 + int(parts[1])
    try: return float(time_str)
    except: return 0

def center_crop_1_1(frame):
    h, w = frame.shape[:2]
    size = min(h, w)
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    return frame[start_y:start_y+size, start_x:start_x+size]

def estimate_frame_motion(cap, frame_idx, sample_step=2):
    """Estimate motion magnitude using a few forward frames (no backward seeking)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, prev_frame = cap.read()
    if not ret: return 0.0
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_accum = 0.0
    count = 0
    
    # Read a few frames forward to estimate motion
    for _ in range(sample_step):
        ret, curr_frame = cap.read()
        if not ret: break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        try:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_accum += mag.mean()
            count += 1
            prev_gray = curr_gray
        except:
            break
            
    return motion_accum / count if count > 0 else 0.0

def adaptive_frame_sampling(cap, total_v_frames, motion_threshold=2.0, max_interval=5, min_interval=1, progress=None):
    """Dynamically sample frames based on motion detection."""
    sampled_indices = []
    current_idx = 0
    
    while current_idx < total_v_frames:
        # Estimate motion at current position
        motion = estimate_frame_motion(cap, current_idx)
        
        # Adjust interval: low motion → larger step, high motion → smaller step
        if motion > motion_threshold:
            interval = min_interval  # High motion: extract more frames
        else:
            interval = max_interval  # Low motion: extract fewer frames
        
        sampled_indices.append(current_idx)
        current_idx += interval
    
    return sampled_indices

def apply_subtle_grain(frame, intensity=0.04):
    """Adds subtle cinematic film grain."""
    noise = np.random.normal(0, 255 * intensity, frame.shape).astype(np.int16)
    grainy_frame = frame.astype(np.int16) + noise
    return np.clip(grainy_frame, 0, 255).astype(np.uint8)

def generate_beat_sync_video(video_files, audio_file, start_time_str, clip_duration, zoom=1.0, motion_speed=0.8, progress=None):
    if progress: progress(0, desc="Initializing...")
    
    start_time = parse_time(start_time_str)
    duration_seconds = float(clip_duration)
    width, height, fps = 1080, 1080, 24
    output_dir = "output"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # --- 1. Audio Beat Analysis ---
    if progress: progress(0.05, desc="Analyzing Audio Beats...")
    print("Analyzing audio beats...")
    y, sr = librosa.load(audio_file, offset=start_time, duration=duration_seconds)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Convert beat times to video frame indices
    beat_indices = [int(t * fps) for t in beat_times]
    beat_indices = [b for b in beat_indices if b < int(duration_seconds * fps)]
    
    # Create a boolean mask for beats
    total_output_frames = int(duration_seconds * fps)
    is_beat_frame = np.zeros(total_output_frames, dtype=bool)
    for b in beat_indices:
        if b < total_output_frames:
            is_beat_frame[b] = True

    # --- 2. High-Precision Feature Extraction with Adaptive Sampling ---
    if progress: progress(0.1, desc="Analyzing Video Features...")
    print("AI is analyzing car angles and geometry with adaptive motion sampling...")
    frame_pool, feature_pool, video_ids = [], [], []
    
    total_videos = len(video_files)
    for v_idx, v_path in enumerate(video_files):
        if progress: progress(0.1 + (0.4 * (v_idx / total_videos)), desc=f"Processing video {v_idx+1}/{total_videos}")
        
        cap = cv2.VideoCapture(v_path)
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use adaptive sampling based on motion
        motion_threshold = 2.0 + (2.0 - motion_speed)  # Adjust threshold based on motion_speed param
        sampled_indices = adaptive_frame_sampling(cap, total_v_frames, motion_threshold=motion_threshold, max_interval=5, min_interval=1)
        
        for f_idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                cropped = center_crop_1_1(frame)
                frame_pool.append((v_path, f_idx))
                feature_pool.append(get_semantic_features(cropped))
                video_ids.append(v_idx)
        cap.release()
    
    feature_pool = np.array(feature_pool)
    video_ids = np.array(video_ids)

    if len(feature_pool) == 0:
        raise ValueError("No frames extracted from videos. Check inputs.")

    # --- 3. Build the "Semantic Path" (Sync to Beats) ---
    if progress: progress(0.5, desc="Computing Semantic Path...")
    print("Computing optimized motion trajectory...")
    
    path_indices = []
    
    # Parameters
    # On beats, we are more likely to switch (cut)
    # Off beats, we prefer continuity (same video, next frames)
    
    current_idx = random.randint(0, len(feature_pool) - 1)
    path_indices.append(current_idx)
    
    used_mask = np.zeros(len(feature_pool), dtype=bool)
    # Don't mark as used immediately so we can revisit if needed, but penalize
    
    # History buffer for semantic smoothing
    history = [feature_pool[current_idx]]
    lookback_window = 5
    
    for i in range(total_output_frames - 1):
        # Determine if this is a beat frame (opportunity to cut)
        is_beat = is_beat_frame[i+1] if i+1 < len(is_beat_frame) else False
        
        weights = np.exp(np.linspace(-1, 0, len(history)))
        weights /= weights.sum()
        acc_feat = np.average(history, axis=0, weights=weights)
        
        curr_vid = video_ids[current_idx]
        curr_f_idx = frame_pool[current_idx][1]
        
        # Calculate Semantic Distances
        dists = np.linalg.norm(feature_pool - acc_feat, axis=1)
        
        # Base penalties
        switch_penalties = np.zeros_like(dists)
        
        # Logic:
        # If BEAT: Encourage switching (lower penalty for diff video), or sharp semantic turn
        # If NO BEAT: Enforce continuity (high penalty for switching)
        
        if is_beat:
             # On beat: Reduce switch penalty to encourage cuts
             # But still want semantic relevance
             switch_penalties = (video_ids != curr_vid).astype(float) * 0.5 
             # Bonus for "visual impact" or change?
        else:
            # Off beat: High penalty for switching
            switch_penalties = (video_ids != curr_vid).astype(float) * 100.0 
        
        # Sequence Bonus (Frames that naturally follow)
        seq_bonuses = np.zeros_like(dists)
        # Check next few frames in pool to see if they are temporal successors
        # This is a heuristic; assuming pool is somewhat ordered per video
        # A better way is to explicitly find the index in pool that corresponds to (curr_vid, curr_f_idx + step)
        
        # Simplified continuity check:
        # If the next element in the pool is the same video and next timestamp, give huge bonus
        if current_idx + 1 < len(video_ids):
             if video_ids[current_idx+1] == curr_vid:
                 # Check frame gap
                 gap = frame_pool[current_idx+1][1] - curr_f_idx
                 if 0 < gap <= 5: # Close enough
                     seq_bonuses[current_idx+1] = 5.0 # Strong pull to continue
        
        # Combine costs: Minimize (Dist + SwitchPenalty - SeqBonus)
        total_cost = dists + switch_penalties - seq_bonuses
        
        # Avoid exact repetition of the immediate last few frames
        total_cost[current_idx] = np.inf 
        
        next_idx = np.argmin(total_cost)
        
        path_indices.append(next_idx)
        current_idx = next_idx
        
        # Update history
        history.append(feature_pool[current_idx])
        if len(history) > lookback_window:
            history.pop(0)

    # --- 4. Render the Path (Optimized I/O) ---
    if progress: progress(0.7, desc="Rendering Video...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_video_path = os.path.join(output_dir, f"raw_{timestamp}.mp4")
    out = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_usage_counter = np.zeros(len(video_files))
    
    print("Stitching frames with optimized I/O...")
    
    # Open all video captures once
    captures = {}
    for v_path in video_files:
        if v_path not in captures:
            captures[v_path] = cv2.VideoCapture(v_path)
            
    try:
        total_frames_render = len(path_indices)
        for i, idx in enumerate(path_indices):
            if progress and i % 10 == 0: 
                progress(0.7 + (0.25 * (i / total_frames_render)), desc="Rendering...")
                
            v_path, f_idx = frame_pool[idx]
            curr_vid = video_ids[idx]
            frame_usage_counter[curr_vid] += 1
            
            cap = captures[v_path]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = center_crop_1_1(frame)
                frame = cv2.resize(frame, (width, height))
                frame = apply_subtle_grain(frame)
                out.write(frame)
    finally:
        # Close all captures
        for cap in captures.values():
            cap.release()
        out.release()

    # --- 5. Final Stats & Audio Merge ---
    if progress: progress(0.95, desc="Merging Audio...")
    final_stats = {
        "total_scenes": len(video_files),
        "total_frames_indexed": len(frame_pool),
        "video_usage_seconds": {os.path.basename(v): round(frame_usage_counter[i]/fps, 2) for i,v in enumerate(video_files)}
    }

    final_output_path = os.path.join(output_dir, f"final_{timestamp}.mp4")
    try:
        v_clip = VideoFileClip(raw_video_path)
        # Use the original audio clip logic
        # Save temp audio segment
        temp_audio = "temp_audio.wav"
        sf.write(temp_audio, y, sr) # y is already the cut segment
        
        audio_sub = AudioFileClip(temp_audio)
        final = v_clip.with_audio(audio_sub)
        final.write_videofile(final_output_path, codec="libx264", audio_codec="aac", fps=fps, logger=None)
        
        v_clip.close(); audio_sub.close(); final.close()
        if os.path.exists(temp_audio): os.remove(temp_audio)
        
        if progress: progress(1.0, desc="Done!")
        return final_output_path, final_stats
    except Exception as e:
        print(f"Merge error: {e}")
        return raw_video_path, final_stats

