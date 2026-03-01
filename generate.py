import os
import random
import numpy as np
import cv2
import librosa
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import datetime
from PIL import Image
from moviepy import VideoFileClip, AudioFileClip
from scenedetect import detect, ContentDetector

# 1. Device Setup for MacBook M4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using Device: {device}")

# 2. Semantic Model: DINOv2 (Small - ViT-S/14)
print("Loading DINOv2...")
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

class VideoStabilizer:
    def __init__(self, smoothing=0.85):
        self.last_gray = None
        self.dx, self.dy = 0, 0
        self.smoothing = smoothing

    def reset(self):
        self.last_gray = None
        self.dx, self.dy = 0, 0

    def process(self, frame, target_w, target_h, zoom=1.0):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patch_size = 512
        if h < patch_size or w < patch_size: patch_size = min(h, w) // 2 * 2
        y1, x1 = (h - patch_size) // 2, (w - patch_size) // 2
        curr_patch = gray[y1:y1+patch_size, x1:x1+patch_size].astype(np.float32)
        
        if self.last_gray is not None:
            shift, _ = cv2.phaseCorrelate(self.last_gray, curr_patch)
            if abs(shift[0]) < 100 and abs(shift[1]) < 100:
                self.dx += shift[0]
                self.dy += shift[1]
            self.dx *= self.smoothing
            self.dy *= self.smoothing
            
        self.last_gray = curr_patch
        side = min(h, w)
        crop_size = int(side / zoom)
        cx, cy = w // 2 + self.dx, h // 2 + self.dy
        half = crop_size // 2
        cx = np.clip(cx, half, w - half)
        cy = np.clip(cy, half, h - half)
        x1, y1 = int(cx - half), int(cy - half)
        cropped = frame[y1:y1+crop_size, x1:x1+crop_size]
        return cv2.resize(cropped, (target_w, target_h))

def generate_beat_sync_video(video_files, audio_file, start_time_str, clip_duration, zoom=1.0, motion_speed=0.5):
    start_time = parse_time(start_time_str)
    duration_seconds = float(clip_duration)
    width, height, fps = 1080, 1080, 24
    output_dir = "output"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    print("Indexing semantic frames...")
    frame_pool, feature_pool, video_ids = [], [], []
    total_scenes_count = 0
    
    for v_idx, v_path in enumerate(video_files):
        scene_list = detect(v_path, ContentDetector(threshold=27.0))
        cap = cv2.VideoCapture(v_path)
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not scene_list: 
            scene_list = [(0, total_v_frames)]
        else: 
            scene_list = [(s[0].get_frames(), s[1].get_frames()) for s in scene_list]
        
        total_scenes_count += len(scene_list)

        for s_start, s_end in scene_list:
            # Step 2 for better slow-motion granularity without excessive memory/time
            for f_idx in range(s_start, s_end, 2):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    frame_pool.append((v_path, f_idx))
                    feature_pool.append(get_semantic_features(frame))
                    video_ids.append(v_idx)
        cap.release()
    
    feature_pool = np.array(feature_pool)
    video_ids = np.array(video_ids)
    
    y, sr = librosa.load(audio_file, offset=start_time, duration=duration_seconds)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_frame_indices = [int(bt * fps) for bt in librosa.frames_to_time(beat_frames, sr=sr)]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_video_path = os.path.join(output_dir, f"raw_{timestamp}.mp4")
    out = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    total_output_frames = int(duration_seconds * fps)
    
    # Use float for pool index to allow precise slow-motion control
    pool_idx_float = float(random.randint(0, len(frame_pool) - 1))
    
    # Track usage of each video
    video_usage = np.zeros(len(video_files)) # Beat-based jumps
    frame_usage_counter = np.zeros(len(video_files)) # Total frames contributed
    
    stabilizer = VideoStabilizer()
    last_v_path, last_f_idx = None, -1
    
    for f_num in range(total_output_frames):
        current_idx = int(pool_idx_float) % len(frame_pool)
        curr_vid = video_ids[current_idx]
        v_path, f_idx = frame_pool[current_idx]
        
        frame_usage_counter[curr_vid] += 1
        
        if v_path != last_v_path or abs(f_idx - last_f_idx) > 10:
            stabilizer.reset()

        cap = cv2.VideoCapture(v_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read(); cap.release()
        
        if not ret:
            pool_idx_float = float(random.randint(0, len(frame_pool)-1))
            continue

        out.write(stabilizer.process(frame, width, height, zoom=zoom))
        last_v_path, last_f_idx = v_path, f_idx

        if f_num in beat_frame_indices:
            # Beat Jump: Intelligent Match Cut with Variety Enforcement
            curr_feat = feature_pool[current_idx]
            dists = np.linalg.norm(feature_pool - curr_feat, axis=1)
            
            # Variety Enforcement: Apply a large penalty to videos already used
            # This makes "fresh" videos more attractive
            penalty = video_usage[video_ids] * 5.0 # Large penalty factor
            dists += penalty
            
            # Prefer jumping to a DIFFERENT video
            dists[video_ids == curr_vid] += 10.0 
            
            # Select from the top k best matches after penalty
            top_k = min(50, len(frame_pool) - 1)
            candidates = np.argpartition(dists, top_k)[:top_k]
            new_idx = random.choice(candidates)
            
            pool_idx_float = float(new_idx)
            video_usage[video_ids[new_idx]] += 1
        else:
            # Smooth Sequential Progression
            pool_idx_float += motion_speed

    out.release()

    # Calculate final stats: how many seconds each video actually appeared in the output
    final_stats = {
        "total_scenes": total_scenes_count,
        "total_frames_indexed": len(frame_pool),
        "video_usage_seconds": {}
    }
    
    for i, v_path in enumerate(video_files):
        v_name = os.path.basename(v_path)
        # Accurate calculation: total frames contributed by this video / fps
        final_stats["video_usage_seconds"][v_name] = round(frame_usage_counter[i] / fps, 2)

    final_output_path = os.path.join(output_dir, f"final_{timestamp}.mp4")
    try:
        v_clip = VideoFileClip(raw_video_path)
        audio_sub = AudioFileClip(audio_file).subclipped(start_time, start_time + duration_seconds)
        if audio_sub.duration > v_clip.duration: audio_sub = audio_sub.subclipped(0, v_clip.duration)
        final = v_clip.with_audio(audio_sub)
        final.write_videofile(final_output_path, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True, fps=fps, logger=None)
        v_clip.close(); audio_sub.close(); final.close()
        return final_output_path, final_stats
    except Exception as e:
        print(f"Error: {e}")
        return raw_video_path, final_stats
