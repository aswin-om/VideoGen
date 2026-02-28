import os
import random
import glob
import cv2

fps = 24
height = 1080
width = 1080
duration_seconds = 30

output_dir = "output"
frames_dir = "all_frames"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_random_video():
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
            
        # Resize frame to match target dimensions
        resized_img = cv2.resize(img, (width, height))
        out.write(resized_img)
        
        if idx % 50 == 0 or idx == num_frames_needed:
            print(f"Processed {idx}/{num_frames_needed} frames")

    out.release()
    print("Done! Video successfully created.")

if __name__ == "__main__":
    generate_random_video()