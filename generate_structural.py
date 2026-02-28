import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

fps = 24
height = 1920
width = 1080
duration_seconds = 15

output_dir = "output"
frames_dir = "all_frames"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def compute_image_features(image_path):
    """
    Computes a structural feature vector for an image.
    Uses HOG (Histogram of Oriented Gradients) to capture the structure/shape 
    while being relatively invariant to color and lighting.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Resize to a smaller standard size for faster and consistent feature extraction
    # We use a square aspect ratio here purely for feature extraction
    feature_size = (128, 128)
    img_resized = cv2.resize(img, feature_size)
    
    # Convert to grayscale as we only care about structure, not color
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    
    # Compute HOG features
    features = hog.compute(gray)
    
    # Flatten and normalize the feature vector
    features = features.flatten()
    if np.linalg.norm(features) > 0:
        features = features / np.linalg.norm(features)
        
    return features

def find_similar_sequence(start_image_path, all_features, all_paths, num_frames_needed):
    """
    Finds a sequence of structurally similar images starting from the seed image.
    """
    # Find the index of the start image
    try:
        current_idx = all_paths.index(start_image_path)
    except ValueError:
        print("Start image not found in the list.")
        return []
        
    sequence_indices = [current_idx]
    
    # To avoid reusing the same exact images, we'll keep track of used indices
    used_indices = set([current_idx])
    
    print("Finding structurally similar frames...")
    # Iteratively find the next most similar image
    for _ in tqdm(range(num_frames_needed - 1)):
        current_features = all_features[current_idx].reshape(1, -1)
        
        # Calculate distances to all other images (using cosine distance)
        # Cosine distance is good for normalized high-dimensional vectors
        distances = cdist(current_features, all_features, metric='cosine')[0]
        
        # Mask out already used indices by setting their distance to infinity
        for used_idx in used_indices:
            distances[used_idx] = np.inf
            
        # Find the index of the closest structurally similar image
        next_idx = np.argmin(distances)
        
        sequence_indices.append(next_idx)
        used_indices.add(next_idx)
        
        # Update current index for the next iteration
        current_idx = next_idx
        
    sequence_paths = [all_paths[i] for i in sequence_indices]
    return sequence_paths

def generate_structural_video():
    print(f"Scanning for frames in '{frames_dir}'...")
    all_frames = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        all_frames.extend(glob.glob(os.path.join(frames_dir, ext)))
    
    if not all_frames:
        print(f"Error: No frames found in '{frames_dir}' folder.")
        return

    num_frames_needed = duration_seconds * fps
    print(f"Total available frames: {len(all_frames)}")
    print(f"Target video duration: {duration_seconds}s at {fps} fps ({num_frames_needed} frames)")

    if len(all_frames) < num_frames_needed:
         print(f"Warning: Only {len(all_frames)} frames available, but {num_frames_needed} needed.")
         print("The algorithm currently requires unique frames. Reducing target frames.")
         num_frames_needed = len(all_frames)

    print("Computing structural features for all frames (this may take a while)...")
    valid_paths = []
    features_list = []
    
    for path in tqdm(all_frames):
        feats = compute_image_features(path)
        if feats is not None:
            valid_paths.append(path)
            features_list.append(feats)
            
    if len(valid_paths) < num_frames_needed:
         print(f"Not enough valid frames ({len(valid_paths)}) to make the video.")
         num_frames_needed = len(valid_paths)

    features_array = np.array(features_list)

    # Pick a random starting frame
    start_frame = valid_paths[np.random.randint(0, len(valid_paths))]
    print(f"Seed frame selected: {start_frame}")

    # Find the sequence of structurally similar frames
    selected_frames = find_similar_sequence(start_frame, features_array, valid_paths, num_frames_needed)

    output_path = os.path.join(output_dir, "structural_video.mp4")
    
    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Stitching {len(selected_frames)} frames into video at: {output_path}")

    for idx, frame_path in enumerate(tqdm(selected_frames), start=1):
        img = cv2.imread(frame_path)
        if img is None:
            continue
            
        # Resize frame to match target dimensions
        resized_img = cv2.resize(img, (width, height))
        out.write(resized_img)

    out.release()
    print("Done! Structural video successfully created.")

if __name__ == "__main__":
    generate_structural_video()
