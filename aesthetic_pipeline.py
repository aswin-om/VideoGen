import argparse
import os
import sys
from generate import generate_beat_sync_video, RESOLUTION_PRESETS, FPS_PRESETS


def main():
    parser = argparse.ArgumentParser(description="Create a slow-motion aesthetic edit.")
    parser.add_argument("--videos", nargs="+", required=True, help="List of input video files.")
    parser.add_argument("--audio", required=True, help="Background music file.")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration of the final video in seconds.")
    parser.add_argument("--start_time", default="00:00", help="Start time in the audio file (MM:SS).")
    parser.add_argument("--resolution", default="1080p Square (1080×1080)", choices=RESOLUTION_PRESETS.keys(),
                        help="Output resolution preset.")
    parser.add_argument("--fps", default="60 fps (Interpolated)", choices=FPS_PRESETS.keys(),
                        help="Target frame rate.")
    parser.add_argument("--speed", type=float, default=0.35, help="Base motion speed multiplier (lower = slower).")
    parser.add_argument("--intensity", type=float, default=0.4, help="Aesthetic intensity (noise/trails).")
    parser.add_argument("--crop_mode", default="person", choices=["center", "person", "content"],
                        help="Smart cropping mode. 'person' uses YOLO detection for people/cars.")
    
    args = parser.parse_args()

    # Pre-validation
    for v in args.videos:
        if not os.path.exists(v):
            print(f"Error: Video file not found: {v}")
            sys.exit(1)
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    print("\n--- Starting Aesthetic Slow-Mo Pipeline ---")
    print(f"Inputs: {len(args.videos)} videos, Audio: {args.audio}")
    print(f"Target: {args.duration}s @ {args.resolution}, {args.fps}")
    print(f"Aesthetic settings: speed={args.speed}, intensity={args.intensity}")
    print("-------------------------------------------\n")

    try:
        final_path, stats = generate_beat_sync_video(
            video_files=args.videos,
            audio_file=args.audio,
            start_time_str=args.start_time,
            clip_duration=args.duration,
            zoom=1.1,                 # Slight zoom for more dynamic feel
            motion_speed=args.speed,
            step_repeat=1,            # Smooth motion
            source_stride=1,
            noise_intensity=args.intensity * 0.15,
            crop_mode=args.crop_mode,
            resolution=args.resolution,
            target_fps=args.fps,
            speed_ramp=0.5,           # Energy-based speed ramping
            step_print=args.intensity, # Motion trails (step printing)
            speed_curve="⏩ Fast → Slow → Fast", # Cinematic aesthetic curve
            status_cb=lambda fn, msg: print(f"[{fn}] {msg}"),
        )
        
        print("\n--- Pipeline Complete! ---")
        print(f"Final output: {final_path}")
        print(f"Stats: {stats}")
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
