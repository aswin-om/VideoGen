import os
import json
import datetime
import numpy as np

from audio import parse_time, detect_beats, analyze_audio_features, compute_energy_envelope
from detection import prescan_person_positions, analyze_video_motion
from timeline import build_video_meta, build_timeline
from render import (
    cleanup_old_generations, render_video, ffmpeg_mux_audio,
    interpolate_60fps, reset_cancel, is_cancelled,
)

RESOLUTION_PRESETS = {
    "720p Square (720×720)": (720, 720),
    "1080p Square (1080×1080)": (1080, 1080),
    "1080p Vertical (1080×1920)": (1080, 1920),
    "1080p Landscape (1920×1080)": (1920, 1080),
}

FPS_PRESETS = {
    "24 fps (Cinematic)": 24,
    "30 fps (Default)": 30,
    "60 fps (Interpolated)": 60,
}


def log_generation(log_data):
    log_file = os.path.join("output", "generations.jsonl")
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")
    except Exception as e:
        print(f"Logging error: {e}")


def calculate_optimal_parameters(audio_features, motion_scores):
    """Map audio/video analysis to recommended generation parameters."""
    bpm = audio_features["bpm"]
    energy = audio_features["energy"]
    beat_strength = audio_features["beat_strength"]
    tempo_cat = audio_features["tempo_category"]
    avg_motion = np.mean(list(motion_scores.values())) if motion_scores else 0.5

    reasoning = []

    # Step repeat
    if bpm < 80:
        step_repeat = 3
        reasoning.append(f"Slow tempo ({bpm:.0f} BPM) → step_repeat=3 for emphasis")
    elif bpm < 100:
        step_repeat = 2
        reasoning.append(f"Medium-slow tempo ({bpm:.0f} BPM) → step_repeat=2")
    else:
        step_repeat = 1
        reasoning.append(f"Tempo {bpm:.0f} BPM → step_repeat=1 for smooth flow")

    # Source stride
    if bpm < 90:
        base_stride = 1
    elif bpm < 120:
        base_stride = 1 if beat_strength < 0.5 else 2
    elif bpm < 150:
        base_stride = 2 if beat_strength < 0.6 else 3
    else:
        base_stride = 3 if beat_strength < 0.7 else 4
    source_stride = min(8, max(1, base_stride))
    reasoning.append(f"BPM {bpm:.0f} + beat strength {beat_strength:.2f} → source_stride={source_stride}")

    # Motion speed
    if avg_motion < 0.3:
        motion_speed = 1.2
        reasoning.append(f"Low video motion ({avg_motion:.2f}) → motion_speed=1.2")
    elif avg_motion < 0.6:
        motion_speed = 0.8
        reasoning.append(f"Medium video motion ({avg_motion:.2f}) → motion_speed=0.8")
    else:
        motion_speed = 0.5
        reasoning.append(f"High video motion ({avg_motion:.2f}) → motion_speed=0.5")

    if tempo_cat in ("fast", "very_fast"):
        motion_speed *= 1.2
        reasoning.append(f"Fast song → motion_speed adjusted to {motion_speed:.2f}")
    motion_speed = round(min(2.0, max(0.05, motion_speed)), 2)

    # Zoom
    if energy < 0.3:
        zoom = 1.0
        reasoning.append(f"Low energy ({energy:.2f}) → zoom=1.0")
    elif energy < 0.6:
        zoom = 1.05
        reasoning.append(f"Medium energy ({energy:.2f}) → zoom=1.05")
    elif energy < 0.8:
        zoom = 1.15
        reasoning.append(f"High energy ({energy:.2f}) → zoom=1.15")
    else:
        zoom = 1.25
        reasoning.append(f"Very high energy ({energy:.2f}) → zoom=1.25")

    return {
        "zoom": zoom,
        "motion_speed": motion_speed,
        "step_repeat": step_repeat,
        "source_stride": source_stride,
        "reasoning": reasoning,
        "audio_analysis": audio_features,
        "avg_video_motion": round(avg_motion, 2),
    }


def analyze_and_recommend_settings(video_files, audio_file, start_time_str, clip_duration):
    """Analyze videos and audio, return recommended settings."""
    start_time = parse_time(start_time_str)
    duration_seconds = float(clip_duration)
    audio_features = analyze_audio_features(audio_file, start_time, duration_seconds)
    motion_scores = analyze_video_motion(video_files)
    return calculate_optimal_parameters(audio_features, motion_scores)


def generate_beat_sync_video(
    video_files,
    audio_file,
    start_time_str,
    clip_duration,
    zoom=1.0,
    motion_speed=0.8,
    step_repeat=1,
    source_stride=1,
    noise_intensity=0.0,
    crop_mode="center",
    resolution="720p Square (720×720)",
    target_fps="30 fps (Default)",
    speed_ramp=0.0,
    step_print=0.0,
    progress=None,
):
    if progress:
        progress(0.0, desc="Initializing...")

    reset_cancel()

    start_time = parse_time(start_time_str)
    duration_seconds = float(clip_duration)
    video_files = list(video_files)[:5]

    width, height = RESOLUTION_PRESETS.get(resolution, (720, 720))
    render_fps = FPS_PRESETS.get(target_fps, 30)
    do_interpolate = render_fps > 30
    if do_interpolate:
        render_fps = 30  # Render at 30, interpolate to 60

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    cleanup_old_generations(output_dir)

    # Beat detection with librosa
    if progress:
        progress(0.08, desc="Analyzing beats (librosa)...")
    beat_set = detect_beats(audio_file, start_time, duration_seconds, render_fps)

    # Energy envelope for speed ramping
    energy_envelope = None
    if speed_ramp > 0:
        if progress:
            progress(0.12, desc="Computing energy envelope...")
        energy_envelope = compute_energy_envelope(
            audio_file, start_time, duration_seconds, render_fps
        )

    if progress:
        progress(0.2, desc="Indexing videos...")
    video_meta = build_video_meta(video_files)

    if not any(m["total"] > 0 for m in video_meta):
        raise ValueError("No frames extracted from videos. Check inputs.")

    # YOLO prescan for person tracking and smart clip selection
    person_cache = None
    if crop_mode == "person":
        if progress:
            progress(0.30, desc="Detecting subjects (YOLO)...")
        person_cache = prescan_person_positions(video_files)

    if progress:
        progress(0.40, desc="Building timeline...")

    total_output_frames = int(duration_seconds * render_fps)
    path = build_timeline(
        video_files, video_meta, beat_set, total_output_frames,
        motion_speed, step_repeat, source_stride, render_fps,
        person_cache=person_cache,
        energy_envelope=energy_envelope,
        ramp_intensity=speed_ramp,
    )

    if is_cancelled():
        raise InterruptedError("Generation cancelled.")

    # Render
    if progress:
        progress(0.55, desc="Rendering video...")
    raw_path, frame_usage, timestamp = render_video(
        path, video_files, person_cache, crop_mode, zoom,
        width, height, render_fps, noise_intensity, output_dir, progress,
        step_print_intensity=step_print,
    )

    # 60fps interpolation
    if do_interpolate:
        if progress:
            progress(0.88, desc="Interpolating to 60fps...")
        interp_path = os.path.join(output_dir, f"interp_{timestamp}.mp4")
        try:
            interpolate_60fps(raw_path, interp_path)
            os.remove(raw_path)
            raw_path = interp_path
        except Exception as e:
            print(f"60fps interpolation failed, using 30fps: {e}")

    # Mux audio
    if progress:
        progress(0.92, desc="Merging audio...")

    final_stats = {
        "total_scenes": len(video_files),
        "total_frames_indexed": int(sum(max(0, m["total"]) for m in video_meta)),
        "video_usage_seconds": {
            os.path.basename(v): round(float(frame_usage[i]) / render_fps, 2)
            for i, v in enumerate(video_files)
        },
    }

    final_output_path = os.path.join(output_dir, f"final_{timestamp}.mp4")

    log_payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_videos": video_files,
        "audio_file": audio_file,
        "parameters": {
            "start_time": start_time_str,
            "duration": clip_duration,
            "zoom": zoom,
            "motion_speed": motion_speed,
            "step_repeat": step_repeat,
            "source_stride": source_stride,
            "noise_intensity": noise_intensity,
            "crop_mode": crop_mode,
            "resolution": resolution,
            "target_fps": target_fps,
            "speed_ramp": speed_ramp,
            "step_print": step_print,
        },
        "stats": final_stats,
    }

    try:
        ffmpeg_mux_audio(
            raw_video_path=raw_path,
            audio_file=audio_file,
            start_time=start_time,
            duration_seconds=duration_seconds,
            final_output_path=final_output_path,
        )
        # Clean up raw/interp file after successful mux
        if os.path.exists(raw_path) and raw_path != final_output_path:
            os.remove(raw_path)

        if progress:
            progress(1.0, desc="Done!")

        log_payload["output_video"] = final_output_path
        log_payload["status"] = "success"
        log_generation(log_payload)
        return final_output_path, final_stats

    except Exception as e:
        print(f"Merge error: {e}")
        log_payload["output_video"] = raw_path
        log_payload["status"] = "audio_merge_failed"
        log_payload["error"] = str(e)
        log_generation(log_payload)
        return raw_path, final_stats
