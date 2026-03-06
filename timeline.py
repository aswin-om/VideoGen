import numpy as np
import cv2

PHI = 1.6180339887  # Golden ratio


def build_video_meta(video_files):
    """Get frame count and FPS for each video."""
    meta = []
    for v_path in video_files:
        cap = cv2.VideoCapture(v_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        cap.release()
        meta.append({"path": v_path, "total": total, "fps": src_fps})
    return meta


def build_timeline(video_files, video_meta, beat_set, total_output_frames,
                   motion_speed, step_repeat, source_stride, fps,
                   person_cache=None, energy_envelope=None, ramp_intensity=0.0):
    """
    Build frame-by-frame timeline.
    Uses YOLO detection density for smart clip selection when person_cache is provided.
    When energy_envelope and ramp_intensity > 0, applies speed ramping:
      quiet → slow-mo (1/φ ≈ 0.618×), loud → fast (φ ≈ 1.618×).
    """
    path = []
    active_vid = 0
    while active_vid < len(video_meta) and video_meta[active_vid]["total"] <= 0:
        active_vid += 1
    if active_vid >= len(video_meta):
        raise ValueError("No readable frames in uploaded videos.")

    source_positions = [0.0] * len(video_files)
    speed_scale = max(0.25, float(motion_speed))
    repeat_count = max(1, int(step_repeat))
    stride_count = max(1, int(source_stride))
    frames_since_switch = 0
    min_shot_len = 10

    for i in range(total_output_frames):
        # Beat-triggered switch
        if i in beat_set and frames_since_switch >= min_shot_len and len(video_files) > 1:
            active_vid = _pick_best_video(
                active_vid, video_files, video_meta, source_positions, person_cache
            )
            frames_since_switch = 0

        # End-of-clip switch
        curr_total = video_meta[active_vid]["total"]
        if source_positions[active_vid] >= max(0, curr_total - 1) and len(video_files) > 1:
            for shift in range(1, len(video_files) + 1):
                cand = (active_vid + shift) % len(video_files)
                if (video_meta[cand]["total"] > 0 and
                        source_positions[cand] < max(0, video_meta[cand]["total"] - 1)):
                    active_vid = cand
                    frames_since_switch = 0
                    break

        curr_total = video_meta[active_vid]["total"]
        curr_fps = video_meta[active_vid]["fps"]
        frame_idx = int(min(max(0, curr_total - 1), round(source_positions[active_vid])))

        # Speed ramping: φ^((2e-1) * intensity)
        ramp_mult = _speed_multiplier(energy_envelope, i, ramp_intensity)
        advance = max(0.25, (curr_fps / fps) * speed_scale * ramp_mult)

        if (i + 1) % repeat_count == 0:
            source_positions[active_vid] = min(
                max(0, curr_total - 1),
                source_positions[active_vid] + (advance * stride_count)
            )
        path.append((video_files[active_vid], frame_idx, active_vid))
        frames_since_switch += 1

    return path


def _speed_multiplier(energy_envelope, frame_idx, ramp_intensity):
    """
    Map energy to speed via golden ratio for perceptually balanced dynamics.
    φ^((2e-1)*intensity) gives:
      energy=0 → 1/φ ≈ 0.618× (dreamy slow-mo)
      energy=0.5 → 1.0× (normal)
      energy=1 → φ ≈ 1.618× (energetic fast-forward)
    The fast/slow ratio is φ² ≈ 2.618 — a naturally pleasing dynamic range.
    """
    if energy_envelope is None or ramp_intensity <= 0:
        return 1.0

    idx = min(frame_idx, len(energy_envelope) - 1)
    energy = float(energy_envelope[idx])
    exponent = (2.0 * energy - 1.0) * ramp_intensity
    return PHI ** exponent


def _pick_best_video(current_vid, video_files, video_meta, source_positions,
                     person_cache):
    """
    Pick the best video to switch to.
    When person_cache is available, prefer clips with more detected subjects.
    Falls back to round-robin otherwise.
    """
    if person_cache is None or len(video_files) <= 1:
        return _round_robin_next(current_vid, video_files, video_meta)

    best_score = -1.0
    best_cand = current_vid

    for shift in range(1, len(video_files) + 1):
        cand = (current_vid + shift) % len(video_files)
        total = video_meta[cand]["total"]
        if total <= 0:
            continue

        src_pos = int(source_positions[cand])
        if src_pos >= total - 1:
            continue

        v_path = video_files[cand]
        detections = person_cache.get(v_path, {})

        # Detection density in the next 30 frames
        window = min(30, total - src_pos)
        hits = sum(1 for f in range(src_pos, src_pos + window) if f in detections)
        score = hits / max(1, window)

        # Variety bonus for less-used videos
        remaining_ratio = (total - src_pos) / max(1, total)
        score += remaining_ratio * 0.1

        if score > best_score:
            best_score = score
            best_cand = cand

    return best_cand


def _round_robin_next(current_vid, video_files, video_meta):
    """Simple round-robin to the next available video."""
    for shift in range(1, len(video_files) + 1):
        cand = (current_vid + shift) % len(video_files)
        if video_meta[cand]["total"] > 0:
            return cand
    return current_vid
