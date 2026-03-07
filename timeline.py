import numpy as np
import cv2

PHI = 1.6180339887  # Golden ratio


# ---------------------------------------------------------------------------
# Speed-curve presets
# Each function takes total_frames and returns an array of per-frame speed
# multipliers (centred around 1.0).
# ---------------------------------------------------------------------------

def _curve_none(n):
    return np.ones(n)


def _curve_fast_slow_fast(n):
    """Fast → slow → fast  (U-shape cosine)."""
    t = np.linspace(0, 2 * np.pi, n)
    # cos goes 1 → -1 → 1; remap to 2.0 → 0.4 → 2.0
    return 0.4 + 1.6 * (0.5 + 0.5 * np.cos(t))


def _curve_slow_fast_slow(n):
    """Slow → fast → slow  (inverted U / ease-in-out)."""
    t = np.linspace(0, 2 * np.pi, n)
    return 0.4 + 1.6 * (0.5 - 0.5 * np.cos(t))


def _curve_ramp_up(n):
    """Gradually speed up over the clip."""
    return np.linspace(0.4, 2.0, n)


def _curve_ramp_down(n):
    """Start fast, gradually slow down."""
    return np.linspace(2.0, 0.4, n)


def _curve_pulse(n):
    """Periodic fast/slow pulses (3 cycles)."""
    t = np.linspace(0, 6 * np.pi, n)
    return 1.0 + 0.8 * np.sin(t)


SPEED_CURVES = {
    "None": _curve_none,
    "⏩ Fast → Slow → Fast": _curve_fast_slow_fast,
    "🐢 Slow → Fast → Slow": _curve_slow_fast_slow,
    "📈 Ramp Up": _curve_ramp_up,
    "📉 Ramp Down": _curve_ramp_down,
    "💓 Pulse": _curve_pulse,
}


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
                   person_cache=None, energy_envelope=None, ramp_intensity=0.0,
                   speed_curve="None", direction_flip_set=None):
    """
    Build frame-by-frame timeline with intelligent direction flipping.
    """
    path = []
    active_vid = 0
    while active_vid < len(video_meta) and video_meta[active_vid]["total"] <= 0:
        active_vid += 1
    if active_vid >= len(video_meta):
        raise ValueError("No readable frames in uploaded videos.")

    source_positions = [0.0] * len(video_files)
    usage_counts = [0] * len(video_files)
    speed_scale = max(0.25, float(motion_speed))
    repeat_count = max(1, int(step_repeat))
    stride_count = max(1, int(source_stride))
    frames_since_switch = 0
    min_shot_len = 15 # Slightly longer minimum shot for stability
    
    # 1 = Forward, -1 = Reverse
    direction = 1.0
    direction_flip_set = direction_flip_set or set()

    curve_fn = SPEED_CURVES.get(speed_curve, _curve_none)
    curve_mults = curve_fn(total_output_frames)

    for i in range(total_output_frames):
        # Direction Toggle (Onsets / Drops)
        if i in direction_flip_set and frames_since_switch > 20:
            direction *= -1.0
            # Small random jump when flipping to add energy
            source_positions[active_vid] = min(
                max(0, video_meta[active_vid]["total"] - 1),
                source_positions[active_vid] + direction * (fps * 0.3)
            )

        # Beat-triggered switch
        if i in beat_set and frames_since_switch >= min_shot_len and len(video_files) > 1:
            active_vid = _pick_best_video(
                active_vid, video_files, video_meta, source_positions, 
                person_cache, usage_counts
            )
            frames_since_switch = 0
            direction = 1.0

        # End-of-clip switch (or start-of-clip if reversed)
        curr_total = video_meta[active_vid]["total"]
        pos = source_positions[active_vid]
        
        # Only switch at pos <= 0 if we are moving backwards
        at_end = pos >= max(0, curr_total - 1) and direction > 0
        at_start = pos <= 0 and direction < 0
        
        if (at_end or at_start) and len(video_files) > 1:
            active_vid = _pick_best_video(
                active_vid, video_files, video_meta, source_positions, 
                person_cache, usage_counts
            )
            frames_since_switch = 0
            direction = 1.0

        curr_total = video_meta[active_vid]["total"]
        curr_fps = video_meta[active_vid]["fps"]
        frame_idx = int(min(max(0, curr_total - 1), round(source_positions[active_vid])))

        # ... (rest of speed logic) ...
        ramp_mult = _speed_multiplier(energy_envelope, i, ramp_intensity)
        curve_mult = float(curve_mults[i])
        target_advance_per_frame = (curr_fps / fps) * speed_scale * ramp_mult * curve_mult
        advance = target_advance_per_frame * repeat_count * stride_count * direction

        if (i + 1) % repeat_count == 0:
            source_positions[active_vid] = min(
                max(0, curr_total - 1),
                max(0, source_positions[active_vid] + advance)
            )
        path.append((video_files[active_vid], frame_idx, active_vid))
        usage_counts[active_vid] += 1
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
                     person_cache, usage_counts):
    """
    Pick the best video to switch to.
    When person_cache is available, prefer clips with more detected subjects.
    Uses usage_counts to force variety across all clips.
    """
    if len(video_files) <= 1:
        return current_vid

    best_score = -1e9
    best_cand = current_vid
    
    total_usage = sum(usage_counts) + 1

    # First, try to find the best candidate AMONG OTHER videos
    for shift in range(1, len(video_files)):
        cand = (current_vid + shift) % len(video_files)
        total = video_meta[cand]["total"]
        if total <= 0:
            continue

        src_pos = int(source_positions[cand])
        if src_pos >= total - 1:
            continue

        v_path = video_files[cand]
        detections = person_cache.get(v_path, {}) if person_cache else {}

        # 1. Detection score (normalized 0-1)
        window = 30
        hits = sum(1 for f in range(src_pos, src_pos + window) if f in detections)
        detection_score = hits / window

        # 2. Usage penalty (heavily penalize frequently used clips)
        usage_ratio = usage_counts[cand] / total_usage
        usage_penalty = usage_ratio * 5.0 # High penalty for repeats
        
        # 3. Variety bonus (favor clips with 0 usage)
        variety_bonus = 2.0 if usage_counts[cand] == 0 else 0.0

        # 4. Remaining content bonus
        remaining_ratio = (total - src_pos) / max(1, total)
        content_bonus = remaining_ratio * 0.5

        score = detection_score - usage_penalty + variety_bonus + content_bonus

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
