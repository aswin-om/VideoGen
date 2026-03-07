import subprocess
import numpy as np
import librosa

BPM_MIN = 60.0
BPM_MAX = 140.0


def parse_time(time_str):
    if isinstance(time_str, (int, float)):
        return float(time_str)
    parts = str(time_str).split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    try:
        return float(time_str)
    except Exception:
        return 0.0


def detect_beats(audio_file, start_time, duration_seconds, fps):
    """Detect beat positions using librosa's beat_track for accurate rhythm sync."""
    total_frames = int(duration_seconds * fps)
    try:
        y, sr = librosa.load(audio_file, sr=22050, offset=start_time,
                             duration=duration_seconds, mono=True)
        if len(y) < sr * 0.1:
            fallback_step = int(max(1, round(fps * 0.55)))
            return set(range(0, total_frames, fallback_step))

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return {int(round(t * fps)) for t in beat_times
                if 0 <= int(round(t * fps)) < total_frames}

    except Exception:
        fallback_step = int(max(1, round(fps * 0.55)))
        return set(range(0, total_frames, fallback_step))


def analyze_audio_features(audio_file, start_time, duration_seconds):
    """Extract BPM, energy, beat strength using librosa."""
    try:
        y, sr = librosa.load(audio_file, sr=22050, offset=start_time,
                             duration=duration_seconds, mono=True)

        if len(y) < sr * 0.1:
            return _default_features()

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        raw_bpm = float(tempo)
        bpm, tempo_factor, bpm_note = normalize_bpm(raw_bpm)

        rms = librosa.feature.rms(y=y)[0]
        energy = min(1.0, float(np.mean(rms)) * 5)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strength = min(1.0, float(np.std(onset_env) / (np.mean(onset_env) + 1e-6)) / 2)

        if bpm < 80:
            tempo_category = "slow"
        elif bpm < 120:
            tempo_category = "medium"
        elif bpm < 160:
            tempo_category = "fast"
        else:
            tempo_category = "very_fast"

        return {
            "bpm": round(bpm, 1),
            "raw_bpm": round(raw_bpm, 1),
            "energy": round(energy, 2),
            "beat_strength": round(beat_strength, 2),
            "tempo_category": tempo_category,
            "audio_tempo_factor": round(tempo_factor, 3),
            "bpm_note": bpm_note,
        }

    except Exception as e:
        print(f"Audio analysis error: {e}")
        return _default_features()


def _default_features():
    return {
        "bpm": 120,
        "raw_bpm": 120,
        "energy": 0.5,
        "beat_strength": 0.5,
        "tempo_category": "medium",
        "audio_tempo_factor": 1.0,
        "bpm_note": "default_fallback",
    }


def normalize_bpm(raw_bpm):
    """
    Normalize BPM into the 60-140 window.
    - If BPM is too fast, keep halving while possible.
    - If a further halving would drop below 60, keep BPM at 60 and
      flag an audio slowdown factor for final mux.
    Returns: (normalized_bpm, audio_tempo_factor, note)
    """
    bpm = float(max(0.0, raw_bpm))
    tempo_factor = 1.0

    if bpm <= 0:
        return 120.0, 1.0, "invalid_bpm_fallback"

    if bpm <= BPM_MAX and bpm >= BPM_MIN:
        return bpm, 1.0, "in_range"

    if bpm > BPM_MAX:
        while bpm > BPM_MAX:
            halved = bpm / 2.0
            if halved < BPM_MIN:
                tempo_factor = BPM_MIN / bpm
                return BPM_MIN, tempo_factor, "too_fast_slowed_audio"
            bpm = halved
        return bpm, 1.0, "too_fast_halved"

    # For very slow tracks, clamp to minimum analysis BPM.
    return BPM_MIN, 1.0, "too_slow_clamped"


def detect_multiband_switches(audio_file, start_time, duration_seconds, fps):
    """
    Analyzes bass (low-end) and high-frequency transients separately to find 
    the most rhythmically impactful switch points.
    """
    try:
        y, sr = librosa.load(audio_file, sr=22050, offset=start_time,
                             duration=duration_seconds, mono=True)
        if len(y) < sr * 0.1:
            return set()

        # 1. BASS ANALYSIS (Kick/Sub) - Focus on Mel bins < 200Hz
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        bass_onset = librosa.onset.onset_strength(S=librosa.power_to_db(S[:15, :]), sr=sr)
        # Relaxed delta from 1.5 to 1.1 for higher sensitivity
        bass_peaks = librosa.util.peak_pick(bass_onset, pre_max=5, post_max=5, pre_avg=5, post_avg=5, delta=1.1, wait=10)
        
        # 2. HIGH FREQUENCY ANALYSIS (Snare/Percussion)
        _, y_percussive = librosa.effects.hpss(y)
        high_onset = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        # Relaxed delta from 2.0 to 1.4
        high_peaks = librosa.util.peak_pick(high_onset, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=1.4, wait=8)

        bass_times = librosa.frames_to_time(bass_peaks, sr=sr)
        high_times = librosa.frames_to_time(high_peaks, sr=sr)
        
        all_switches = sorted(list(set(bass_times) | set(high_times)))
        
        # Fallback: If we found very few multiband points, merge with regular beats
        min_density = duration_seconds / 5.0 # At least one switch every 5s
        if len(all_switches) < min_density:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            all_switches = sorted(list(set(all_switches) | set(beat_times)))

        min_gap = 0.4
        filtered_switches = set()
        last_t = -1.0
        
        for t in all_switches:
            if 0 <= t < duration_seconds and (t - last_t) >= min_gap:
                filtered_switches.add(int(round(t * fps)))
                last_t = t
                
        return filtered_switches
    except Exception as e:
        print(f"Multiband detection error: {e}")
        return set()


def detect_direction_flips(audio_file, start_time, duration_seconds, fps):
    """
    Identify regions where video playback direction should flip (Forward/Reverse).
    Triggered by high onset strength peaks (drum fills, drops, etc.).
    Returns a set of frame indices where direction should toggle.
    """
    total_frames = int(duration_seconds * fps)
    try:
        y, sr = librosa.load(audio_file, sr=22050, offset=start_time,
                             duration=duration_seconds, mono=True)
        if len(y) < sr * 0.1:
            return set()

        # onset_strength captures high energy changes
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Use a higher threshold for direction flips than normal beats
        peaks = librosa.util.peak_pick(onset_env, pre_max=7, post_max=7, pre_avg=7, post_avg=7, delta=0.5, wait=15)
        
        peak_times = librosa.frames_to_time(peaks, sr=sr)
        # Only flip every ~2 seconds at most to avoid headache
        flip_frames = sorted([int(round(t * fps)) for t in peak_times])
        
        safe_flips = set()
        last_flip = -100
        min_gap = int(fps * 1.5)
        
        for f in flip_frames:
            if 0 <= f < total_frames and (f - last_flip) >= min_gap:
                safe_flips.add(f)
                last_flip = f
                
        return safe_flips
    except Exception:
        return set()


def compute_energy_envelope(audio_file, start_time, duration_seconds, fps):
    """
    Compute a smoothed per-frame energy envelope for speed ramping.
    Returns numpy array of length total_frames with values normalized 0-1.
    Heavy smoothing ensures buttery speed transitions.
    """
    total_frames = int(duration_seconds * fps)

    try:
        y, sr = librosa.load(audio_file, sr=22050, offset=start_time,
                             duration=duration_seconds, mono=True)
        if len(y) < sr * 0.1:
            return np.full(total_frames, 0.5)

        # RMS energy per librosa frame
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Resample to video frame rate
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr,
                                           hop_length=hop_length)
        frame_times = np.arange(total_frames) / fps
        energy = np.interp(frame_times, rms_times, rms)

        # Normalize to 0-1
        e_min, e_max = energy.min(), energy.max()
        if e_max - e_min > 1e-6:
            energy = (energy - e_min) / (e_max - e_min)
        else:
            return np.full(total_frames, 0.5)

        # Heavy gaussian-like smoothing (~1s window) for seamless transitions
        kernel_size = max(3, int(fps * 1.0)) | 1  # ensure odd
        kernel = np.ones(kernel_size) / kernel_size
        energy = np.convolve(energy, kernel, mode='same')

        # Second pass with wider kernel for extra smoothness
        wide_kernel_size = max(3, int(fps * 0.5)) | 1
        wide_kernel = np.ones(wide_kernel_size) / wide_kernel_size
        energy = np.convolve(energy, wide_kernel, mode='same')

        # Re-normalize after smoothing
        e_min, e_max = energy.min(), energy.max()
        if e_max - e_min > 1e-6:
            energy = (energy - e_min) / (e_max - e_min)

        return energy

    except Exception:
        return np.full(total_frames, 0.5)
