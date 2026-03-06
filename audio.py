import subprocess
import numpy as np
import librosa


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
        bpm = float(tempo)

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
            "energy": round(energy, 2),
            "beat_strength": round(beat_strength, 2),
            "tempo_category": tempo_category,
        }

    except Exception as e:
        print(f"Audio analysis error: {e}")
        return _default_features()


def _default_features():
    return {"bpm": 120, "energy": 0.5, "beat_strength": 0.5, "tempo_category": "medium"}


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
