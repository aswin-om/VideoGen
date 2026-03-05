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
