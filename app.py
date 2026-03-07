import gradio as gr
import os
import time
import queue
import threading
import datetime
from generate import (
    generate_beat_sync_video, analyze_and_recommend_settings,
    RESOLUTION_PRESETS, FPS_PRESETS,
)
from timeline import SPEED_CURVES
from render import cancel_generation


FIXED_SPEED_RAMP = 1.0
DEFAULT_MUSIC_DIR = "/Users/aswinharidas/Downloads/Music"
SUPPORTED_AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")


def list_music_files(folder):
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if name.lower().endswith(SUPPORTED_AUDIO_EXTS):
                files.append(os.path.join(root, name))
    return sorted(files)


def refresh_music_choices(folder):
    choices = list_music_files(folder)
    value = choices[0] if choices else None
    return gr.Dropdown(choices=choices, value=value)


INITIAL_MUSIC_CHOICES = list_music_files(DEFAULT_MUSIC_DIR)


def _format_live_status(lines):
    body = "\n".join(lines[-14:]) if lines else "- Initializing..."
    return "### ⚙️ Generating...\n" + body


def run_generator(video_files, audio_file, start_time, duration,
                  noise_intensity, step_print, speed_curve,
                  resolution, target_fps, music_file_path,
                  crop_mode, progress=gr.Progress()):
    if not video_files:
        yield None, "Please upload videos."
        return
    chosen_audio = audio_file.name if audio_file is not None else music_file_path
    if not chosen_audio:
        yield None, "Please upload an audio file."
        return

    video_paths = [v.name for v in video_files[:5]]
    audio_path = chosen_audio
    safe_duration = max(5, min(float(duration), 45.0))

    crop_key = "center"
    if "Person" in crop_mode:
        crop_key = "person"
    elif "Content" in crop_mode:
        crop_key = "content"


    status_q = queue.Queue()
    status_lines = []
    result = {"output_video": None, "stats": None, "recs": None, "error": None}
    done = threading.Event()

    def push_status(fn_name, text):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        status_q.put(f"- `{fn_name}`: {text} (`{stamp}`)")

    def worker():
        try:
            push_status("generate.analyze_and_recommend_settings", "Analyzing audio + source motion")
            recs = analyze_and_recommend_settings(video_paths, audio_path,
                                                  start_time, safe_duration)

            zoom = recs['zoom']
            motion_speed = recs['motion_speed']
            step_repeat = recs['step_repeat']
            source_stride = recs['source_stride']
            audio_tempo_factor = recs['audio_analysis'].get('audio_tempo_factor', 1.0)

            output_video, stats = generate_beat_sync_video(
                video_files=video_paths,
                audio_file=audio_path,
                start_time_str=start_time,
                clip_duration=safe_duration,
                zoom=zoom,
                motion_speed=motion_speed,
                step_repeat=step_repeat,
                source_stride=source_stride,
                noise_intensity=noise_intensity,
                crop_mode=crop_key,
                resolution=resolution,
                target_fps=target_fps,
                speed_ramp=FIXED_SPEED_RAMP,
                step_print=step_print,
                speed_curve=speed_curve,
                audio_tempo_factor=audio_tempo_factor,
                status_cb=push_status,
                progress=None,
            )

            result["recs"] = recs
            result["output_video"] = output_video
            result["stats"] = stats
        except Exception as e:
            result["error"] = e
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    yield None, _format_live_status(status_lines)

    while not done.is_set() or not status_q.empty():
        updated = False
        while not status_q.empty():
            status_lines.append(status_q.get())
            updated = True
        if updated:
            yield None, _format_live_status(status_lines)
        time.sleep(0.2)

    try:
        if result["error"] is not None:
            raise result["error"]

        recs = result["recs"]
        output_video = result["output_video"]
        stats = result["stats"]

        zoom = recs['zoom']
        motion_speed = recs['motion_speed']
        step_repeat = recs['step_repeat']
        source_stride = recs['source_stride']
        audio_tempo_factor = recs['audio_analysis'].get('audio_tempo_factor', 1.0)

        w, h = RESOLUTION_PRESETS.get(resolution, (720, 720))
        fps_val = FPS_PRESETS.get(target_fps, 30)

        analysis = recs['audio_analysis']
        msg = f"### ✅ Generation Complete\n"
        msg += f"- **Output:** {w}×{h} @ {fps_val}fps\n"
        msg += f"- **Duration:** {safe_duration}s\n"
        msg += f"- **Auto params:** zoom={zoom}, speed={motion_speed}, "
        msg += f"repeat={step_repeat}x, stride={source_stride}\n"
        msg += f"- **Audio:** BPM={analysis['bpm']}, "
        msg += f"Energy={analysis['energy']}, "
        msg += f"Tempo={analysis['tempo_category']}\n"
        if 'raw_bpm' in analysis and analysis['raw_bpm'] != analysis['bpm']:
            msg += f"- **BPM normalized:** raw={analysis['raw_bpm']} → used={analysis['bpm']}\n"
        if abs(float(audio_tempo_factor) - 1.0) > 1e-6:
            msg += f"- **Audio tempo factor:** {audio_tempo_factor}x\n"
        msg += f"- **Scenes:** {stats['total_scenes']} | "
        msg += f"**Frames Indexed:** {stats['total_frames_indexed']}\n\n"
        msg += "#### 📊 Video Usage:\n"

        sorted_usage = sorted(stats['video_usage_seconds'].items(),
                              key=lambda x: x[1], reverse=True)
        for name, secs in sorted_usage:
            # Clean up Gradio/tmp filenames
            clean_name = name.split('/')[-1]
            if clean_name.count('_') > 4:
                # Likely a temp name like _._d.e.v.u_.__173...
                # Try to find the original-ish part
                parts = clean_name.split('_')
                if len(parts) > 2:
                    clean_name = "_".join(parts[:-2])[:30] + "..." + parts[-1][-4:]
            
            msg += f"- `{clean_name}`: {secs}s\n" if secs > 0 else \
                   f"- `{clean_name}`: Not used\n"

        yield output_video, msg

    except InterruptedError:
        yield None, "### ⏹ Generation cancelled."
    except Exception as e:
        yield None, f"Error: {e}"


def stop_generation():
    cancel_generation()
    return "### ⏹ Stopping generation..."


custom_css = """
body, .gradio-container { background-color: #000000 !important; color: #a3a3a3 !important; }
.gr-button, button { background: #333333 !important; border: 1px solid #404040 !important; color: #a3a3a3 !important; }
#generate_btn { background: #f97316 !important; border: 1px solid #ea580c !important; color: white !important; }
#auto_calc_btn { background: #8b5cf6 !important; border: 1px solid #7c3aed !important; color: white !important; }
#cancel_btn { background: #dc2626 !important; border: 1px solid #b91c1c !important; color: white !important; }
.gr-input, .gr-box, input, textarea, select { background-color: #1a1a1a !important; border-color: #333333 !important; color: #d4d4d4 !important; border-radius: 4px !important; }
footer { display: none !important; }
"""

with gr.Blocks() as demo:
    gr.Markdown("# 🎥 VideoGen M5")
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(label="Videos", file_count="multiple")
            audio_input = gr.File(label="Audio", file_types=["audio"])
            music_dir_input = gr.Textbox(
                label="Music Folder",
                value=DEFAULT_MUSIC_DIR,
                info="Pick audio directly from this folder",
            )
            with gr.Row():
                music_file_input = gr.Dropdown(
                    label="Music Library",
                    choices=INITIAL_MUSIC_CHOICES,
                    value=INITIAL_MUSIC_CHOICES[0] if INITIAL_MUSIC_CHOICES else None,
                )
                refresh_music_btn = gr.Button("Refresh")

            with gr.Row():
                start_time_input = gr.Textbox(label="Start", value="00:00")
                duration_input = gr.Number(label="Duration (s)", value=15, precision=1)

            with gr.Row():
                resolution_input = gr.Dropdown(
                    label="Resolution",
                    choices=list(RESOLUTION_PRESETS.keys()),
                    value="1080p Square (1080×1080)",
                )
                fps_input = gr.Dropdown(
                    label="FPS",
                    choices=list(FPS_PRESETS.keys()),
                    value="60 fps (Interpolated)",
                )

            with gr.Row():
                noise_intensity_input = gr.Slider(label="Noise", minimum=0.0,
                                                  maximum=0.2, value=0.07, step=0.01)
                step_print_input = gr.Slider(
                    label="Step Print", minimum=0.0, maximum=1.0,
                    value=0.5, step=0.05,
                    info="Low shutter / motion trail effect (0=off, 1=max)"
                )

            crop_mode_input = gr.Dropdown(
                label="Crop Mode",
                choices=["Center", "Person/Car (YOLO)", "Content-Aware"],
                value="Person/Car (YOLO)",
                info="Smart tracking for person/car, or content-aware (faster)",
            )

            speed_curve_input = gr.Dropdown(
                label="Speed Curve",
                choices=list(SPEED_CURVES.keys()),
                value="⏩ Fast → Slow → Fast",
                info="Controls playback speed over the clip duration",
            )

            with gr.Row():
                generate_btn = gr.Button("▶ Generate", variant="primary",
                                         elem_id="generate_btn")
                aesthetic_btn = gr.Button("✨ Aesthetic Slow-Mo", elem_id="auto_calc_btn")
                cancel_btn = gr.Button("⏹ Stop", elem_id="cancel_btn")

        with gr.Column(scale=2):
            video_output = gr.Video(label=None)
            status_output = gr.Markdown("### 🕒 Waiting for Input...")

    def apply_aesthetic_preset():
        return [
            "1080p Square (1080×1080)",
            "60 fps (Interpolated)",
            0.08,
            0.55,
            "⏩ Fast → Slow → Fast",
            "Person/Car (YOLO)"
        ]

    aesthetic_btn.click(
        fn=apply_aesthetic_preset,
        outputs=[resolution_input, fps_input, noise_intensity_input, 
                 step_print_input, speed_curve_input, crop_mode_input]
    )

    # Generate
    gen_event = generate_btn.click(
        fn=run_generator,
        inputs=[video_input, audio_input, start_time_input, duration_input,
                noise_intensity_input, step_print_input, speed_curve_input,
                resolution_input, fps_input, music_file_input, crop_mode_input],
        outputs=[video_output, status_output],
    )

    refresh_music_btn.click(
        fn=refresh_music_choices,
        inputs=[music_dir_input],
        outputs=[music_file_input],
    )

    # Cancel
    cancel_btn.click(fn=stop_generation, outputs=[status_output],
                     cancels=[gen_event])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=custom_css,
        theme=gr.themes.Base(primary_hue="orange", neutral_hue="neutral").set(
            body_background_fill="#000000",
            block_background_fill="#000000",
            block_border_color="#333333",
            input_background_fill="#1a1a1a",
            button_primary_background_fill="#f97316",
        ),
    )
