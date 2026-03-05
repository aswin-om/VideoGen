import gradio as gr
import os
from generate import (
    generate_beat_sync_video, analyze_and_recommend_settings,
    RESOLUTION_PRESETS, FPS_PRESETS,
)
from render import cancel_generation


def auto_calculate_settings(video_files, audio_file, start_time, duration,
                            progress=gr.Progress()):
    if not video_files:
        return 1.0, 0.8, 1, 1, "❌ Please upload videos first."
    if audio_file is None:
        return 1.0, 0.8, 1, 1, "❌ Please upload an audio file first."

    video_paths = [v.name for v in video_files[:5]]
    audio_path = audio_file.name
    safe_duration = max(5, min(float(duration), 45.0))

    try:
        if progress:
            progress(0.0, desc="Analyzing audio...")
        recs = analyze_and_recommend_settings(video_paths, audio_path,
                                              start_time, safe_duration)
        msg = "### 🤖 AI Analysis Complete\n\n"
        msg += f"**Audio:** BPM={recs['audio_analysis']['bpm']}, "
        msg += f"Energy={recs['audio_analysis']['energy']}, "
        msg += f"Beats={recs['audio_analysis']['beat_strength']}, "
        msg += f"Tempo={recs['audio_analysis']['tempo_category']}\n\n"
        msg += f"**Video Motion:** {recs['avg_video_motion']}\n\n"
        msg += "**Reasoning:**\n"
        for r in recs['reasoning']:
            msg += f"- {r}\n"
        msg += "\n✅ Sliders updated!"

        return recs['zoom'], recs['motion_speed'], recs['step_repeat'], \
            recs['source_stride'], msg
    except Exception as e:
        return 1.0, 0.8, 1, 1, f"❌ Analysis error: {e}"


def run_generator(video_files, audio_file, start_time, duration, zoom,
                  motion_speed, step_repeat, source_stride, noise_intensity,
                  crop_mode, resolution, target_fps, progress=gr.Progress()):
    if not video_files:
        return None, "Please upload videos."
    if audio_file is None:
        return None, "Please upload an audio file."

    video_paths = [v.name for v in video_files[:5]]
    audio_path = audio_file.name
    safe_duration = max(5, min(float(duration), 45.0))

    # Map UI crop mode label to internal key
    crop_key = {"Center": "center", "Content Aware": "content",
                "Person Tracking (YOLO)": "person"}.get(crop_mode, "center")

    try:
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
            progress=progress,
        )

        w, h = RESOLUTION_PRESETS.get(resolution, (720, 720))
        fps_val = FPS_PRESETS.get(target_fps, 30)

        msg = f"### ✅ Generation Complete\n"
        msg += f"- **Output:** {w}×{h} @ {fps_val}fps\n"
        msg += f"- **Duration:** {safe_duration}s\n"
        msg += f"- **Settings:** repeat {int(step_repeat)}x, stride {int(source_stride)}\n"
        msg += f"- **Scenes:** {stats['total_scenes']} | "
        msg += f"**Frames Indexed:** {stats['total_frames_indexed']}\n\n"
        msg += "#### 📊 Video Usage:\n"

        sorted_usage = sorted(stats['video_usage_seconds'].items(),
                              key=lambda x: x[1], reverse=True)
        for name, secs in sorted_usage:
            msg += f"- `{name}`: {secs}s\n" if secs > 0 else \
                   f"- `{name}`: Not used\n"

        return output_video, msg

    except InterruptedError:
        return None, "### ⏹ Generation cancelled."
    except Exception as e:
        return None, f"Error: {e}"


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

            with gr.Row():
                start_time_input = gr.Textbox(label="Start", value="00:00")
                duration_input = gr.Number(label="Duration (s)", value=15, precision=1)

            with gr.Row():
                resolution_input = gr.Dropdown(
                    label="Resolution",
                    choices=list(RESOLUTION_PRESETS.keys()),
                    value="720p Square (720×720)",
                )
                fps_input = gr.Dropdown(
                    label="FPS",
                    choices=list(FPS_PRESETS.keys()),
                    value="30 fps (Default)",
                )

            with gr.Row():
                zoom_input = gr.Slider(label="Zoom", minimum=1.0, maximum=1.5,
                                       value=1.0, step=0.01)
                speed_input = gr.Slider(label="Motion Speed", minimum=0.05,
                                        maximum=2.0, value=0.8, step=0.05)

            with gr.Row():
                step_repeat_input = gr.Slider(label="Step Repeat", minimum=1,
                                              maximum=8, value=1, step=1)
                source_stride_input = gr.Slider(label="Source Stride", minimum=1,
                                                maximum=8, value=1, step=1)

            noise_intensity_input = gr.Slider(label="Noise", minimum=0.0,
                                              maximum=0.2, value=0.0, step=0.01)
            crop_mode_input = gr.Radio(
                label="Crop Mode",
                choices=["Center", "Content Aware", "Person Tracking (YOLO)"],
                value="Center",
            )

            with gr.Row():
                auto_calc_btn = gr.Button("🤖 Auto-Calculate",
                                          elem_id="auto_calc_btn")
                generate_btn = gr.Button("▶ Generate", variant="primary",
                                         elem_id="generate_btn")
                cancel_btn = gr.Button("⏹ Stop", elem_id="cancel_btn")

        with gr.Column(scale=2):
            video_output = gr.Video(label=None)
            status_output = gr.Markdown("### 🕒 Waiting for Input...")

    # Auto-calculate
    auto_calc_btn.click(
        fn=auto_calculate_settings,
        inputs=[video_input, audio_input, start_time_input, duration_input],
        outputs=[zoom_input, speed_input, step_repeat_input,
                 source_stride_input, status_output],
    )

    # Generate
    gen_event = generate_btn.click(
        fn=run_generator,
        inputs=[video_input, audio_input, start_time_input, duration_input,
                zoom_input, speed_input, step_repeat_input, source_stride_input,
                noise_intensity_input, crop_mode_input, resolution_input,
                fps_input],
        outputs=[video_output, status_output],
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
