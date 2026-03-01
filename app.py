import gradio as gr
import os
from generate import generate_beat_sync_video

def run_generator(video_files, audio_file, start_time, duration, zoom, motion_speed):
    if not video_files: return None, "Please upload videos."
    if audio_file is None: return None, "Please upload an audio file."
    
    video_paths = [v.name for v in video_files]
    audio_path = audio_file.name
    
    try:
        output_video, stats = generate_beat_sync_video(
            video_files=video_paths,
            audio_file=audio_path,
            start_time_str=start_time,
            clip_duration=duration,
            zoom=zoom,
            motion_speed=motion_speed
        )
        
        # Format verbose stats
        status_msg = f"### ✅ Generation Complete\n"
        status_msg += f"- **Total Scenes Extracted:** {stats['total_scenes']}\n"
        status_msg += f"- **Frames Indexed (DINOv2):** {stats['total_frames_indexed']}\n\n"
        status_msg += "#### 📊 Video Usage (Screen Time):\n"
        
        # Sort by usage for better readability
        sorted_usage = sorted(stats['video_usage_seconds'].items(), key=lambda x: x[1], reverse=True)
        for v_name, seconds in sorted_usage:
            if seconds > 0:
                status_msg += f"- `{v_name}`: {seconds}s\n"
            else:
                status_msg += f"- `{v_name}`: Not used (penalty too high/no matches)\n"
                
        return output_video, status_msg
    except Exception as e:
        return None, f"Error: {str(e)}"

custom_css = """
body, .gradio-container { background-color: #000000 !important; color: #a3a3a3 !important; }
.gr-button, button { background: #333333 !important; border: 1px solid #404040 !important; color: #a3a3a3 !important; }
#generate_btn { background: #f97316 !important; border: 1px solid #ea580c !important; color: white !important; }
.gr-input, .gr-box, input, textarea, select { background-color: #1a1a1a !important; border-color: #333333 !important; color: #d4d4d4 !important; border-radius: 4px !important; }
footer { display: none !important; }
"""

with gr.Blocks() as demo:
    gr.Markdown("# 🎥 VideoGen M4 (Semantic Match Cuts)")
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(label="Videos", file_count="multiple")
            audio_input = gr.File(label="Audio", file_types=["audio"])
            
            with gr.Row():
                start_time_input = gr.Textbox(label="Start", value="00:00")
                duration_input = gr.Number(label="Duration", value=15, precision=1)
            
            with gr.Row():
                zoom_input = gr.Slider(label="Stabilization Zoom", minimum=1.0, maximum=1.5, value=1.0, step=0.01)
                speed_input = gr.Slider(label="Motion Speed", minimum=0.05, maximum=2.0, value=0.25, step=0.05)
            
            generate_btn = gr.Button("Generate Video", variant="primary", elem_id="generate_btn")
            
        with gr.Column(scale=2):
            video_output = gr.Video(label=None)
            status_output = gr.Markdown("### 🕒 Waiting for Input...")

    generate_btn.click(
        fn=run_generator,
        inputs=[video_input, audio_input, start_time_input, duration_input, zoom_input, speed_input],
        outputs=[video_output, status_output]
    )

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
        )
    )
