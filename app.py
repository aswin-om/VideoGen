import gradio as gr
import os
from generate import generate_beat_sync_video

def run_generator(video_files, audio_file, start_time, duration):
    if not video_files:
        return None
    if audio_file is None:
        return None
    
    video_paths = [v.name for v in video_files]
    audio_path = audio_file.name
    
    try:
        output_video = generate_beat_sync_video(
            video_files=video_paths,
            audio_file=audio_path,
            start_time_str=start_time,
            clip_duration=duration
        )
        return output_video
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Custom CSS for the requested styling
custom_css = """
body, .gradio-container {
    background-color: #262626 !important; /* Neutral gray, no pitch black */
    color: #a3a3a3 !important; /* Soft gray text */
}

/* All buttons except primary to be neutral gray */
.gr-button, .lg.primary, .secondary, button {
    background: #333333 !important;
    border: 1px solid #404040 !important;
    color: #a3a3a3 !important;
}

/* Only the primary generate button should be orange */
#generate_btn {
    background: #f97316 !important; 
    border: 1px solid #ea580c !important;
    color: white !important;
}

/* Minimal inputs */
.gr-input, .gr-box, input, textarea {
    background-color: #2d2d2d !important;
    border-color: #404040 !important;
    color: #d4d4d4 !important;
    border-radius: 4px !important;
}

/* Hide footer */
footer { display: none !important; }

/* Remove excessive padding/margins for minimal look */
.gap-4 { gap: 0.75rem !important; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base(
    primary_hue="orange",
    neutral_hue="neutral",
    spacing_size="sm",
    radius_size="md",
).set(
    body_background_fill="#262626",
    block_background_fill="#262626",
    block_border_color="#404040",
    input_background_fill="#2d2d2d",
    button_primary_background_fill="#f97316",
    button_secondary_background_fill="#333333",
)) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(label="Videos", file_count="multiple")
            audio_input = gr.File(label="Audio", file_types=["audio"])
            
            with gr.Row():
                start_time_input = gr.Textbox(label="Start", value="0:50")
                duration_input = gr.Number(label="Duration", value=15, precision=1)
            
            generate_btn = gr.Button("Generate Video", variant="primary", elem_id="generate_btn")
            
        with gr.Column(scale=2):
            video_output = gr.Video(label=None) # Cleaner video output

    generate_btn.click(
        fn=run_generator,
        inputs=[video_input, audio_input, start_time_input, duration_input],
        outputs=[video_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
