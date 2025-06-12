import os
import argparse
import json
import torch
import gradio as gr
import whisperx
import subprocess
from pathlib import Path


# Constants and defaults
DEFAULT_MODEL = "large-v3"
OUTPUT_FORMAT_CHOICES = ["all", "txt", "html", "srt", "vtt", "tsv", "json", "aud"]
COMPUTE_TYPE_CHOICES = ["float16", "float32", "int8"]
INTERPOLATE_METHOD_CHOICES = ["nearest", "linear", "ignore"]
VAD_METHOD_CHOICES = ["pyannote", "silero"]
SEGMENT_RESOLUTION_CHOICES = ["sentence", "chunk"]
TASK_CHOICES = ["transcribe", "translate"]

# Determine available device for inference
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INTRO_TEXT = """\
# WhisperX Gradio Interface

Transcribe audio files with word-level timestamps using WhisperX.

WhisperX adds word-level timestamps to Whisper's transcriptions through phoneme-level
alignment with a fine-tuned speech recognition model. It also supports speaker diarization.

### Notes
- The transcription process may take a while depending on the model size and audio length
- For diarization to work, you may need to set the `HF_TOKEN` environment variable
- For custom alignment models, provide the model name or path
"""

class WhisperXManager:
    """Simple class to manage WhisperX models loading and caching"""
    def __init__(self):
        self.asr_model = None
        self.asr_options = None
        self.model_name = None
        self.device = None
        self.compute_type = None


    def get_asr_model(self, model_name, device, compute_type):
        # Only reload if parameters have changed
        if self.asr_model is None or model_name != self.model_name or device != self.device or compute_type != self.compute_type:

            print(f"Loading Whisper {model_name} model on {device} with compute_type={compute_type}...")

            self.asr_model = whisperx.load_model(
                model_name,
                device=device,
                compute_type=compute_type
            )

            self.model_name = model_name
            self.device = device
            self.compute_type = compute_type

        return self.asr_model

def format_timestamp(seconds, always_include_hours=False, decimal_marker="."):
    """Format seconds into a timestamp string"""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )

def is_video_file(file_path):
    #TODO make this more robust; eg check streams using ffprobe
    try:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']  # Add other video extensions if needed
        file_extension = os.path.splitext(file_path)[1].lower()
        is_video = file_extension in video_extensions
        return is_video
    except Exception as e:
        print(f"Error checking if file is a video: {e}")
        return False

def extract_audio_from_video(video_file, output_audio_file):
    try:
        print(f"Extracting audio from video file: {video_file}...")
        command = ['ffmpeg', '-i', video_file, '-q:a', '0', '-map', 'a', output_audio_file, '-y']
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio extracted to: {output_audio_file}")
    except Exception as e:
        print(f"Error extracting audio from video: {e}")

    print(f"Audio extracted from video file: {video_file} and saved at: {output_audio_file}")

def transcribe_audio(
    # Input file
    #audio_file,
    input_file,

    # Basic options
    model="large-v3",
    task="transcribe",
    language="en",
    device="cuda",
    compute_type="float16",
    batch_size=16,

    # Alignment options
    align_model=None,
    no_align=False,
    interpolate_method="nearest",
    return_char_alignments=False,

    # VAD options
    vad_method="pyannote",
    vad_onset=0.500,
    vad_offset=0.363,
    chunk_size=30,

    # Diarization options
    diarize=False,
    min_speakers=None,
    max_speakers=None,

    # Decoding options
    temperature=0,
    best_of=5,
    beam_size=5,
    patience=1.0,
    length_penalty=1.0,
    suppress_tokens="-1",
    suppress_numerals=False,
    condition_on_previous_text=True,
    initial_prompt=None,
    word_timestamps=True,

    # Format options
    highlight_words=False,
    max_line_width=None,
    max_line_count=None,
    segment_resolution="sentence",

    # Advanced options
    repetition_penalty=1.0,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    hotwords=None
):
    # Create a permanent output directory
    output_dir = Path('whisperx_output')
    output_dir.mkdir(exist_ok=True)

    # Generate a unique timestamp for this run
    import time
    timestamp = int(time.time())
    file_prefix = f"transcript_{timestamp}"

    try:
        # for video input file, extract as audio
        if is_video_file(input_file):
            audio_file = os.path.splitext(input_file)[0] + ".mp3"
            extract_audio_from_video(input_file, audio_file)
            file_path = audio_file
        else:
            audio_file = input_file
            
        # Handle both string paths and FileData objects
        if isinstance(audio_file, dict) and 'path' in audio_file:
            audio_path = audio_file['path']  # Extract path from FileData
        else:
            audio_path = audio_file  # Already a string path

        # Get model manager
        model_manager = MODEL_MANAGER.get_asr_model(model, device, compute_type)

        # Parse advanced options
        if min_speakers:
            min_speakers = int(min_speakers)
        if max_speakers:
            max_speakers = int(max_speakers)

        # Parse hotwords if provided
        hotwords_list = None
        if hotwords:
            try:
                hotwords_list = [hw.strip() for hw in hotwords.split(",")]
            except:
                print("Error parsing hotwords, using default")

        # Create empty dict for HF token if diarization is enabled
        hf_token = None  # Can be set via environment variable HF_TOKEN if needed

        try:
            # Step 1: Transcribe with Whisper
            print("Transcribing audio...")

            # FasterWhisperPipeline API has different parameters
            # Only pass the parameters that are available in the current version
            result = model_manager.transcribe(
                audio_path,
                language=language,
                task=task,
                batch_size=batch_size,
                chunk_size=chunk_size
            )

            # Step 2: Perform alignment if enabled
            if not no_align:
                print("Aligning results...")

                # Make sure we have a valid language code
                detected_language = result.get("language", "en")  # Default to English if not detected
                if language and language.strip():
                    detected_language = language.strip()

                print(f"Using language code: {detected_language} for alignment")

                # First load the alignment model - don't pass model_name if it's empty
                # This will make it use the default model for the detected language
                if align_model and align_model.strip():
                    alignment_model, align_metadata = whisperx.load_align_model(
                        language_code=detected_language,
                        device=device,
                        model_name=align_model
                    )
                else:
                    alignment_model, align_metadata = whisperx.load_align_model(
                        language_code=detected_language,
                        device=device
                    )

                # Then perform alignment
                result = whisperx.align(
                    result["segments"],
                    alignment_model,
                    align_metadata,
                    audio_path,
                    device,
                    return_char_alignments=return_char_alignments,
                    interpolate_method=interpolate_method
                )

            # Step 3: Perform diarization if enabled
            if diarize:
                print("Performing diarization...")
                diarize_model = whisperx.diarize.DiarizationPipeline(
                    device=device,
                    use_auth_token=hf_token
                )
                diarize_segments = diarize_model(
                    audio_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)

            # Format and save results
            print("Processing complete.")

            # Save outputs in multiple formats
            output_files = []

            # 1. JSON format
            json_path = output_dir / f"{file_prefix}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            output_files.append(str(json_path))

            # 2. Plain text format
            txt_path = output_dir / f"{file_prefix}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                oldspeaker = ''
                for segment in result["segments"]:
                    if diarize and "speaker" in segment:
                        # if still the same speaker, just print segment
                        if segment['speaker'] == oldspeaker:
                            f.write(f"{segment['text'].strip()}\n")
                            continue
                        # if new speaker, print speaker label
                        f.write("\n")
                        f.write(f"[{segment['speaker']}]: {segment['text'].strip()}\n")
                        oldspeaker = segment['speaker']
                    else:
                        f.write(f"{segment['text'].strip()}\n")
            output_files.append(str(txt_path))

            # 3. SRT format (subtitle)
            srt_path = output_dir / f"{file_prefix}.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(result["segments"], start=1):
                    start_time = format_timestamp(segment["start"], always_include_hours=True, decimal_marker=",")
                    end_time = format_timestamp(segment["end"], always_include_hours=True, decimal_marker=",")

                    text = segment["text"].strip().replace("-->", "->")
                    if diarize and "speaker" in segment:
                        text = f"[{segment['speaker']}]: {text}"

                    f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
            output_files.append(str(srt_path))

            # 4. VTT format (web subtitle)
            vtt_path = output_dir / f"{file_prefix}.vtt"
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for i, segment in enumerate(result["segments"], start=1):
                    start_time = format_timestamp(segment["start"], always_include_hours=False, decimal_marker=".")
                    end_time = format_timestamp(segment["end"], always_include_hours=False, decimal_marker=".")

                    text = segment["text"].strip().replace("-->", "->")
                    if diarize and "speaker" in segment:
                        text = f"[{segment['speaker']}]: {text}"

                    f.write(f"{start_time} --> {end_time}\n{text}\n\n")
            output_files.append(str(vtt_path))

            # 5. HTML
            html_path = output_dir / f"{file_prefix}.html"
            css = """
              body {
                color-scheme: light dark;
                color: #ffffffde;
                background-color: #242424;
              }
              
              span.timestamp {
                font-family: monospace;
              }
              
              div.segments {
                width: 90%;
              }
              
              p.segment {
                display: flex;
                align-items: center;
                gap: 1rem;
                text-align: left;
              }
              
              .speaker.speakerid {
                font-family: monospace;
              }
              
              .speaker.SPEAKER_00 {
                color: #59ffa1;
              }
              
              .speaker.SPEAKER_01 {
                color: #597aff;
              }
              
              .speaker.SPEAKER_02 {
                color: #f04f4f;
              }
              
              .speaker.SPEAKER_03 {
                color: #e29b16;
              }
              
              .speaker.SPEAKER_04 {
                color: #16c5e2;
              }
              
              .speaker.SPEAKER_05 {
                color: #8316e2;
              }
              
              .speaker.SPEAKER_06 {
                color: #e216af;;
              }
              
              .speaker.SPEAKER_07 {
                color: #16e29b;
              }
              
              .speaker.SPEAKER_08 {
                color: #3fe216;
              }
              
              .speaker.SPEAKER_09 {
                color: #9b16e2;
              }
              
              .speaker.SPEAKER_10 {
                color: #16bae2;
              }
              
              .speaker.SPEAKER_11 {
                color: #a9e216;
              }
              
              .speaker.SPEAKER_12 {
                color: #1676e2;
              }
              
              .speaker.SPEAKER_13 {
                color: #dcbeff;
              }
              
              .speaker.SPEAKER_14 {
                color: #aaffc3;
              }
              
              .speaker.SPEAKER_15 {
                color: #ffd8b1;
              }
              
              .speaker.SPEAKER_16 {
                color: #fabed4;
              }
              
              .speaker.SPEAKER_17 {
                color: #16e287;
              }
              
              .speaker.SPEAKER_18 {
                color: #16bae2;
              }
              
              .speaker.SPEAKER_19 {
                color: #a9a9a9;
              }
            """
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("<html>\n")
                f.write(f"\t<head>\t\t<style>{css}\t\t</style>\t</head>")
                f.write(f"\t<body>\n\t\t<div class=\"segments\">")
                oldspeaker = ''
                for segment in result["segments"]:

                    if diarize and "speaker" in segment:
                        if segment['speaker'] != oldspeaker:
                          f.write(f"\t\t\t<br /><p class=\"speaker speakerid\">{segment['speaker']}:</p>\n")
                          oldspeaker = segment['speaker']
                        f.write(f"\t\t\t<p class=\"segment speaker {segment['speaker']}\">{segment['text'].strip()}</p>\n")
                    else:
                        f.write(f"\t\t<p>{segment['text'].strip()}</p>\n")
                f.write("\t\t</div>\t</body>\n</html>\n")
            output_files.append(str(html_path))

            # Read the TXT file content for display (no timestamps)
            with open(txt_path, "r", encoding="utf-8") as f:
                transcript = f.read()

            # Return transcript and output files
            return transcript, output_files

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}", []

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", []

def gradio_app():
    with gr.Blocks() as app:
        gr.Markdown(INTRO_TEXT)

        with gr.Column():
            # Input file
            #audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            file_input = gr.File(label="Upload Audio/Video File")

            with gr.Accordion("Basic Options", open=True):
                model = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                    value=DEFAULT_MODEL,
                    label="Whisper Model"
                )
                task = gr.Dropdown(
                    choices=TASK_CHOICES,
                    value="transcribe",
                    label="Task"
                )
                language = gr.Textbox(
                    value="en",
                    placeholder="Auto-detect if empty",
                    label="Language Code (e.g., en, fr, de)"
                )

                device = gr.Dropdown(
                    choices=["cuda", "cpu"],
                    value=DEFAULT_DEVICE,
                    label="Device"
                )
                compute_type = gr.Dropdown(
                    choices=COMPUTE_TYPE_CHOICES,
                    value="float16",
                    label="Compute Type"
                )
                batch_size = gr.Number(
                    value=16,
                    minimum=1,
                    maximum=64,
                    step=1,
                    label="Batch Size"
                )

            with gr.Accordion("Alignment Options", open=False):
                align_model = gr.Textbox(
                    value="",
                    placeholder="Default alignment model if empty",
                    label="Alignment Model Name"
                )
                no_align = gr.Checkbox(
                    value=False,
                    label="Skip Alignment"
                )

                interpolate_method = gr.Dropdown(
                    choices=INTERPOLATE_METHOD_CHOICES,
                    value="nearest",
                    label="Interpolation Method"
                )
                return_char_alignments = gr.Checkbox(
                    value=False,
                    label="Return Character Alignments"
                )

            with gr.Accordion("VAD Options", open=False):
                vad_method = gr.Dropdown(
                    choices=VAD_METHOD_CHOICES,
                    value="pyannote",
                    label="VAD Method"
                )
                vad_onset = gr.Number(
                    value=0.500,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.001,
                    label="VAD Onset"
                )
                vad_offset = gr.Number(
                    value=0.363,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.001,
                    label="VAD Offset"
                )
                chunk_size = gr.Number(
                    value=30,
                    minimum=1,
                    maximum=120,
                    step=1,
                    label="Chunk Size (s)"
                )

            with gr.Accordion("Diarization Options", open=False):
                diarize = gr.Checkbox(
                    value=False,
                    label="Enable Speaker Diarization"
                )
                min_speakers = gr.Textbox(
                    value="",
                    placeholder="Auto-detect if empty",
                    label="Min Speakers"
                )
                max_speakers = gr.Textbox(
                    value="",
                    placeholder="Auto-detect if empty",
                    label="Max Speakers"
                )

            with gr.Accordion("Decoding Options", open=False):
                temperature = gr.Number(
                    value=0,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    label="Temperature"
                )
                best_of = gr.Number(
                    value=5,
                    minimum=1,
                    maximum=10,
                    step=1,
                    label="Best of"
                )
                beam_size = gr.Number(
                    value=5,
                    minimum=1,
                    maximum=10,
                    step=1,
                    label="Beam Size"
                )

                patience = gr.Number(
                    value=1.0,
                    minimum=0.0,
                    maximum=5.0,
                    step=0.1,
                    label="Patience"
                )
                length_penalty = gr.Number(
                    value=1.0,
                    minimum=0.0,
                    maximum=5.0,
                    step=0.1,
                    label="Length Penalty"
                )
                suppress_tokens = gr.Textbox(
                    value="-1",
                    label="Suppress Tokens"
                )

                suppress_numerals = gr.Checkbox(
                    value=False,
                    label="Suppress Numerals"
                )
                condition_on_previous_text = gr.Checkbox(
                    value=True,
                    label="Condition on Previous Text"
                )
                initial_prompt = gr.Textbox(
                    value="",
                    placeholder="Initial prompt if any",
                    label="Initial Prompt"
                )

            with gr.Accordion("Format Options", open=False):
                highlight_words = gr.Checkbox(
                    value=False,
                    label="Highlight Words"
                )
                max_line_width = gr.Textbox(
                    value="",
                    placeholder="Auto if empty",
                    label="Max Line Width"
                )
                max_line_count = gr.Textbox(
                    value="",
                    placeholder="Auto if empty",
                    label="Max Line Count"
                )
                segment_resolution = gr.Dropdown(
                    choices=SEGMENT_RESOLUTION_CHOICES,
                    value="sentence",
                    label="Segment Resolution"
                )

            with gr.Accordion("Advanced Options", open=False):
                repetition_penalty = gr.Number(
                    value=1.0,
                    minimum=0.0,
                    maximum=5.0,
                    step=0.1,
                    label="Repetition Penalty"
                )
                compression_ratio_threshold = gr.Number(
                    value=2.4,
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    label="Compression Ratio Threshold"
                )

                log_prob_threshold = gr.Number(
                    value=-1.0,
                    minimum=-10.0,
                    maximum=0.0,
                    step=0.1,
                    label="Log Probability Threshold"
                )
                no_speech_threshold = gr.Number(
                    value=0.6,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    label="No Speech Threshold"
                )
                hotwords = gr.Textbox(
                    value="",
                    placeholder="Comma-separated hotwords",
                    label="Hotwords"
                )

            # Transcribe button
            transcribe_btn = gr.Button("Transcribe", variant="primary")

            # Output
            transcript_output = gr.Textbox(
                label="Transcript",
                lines=20
            )

            output_files = gr.File(
                label="Download Transcription Files",
                file_count="multiple"
            )

            # Connect the transcribe button to the transcription function
            transcribe_btn.click(
                fn=transcribe_audio,
                inputs=[
                    # Input file
                    #audio_input,
                    file_input,

                    # Basic options
                    model, task, language, device, compute_type, batch_size,

                    # Alignment options
                    align_model, no_align, interpolate_method, return_char_alignments,

                    # VAD options
                    vad_method, vad_onset, vad_offset, chunk_size,

                    # Diarization options
                    diarize, min_speakers, max_speakers,

                    # Decoding options
                    temperature, best_of, beam_size, patience, length_penalty,
                    suppress_tokens, suppress_numerals, condition_on_previous_text,
                    initial_prompt, gr.Checkbox(value=True, visible=False),  # word_timestamps always True

                    # Format options
                    highlight_words, max_line_width, max_line_count, segment_resolution,

                    # Advanced options
                    repetition_penalty, compression_ratio_threshold,
                    log_prob_threshold, no_speech_threshold, hotwords
                ],
                outputs=[transcript_output, output_files]
            )

    return app

# Apply monkey patch to fix Gradio's URL handling behind a proxy
import gradio.route_utils
original_get_api_call_path = gradio.route_utils.get_api_call_path

def patched_get_api_call_path(request):
    try:
        return original_get_api_call_path(request)
    except ValueError:
        path = request.url.path
        if not path:
            path = "/"
        return f"{path}/api"

# Apply the patch
gradio.route_utils.get_api_call_path = patched_get_api_call_path

# Initialize model manager
MODEL_MANAGER = WhisperXManager()

app = gradio_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", type=str, default='0.0.0.0', \
                        help="The host or IP to bind to. If None, bind to localhost.") # None
    parser.add_argument("--server_port", type=int, default='7860', \
                        help="The port to bind to.") # 7860
    parser.add_argument("--root_path", type=str, default='/', \
                        help="Root path.")
    _args = parser.parse_args()
    app.queue()
    app.launch(server_name=_args.server_name, server_port=_args.server_port, root_path=_args.root_path)
