# 🎙️ WhisperX Gradio Interface 🎧

[![Python 3.10](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Gradio](https://img.shields.io/badge/Gradio-5.0%2B-orange.svg)](https://gradio.app/)
[![WhisperX](https://img.shields.io/badge/WhisperX-Latest-green.svg)](https://github.com/m-bain/whisperx)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A user-friendly Gradio interface for [WhisperX](https://github.com/m-bain/whisperx) that extends OpenAI's Whisper model with word-level timestamps through phoneme-level alignment and speaker diarization.

## ✨ Features

- 🎯 **Word-Level Timestamps**: Precise timing for each word in transcriptions
- 👥 **Speaker Diarization**: Identify who said what in multi-speaker audio
- 🌍 **Multilingual Support**: Transcribe and translate in numerous languages
- 🔊 **Voice Activity Detection**: Advanced speech detection with PyAnnote or Silero
- ⚙️ **Customizable Models**: Choose from tiny to large Whisper models
- 📄 **Multiple Output Formats**: SRT, VTT, TXT, and JSON outputs
- 🎚️ **Advanced Controls**: Fine-tune transcription parameters for best results
- 🌐 **Web Interface**: Intuitive UI powered by Gradio

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (recommended)
- CUDA-compatible GPU (for optimal performance)
- `ffmpeg` installed on your system

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/c3-whisperx-gradio.git
   cd c3-whisperx-gradio
   ```

2. Install WhisperX:
   ```bash
   pip install git+https://github.com/m-bain/whisperx.git
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:7860`

## 🧩 Usage Examples

### Basic Transcription

For quick transcription with default settings:
1. Upload an audio file
2. Select model size (usually "small" or "medium" is sufficient)
3. Click "Transcribe"

### Multilingual Audio

For non-English audio:
1. Upload audio file
2. Set language code (e.g., "fr" for French, "de" for German)
3. Click "Transcribe"

### Speaker Diarization

To identify speakers in a conversation:
1. Upload audio file
2. Expand "Diarization Options"
3. Check "Enable Speaker Diarization"
4. Optionally set min/max speaker count
5. Click "Transcribe"

## 🐳 Docker Support

### Using Pre-built Image

Pull and run the Docker image:

```bash
# Pull the image
docker pull ghcr.io/yourusername/c3-whisperx-gradio:latest

# Run the container with your Hugging Face token
docker run -p 7860:7860 --gpus all -e HF_TOKEN=your_huggingface_token ghcr.io/yourusername/c3-whisperx-gradio
```

### Building Locally

Build and run the application using Docker:

```bash
# Build the image
docker build -t whisperx-gradio .

# Run the container with your Hugging Face token
docker run -p 7860:7860 --gpus all -e HF_TOKEN=your_huggingface_token whisperx-gradio
```

### About HF_TOKEN

The `HF_TOKEN` environment variable is required for speaker diarization with PyAnnote. You can obtain this token from your [Hugging Face account settings](https://huggingface.co/settings/tokens).

## ⚙️ Advanced Configuration

The interface offers extensive customization through expandable accordion sections:

### Basic Options

- **Whisper Model**: Select model size (`tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`)
- **Task**: Choose between `transcribe` (original language) and `translate` (to English)
- **Language Code**: Specify the audio language (e.g., "en", "fr", "de") or leave empty for auto-detection
- **Device**: Processing hardware (`cuda` for GPU, `cpu` for CPU)
- **Compute Type**: Numerical precision (`float16`, `float32`, `int8`)
- **Batch Size**: Number of audio segments processed in parallel (1-64)

### Alignment Options

- **Alignment Model**: Custom alignment model name/path or leave empty for default
- **Skip Alignment**: Disable phoneme-level alignment
- **Interpolation Method**: How to handle non-aligned words (`nearest`, `linear`, `ignore`)
- **Return Character Alignments**: Include character-level timing in output

### VAD Options (Voice Activity Detection)

- **VAD Method**: Speech detection method (`pyannote` or `silero`)
- **VAD Onset**: Threshold to start a new speech segment (0.0-1.0)
- **VAD Offset**: Threshold to end a speech segment (0.0-1.0)
- **Chunk Size**: Maximum duration in seconds for speech chunks (1-120s)

### Diarization Options

- **Enable Speaker Diarization**: Identify different speakers in the audio
- **Min Speakers**: Minimum number of speakers to detect (empty for auto-detection)
- **Max Speakers**: Maximum number of speakers to detect (empty for auto-detection)

### Decoding Options

- **Temperature**: Randomness in model predictions (0.0-2.0)
- **Best of**: Transcription candidates to generate when temperature > 0
- **Beam Size**: Number of beams for beam search decoding
- **Patience**: Beam search patience factor
- **Length Penalty**: Penalty for longer transcripts
- **Suppress Tokens**: Comma-separated list of token IDs to suppress
- **Suppress Numerals**: Avoid transcribing numbers
- **Condition on Previous Text**: Use previous segments as context
- **Initial Prompt**: Text provided as a prompt for the first window

### Format Options

- **Highlight Words**: Underline words as they're spoken (for SRT/VTT)
- **Max Line Width**: Maximum characters per line (empty for auto)
- **Max Line Count**: Maximum lines per segment (empty for auto)
- **Segment Resolution**: Segmentation method (`sentence` or `chunk`)

### Advanced Options

- **Repetition Penalty**: Penalize repetition in generated text
- **Compression Ratio Threshold**: Filter segments based on compression ratio
- **Log Probability Threshold**: Filter segments based on log probability
- **No Speech Threshold**: Confidence threshold for "no speech" detection
- **Hotwords**: Comma-separated list of words the model should prefer to detect

## 🔍 Implementation Details

This application is built on:

- **WhisperX**: For advanced speech recognition with word-level alignment
- **Whisper Models**: OpenAI's speech recognition models
- **PyAnnote**: For speaker diarization (requires HF token)
- **Gradio**: For the web interface

## ⚠️ Performance Considerations

- Use larger models (medium, large) for higher accuracy, smaller models for speed
- GPU acceleration is recommended for faster processing
- For long files, consider chunking or using VAD to segment the audio
- Speaker diarization may require more computational resources

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperx) by Max Bain and colleagues
- [OpenAI Whisper](https://github.com/openai/whisper) for the base speech recognition model
- [PyAnnote](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [Gradio](https://gradio.app/) for the web interface framework