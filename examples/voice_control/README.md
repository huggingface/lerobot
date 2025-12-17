# Voice Assistant Examples

Voice-enabled robot assistant examples using speech-to-text (STT), and text-to-speech (TTS).

## Overview

These examples demonstrate how to build a voice interface for robot control:

1. **Hold SPACE** → Push-to-talk recording starts
2. **Release SPACE** → Recording stops
3. **STT (Whisper)** → Converts speech to text (high-level task prompt)
4. **Pi0.5** → Generates robot response/utterance
5. **TTS (Kokoro)** → Speaks the response back

## Requirements

```bash
pip install torch transformers sounddevice numpy pynput kokoro>=0.9.2
```

## Usage

### With Pi0.5 Model

```bash
python examples/voice_assistant/voice_assistant_pi05.py \
    --pretrained_path path/to/pi05/checkpoint
```

## How It Works

### Pi0.5 Voice Integration

Pi0.5 can generate robot utterances as part of its subtask prediction. The flow:

1. **High-level prompt**: User voice command is transcribed and formatted as a task prompt
2. **Subtask generation**: Pi0.5 autoregressively generates a response
3. **Utterance extraction**: If the response contains `<utterance>...</utterance>` tags, the content is extracted
4. **TTS output**: The response is spoken back to the user

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pretrained_path` | None | Path to Pi0.5 checkpoint |
| `--record_seconds` | 5.0 | Audio recording duration |
| `--max_response_tokens` | 100 | Max tokens in generated response |