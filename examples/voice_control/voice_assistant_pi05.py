#!/usr/bin/env python
"""
Voice Assistant with Pi0.5: Microphone → STT → Pi0.5 → TTS → Speaker

This example demonstrates how to use Pi0.5 as a conversational robot assistant:
1. Hold SPACE to record your voice command
2. Speech-to-text (Whisper) converts speech to text
3. Text is fed as a high-level prompt to Pi0.5
4. Pi0.5 generates a response (robot utterance)
5. Text-to-speech (Kokoro) speaks the response back

Requirements:
    pip install torch transformers sounddevice numpy pynput kokoro>=0.9.2

Usage:
    python examples/voice_assistant/voice_assistant_pi05.py \
        --pretrained_path lerobot/pi0.5-base
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import re
import subprocess
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from pynput import keyboard
from transformers import AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

SAMPLE_RATE = 16000


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Pi05VoiceAssistant:
    """Voice assistant using Pi0.5 for generating robot utterances."""

    def __init__(
        self,
        pretrained_path: str | None = None,
        max_response_tokens: int = 100,
        max_record_seconds: float = 30.0,
    ):
        self.device = get_device()
        self.dtype = torch.float32 if self.device.type == "mps" else torch.bfloat16
        self.max_response_tokens = max_response_tokens
        self.max_record_seconds = max_record_seconds

        # Push-to-talk state
        self._recording = False
        self._audio_chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

        print(f"Using device: {self.device}")
        self._load_models(pretrained_path)

    def _load_models(self, pretrained_path: str | None):
        print("Loading STT (Whisper tiny)...")
        self.stt_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        self.stt_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny.en", torch_dtype=self.dtype
        ).to(self.device)

        print("Loading Pi0.5 model...")
        self._load_pi05(pretrained_path)

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        self._load_tts()
        print("Ready!\n")

    def _load_pi05(self, pretrained_path: str | None):
        """Load Pi0.5 model for utterance generation."""
        config = PI05Config()
        config.dtype = "float32" if self.device.type == "mps" else "bfloat16"

        self.pi05_model = PI05Pytorch(config)

        if pretrained_path:
            try:
                from safetensors.torch import load_file
                state_dict = load_file(f"{pretrained_path}/model.safetensors")
                self.pi05_model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded Pi0.5 weights from {pretrained_path}")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
                print("Using randomly initialized model for demo purposes")

        self.pi05_model = self.pi05_model.to(self.device)
        self.pi05_model.eval()

    def _load_tts(self):
        try:
            print("Loading TTS (Kokoro 82M)...")
            from kokoro import KPipeline

            self.tts_pipeline = KPipeline(lang_code="a")  # American English
            self.tts_voice = "af_heart"
            self.tts_type = "kokoro"
            print("Kokoro loaded!")
        except Exception as e:
            print(f"Kokoro not available ({e})")
            print("Using macOS `say` for TTS")
            self.tts_pipeline = None
            self.tts_type = "system"

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - collects chunks while recording."""
        if self._recording:
            self._audio_chunks.append(indata.copy())

    def _start_recording(self):
        """Start recording audio."""
        if self._recording:
            return
        self._recording = True
        self._audio_chunks = []
        print("🎤 Recording... (release SPACE to stop)")

    def _stop_recording(self) -> np.ndarray | None:
        """Stop recording and return the audio."""
        if not self._recording:
            return None
        self._recording = False

        if not self._audio_chunks:
            return None

        audio = np.concatenate(self._audio_chunks, axis=0).flatten()
        duration = len(audio) / SAMPLE_RATE
        volume = np.abs(audio).max()
        print(f"Recorded {duration:.1f}s, volume: {volume:.4f}")

        if volume < 0.001:
            print("⚠️  Very low audio - check microphone permissions!")
            return None

        return audio

    def wait_for_spacebar(self) -> np.ndarray | None:
        """Wait for spacebar press, record while held, return audio on release."""
        audio_result = None
        recording_done = threading.Event()

        def on_press(key):
            if key == keyboard.Key.space:
                self._start_recording()

        def on_release(key):
            nonlocal audio_result
            if key == keyboard.Key.space and self._recording:
                audio_result = self._stop_recording()
                recording_done.set()
                return False  # Stop listener

        # Start audio stream
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
        )

        with self._stream:
            print("\n⏳ Press and hold SPACE to speak...")
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                # Wait for recording to complete or timeout
                recording_done.wait(timeout=self.max_record_seconds)
                if self._recording:
                    audio_result = self._stop_recording()

        return audio_result

    def transcribe(self, audio: np.ndarray) -> str:
        start = time.perf_counter()
        inputs = self.stt_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)
        tokens = self.stt_model.generate(input_features)
        text = self.stt_processor.batch_decode(tokens, skip_special_tokens=True)[0]
        print(f"STT: {time.perf_counter() - start:.2f}s")
        return text.strip()

    def _create_dummy_images(self, batch_size: int = 1) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Create placeholder images for Pi0.5 when no camera is available."""
        image_shape = (batch_size, 3, 224, 224)
        dummy_image = torch.zeros(image_shape, dtype=torch.float32, device=self.device)
        dummy_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        return [dummy_image], [dummy_mask]

    def _tokenize_prompt(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the user prompt for Pi0.5."""
        prompt = f"User request: {text}\nRobot response:"
        tokenized = self.tokenizer(
            [prompt],
            max_length=200,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = tokenized["input_ids"].to(self.device)
        masks = tokenized["attention_mask"].to(self.device, dtype=torch.bool)
        return tokens, masks

    def generate_response(self, user_text: str) -> str:
        """Generate robot utterance using Pi0.5's language generation."""
        start = time.perf_counter()

        images, img_masks = self._create_dummy_images()
        tokens, masks = self._tokenize_prompt(user_text)

        with torch.no_grad():
            generated_tokens = self.pi05_model._generate_subtask_tokens(
                images=images,
                img_masks=img_masks,
                tokens=tokens,
                masks=masks,
                tokenizer=self.tokenizer,
                max_length=self.max_response_tokens,
                device=self.device,
            )

        # Decode generated tokens
        valid_tokens = generated_tokens[0][generated_tokens[0] != 0]
        response = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)

        # Extract utterance if marked with special tokens
        response = self._extract_utterance(response)

        print(f"Pi0.5: {time.perf_counter() - start:.2f}s")
        return response.strip()

    def _extract_utterance(self, text: str) -> str:
        """Extract utterance from between <utterance> tokens if present."""
        pattern = r"<utterance>(.*?)</utterance>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def speak(self, text: str):
        start = time.perf_counter()
        if self.tts_type == "kokoro":
            generator = self.tts_pipeline(text, voice=self.tts_voice)
            audio_chunks = [audio for _, _, audio in generator]
            if audio_chunks:
                audio = np.concatenate(audio_chunks)
                sd.play(audio, 24000)
                sd.wait()
        else:
            subprocess.run(["say", text], check=True)
        print(f"TTS: {time.perf_counter() - start:.2f}s")

    def run(self):
        print("=" * 50)
        print("Pi0.5 Voice Assistant")
        print("=" * 50)
        print("• Hold SPACE to record your voice command")
        print("• Release SPACE when done speaking")
        print("• Press Ctrl+C to exit")
        print("=" * 50)

        while True:
            try:
                audio = self.wait_for_spacebar()

                if audio is None:
                    print("(no audio captured)\n")
                    continue

                user_text = self.transcribe(audio)

                if not user_text:
                    print("(no speech detected)\n")
                    continue

                print(f"You: {user_text}")

                response = self.generate_response(user_text)
                print(f"Robot: {response}\n")

                self.speak(response)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


def main():
    parser = argparse.ArgumentParser(description="Pi0.5 Voice Assistant")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained Pi0.5 model (optional)",
    )
    parser.add_argument(
        "--max_response_tokens",
        type=int,
        default=100,
        help="Maximum tokens in generated response",
    )
    parser.add_argument(
        "--max_record_seconds",
        type=float,
        default=30.0,
        help="Maximum recording duration in seconds",
    )
    args = parser.parse_args()

    assistant = Pi05VoiceAssistant(
        pretrained_path=args.pretrained_path,
        max_response_tokens=args.max_response_tokens,
        max_record_seconds=args.max_record_seconds,
    )
    assistant.run()


if __name__ == "__main__":
    main()
