#!/usr/bin/env python3
"""
Play a WAV file through the G1 robot's speaker.

Requirements:
    - WAV file must be 16kHz, mono, 16-bit PCM
    
Usage:
    python test_speaker_wav.py en7 path/to/audio.wav
"""

import sys
import time
import struct
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


def read_wav(filename):
    """Read WAV file and return PCM data."""
    try:
        with open(filename, 'rb') as f:
            def read(fmt):
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

            # Read RIFF header
            chunk_id, = read('<I')
            if chunk_id != 0x46464952:  # "RIFF"
                print(f"[ERROR] Not a valid WAV file (invalid RIFF header)")
                return [], -1, -1, False

            _chunk_size, = read('<I')
            format_tag, = read('<I')
            if format_tag != 0x45564157:  # "WAVE"
                print(f"[ERROR] Not a valid WAV file (invalid WAVE format)")
                return [], -1, -1, False

            # Read fmt chunk
            subchunk1_id, = read('<I')
            subchunk1_size, = read('<I')

            # Skip JUNK chunk if present
            if subchunk1_id == 0x4B4E554A:  # "JUNK"
                f.seek(subchunk1_size, 1)
                subchunk1_id, = read('<I')
                subchunk1_size, = read('<I')

            if subchunk1_id != 0x20746D66:  # "fmt "
                print(f"[ERROR] Invalid fmt chunk")
                return [], -1, -1, False

            if subchunk1_size not in [16, 18]:
                print(f"[ERROR] Unsupported fmt chunk size: {subchunk1_size}")
                return [], -1, -1, False

            audio_format, = read('<H')
            if audio_format != 1:
                print(f"[ERROR] Only PCM format supported, got format {audio_format}")
                return [], -1, -1, False

            num_channels, = read('<H')
            sample_rate, = read('<I')
            _byte_rate, = read('<I')
            _block_align, = read('<H')
            bits_per_sample, = read('<H')

            if bits_per_sample != 16:
                print(f"[ERROR] Only 16-bit samples supported, got {bits_per_sample}-bit")
                return [], -1, -1, False

            if subchunk1_size == 18:
                extra_size, = read('<H')
                if extra_size != 0:
                    f.seek(extra_size, 1)

            # Find data chunk
            while True:
                subchunk2_id, subchunk2_size = read('<II')
                if subchunk2_id == 0x61746164:  # "data"
                    break
                f.seek(subchunk2_size, 1)

            # Read PCM data
            raw_pcm = f.read(subchunk2_size)
            if len(raw_pcm) != subchunk2_size:
                print("[ERROR] Failed to read full PCM data")
                return [], -1, -1, False

            return list(raw_pcm), sample_rate, num_channels, True

    except Exception as e:
        print(f"[ERROR] Failed to read WAV file: {e}")
        return [], -1, -1, False


def play_pcm_stream(client, pcm_list, app_name="example", chunk_size=96000):
    """
    Play PCM audio in chunks.
    
    Args:
        client: AudioClient instance
        pcm_list: List of PCM bytes
        app_name: Application name for this audio stream
        chunk_size: Bytes per chunk (96000 = ~3 seconds at 16kHz)
    """
    pcm_data = bytes(pcm_list)
    stream_id = str(int(time.time() * 1000))
    offset = 0
    chunk_index = 0
    total_size = len(pcm_data)

    print(f"Playing audio: {total_size} bytes in {(total_size // chunk_size) + 1} chunks")

    while offset < total_size:
        remaining = total_size - offset
        current_chunk_size = min(chunk_size, remaining)
        chunk = pcm_data[offset:offset + current_chunk_size]

        # Send chunk
        ret_code, _ = client.PlayStream(app_name, stream_id, chunk)
        if ret_code != 0:
            print(f"[ERROR] Failed to send chunk {chunk_index}, return code: {ret_code}")
            break
        else:
            print(f"[INFO] Sent chunk {chunk_index}/{(total_size // chunk_size)}")

        offset += current_chunk_size
        chunk_index += 1
        time.sleep(1.0)  # Wait between chunks


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <network_interface> <wav_file>")
        print("Example: python3 test_speaker_wav.py en7 audio.wav")
        print("\nWAV file requirements:")
        print("  - Sample rate: 16000 Hz")
        print("  - Channels: 1 (mono)")
        print("  - Bit depth: 16-bit")
        sys.exit(1)

    network_interface = sys.argv[1]
    wav_path = sys.argv[2]

    # Initialize communication
    print(f"Initializing on {network_interface}...")
    ChannelFactoryInitialize(0)

    # Create audio client
    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()
    print("Audio client initialized!")

    # Read WAV file
    print(f"Reading WAV file: {wav_path}")
    pcm_list, sample_rate, num_channels, is_ok = read_wav(wav_path)
    
    if not is_ok:
        print("[ERROR] Failed to read WAV file")
        sys.exit(1)

    print(f"WAV info:")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Channels: {num_channels}")
    print(f"  - Size: {len(pcm_list)} bytes")

    # Verify format
    if sample_rate != 16000:
        print(f"[ERROR] Sample rate must be 16000 Hz, got {sample_rate} Hz")
        print("Use ffmpeg to convert:")
        print(f"  ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav")
        sys.exit(1)

    if num_channels != 1:
        print(f"[ERROR] Must be mono (1 channel), got {num_channels} channels")
        print("Use ffmpeg to convert:")
        print(f"  ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav")
        sys.exit(1)

    # Play audio
    print("Playing audio...")
    play_pcm_stream(audio_client, pcm_list, "test_wav")
    
    # Wait for playback to finish
    duration_seconds = len(pcm_list) / (16000 * 2)  # 16kHz, 16-bit (2 bytes)
    print(f"Waiting {duration_seconds:.1f} seconds for playback...")
    time.sleep(duration_seconds + 1)

    # Stop playback
    print("Stopping playback...")
    audio_client.PlayStop("test_wav")
    
    print("Done!")


if __name__ == "__main__":
    main()

