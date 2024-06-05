import argparse
import json
import shutil
import subprocess
import sys
import time
from typing import Iterator
import os
import requests
import audioop
import numpy as np
import audioop
from pydub import AudioSegment
import io

server_url = os.getenv("SERVER_URL", "http://15.236.247.75:5000")
speaker_file_path = "french_speaker3.json"
output_file = "output_french.wav"
text = "My name is Yoann, dans les sombres catacombes de Paris. Exploitant leurs talents uniques, quelque chose d'étonnant se produisit. commença à s'effriter."


def convert_wav_chunk_to_ulaw_chunk(wav_chunk, sample_width=2): 
    # The sample_width parameter corresponds to the number of bytes used per sample, default is 2 for 16-bit audio
    
    if sample_width not in {1, 2, 4}:
        raise ValueError("sample_width must be 1, 2, or 4")
        
    # Convert the WAV audio chunk to u-Law encoding 
    try:
        ulaw_chunk = audioop.lin2ulaw(wav_chunk, sample_width) 
    except audioop.error as e:
        print(f"Error converting WAV chunk to u-Law: {e}")
        return None
    
    return ulaw_chunk


# Constants for µ-law encoding
MU = 255
MAX = 32768

def linear_to_ulaw(sample):
    """
    Convert a single 16-bit PCM sample to µ-law.
    """
    # Get the sign and the magnitude of the sample
    sign = np.sign(sample)
    magnitude = np.abs(sample)
    
    # Apply the µ-law algorithm
    magnitude = np.log1p(MU * magnitude / MAX) / np.log1p(MU)
    
    # Scale to 8-bit range and add the sign bit
    ulaw_sample = sign * (magnitude * 127 + 128)
    
    # Ensure the sample is in the range [0, 255]
    ulaw_sample = np.clip(ulaw_sample, 0, 255)
    
    return np.uint8(ulaw_sample)

def convert_wav_chunk_to_ulaw_chunk2(wav_chunk):
    """
    Convert a chunk of 16-bit PCM WAV data to µ-law encoded data.
    
    Parameters:
    wav_chunk (bytes): A bytes object containing 16-bit PCM samples.
    
    Returns:
    bytes: A bytes object containing µ-law encoded samples.
    """
    # Convert the byte data to a numpy array of 16-bit integers
    wav_samples = np.frombuffer(wav_chunk, dtype=np.int16)
    
    # Apply the µ-law conversion to each sample
    ulaw_samples = np.array([linear_to_ulaw(sample) for sample in wav_samples], dtype=np.uint8)
    
    # Convert the numpy array of µ-law samples back to bytes
    ulaw_chunk = ulaw_samples.tobytes()
    
    return ulaw_chunk



def is_installed(lib_name: str) -> bool:
    lib = shutil.which(lib_name)
    if lib is None:
        return False
    return True


def save(audio: bytes, filename: str) -> None:
    with open(filename, "wb") as f:
        f.write(audio)


import wave


def wav_to_ulaw(wav_file):
    segment = AudioSegment.from_wav(wav_file)
    segment = segment.set_frame_rate(8000)
    segment = segment.set_channels(1)
    segment = segment.set_sample_width(2)
    ulaw_audio = audioop.lin2ulaw(segment.raw_data, 2)
    return ulaw_audio

def save_wav_chunk_to_file(stream, filename):
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
    with open(filename, "wb") as f:
        header = [chunks[0]]
        ulaw_audio = wav_to_ulaw(io.BytesIO(b''.join(chunks)))
        f.write(ulaw_audio)
        #f.write(b''.join(header + chunks[1:200]))

    

def tts(text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
    start = time.perf_counter()
    speaker["text"] = text
    speaker["language"] = language
    speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
    res = requests.post(
        f"{server_url}/tts_stream",
        json=speaker,
        stream=True,
    )
    end = time.perf_counter()
    print(f"Time to make POST: {end-start}s", file=sys.stderr)

    if res.status_code != 200:
        print("Error:", res.text)
        sys.exit(1)

    first = True
    for chunk in res.iter_content(chunk_size=512):
        if first:
            end = time.perf_counter()
            print(f"Time to first chunk: {end-start}s", file=sys.stderr)
            first = False
        if chunk:
            yield chunk

    print("⏱️ response.elapsed:", res.elapsed)


def get_speaker(ref_audio,server_url):
    files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
    response = requests.post(f"{server_url}/clone_speaker", files=files)
    return response.json()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default=text,
        help="text input for TTS"
    )
    parser.add_argument(
        "--language",
        default="fr",
        help="Language to use default is 'en'  (English)"
    )
    parser.add_argument(
        "--output_file",
        default=output_file,
        help="Save TTS output to given filename"
    )
    parser.add_argument(
        "--ref_file",
        default=None,
        help="Reference audio file to use, when not given will use default"
    )
    parser.add_argument(
        "--server_url",
        default=server_url,
        help="Server url http://localhost:8000 default, change to your server location "
    )
    parser.add_argument(
        "--stream_chunk_size",
        default="30",
        help="Stream chunk size , 20 default, reducing will get faster latency but may degrade quality"
    )
    args = parser.parse_args()

    with open(speaker_file_path, "r") as file:
        speaker = json.load(file)

    if args.ref_file is not None:
        print("Computing the latents for a new reference...")
        speaker = get_speaker(args.ref_file, args.server_url)

    audio = save_wav_chunk_to_file(
        tts(
            args.text,
            speaker,
            args.language,
            args.server_url,
            args.stream_chunk_size
        ), "nejzj3.ulaw"
    )
